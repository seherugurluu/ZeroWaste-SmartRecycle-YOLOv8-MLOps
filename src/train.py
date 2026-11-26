"""ResNet50 transfer-learning eğitim akışı."""

from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    for key in ("data_dir", "output_dir"):
        value = config.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            config[key] = str(PROJECT_ROOT / value)
    return config


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_loaders(data_dir: Path, image_size: int, batch_size: int, workers: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    valid = datasets.ImageFolder(data_dir / "valid", transform=eval_transform)
    test = datasets.ImageFolder(data_dir / "test", transform=eval_transform)
    if (
        train.class_to_idx != valid.class_to_idx
        or train.class_to_idx != test.class_to_idx
    ):
        raise ValueError("train, valid ve test sınıf klasörleri aynı olmalıdır.")
    class_counts = torch.bincount(
        torch.tensor(train.targets), minlength=len(train.classes)
    )
    class_weights = 1.0 / class_counts.float().clamp_min(1)
    sample_weights = class_weights[torch.tensor(train.targets)]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )
    common = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train, batch_size=batch_size, sampler=sampler, **common),
        DataLoader(valid, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=batch_size, shuffle=False, **common),
        train.classes,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    parser = ArgumentParser(
        description="ResNet50 ile atık görüntüsü sınıflandırması eğitir."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    requested_device = config.get("device", "cuda")
    device = torch.device(
        requested_device
        if requested_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    train_loader, valid_loader, _, classes = make_loaders(
        Path(config["data_dir"]),
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["workers"]),
    )
    model = build_model(len(classes), bool(config.get("pretrained", True))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-4
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        valid_loss, valid_accuracy = run_epoch(model, valid_loader, criterion, device)
        print(
            f"epoch={epoch}/{config['epochs']} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} valid_loss={valid_loss:.4f} valid_acc={valid_accuracy:.4f}",
            flush=True,
        )
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(
                {"model": model.state_dict(), "classes": classes},
                output_dir / "best_model.pth",
            )
    print(f"En iyi model: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
