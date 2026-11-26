"""Eğitilmiş CNN modelini test eder ve sınıflandırma raporu üretir."""

from argparse import ArgumentParser
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def main() -> None:
    parser = ArgumentParser(description="CNN modelini test veri setiyle değerlendirir.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    model_path = args.model or PROJECT_ROOT / config["output_dir"] / "best_model.pth"
    report_dir = args.report_dir or PROJECT_ROOT / "reports"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((int(config["image_size"]), int(config["image_size"]))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir / "test", transform=transform)
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    report = {
        "model": str(model_path), "data": str(data_dir / "test"), "classes": classes,
        "metrics": {"accuracy": accuracy_score(targets, predictions), "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0)},
        "classification_report": classification_report(targets, predictions, target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
