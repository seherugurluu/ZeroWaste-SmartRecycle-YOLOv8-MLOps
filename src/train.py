"""YOLOv8 model eğitimini yapılandırma dosyasından çalıştırır."""

from argparse import ArgumentParser
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(config_path: Path) -> dict:
    """Eğitim YAML dosyasını okur ve göreli yolları proje köküne göre çözer."""
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    for key in ("data", "model"):
        value = config.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            candidate = PROJECT_ROOT / value
            if candidate.exists():
                config[key] = str(candidate)

    project = config.get("project")
    if isinstance(project, str) and not Path(project).is_absolute():
        config["project"] = str(PROJECT_ROOT / project)

    return config


def parse_args():
    parser = ArgumentParser(description="YOLOv8 waste detection modelini eğitir.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Eğitim ayarlarının bulunduğu YAML dosyası.",
    )
    parser.add_argument(
        "--epochs", type=int, help="YAML ayarını geçici olarak geçersiz kılar."
    )
    parser.add_argument(
        "--batch", type=int, help="YAML ayarını geçici olarak geçersiz kılar."
    )
    parser.add_argument(
        "--imgsz", type=int, help="YAML ayarını geçici olarak geçersiz kılar."
    )
    parser.add_argument(
        "--fraction", type=float, help="Eğitim verisinin kullanılacak oranı."
    )
    parser.add_argument(
        "--device", help="Örneğin cpu veya 0; YAML ayarını geçersiz kılar."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for key in ("epochs", "batch", "imgsz", "fraction", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    model = YOLO(config.pop("model"))
    model.train(**config)


if __name__ == "__main__":
    main()
