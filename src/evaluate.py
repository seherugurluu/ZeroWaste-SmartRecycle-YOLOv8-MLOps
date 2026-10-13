"""Eğitilmiş YOLOv8 modelini değerlendirir ve rapor üretir."""

from argparse import ArgumentParser
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


def parse_args():
    parser = ArgumentParser(description="YOLOv8 modelini test verisiyle değerlendirir.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def _metric(metrics, name: str):
    value = getattr(metrics.box, name, None)
    return None if value is None else float(value)


def build_report(metrics, model_path: Path, data_path: Path, split: str) -> dict:
    """Ultralytics sonuçlarını JSON'a yazılabilir basit bir sözlüğe dönüştürür."""
    names = getattr(metrics, "names", {})
    per_class = {}
    class_maps = getattr(metrics.box, "maps", [])
    for class_id, value in enumerate(class_maps):
        class_name = (
            names.get(class_id, str(class_id))
            if isinstance(names, dict)
            else str(class_id)
        )
        per_class[class_name] = {"map50_95": float(value)}

    return {
        "model": str(model_path),
        "data": str(data_path),
        "split": split,
        "metrics": {
            "precision": _metric(metrics, "mp"),
            "recall": _metric(metrics, "mr"),
            "map50": _metric(metrics, "map50"),
            "map50_95": _metric(metrics, "map"),
        },
        "per_class": per_class,
    }


def write_markdown(report: dict, output_path: Path) -> None:
    values = report["metrics"]
    lines = [
        "# Model Değerlendirme Raporu",
        "",
        f"- Model: `{report['model']}`",
        f"- Veri: `{report['data']}`",
        f"- Bölüm: `{report['split']}`",
        "",
        "## Başarı Metrikleri",
        "",
        "| Metrik | Değer |",
        "|---|---:|",
    ]
    for name, value in values.items():
        formatted = "N/A" if value is None else f"{value:.4f}"
        lines.append(f"| {name} | {formatted} |")

    if report["per_class"]:
        lines.extend(
            ["", "## Sınıf Bazlı mAP50-95", "", "| Sınıf | Değer |", "|---|---:|"]
        )
        for class_name, class_metrics in report["per_class"].items():
            lines.append(f"| {class_name} | {class_metrics['map50_95']:.4f} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    if model.task != "detect":
        raise SystemExit(
            f"Uyumsuz model görevi: {model.task}. "
            "Bu değerlendirme, data.yaml içindeki images/labels yapısı için "
            "bir detection modeli bekliyor."
        )

    metrics = model.val(data=str(args.data), split=args.split, plots=True)
    report = build_report(metrics, args.model, args.data, args.split)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.report_dir / "evaluation_report.md")
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
