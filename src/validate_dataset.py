"""Klasör tabanlı CNN sınıflandırma veri setini doğrular."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data_classification" / "split"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset(dataset_dir: Path = DATASET_DIR) -> list[str]:
    errors = []
    split_classes = {}
    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            errors.append(f"{split}: klasör bulunamadı: {split_dir}")
            continue
        classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        split_classes[split] = classes
        if not classes:
            errors.append(f"{split}: sınıf klasörü bulunamadı")
        for class_name in classes:
            files = [path for path in (split_dir / class_name).iterdir() if path.is_file()]
            invalid = [path for path in files if path.suffix.lower() not in IMAGE_EXTENSIONS]
            if invalid:
                errors.append(f"{split}/{class_name}: desteklenmeyen dosya: {invalid[0].name}")
            if not files:
                errors.append(f"{split}/{class_name}: görüntü bulunamadı")
    if split_classes:
        expected = split_classes.get("train", [])
        for split, classes in split_classes.items():
            if classes != expected:
                errors.append(f"{split}: sınıflar train ile aynı değil")
    return errors


def main() -> None:
    errors = validate_dataset()
    if errors:
        print(f"HATA SAYISI: {len(errors)}")
        print("\n".join(errors[:20]))
        raise SystemExit(1)
    for split in ("train", "valid", "test"):
        count = sum(1 for path in (DATASET_DIR / split).rglob("*") if path.is_file())
        print(f"{split}: {count} görüntü kontrol edildi.")
    print("Classification dataset kontrolu basarili!")


if __name__ == "__main__":
    main()
