from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLASS_COUNT = 6
SPLITS = ["train", "valid", "test"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data"


def validate_label_file(label_file: Path) -> list[str]:
    """Bir YOLO etiket dosyasındaki satırları kontrol eder."""
    errors = []

    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [f"{label_file}: dosya okunamadi: {error}"]

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(f"{label_file} - satir {line_number}: 5 deger bekleniyor")
            continue

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_file} - satir {line_number}: sayisal olmayan deger")
            continue

        if not 0 <= class_id < CLASS_COUNT:
            errors.append(
                f"{label_file} - satir {line_number}: " f"gecersiz class id: {class_id}"
            )

        if not all(0 <= value <= 1 for value in values):
            errors.append(
                f"{label_file} - satir {line_number}: "
                "koordinatlar 0 ile 1 arasinda olmali"
            )

    return errors


def main() -> None:
    total_labels = 0
    errors = []

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            errors.append(f"{split}: images veya labels klasoru bulunamadi")
            continue

        image_stems = {file.stem for file in images_dir.iterdir() if file.is_file()}
        label_files = sorted(labels_dir.glob("*.txt"))
        label_stems = {file.stem for file in label_files}
        total_labels += len(label_files)

        for missing_label in sorted(image_stems - label_stems):
            errors.append(f"{split}: etiketi bulunamayan gorsel: {missing_label}")

        for missing_image in sorted(label_stems - image_stems):
            errors.append(f"{split}: gorseli bulunamayan etiket: {missing_image}")

        worker_count = min(16, max(1, len(label_files)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for file_errors in executor.map(validate_label_file, label_files):
                errors.extend(file_errors)

        print(
            f"{split}: {len(image_stems)} gorsel, "
            f"{len(label_files)} label dosyasi kontrol edildi.",
            flush=True,
        )

    print()
    print(f"Toplam label dosyasi: {total_labels}")

    if errors:
        print(f"HATA SAYISI: {len(errors)}")
        for error in errors[:20]:
            print(error)
    else:
        print("Dataset kontrolu basarili!")


if __name__ == "__main__":
    main()
