from pathlib import Path

DATASET_DIR = Path("data")
CLASS_COUNT = 6

splits = ["train", "valid", "test"]

total_labels = 0
errors = []

for split in splits:
    labels_dir = DATASET_DIR / split / "labels"

    if not labels_dir.exists():
        errors.append(f"{split}: labels klasoru bulunamadi")
        continue

    label_files = list(labels_dir.glob("*.txt"))
    total_labels += len(label_files)

    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    errors.append(
                        f"{label_file} - satir {line_number}: "
                        "5 deger bekleniyor"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(
                        f"{label_file} - satir {line_number}: "
                        "sayisal olmayan deger"
                    )
                    continue

                if not 0 <= class_id < CLASS_COUNT:
                    errors.append(
                        f"{label_file} - satir {line_number}: "
                        f"gecersiz class id: {class_id}"
                    )

                if not all(0 <= value <= 1 for value in values):
                    errors.append(
                        f"{label_file} - satir {line_number}: "
                        "koordinatlar 0 ile 1 arasinda olmali"
                    )

    print(f"{split}: {len(label_files)} label dosyasi kontrol edildi.")

print()
print(f"Toplam label dosyasi: {total_labels}")

if errors:
    print(f"HATA SAYISI: {len(errors)}")
    for error in errors[:20]:
        print(error)
else:
    print("Dataset label kontrolu basarili!")