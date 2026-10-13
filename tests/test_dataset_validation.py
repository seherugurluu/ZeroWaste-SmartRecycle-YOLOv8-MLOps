from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data"
VALIDATOR = PROJECT_ROOT / "src" / "validate_dataset.py"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from validate_dataset import validate_label_file  # noqa: E402


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="DVC veri seti CI ortamında yok")
def test_dataset_split_counts_and_image_label_pairs():
    """Veri setindeki beklenen görsel sayısını ve dosya eşleşmelerini kontrol eder."""
    expected_counts = {
        "train": 7324,
        "valid": 2098,
        "test": 1042,
    }

    for split, expected_count in expected_counts.items():
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        image_stems = {file.stem for file in images_dir.iterdir() if file.is_file()}
        label_files = list(labels_dir.glob("*.txt"))
        label_stems = {file.stem for file in label_files}

        assert len(image_stems) == expected_count
        assert len(label_files) == expected_count
        assert image_stems == label_stems


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="DVC veri seti CI ortamında yok")
def test_dataset_validator_passes():
    """Ana doğrulama betiği veri setini hatasız kabul etmelidir."""
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Dataset kontrolu basarili!" in result.stdout


def test_validate_label_file_accepts_valid_yolo_labels(tmp_path):
    label_file = tmp_path / "valid.txt"
    label_file.write_text("0 0.5 0.5 0.25 0.25\n5 0 1 1 0\n", encoding="utf-8")

    assert validate_label_file(label_file) == []


def test_validate_label_file_rejects_invalid_class_id(tmp_path):
    label_file = tmp_path / "invalid-class.txt"
    label_file.write_text("6 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    errors = validate_label_file(label_file)

    assert len(errors) == 1
    assert "gecersiz class id" in errors[0]


def test_validate_label_file_rejects_invalid_format_and_coordinates(tmp_path):
    label_file = tmp_path / "invalid-format.txt"
    label_file.write_text(
        "0 0.5 0.5\n" "1 1.2 0.5 0.25 0.25\n" "a 0.5 0.5 0.25 0.25\n",
        encoding="utf-8",
    )

    errors = validate_label_file(label_file)

    assert len(errors) == 3
    assert any("5 deger bekleniyor" in error for error in errors)
    assert any("koordinatlar 0 ile 1 arasinda olmali" in error for error in errors)
    assert any("sayisal olmayan deger" in error for error in errors)
