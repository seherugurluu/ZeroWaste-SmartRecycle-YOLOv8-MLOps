from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data"
VALIDATOR = PROJECT_ROOT / "src" / "validate_dataset.py"


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

        image_stems = {
            file.stem for file in images_dir.iterdir() if file.is_file()
        }
        label_files = list(labels_dir.glob("*.txt"))
        label_stems = {file.stem for file in label_files}

        assert len(image_stems) == expected_count
        assert len(label_files) == expected_count
        assert image_stems == label_stems


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
