from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data_classification" / "split"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from validate_dataset import validate_dataset  # noqa: E402


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="Classification veri seti yok")
def test_classification_dataset_is_valid():
    assert validate_dataset(DATASET_DIR) == []


def test_validator_rejects_missing_split(tmp_path):
    (tmp_path / "train" / "plastic").mkdir(parents=True)
    errors = validate_dataset(tmp_path)
    assert any("valid: klasör bulunamadı" in error for error in errors)


def test_validator_accepts_folder_dataset(tmp_path):
    for split in ("train", "valid", "test"):
        for class_name in ("glass", "plastic"):
            folder = tmp_path / split / class_name
            folder.mkdir(parents=True)
            (folder / "sample.jpg").write_bytes(b"image")
    assert validate_dataset(tmp_path) == []
