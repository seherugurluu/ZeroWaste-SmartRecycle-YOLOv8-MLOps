"""Ham sınıf klasörlerini train/valid/test olarak böler."""

from argparse import ArgumentParser
from pathlib import Path
import random
import shutil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare(input_dir: Path, output_dir: Path, seed: int = 42) -> dict[str, dict[str, int]]:
    rng = random.Random(seed)
    classes = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"Sınıf klasörü bulunamadı: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for class_name in classes:
        files = sorted(path for path in (input_dir / class_name).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(files)
        train_end = int(len(files) * 0.70)
        valid_end = train_end + int(len(files) * 0.15)
        groups = {"train": files[:train_end], "valid": files[train_end:valid_end], "test": files[valid_end:]}
        result[class_name] = {}
        for split, split_files in groups.items():
            destination = output_dir / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for source in split_files:
                shutil.copy2(source, destination / source.name)
            result[class_name][split] = len(split_files)
    return result


def main() -> None:
    parser = ArgumentParser(description="CNN veri seti splitlerini oluşturur.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.input, args.output, args.seed))


if __name__ == "__main__":
    main()
