import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    pairs = []
    for image in images:
        candidates = [
            image.with_suffix(".txt"),
            input_dir / "labels" / f"{image.stem}.txt",
        ]
        label = next((c for c in candidates if c.exists()), None)
        if label:
            pairs.append((image, label))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pasta com imagens e labels")
    parser.add_argument("--output", default="dataset", help="Pasta YOLO de saida")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta de entrada inexistente: {input_dir}")

    pairs = find_pairs(input_dir)
    if not pairs:
        raise RuntimeError("Nenhum par imagem/label encontrado.")

    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.2)
    splits = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}

    for split, split_pairs in splits.items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / f"{image.stem}.txt")
        print(f"{split}: {len(split_pairs)} pares")

    (output_dir / "data.yaml").write_text(
        "path: dataset\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: parafuso\n",
        encoding="utf-8",
    )
    print(f"Dataset preparado em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
