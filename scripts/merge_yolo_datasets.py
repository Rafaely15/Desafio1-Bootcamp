import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def split_paths(dataset: Path, split: str) -> tuple[Path, Path]:
    split_dir = "valid" if split == "val" and (dataset / "valid").exists() else split
    roboflow_images = dataset / split_dir / "images"
    roboflow_labels = dataset / split_dir / "labels"
    if roboflow_images.exists() or roboflow_labels.exists():
        return roboflow_images, roboflow_labels
    return dataset / "images" / split, dataset / "labels" / split


def copy_split(source: Path, output: Path, split: str, prefix: str) -> tuple[int, int]:
    image_dir, label_dir = split_paths(source, split)
    out_images = output / "images" / split
    out_labels = output / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if image_dir.exists() else []
    copied_images = 0
    copied_labels = 0
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        new_stem = f"{prefix}_{image.stem}"
        shutil.copy2(image, out_images / f"{new_stem}{image.suffix.lower()}")
        copied_images += 1
        if label.exists():
            shutil.copy2(label, out_labels / f"{new_stem}.txt")
            copied_labels += 1
        else:
            (out_labels / f"{new_stem}.txt").write_text("", encoding="utf-8")
    return copied_images, copied_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Mescla datasets YOLO sem sobrescrever os originais.")
    parser.add_argument("--sources", nargs="+", required=True, help="Pastas dos datasets YOLO")
    parser.add_argument("--output", default="dataset_combined", help="Pasta de saida")
    args = parser.parse_args()

    output = Path(args.output)
    totals = {}
    for idx, source_raw in enumerate(args.sources, start=1):
        source = Path(source_raw)
        if not source.exists():
            raise FileNotFoundError(f"Dataset inexistente: {source}")
        prefix = f"ds{idx}"
        for split in ("train", "val", "test"):
            images, labels = copy_split(source, output, split, prefix)
            totals.setdefault(split, [0, 0])
            totals[split][0] += images
            totals[split][1] += labels
            print(f"{source.name} -> {split}: {images} imagens, {labels} labels")

    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: parafuso\n",
        encoding="utf-8",
    )

    print("\nDataset combinado criado em:", output.resolve())
    for split, (images, labels) in totals.items():
        print(f"{split}: {images} imagens, {labels} labels")


if __name__ == "__main__":
    main()
