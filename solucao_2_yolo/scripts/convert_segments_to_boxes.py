import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def split_names(dataset: Path) -> list[tuple[str, str]]:
    names = []
    for split in ("train", "valid", "test"):
        if (dataset / split).exists():
            out_split = "val" if split == "valid" else split
            names.append((split, out_split))
    if not names:
        for split in ("train", "val", "test"):
            if (dataset / "images" / split).exists():
                names.append((split, split))
    return names


def paths_for(dataset: Path, split: str) -> tuple[Path, Path]:
    if (dataset / split / "images").exists():
        return dataset / split / "images", dataset / split / "labels"
    return dataset / "images" / split, dataset / "labels" / split


def convert_line(line: str) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    cls = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        x, y, w, h = values
        return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
    if len(values) < 6 or len(values) % 2 != 0:
        raise ValueError(f"linha invalida: {line}")

    xs = values[0::2]
    ys = values[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w == 0 or h == 0:
        return None
    xc = x1 + w / 2
    yc = y1 + h / 2
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte labels YOLO segmentacao/poligono para bounding boxes.")
    parser.add_argument("--input", required=True, help="Dataset original")
    parser.add_argument("--output", default="dataset_detect", help="Dataset convertido para deteccao")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset inexistente: {input_dir}")

    total_labels = 0
    total_boxes = 0
    for in_split, out_split in split_names(input_dir):
        image_dir, label_dir = paths_for(input_dir, in_split)
        out_images = output_dir / "images" / out_split
        out_labels = output_dir / "labels" / out_split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        for image in images:
            shutil.copy2(image, out_images / image.name)
            src_label = label_dir / f"{image.stem}.txt"
            converted = []
            if src_label.exists():
                for raw in src_label.read_text(encoding="utf-8").splitlines():
                    line = convert_line(raw)
                    if line:
                        converted.append(line)
            (out_labels / f"{image.stem}.txt").write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
            total_labels += 1
            total_boxes += len(converted)
        print(f"{out_split}: {len(images)} imagens convertidas")

    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: parafuso\n",
        encoding="utf-8",
    )
    print(f"Labels convertidos: {total_labels}")
    print(f"Boxes geradas: {total_boxes}")
    print(f"Dataset de deteccao salvo em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
