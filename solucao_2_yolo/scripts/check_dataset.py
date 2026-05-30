import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def split_paths(dataset: Path, split: str) -> tuple[Path, Path]:
    split_dir = "valid" if split == "val" and (dataset / "valid").exists() else split
    roboflow_images = dataset / split_dir / "images"
    roboflow_labels = dataset / split_dir / "labels"
    if roboflow_images.exists() or roboflow_labels.exists():
        return roboflow_images, roboflow_labels
    return dataset / "images" / split, dataset / "labels" / split


def image_map(images_dir: Path) -> dict[str, Path]:
    if not images_dir.exists():
        return {}
    return {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}


def label_map(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.exists():
        return {}
    return {p.stem: p for p in labels_dir.glob("*.txt")}


def check_label(label: Path) -> tuple[list[str], set[int]]:
    errors = []
    classes = set()
    for line_no, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label}:{line_no} linha mal formatada: {line}")
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{label}:{line_no} valores nao numericos: {line}")
            continue
        classes.add(cls)
        if any(v < 0 or v > 1 for v in coords):
            errors.append(f"{label}:{line_no} coordenadas fora de 0..1: {line}")
    return errors, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Pasta do dataset YOLO")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    lines = [f"Relatorio do dataset: {dataset.resolve()}"]
    all_errors = []
    all_classes = set()

    for split in ("train", "val", "test"):
        images_dir, labels_dir = split_paths(dataset, split)
        images = image_map(images_dir)
        labels = label_map(labels_dir)
        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))

        lines.extend(
            [
                "",
                f"[{split}]",
                f"pasta imagens: {images_dir} - {'OK' if images_dir.exists() else 'FALTANDO'}",
                f"pasta labels: {labels_dir} - {'OK' if labels_dir.exists() else 'FALTANDO'}",
                f"imagens: {len(images)}",
                f"labels: {len(labels)}",
                f"imagens sem label: {len(missing_labels)}",
                f"labels sem imagem: {len(orphan_labels)}",
            ]
        )
        if missing_labels[:20]:
            lines.append("exemplos imagens sem label: " + ", ".join(missing_labels[:20]))
        if orphan_labels[:20]:
            lines.append("exemplos labels sem imagem: " + ", ".join(orphan_labels[:20]))

        for label in labels.values():
            errors, classes = check_label(label)
            all_errors.extend(errors)
            all_classes.update(classes)

    lines.extend(["", f"classes encontradas: {sorted(all_classes)}", f"linhas com erro: {len(all_errors)}"])
    lines.extend(all_errors[:200])
    if len(all_errors) > 200:
        lines.append(f"... mais {len(all_errors) - 200} erros omitidos")

    report = "\n".join(lines)
    print(report)
    Path("dataset_report.txt").write_text(report, encoding="utf-8")
    print("\nRelatorio salvo em dataset_report.txt")


if __name__ == "__main__":
    main()
