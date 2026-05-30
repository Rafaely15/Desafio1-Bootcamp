import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def dataset_has_images(data_yaml: Path) -> bool:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = (data_yaml.parent / data.get("path", ".")).resolve()
    train = root / data["train"]
    return train.exists() and any(train.glob("*.*"))


def make_ultralytics_data_yaml(data_yaml: Path) -> Path:
    """Gera um data.yaml com path absoluto para evitar resolucao errada no Windows."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw_root = Path(data.get("path", "."))
    root = raw_root if raw_root.is_absolute() else (data_yaml.parent / raw_root).resolve()
    data["path"] = root.as_posix()

    fixed_yaml = data_yaml.parent / "data.ultralytics.yaml"
    fixed_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return fixed_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--model", default="yolo11s.pt", help="Use yolo11n.pt para treino rapido/mobile")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs_screws")
    parser.add_argument("--name", default="yolo11_screws")
    parser.add_argument("--check-only", action="store_true", help="Valida caminhos e sai sem iniciar treinamento")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml inexistente: {data_yaml}")
    if not dataset_has_images(data_yaml):
        raise RuntimeError("Dataset vazio ou caminho train invalido no data.yaml.")
    train_yaml = make_ultralytics_data_yaml(data_yaml)

    device = args.device if args.device is not None else (0 if has_gpu() else "cpu")
    project = Path(args.project)
    if not project.is_absolute():
        project = PROJECT_ROOT / project
    print(f"Modelo base: {args.model}")
    print("yolo11n.pt e mais leve e rapido; yolo11s.pt tende a ser mais preciso.")
    print("Para smartphone, prefira modelo menor. Para servidor, modelos maiores podem compensar.")
    print("imgsz=640 e mais rapido; imgsz=960 ou 1280 pode melhorar parafusos pequenos.")
    print(f"Dispositivo: {device}")
    print(f"data.yaml usado pelo Ultralytics: {train_yaml}")
    print(f"Pasta de resultados: {project / args.name}")

    if args.check_only:
        print("Check-only OK: caminhos validados, treinamento nao iniciado.")
        return

    try:
        model = YOLO(args.model)
        model.train(
            data=str(train_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            device=device,
            project=str(project),
            name=args.name,
            close_mosaic=10,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("Modelo base ausente. O Ultralytics tentara baixar se houver internet.") from exc
    except RuntimeError as exc:
        text = str(exc).lower()
        if "cuda" in text:
            raise RuntimeError("Erro de CUDA. Tente --device cpu ou reduza --batch/--imgsz.") from exc
        if "out of memory" in text or "memory" in text:
            raise RuntimeError("Falta de memoria de GPU. Reduza --batch ou --imgsz.") from exc
        raise


if __name__ == "__main__":
    main()
