import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def make_ultralytics_data_yaml(data_yaml: Path) -> Path:
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
    parser.add_argument("--model", default="runs_screws/yolo11_screws/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    model_path = Path(args.model).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml inexistente: {data_yaml}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo treinado inexistente: {model_path}")
    val_yaml = make_ultralytics_data_yaml(data_yaml)

    model = YOLO(str(model_path))
    metrics = model.val(data=str(val_yaml), conf=args.conf)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    report = f"""Validacao do modelo
modelo: {model_path}
data: {data_yaml}

precision: {precision:.4f}
recall: {recall:.4f}
mAP50: {map50:.4f}
mAP50-95: {map5095:.4f}

Alta precision significa menos falsos parafusos.
Alto recall significa menos parafusos perdidos.
Para contagem, precision e recall precisam estar equilibrados.
"""
    print(report)
    Path("validation_report.txt").write_text(report, encoding="utf-8")
    print("Relatorio salvo em validation_report.txt")


if __name__ == "__main__":
    main()
