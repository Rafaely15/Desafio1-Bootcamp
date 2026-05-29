from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_MODEL_CACHE: dict[str, YOLO] = {}


def _safe_name(path: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{path.stem}_{stamp}{path.suffix.lower() or '.jpg'}"


def contar_parafusos(
    image_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    conf: float = 0.25,
    funcionario_id: str | None = None,
    funcionario_nome: str | None = None,
    iou: float = 0.45,
    imgsz: int | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem inexistente: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Formato de imagem nao suportado: {image_path.suffix}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo inexistente: {model_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Nao foi possivel ler a imagem: {image_path}")

    model_key = str(model_path.resolve())
    if model_key not in _MODEL_CACHE:
        _MODEL_CACHE[model_key] = YOLO(model_key)
    model = _MODEL_CACHE[model_key]
    predict_kwargs = {"source": str(image_path), "conf": conf, "iou": iou, "verbose": False}
    if imgsz:
        predict_kwargs["imgsz"] = imgsz
    results = model.predict(**predict_kwargs)
    result = results[0]

    boxes = result.boxes
    total = 0 if boxes is None else len(boxes)
    confidences = []

    annotated = image.copy()
    if boxes is not None and total > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        confidences = [float(value) for value in confs]
        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 130, 255), 2)
            cv2.putText(
                annotated,
                f"parafuso {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 130, 255),
                2,
                cv2.LINE_AA,
            )

    media = float(np.mean(confidences)) if confidences else 0.0
    cv2.rectangle(annotated, (0, 0), (min(560, annotated.shape[1]), 48), (255, 255, 255), -1)
    cv2.putText(
        annotated,
        f"Total de parafusos: {total}",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 70, 140),
        2,
        cv2.LINE_AA,
    )

    result_path = output_dir / _safe_name(image_path)
    if not cv2.imwrite(str(result_path), annotated):
        raise RuntimeError(f"Nao foi possivel salvar a imagem processada: {result_path}")

    data_hora = datetime.now().isoformat(timespec="seconds")
    payload = {
        "funcionario_id": funcionario_id or "",
        "funcionario_nome": funcionario_nome or "",
        "imagem": str(image_path),
        "total_parafusos": total,
        "confianca_media": round(media, 4),
        "imagem_resultado": str(result_path),
        "data_hora": data_hora,
    }

    if csv_path is not None:
        _append_csv(Path(csv_path), payload)

    return payload


def _append_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "nome_da_imagem",
        "total_detectado",
        "confianca_media",
        "data_hora",
        "caminho_imagem_original",
        "caminho_imagem_resultado",
        "funcionario_id",
        "funcionario_nome",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "nome_da_imagem": Path(row["imagem"]).name,
                "total_detectado": row["total_parafusos"],
                "confianca_media": row["confianca_media"],
                "data_hora": row["data_hora"],
                "caminho_imagem_original": row["imagem"],
                "caminho_imagem_resultado": row["imagem_resultado"],
                "funcionario_id": row["funcionario_id"],
                "funcionario_nome": row["funcionario_nome"],
            }
        )
