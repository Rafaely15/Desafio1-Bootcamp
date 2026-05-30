from pathlib import Path

from ultralytics import YOLO

MODEL = Path("runs_screws/yolo11_screws/weights/best.pt")

if not MODEL.exists():
    raise FileNotFoundError(f"Modelo inexistente: {MODEL}")

YOLO(str(MODEL)).export(format="ncnn")
