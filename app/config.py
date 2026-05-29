from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "runs_screws" / "yolo11_screws-2" / "weights" / "best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "app" / "static" / "results"
DATABASE_URL = f"sqlite:///{BASE_DIR / 'contagens.db'}"

for directory in (UPLOAD_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
