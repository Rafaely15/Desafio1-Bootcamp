from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "app" / "static" / "results"
DATABASE_URL = f"sqlite:///{BASE_DIR / 'contagens.db'}"

for directory in (UPLOAD_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
