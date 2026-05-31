"""Persistencia local para registros de contagem de parafusos."""

from __future__ import annotations

import csv
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATABASE_DIR = ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "contador_parafusos.db"
RECORDS_DIR = ROOT / "records"

TABLE_COLUMNS = [
    "id",
    "timestamp",
    "date",
    "employee_name",
    "employee_id",
    "sector",
    "order_id",
    "automatic_count",
    "final_count",
    "correction",
    "status",
    "image_original_path",
    "image_result_path",
    "csv_export_path",
    "notes",
    "metadata_json",
]

CSV_COLUMNS = [
    "id",
    "timestamp",
    "date",
    "employee_name",
    "employee_id",
    "sector",
    "order_id",
    "automatic_count",
    "final_count",
    "correction",
    "status",
    "image_original_path",
    "image_result_path",
    "notes",
]


def sanitize_filename(value: str) -> str:
    """Remove caracteres problemáticos para nomes de arquivos no Windows."""
    value = (value or "sem_id").strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value, flags=re.IGNORECASE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sem_id"


def get_record_dirs(record_date: str | None = None) -> dict[str, Path]:
    """Retorna e cria a estrutura records/YYYY-MM-DD."""
    record_date = record_date or date.today().isoformat()
    base = RECORDS_DIR / record_date
    dirs = {
        "base": base,
        "originals": base / "originals",
        "detections": base / "detections",
        "exports": base / "exports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def init_db() -> Path:
    """Cria o banco SQLite e a tabela principal quando necessario."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS count_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                date TEXT,
                employee_name TEXT,
                employee_id TEXT,
                sector TEXT,
                order_id TEXT,
                automatic_count INTEGER,
                final_count INTEGER,
                correction INTEGER,
                status TEXT,
                image_original_path TEXT,
                image_result_path TEXT,
                csv_export_path TEXT,
                notes TEXT,
                metadata_json TEXT
            )
            """
        )
        conn.commit()
    return DATABASE_PATH


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _connect() -> sqlite3.Connection:
    init_db()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_count_record(record: dict) -> int:
    """Salva um registro e atualiza o CSV diario."""
    init_db()
    record = record.copy()
    record.setdefault("csv_export_path", "")
    columns = [c for c in TABLE_COLUMNS if c != "id"]
    values = [record.get(c) for c in columns]
    placeholders = ", ".join(["?"] * len(columns))
    with sqlite3.connect(DATABASE_PATH) as conn:
        cur = conn.execute(
            f"INSERT INTO count_records ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        record_id = int(cur.lastrowid)
        conn.commit()
    export_path = export_daily_csv(str(record["date"]))
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("UPDATE count_records SET csv_export_path = ? WHERE id = ?", (str(export_path), record_id))
        conn.commit()
    return record_id


def load_records_by_date(date: str) -> pd.DataFrame:
    """Carrega registros de um dia."""
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM count_records WHERE date = ? ORDER BY timestamp DESC",
            conn,
            params=(date,),
        )


def load_records_by_employee(employee_id: str) -> pd.DataFrame:
    """Carrega registros de um funcionario."""
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM count_records WHERE employee_id = ? ORDER BY timestamp DESC",
            conn,
            params=(employee_id,),
        )


def load_records_by_employee_and_date(employee_id: str, date: str) -> pd.DataFrame:
    """Carrega registros de um funcionario em um dia."""
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM count_records WHERE employee_id = ? AND date = ? ORDER BY timestamp DESC",
            conn,
            params=(employee_id, date),
        )


def export_daily_csv(date: str) -> Path:
    """Exporta ou atualiza o CSV geral do dia."""
    dirs = get_record_dirs(date)
    csv_path = dirs["exports"] / f"contagens_{date}.csv"
    df = load_records_by_date(date)
    if df.empty:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
    else:
        df[[c for c in CSV_COLUMNS if c in df.columns]].to_csv(csv_path, index=False)
    return csv_path


def export_employee_csv(employee_id: str) -> Path:
    """Exporta registros de um funcionario."""
    today = date.today().isoformat()
    dirs = get_record_dirs(today)
    safe_id = sanitize_filename(employee_id)
    csv_path = dirs["exports"] / f"contagens_{safe_id}_{today}.csv"
    df = load_records_by_employee(employee_id)
    if df.empty:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
    else:
        df[[c for c in CSV_COLUMNS if c in df.columns]].to_csv(csv_path, index=False)
    return csv_path


def get_daily_summary(date: str) -> dict[str, int]:
    """Resumo operacional do dia."""
    df = load_records_by_date(date)
    if df.empty:
        return {"records": 0, "total_screws": 0, "corrections": 0}
    return {
        "records": int(len(df)),
        "total_screws": int(df["final_count"].fillna(0).sum()),
        "corrections": int((df["status"] == "corrigida").sum()),
    }


def list_employees() -> pd.DataFrame:
    """Lista funcionarios que ja usaram o sistema."""
    with _connect() as conn:
        return pd.read_sql_query(
            """
            SELECT employee_id, employee_name, sector, COUNT(*) AS total_records,
                   MAX(timestamp) AS last_seen
            FROM count_records
            GROUP BY employee_id, employee_name, sector
            ORDER BY last_seen DESC
            """,
            conn,
        )

