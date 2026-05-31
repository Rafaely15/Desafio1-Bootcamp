"""Avaliacao em lote do contador de parafusos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from metrics import summary_metrics
from screw_counter import count_screws, grid_search_parameters, save_debug_grid


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
DEBUG_DIR = OUTPUT_DIR / "debug"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_labels() -> dict[str, int]:
    """Carrega labels conhecidos a partir de outputs/results_v3.csv, se existir."""
    labels_path = OUTPUT_DIR / "results_v3.csv"
    if not labels_path.exists():
        return {}
    df = pd.read_csv(labels_path)
    if "imagem" in df and "contagem_real" in df:
        return dict(zip(df["imagem"].astype(str), df["contagem_real"].astype(int)))
    return {}


def list_images(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted([p for p in data_dir.iterdir() if p.suffix.lower() in EXTS]) if data_dir.exists() else []


def result_row(path: Path, result: dict[str, Any], y_true: int | None = None) -> dict[str, Any]:
    scene = result.get("metrics_internal", {})
    signed = int(result["count"] - y_true) if y_true is not None else None
    return {
        "filename": path.name,
        "y_true": y_true,
        "y_pred": int(result["count"]),
        "absolute_error": abs(signed) if signed is not None else None,
        "signed_error": signed,
        "mode_used": result.get("mode_used"),
        "segmentation_mode": result.get("segmentation_mode"),
        "ref_area": round(float(result.get("ref_area", 0.0)), 2),
        "min_area": result.get("min_area"),
        "watershed_used": result.get("watershed_used"),
        "blur_used": result.get("blur_used"),
        "noise_ratio": round(float(scene.get("noise_ratio", 0.0)), 4),
        "n_components": scene.get("n_components"),
        "n_big_components": scene.get("n_big_components"),
        "n_tiny_components": scene.get("n_tiny_components"),
        "processing_time_ms": round(float(result.get("processing_time_ms", 0.0)), 2),
    }


def evaluate_dataset(data_dir: Path = DATA_DIR, labels: dict[str, int] | None = None, config: dict | None = None) -> pd.DataFrame:
    """Processa imagens, salva debug grids e retorna resultados tabulares."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    labels = labels or default_labels()
    rows = []
    for path in list_images(data_dir):
        img = cv2.imread(str(path))
        if img is None:
            continue
        result = count_screws(img, config=config, return_debug=True)
        y_true = labels.get(path.name)
        save_debug_grid(result, DEBUG_DIR / f"{path.stem}_debug.jpg", title=path.name)
        rows.append(result_row(path, result, y_true=y_true))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    generate_markdown_report(df, OUTPUT_DIR / "relatorio_metricas.md")
    return df


def generate_markdown_report(results_df: pd.DataFrame, output_path: Path = OUTPUT_DIR / "relatorio_metricas.md") -> Path:
    """Gera relatorio simples em Markdown com metricas e erros."""
    metrics = summary_metrics(results_df)
    lines = ["# Relatorio de metricas", ""]
    if metrics:
        lines.extend(
            [
                f"- Imagens avaliadas: {metrics['n']}",
                f"- MAE: {metrics['mae']:.3f}",
                f"- Acuracia exata: {metrics['exact_match_accuracy']:.1%}",
                f"- Vies medio: {metrics['mean_bias_error']:.3f}",
                f"- Supercontagens: {metrics['overestimates']}",
                f"- Subcontagens: {metrics['underestimates']}",
                "",
            ]
        )
    else:
        lines.append("Sem y_true disponivel para calcular metricas agregadas.")
        lines.append("")

    lines.append("## Tabela de resultados")
    lines.append("")
    lines.append(results_df.to_markdown(index=False) if not results_df.empty else "_Sem resultados._")
    lines.append("")

    if "absolute_error" in results_df and results_df["absolute_error"].notna().any():
        worst = results_df.sort_values("absolute_error", ascending=False).head(5)
        under = results_df[results_df["signed_error"].fillna(0) < 0]
        over = results_df[results_df["signed_error"].fillna(0) > 0]
        lines.extend(["## Imagens com maior erro", "", worst.to_markdown(index=False), ""])
        lines.extend(["## Analise de subcontagem", "", under.to_markdown(index=False) if not under.empty else "Nao houve subcontagem nos labels disponiveis.", ""])
        lines.extend(["## Analise de supercontagem", "", over.to_markdown(index=False) if not over.empty else "Nao houve supercontagem nos labels disponiveis.", ""])

    lines.extend(
        [
            "## Limitacoes do OpenCV",
            "",
            "A abordagem classica e leve, explicavel e util como baseline, mas depende de contraste, escala e textura. Em aglomerados densos, objetos encostados podem virar um unico blob; em fundos texturizados, um objeto pode se fragmentar.",
            "",
            "## Evolucao com YOLO",
            "",
            "Com feedback manual e novas imagens anotadas, YOLO e a evolucao natural para generalizacao robusta, pois aprende aparencia, escala, orientacao e contexto diretamente dos exemplos.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_grid_search_from_labels(data_dir: Path = DATA_DIR, labels: dict[str, int] | None = None) -> dict:
    labels = labels or default_labels()
    images, y = [], []
    for path in list_images(data_dir):
        if path.name not in labels:
            continue
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
            y.append(labels[path.name])
    return grid_search_parameters(images, y) if images else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--grid-search", action="store_true")
    args = parser.parse_args()
    if args.grid_search:
        search = run_grid_search_from_labels(args.data_dir)
        print(search.get("best_config", {}))
        print(search.get("best_metrics", {}))
    df = evaluate_dataset(args.data_dir)
    print(df.to_string(index=False))
    print(summary_metrics(df))


if __name__ == "__main__":
    main()

