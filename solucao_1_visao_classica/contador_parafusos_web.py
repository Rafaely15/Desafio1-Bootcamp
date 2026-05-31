"""
contador_parafusos_web.py
Adaptador do pipeline v5 para uso na interface web Streamlit.

Recebe uma imagem BGR em memória, aplica a contagem com screw_counter.py e
retorna:
    contagem, imagem_resultado, imagens_debug, metadados

Novidades v5:
- segmentação adaptativa ao ruído de fundo;
- parâmetros min_area e use_ref_auto expostos para a interface;
- painel DT + Peaks para depuração visual;
- metadados com segment_mode e noise_stats.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from screw_counter import count_screws


METHOD_COLORS = {
    "isolado": (0, 220, 80),
    "watershed": (255, 150, 0),
    "dbscan": (255, 90, 0),
    "descartado": (0, 0, 220),
}


def _ensure_min_width(img: np.ndarray, min_width: int = 640) -> tuple[np.ndarray, float]:
    """Amplia imagens pequenas para melhorar gradiente e segmentação."""
    h, w = img.shape[:2]
    if w >= min_width:
        return img.copy(), 1.0
    scale = min_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (min_width, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def _as_bgr_debug(img: np.ndarray) -> np.ndarray:
    """Garante imagem BGR para codificação/visualização."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def make_dt_peaks_debug(mask: np.ndarray, ref_area: float) -> np.ndarray:
    """
    Cria visualização Distance Transform + Peaks para a máscara inteira.

    É um painel de diagnóstico: regiões mais claras indicam centros prováveis
    dos objetos; pontos verdes indicam picos locais usados como indício de
    objetos fundidos quando o watershed entra.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = (dist / dist.max() * 255).astype(np.uint8)
    else:
        dist_norm = np.zeros_like(mask, dtype=np.uint8)
    dt_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_HOT)

    min_dist = max(int(np.sqrt(max(ref_area, 1.0) / np.pi) * 0.75), 6)
    sigma = max(min_dist / 2.0, 1.0)
    dist_smooth = cv2.GaussianBlur(dist, (0, 0), sigmaX=sigma)
    kernel_r = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * min_dist + 1, 2 * min_dist + 1)
    )
    dist_dilated = cv2.dilate(dist_smooth, kernel_r)
    dt_max = dist_smooth.max()
    peak_mask = np.zeros_like(mask_bin, dtype=np.uint8)
    if dt_max > 0:
        peak_mask[(dist_smooth == dist_dilated) & (dist_smooth > dt_max * 0.30)] = 255

    n_peaks, _, _, cent = cv2.connectedComponentsWithStats(peak_mask, connectivity=8)
    for i in range(1, n_peaks):
        px, py = int(cent[i][0]), int(cent[i][1])
        cv2.circle(dt_color, (px, py), 5, (0, 255, 0), -1)
        cv2.circle(dt_color, (px, py), 6, (0, 0, 0), 1)
    return dt_color


def desenhar_resultado_limpo(
    img_bgr: np.ndarray,
    result: dict[str, Any],
    show_rejected: bool = False,
) -> np.ndarray:
    """
    Desenha apenas componentes aceitos por padrão.
    Componentes rejeitados podem ser mostrados em vermelho se show_rejected=True.
    """
    vis = img_bgr.copy()
    idx = 1

    for det in result.get("detections", []):
        count = int(det.get("count", 0))
        method = det.get("method", "isolado")

        if count <= 0 and not show_rejected:
            continue

        mask = det["mask"].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = METHOD_COLORS.get(method, (0, 220, 80))
        if count <= 0:
            color = METHOD_COLORS["descartado"]

        cv2.drawContours(vis, contours, -1, color, 2)

        cx, cy = int(det.get("cx", 0)), int(det.get("cy", 0))
        if count <= 0:
            label = "X"
        elif count == 1:
            label = str(idx)
            idx += 1
        else:
            label = f"~{count}"
            idx += count

        cv2.putText(
            vis, label, (max(cx - 10, 2), max(cy - 12, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )

    texto = f"Contagem: {int(result.get('count', 0))}"
    cv2.putText(vis, texto, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, texto, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def contar_parafusos_array(
    imagem_bgr: np.ndarray,
    debug: bool = True,
    params: dict[str, Any] | None = None,
) -> tuple[int, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """
    Interface principal usada pelo app.

    params aceitos:
      - ref_area: área de referência de um parafuso. padrão 4021.
      - min_area: área mínima dos componentes. padrão 300.
      - min_width: largura mínima para processar imagens pequenas. padrão 640.
      - use_ref_auto: se True, estima ref_area pela própria imagem. padrão True.
      - show_rejected: desenhar componentes descartados em vermelho. padrão False.
    """
    params = params or {}
    ref_area = float(params.get("ref_area", 4021.0))
    min_area = int(params.get("min_area", 300))
    min_width = int(params.get("min_width", 640))
    use_ref_auto = bool(params.get("use_ref_auto", True))
    show_rejected = bool(params.get("show_rejected", False))

    img_proc, scale = _ensure_min_width(imagem_bgr, min_width=min_width)
    result = count_screws(
        img_proc,
        config={
            "ref_area": ref_area,
            "min_area": min_area,
            "use_ref_auto": use_ref_auto,
            "ensemble": bool(params.get("ensemble", True)),
            "mode": params.get("mode", "auto"),
            "watershed": bool(params.get("watershed", True)),
            "blur": bool(params.get("blur", True)),
            "close_kernel": int(params.get("close_kernel", 9)),
            "open_kernel": int(params.get("open_kernel", 3)),
        },
        return_debug=True,
    )
    saida = desenhar_resultado_limpo(img_proc, result, show_rejected=show_rejected)
    dbg = result.get("debug_images", {})
    dt_peaks = _as_bgr_debug(dbg.get("distance", make_dt_peaks_debug(result["mask"], result.get("ref_area_used", ref_area))))
    peaks = _as_bgr_debug(dbg.get("peaks", result["mask"]))

    debug_imgs: dict[str, np.ndarray] = {}
    if debug:
        debug_imgs = {
            "original": img_proc,
            "enhanced": _as_bgr_debug(dbg.get("enhanced", img_proc)),
            "preprocessed": _as_bgr_debug(dbg.get("preprocessed", img_proc)),
            "mask": _as_bgr_debug(result["mask"]),
            "dt_peaks": dt_peaks,
            "peaks": peaks,
            "components": _as_bgr_debug(dbg.get("components", saida)),
            "resultado": saida,
        }

    accepted = [d for d in result.get("detections", []) if int(d.get("count", 0)) > 0]
    meta = {
        "ref_area_used": float(result.get("ref_area_used", ref_area)),
        "ref_area_input": ref_area,
        "min_area": min_area,
        "use_ref_auto": use_ref_auto,
        "segment_mode": result.get("segment_mode"),
        "mode_used": result.get("mode_used"),
        "noise_stats": result.get("noise_stats", {}),
        "fragmentacao_detectada": bool(result.get("fragmentacao_detectada")),
        "watershed_aplicado": bool(result.get("watershed_aplicado")),
        "blur_used": bool(result.get("blur_used")),
        "processing_time_ms": float(result.get("processing_time_ms", 0.0)),
        "low_confidence": bool(result.get("metrics_internal", {}).get("low_confidence", False)),
        "metrics_internal": result.get("metrics_internal", {}),
        "scale_applied": float(scale),
        "n_detections": len(accepted),
        "methods": [d.get("source", d.get("method")) for d in accepted],
    }
    return int(result["count"]), saida, debug_imgs, meta
