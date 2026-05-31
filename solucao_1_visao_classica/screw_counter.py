"""
Pipeline classico e explicavel para contagem de parafusos com OpenCV.

O modulo implementa uma abordagem em camadas:
- analise automatica da cena;
- multiplos modos de segmentacao;
- detectores por contorno, watershed e agrupamento de fragmentos;
- ensemble leve para escolher a contagem mais coerente sem usar nome de arquivo.

Esta e uma baseline interpretavel. Em cenarios reais muito variados, a evolucao
natural e uma solucao supervisionada, como YOLO, treinada com imagens anotadas.
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    from sklearn.cluster import DBSCAN

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "auto",
    "ensemble": True,
    "ref_area": 4021.0,
    "use_ref_auto": True,
    "min_area": 300,
    "max_area_ratio": 0.12,
    "close_kernel": 9,
    "open_kernel": 3,
    "blur": True,
    "adaptive_threshold": True,
    "watershed": True,
    "min_distance": None,
    "show_rejected": False,
}

SEGMENTATION_MODES = [
    "modo_claro",
    "modo_cinza",
    "modo_bandeja",
    "modo_texturizado",
    "modo_denso",
    "modo_multiescala",
]


@dataclass
class Component:
    """Componente conectado extraido de uma mascara binaria."""

    label: int
    area: float
    bbox: tuple[int, int, int, int]
    cx: float
    cy: float
    aspect_ratio: float
    solidity: float
    extent: float
    mask: np.ndarray
    source: str = "component"
    n_frags: int = 1
    gradient_variance: float = 0.0


def _merge_config(config: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    return cfg


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(size), _odd(size)))


def _enhance_gray(img_bgr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _normalize_u8(img: np.ndarray) -> np.ndarray:
    if img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def _remove_border(mask: np.ndarray, border: int = 3) -> np.ndarray:
    clean = mask.copy()
    clean[:border, :] = 0
    clean[-border:, :] = 0
    clean[:, :border] = 0
    clean[:, -border:] = 0
    return clean


def _component_counts(mask: np.ndarray, min_area: int, image_area: int) -> dict[str, int | float]:
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    n_tiny = sum(a < min_area for a in areas)
    n_big = sum(min_area <= a <= image_area * 0.12 for a in areas)
    n_large = sum(a > image_area * 0.12 for a in areas)
    total = max(len(areas), 1)
    return {
        "n_components": int(len(areas)),
        "n_tiny_components": int(n_tiny),
        "n_big_components": int(n_big),
        "n_large_components": int(n_large),
        "noise_ratio": float(n_tiny / total),
    }


def classify_scene(img_bgr: np.ndarray) -> dict[str, Any]:
    """
    Calcula indicadores da cena e sugere um modo de segmentacao.

    A funcao usa apenas caracteristicas visuais da imagem: brilho, contraste,
    textura, componentes preliminares, densidade e variacao de escala.
    """
    h, w = img_bgr.shape[:2]
    image_area = h * w
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _enhance_gray(img_bgr)
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    light_bg_ratio = float(np.mean(gray > 185))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    noise_estimate = float(np.std(lap))
    texture_ratio = float(np.mean(np.abs(lap) > max(12.0, noise_estimate * 0.45)))

    grad = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, _kernel(3))
    _, prelim = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prelim = cv2.morphologyEx(prelim, cv2.MORPH_CLOSE, _kernel(7))
    prelim = _remove_border(prelim)
    counts = _component_counts(prelim, min_area=max(120, int(image_area * 0.0005)), image_area=image_area)

    n, _, stats, _ = cv2.connectedComponentsWithStats((prelim > 0).astype(np.uint8), 8)
    valid_areas = [
        float(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n)
        if image_area * 0.0005 <= stats[i, cv2.CC_STAT_AREA] <= image_area * 0.15
    ]
    density = float(np.count_nonzero(prelim) / image_area)
    scale_variation = float(np.percentile(valid_areas, 90) / max(np.percentile(valid_areas, 10), 1.0)) if len(valid_areas) >= 4 else 1.0
    touching_score = float(sum(a > np.median(valid_areas) * 1.8 for a in valid_areas) / max(len(valid_areas), 1)) if valid_areas else 0.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    tray_like = bool(np.mean(saturation > 45) > 0.20 and 0.10 < light_bg_ratio < 0.85 and counts["n_big_components"] <= 8)
    textured = bool(noise_estimate > 38 or texture_ratio > 0.24 or counts["noise_ratio"] > 0.82)

    if touching_score > 0.25 or density > 0.15:
        mode = "modo_denso"
    elif scale_variation > 3.0:
        mode = "modo_multiescala"
    elif tray_like:
        mode = "modo_bandeja"
    elif textured and counts["n_big_components"] <= 6:
        mode = "modo_texturizado"
    elif light_bg_ratio > 0.55 or mean_brightness > 170:
        mode = "modo_claro"
    else:
        mode = "modo_cinza"

    return {
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "light_bg_ratio": light_bg_ratio,
        "noise_estimate": noise_estimate,
        "texture_ratio": texture_ratio,
        "density": density,
        "touching_score": touching_score,
        "textured_background": textured,
        "scale_variation": scale_variation,
        "many_touching_objects": bool(touching_score > 0.25 or density > 0.15),
        "large_scale_variation": bool(scale_variation > 3.0),
        "tray_like": tray_like,
        "recommended_mode": mode,
        **counts,
    }


def segment_image(img_bgr: np.ndarray, mode: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Segmenta a imagem usando um dos modos especializados."""
    cfg = _merge_config(config)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _enhance_gray(img_bgr, clip=2.5 if mode in {"modo_texturizado", "modo_denso"} else 2.0)
    pre = enhanced.copy()

    if mode == "modo_claro":
        pre = cv2.GaussianBlur(pre, (_odd(5), _odd(5)), 0) if cfg["blur"] else pre
        _, mask = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(pre, 45, 135)
        mask = cv2.bitwise_or(mask, cv2.dilate(edges, _kernel(3)))
    elif mode == "modo_cinza":
        pre = cv2.bilateralFilter(pre, 7, 45, 45) if cfg["blur"] else pre
        grad = cv2.morphologyEx(pre, cv2.MORPH_GRADIENT, _kernel(3))
        _, mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "modo_bandeja":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        pre = cv2.bilateralFilter(l_channel, 9, 55, 55)
        grad = cv2.morphologyEx(pre, cv2.MORPH_GRADIENT, _kernel(5))
        _, mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "modo_texturizado":
        pre = cv2.GaussianBlur(pre, (_odd(7), _odd(7)), 0)
        bg = cv2.morphologyEx(pre, cv2.MORPH_OPEN, _kernel(31))
        top_hat = cv2.absdiff(pre, bg)
        grad = cv2.morphologyEx(top_hat, cv2.MORPH_GRADIENT, _kernel(3))
        _, mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "modo_denso":
        pre = cv2.bilateralFilter(pre, 7, 60, 60)
        grad = cv2.morphologyEx(pre, cv2.MORPH_GRADIENT, _kernel(3))
        adaptive = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
        _, otsu = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_or(otsu, adaptive)
    elif mode == "modo_multiescala":
        small = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, _kernel(3))
        large = cv2.morphologyEx(cv2.GaussianBlur(enhanced, (7, 7), 0), cv2.MORPH_GRADIENT, _kernel(7))
        mix = cv2.addWeighted(small, 0.65, large, 0.35, 0)
        _, mask = cv2.threshold(mix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pre = mix
    else:
        raise ValueError(f"Modo de segmentacao desconhecido: {mode}")

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(cfg["open_kernel"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(cfg["close_kernel"]))
    if mode in {"modo_denso", "modo_multiescala"}:
        mask = cv2.dilate(mask, _kernel(3), iterations=1)
    mask = _remove_border(mask)
    return {"mask": mask, "preprocessed": pre, "edge_map": cv2.Canny(pre, 50, 150), "mode": mode}


def _extract_components(mask: np.ndarray, config: dict[str, Any], source: str = "contours") -> list[Component]:
    h, w = mask.shape[:2]
    image_area = h * w
    min_area = int(config["min_area"])
    max_area = image_area * float(config["max_area_ratio"])
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    comps: list[Component] = []
    for label in range(1, n):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = (int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP]), int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT]))
        if bw < 4 or bh < 4:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 14:
            continue
        comp_mask = (labels == label).astype(np.uint8)
        sobelx = cv2.Sobel(comp_mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(comp_mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_variance = float(np.var(grad_mag[comp_mask > 0])) if np.any(comp_mask) else 0.0
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = 0.0
        contour_area = 0.0
        if contours:
            merged = np.vstack(contours)
            hull_area = float(cv2.contourArea(cv2.convexHull(merged)))
            contour_area = float(sum(cv2.contourArea(c) for c in contours))
        solidity = float(contour_area / hull_area) if hull_area > 1 else 0.5
        extent = float(area / max(bw * bh, 1))
        if solidity < 0.08 or extent < 0.03:
            continue
        cx, cy = centroids[label]
        comps.append(Component(label, area, (x, y, bw, bh), float(cx), float(cy), float(aspect), solidity, extent, comp_mask, source, 1, gradient_variance))
    return comps


def estimate_ref_area(components: Iterable[Component], fallback: float = 4021.0) -> float:
    """Estima a area de um parafuso isolado por estatistica robusta."""
    areas = sorted(float(c.area) for c in components)
    if not areas:
        return float(fallback)
    if len(areas) == 1:
        return max(float(fallback), min(areas[0], float(fallback)))
    if len(areas) >= 4:
        trimmed = areas[int(len(areas) * 0.20) : max(int(len(areas) * 0.80), 1)]
        ref = float(np.median(trimmed or areas))
    else:
        ref = float(np.median(areas))
    return float(np.clip(ref, fallback * 0.45, fallback * 2.5))


def _distance_peaks(mask: np.ndarray, ref_area: float, min_distance: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    mask_bin = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    if min_distance is None:
        min_distance = max(int(np.sqrt(max(ref_area, 1.0) / np.pi) * 0.62), 5)
    sigma = max(min_distance / 2.5, 1.0)
    smooth = cv2.GaussianBlur(dist, (0, 0), sigmaX=sigma)
    dilated = cv2.dilate(smooth, _kernel(2 * int(min_distance) + 1))
    peak_mask = np.zeros(mask.shape, dtype=np.uint8)
    if smooth.max() > 0:
        peak_mask[(smooth == dilated) & (smooth > smooth.max() * 0.28)] = 255
    n_peaks, _, _, _ = cv2.connectedComponentsWithStats(peak_mask, 8)
    return dist, peak_mask, max(0, n_peaks - 1)


def detector_contours(mask: np.ndarray, config: dict[str, Any], ref_area: float) -> dict[str, Any]:
    comps = _extract_components(mask, config, "contours")
    detections = []
    count = 0
    for comp in comps:
        multiplicity = max(1, int(round(comp.area / ref_area))) if comp.area > ref_area * 1.75 else 1
        count += multiplicity
        detections.append(_detection_from_component(comp, multiplicity, "contours", ref_area))
    return {"name": "detector_contornos", "count": count, "detections": detections, "components": comps}


def detector_watershed(mask: np.ndarray, config: dict[str, Any], ref_area: float) -> dict[str, Any]:
    comps = _extract_components(mask, config, "watershed")
    detections = []
    total = 0
    any_ws = False
    peak_canvas = np.zeros(mask.shape, dtype=np.uint8)
    dist_canvas = np.zeros(mask.shape, dtype=np.float32)
    for comp in comps:
        if comp.area > ref_area * 1.35:
            dist, peaks, n_peaks = _distance_peaks(comp.mask, ref_area, config.get("min_distance"))
            area_est = max(1, int(round(comp.area / ref_area)))
            count = n_peaks if 1 <= n_peaks <= max(area_est + 4, 2) else area_est
            any_ws = True
            peak_canvas = cv2.bitwise_or(peak_canvas, peaks)
            dist_canvas = np.maximum(dist_canvas, dist)
            source = "watershed"
        else:
            count = 1
            source = "isolado"
        total += count
        detections.append(_detection_from_component(comp, count, source, ref_area))
    return {"name": "detector_watershed", "count": total, "detections": detections, "components": comps, "peaks": peak_canvas, "distance": dist_canvas, "watershed_used": any_ws}


def detector_fragmentos_dbscan(mask: np.ndarray, config: dict[str, Any], ref_area: float) -> dict[str, Any]:
    loose_cfg = config.copy()
    loose_cfg["min_area"] = max(50, int(config["min_area"] * 0.22))
    comps = _extract_components(mask, loose_cfg, "dbscan_fragmentos")
    if not comps:
        return {"name": "detector_fragmentos_dbscan", "count": 0, "detections": [], "components": []}
    small = [c for c in comps if c.area < ref_area * 0.70]
    if len(small) < max(4, len(comps) * 0.60):
        return {"name": "detector_fragmentos_dbscan", "count": len(comps), "detections": [_detection_from_component(c, 1, "fragmento_isolado", ref_area) for c in comps], "components": comps}
    pts = np.array([[c.cx, c.cy] for c in small], dtype=np.float32)
    h, w = mask.shape[:2]
    eps = max(np.sqrt(ref_area) * 2.5, min(h, w) * 0.10)
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(pts) if HAS_SKLEARN else np.zeros(len(pts), dtype=int)
    detections = []
    for cluster_id in sorted(set(labels)):
        members = [c for c, label in zip(small, labels) if label == cluster_id]
        union = np.zeros_like(mask, dtype=np.uint8)
        for m in members:
            union = cv2.bitwise_or(union, m.mask)
        x, y, bw, bh = cv2.boundingRect(union)
        area = float(sum(m.area for m in members))
        comp = Component(
            int(cluster_id),
            area,
            (x, y, bw, bh),
            float(np.mean([m.cx for m in members])),
            float(np.mean([m.cy for m in members])),
            max(bw, bh) / max(min(bw, bh), 1),
            0.5,
            area / max(bw * bh, 1),
            union,
            "dbscan",
            len(members),
            float(np.mean([m.gradient_variance for m in members])),
        )
        detections.append(_detection_from_component(comp, 1, "dbscan", ref_area))
    return {"name": "detector_fragmentos_dbscan", "count": len(detections), "detections": detections, "components": comps}


def detector_multiescala(img_bgr: np.ndarray, config: dict[str, Any], ref_area: float) -> dict[str, Any]:
    all_detections: list[dict[str, Any]] = []
    for scale in (0.75, 1.0, 1.25):
        if scale == 1.0:
            img = img_bgr
        else:
            img = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        seg = segment_image(img, "modo_multiescala", config)
        scaled_ref = ref_area * scale * scale
        res = detector_watershed(seg["mask"], config, scaled_ref)
        for det in res["detections"]:
            x, y, w, h = det["bbox"]
            det = det.copy()
            det["bbox"] = [int(round(x / scale)), int(round(y / scale)), int(round(w / scale)), int(round(h / scale))]
            det["source"] = f"multiescala_{scale:g}"
            all_detections.append(det)
    merged = _nms_detections(all_detections, iou_threshold=0.25)
    return {"name": "detector_multiescala", "count": int(sum(d["count"] for d in merged)), "detections": merged, "components": []}


def _detection_from_component(comp: Component, count: int, source: str, ref_area: float) -> dict[str, Any]:
    confidence = float(np.clip(1.0 - abs(comp.area / max(ref_area, 1.0) - max(count, 1)) / max(count, 1), 0.15, 0.98))
    return {
        "bbox": [int(v) for v in comp.bbox],
        "area": float(comp.area),
        "confidence_proxy": confidence,
        "source": source,
        "count": int(count),
        "mask": comp.mask,
        "cx": float(comp.cx),
        "cy": float(comp.cy),
        "aspect_ratio": float(comp.aspect_ratio),
        "solidity": float(comp.solidity),
        "n_frags": int(comp.n_frags),
        "method": source,
    }


def _iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.3) -> list[dict[str, Any]]:
    ordered = sorted(detections, key=lambda d: d.get("confidence_proxy", 0), reverse=True)
    kept: list[dict[str, Any]] = []
    for det in ordered:
        if all(_iou(det["bbox"], other["bbox"]) < iou_threshold for other in kept):
            kept.append(det)
    return kept


def _choose_ensemble(candidates: list[dict[str, Any]], scene: dict[str, Any]) -> dict[str, Any]:
    valid = [c for c in candidates if c["count"] > 0]
    if not valid:
        return min(candidates, key=lambda c: c.get("penalty", 999))

    # Fundo muito fragmentado e baixo contraste costuma representar um unico
    # parafuso quebrado em varios aneis/fragmentos. Nesse caso, favorecemos o
    # modo cinza, que e mais conservador, antes dos detectores multiescala.
    if (
        scene.get("textured_background")
        and scene.get("noise_ratio", 0) > 0.88
        and scene.get("n_tiny_components", 0) > scene.get("n_big_components", 0) * 8
        and scene.get("contrast", 999) < 25
    ):
        conservative = [
            c for c in valid
            if c.get("segmentation_mode") == "modo_cinza" and c.get("count", 0) <= 3
        ]
        if conservative:
            return min(conservative, key=lambda c: (c["count"], c.get("penalty", 999)))

    if scene["many_touching_objects"]:
        pref = ["detector_watershed", "detector_multiescala", "detector_contornos"]
    elif scene["textured_background"] and scene["n_tiny_components"] > scene["n_big_components"] * 4:
        pref = ["detector_fragmentos_dbscan", "detector_watershed", "detector_contornos"]
    elif scene["large_scale_variation"]:
        pref = ["detector_multiescala", "detector_watershed", "detector_contornos"]
    else:
        pref = ["detector_contornos", "detector_watershed", "detector_fragmentos_dbscan"]
    by_name = {c["name"]: c for c in valid}
    counts = [c["count"] for c in valid]
    median_count = float(np.median(counts))
    best = None
    best_score = -1e9
    for rank, name in enumerate(pref):
        cand = by_name.get(name)
        if not cand:
            continue
        conf = float(np.mean([d.get("confidence_proxy", 0.4) for d in cand.get("detections", [])]) if cand.get("detections") else 0.2)
        agreement = 1.0 / (1.0 + abs(cand["count"] - median_count))
        score = conf * 1.2 + agreement - rank * 0.05
        if score > best_score:
            best, best_score = cand, score
    return best or valid[0]


def count_screws(
    img_bgr: np.ndarray,
    config: dict[str, Any] | None = None,
    return_debug: bool = True,
    ref_area: float | None = None,
    min_area: int | None = None,
    use_ref_auto: bool | None = None,
) -> dict[str, Any]:
    """
    Conta parafusos em uma imagem BGR.

    Retorna um dicionario com contagem, modo usado, mascara, imagens de debug,
    deteccoes, metricas internas e tempo de processamento.
    """
    t0 = time.perf_counter()
    cfg = _merge_config(config, ref_area=ref_area, min_area=min_area, use_ref_auto=use_ref_auto)
    scene = classify_scene(img_bgr)
    selected_mode = scene["recommended_mode"] if cfg["mode"] == "auto" else cfg["mode"]
    modes = SEGMENTATION_MODES if cfg["ensemble"] else [selected_mode]

    candidates: list[dict[str, Any]] = []
    debug_images: dict[str, np.ndarray] = {}
    best_seg: dict[str, Any] | None = None
    best_ref = float(cfg["ref_area"])

    for mode in modes:
        seg = segment_image(img_bgr, mode, cfg)
        comps = _extract_components(seg["mask"], cfg)
        ref = estimate_ref_area(comps, float(cfg["ref_area"])) if cfg["use_ref_auto"] else float(cfg["ref_area"])
        contour = detector_contours(seg["mask"], cfg, ref)
        if not cfg["ensemble"]:
            for cand in (contour,):
                c = cand.copy()
                c["segmentation_mode"] = mode
                c["mask"] = seg["mask"]
                c["preprocessed"] = seg["preprocessed"]
                c["edge_map"] = seg["edge_map"]
                c["ref_area"] = ref
                c["penalty"] = abs(c["count"] - max(1, round(np.count_nonzero(seg["mask"]) / max(ref, 1.0))))
                candidates.append(c)
            continue
        ws = detector_watershed(seg["mask"], cfg, ref) if cfg["watershed"] else contour
        db = detector_fragmentos_dbscan(seg["mask"], cfg, ref)
        for cand in (contour, ws, db):
            c = cand.copy()
            c["segmentation_mode"] = mode
            c["mask"] = seg["mask"]
            c["preprocessed"] = seg["preprocessed"]
            c["edge_map"] = seg["edge_map"]
            c["ref_area"] = ref
            c["penalty"] = abs(c["count"] - max(1, round(np.count_nonzero(seg["mask"]) / max(ref, 1.0))))
            candidates.append(c)
        if mode == "modo_multiescala":
            ms = detector_multiescala(img_bgr, cfg, ref)
            ms.update({"segmentation_mode": mode, "mask": seg["mask"], "preprocessed": seg["preprocessed"], "edge_map": seg["edge_map"], "ref_area": ref})
            candidates.append(ms)

    chosen = _choose_ensemble(candidates, scene) if cfg["ensemble"] else candidates[0]
    best_seg = {"mask": chosen["mask"], "preprocessed": chosen["preprocessed"], "edge_map": chosen["edge_map"]}
    best_ref = float(chosen["ref_area"])
    dist, peak_mask, n_peaks = _distance_peaks(best_seg["mask"], best_ref, cfg.get("min_distance"))
    processing_time_ms = (time.perf_counter() - t0) * 1000.0

    detections = chosen.get("detections", [])
    low_confidence = bool(
        len(candidates) >= 2
        and np.std([c["count"] for c in candidates if c["count"] > 0]) > max(1.5, chosen["count"] * 0.35)
    )

    if return_debug:
        debug_images = {
            "original": img_bgr.copy(),
            "preprocessed": best_seg["preprocessed"],
            "enhanced": _enhance_gray(img_bgr),
            "mask": best_seg["mask"],
            "distance": _normalize_u8(dist),
            "peaks": peak_mask,
            "edge_map": best_seg["edge_map"],
            "components": draw_components(img_bgr, detections),
            "result": draw_detections(img_bgr, {"count": chosen["count"], "detections": detections}),
        }

    return {
        "count": int(chosen["count"]),
        "mode": str(selected_mode),
        "mode_used": str(chosen["name"]),
        "segmentation_mode": str(chosen["segmentation_mode"]),
        "segment_mode": str(chosen["segmentation_mode"]),
        "mask": best_seg["mask"],
        "debug_images": debug_images,
        "detections": detections,
        "metrics_internal": {
            **scene,
            "candidate_counts": {f"{c['segmentation_mode']}:{c['name']}": int(c["count"]) for c in candidates},
            "n_peaks": int(n_peaks),
            "low_confidence": low_confidence,
        },
        "scene": scene,
        "ref_area": best_ref,
        "ref_area_used": best_ref,
        "min_area": int(cfg["min_area"]),
        "watershed_used": bool(any(d.get("source") == "watershed" for d in detections)),
        "watershed_aplicado": bool(any(d.get("source") == "watershed" for d in detections)),
        "blur_used": bool(cfg["blur"] or chosen["segmentation_mode"] == "modo_texturizado"),
        "noise_stats": {
            "noise_ratio": float(scene["noise_ratio"]),
            "n_total": int(scene["n_components"]),
            "n_big": int(scene["n_big_components"]),
            "n_tiny": int(scene["n_tiny_components"]),
        },
        "fragmentacao_detectada": bool(chosen["name"] == "detector_fragmentos_dbscan"),
        "processing_time_ms": float(processing_time_ms),
    }


def draw_components(img_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    vis = img_bgr.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 160, 0), 2)
        cv2.putText(vis, str(det.get("count", 1)), (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2, cv2.LINE_AA)
    return vis


METHOD_COLORS = {
    "isolado": (0, 200, 0),
    "contours": (0, 200, 0),
    "watershed": (220, 120, 0),
    "dbscan": (255, 90, 0),
    "fragmento_isolado": (180, 180, 0),
    "descartado": (0, 0, 220),
}


def draw_detections(img_bgr: np.ndarray, result: dict[str, Any], gt: int | None = None) -> np.ndarray:
    """Desenha caixas, contornos e texto de resultado."""
    vis = img_bgr.copy()
    for det in result.get("detections", []):
        color = METHOD_COLORS.get(det.get("source", det.get("method", "contours")), (0, 220, 80))
        mask = det.get("mask")
        if isinstance(mask, np.ndarray):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
        x, y, w, h = det["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        label = str(det.get("count", 1))
        cv2.putText(vis, label, (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    text = f"Detectado: {int(result.get('count', 0))}"
    if gt is not None:
        text += f" | Real: {gt} | Erro: {int(result.get('count', 0)) - gt:+d}"
    cv2.putText(vis, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def make_debug_grid(result: dict[str, Any], title: str = "") -> np.ndarray:
    """Monta uma grade de debug com os principais estagios do pipeline."""
    imgs = result.get("debug_images", {})
    panels = [
        ("Original", imgs.get("original")),
        ("Pre-processada", imgs.get("preprocessed")),
        ("Mascara", imgs.get("mask")),
        ("Distance", imgs.get("distance")),
        ("Peaks", imgs.get("peaks")),
        ("Componentes", imgs.get("components")),
        ("Resultado", imgs.get("result")),
    ]
    valid = [(name, img) for name, img in panels if isinstance(img, np.ndarray)]
    if not valid:
        return np.zeros((320, 640, 3), dtype=np.uint8)
    cell_h, cell_w = 260, 260
    rows = 2
    cols = 4
    grid = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, (name, img) in enumerate(valid[:8]):
        r, c = divmod(idx, cols)
        panel = img
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (cell_w, cell_h - 28), interpolation=cv2.INTER_AREA)
        y0, x0 = r * cell_h, c * cell_w
        grid[y0 + 28 : y0 + cell_h, x0 : x0 + cell_w] = panel
        cv2.putText(grid, name, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    text = f"Contagem: {result['count']} | {result['segmentation_mode']} | {result['mode_used']}"
    if title:
        text = f"{title} | {text}"
    cv2.putText(grid, text[:100], (8, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2, cv2.LINE_AA)
    return grid


def save_debug_grid(result: dict[str, Any], output_path: str | Path, title: str = "") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), make_debug_grid(result, title=title))
    return output_path


def grid_search_parameters(images: list[np.ndarray], labels: list[int]) -> dict[str, Any]:
    """
    Busca simples de parametros para calibracao, separada da inferencia.

    Otimiza MAE e usa acuracia exata como criterio de desempate.
    """
    from metrics import exact_match_accuracy, mean_absolute_error

    search_space = {
        "min_area": [120, 220, 300, 450],
        "max_area_ratio": [0.08, 0.12, 0.18],
        "close_kernel": [5, 9, 13],
        "open_kernel": [1, 3, 5],
        "blur": [True, False],
        "adaptive_threshold": [True, False],
        "watershed": [True, False],
        "min_distance": [None, 12, 18, 26],
    }
    keys = list(search_space)
    rows = []
    best: dict[str, Any] | None = None
    for values in product(*[search_space[k] for k in keys]):
        cfg = dict(zip(keys, values))
        cfg.update({"ensemble": True, "mode": "auto"})
        preds = [count_screws(img, config=cfg, return_debug=False)["count"] for img in images]
        mae = mean_absolute_error(labels, preds)
        acc = exact_match_accuracy(labels, preds)
        over = sum(1 for y, p in zip(labels, preds) if p > y)
        under = sum(1 for y, p in zip(labels, preds) if p < y)
        row = {**cfg, "mae": mae, "exact_match_accuracy": acc, "overestimates": over, "underestimates": under, "predictions": preds}
        rows.append(row)
        if best is None or (mae, -acc, over + under) < (best["mae"], -best["exact_match_accuracy"], best["overestimates"] + best["underestimates"]):
            best = row
    return {"best_config": {k: best[k] for k in keys}, "best_metrics": {k: best[k] for k in ["mae", "exact_match_accuracy", "overestimates", "underestimates", "predictions"]}, "results": rows}


def calibrate_from_single_screw(img_bgr: np.ndarray) -> dict[str, Any]:
    """
    Estima ref_area a partir de uma foto com exatamente 1 parafuso isolado.

    O colaborador centraliza 1 parafuso no campo de visao e fotografa
    na mesma distancia que usara no trabalho. Esta funcao segmenta a imagem
    e mede a area do maior componente valido, devolvendo o valor a ser
    usado como ref_area nas contagens seguintes.

    Returns:
        dict com ref_area (float), confidence ('alta'/'media'/'baixa'),
        message (str orientativo) e debug_image (np.ndarray RGB anotada).
    """
    result = count_screws(
        img_bgr,
        config={"ensemble": True, "use_ref_auto": False, "ref_area": 4021.0},
        return_debug=True,
    )
    detections = [d for d in result.get("detections", []) if float(d.get("area", 0)) > 200]

    if not detections:
        return {
            "ref_area": 4021.0,
            "confidence": "baixa",
            "message": "Nenhum objeto detectado. Tente com iluminacao melhor "
                       "e parafuso centralizado em fundo liso.",
            "debug_image": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            "n_detected": 0,
        }

    largest = max(detections, key=lambda d: float(d.get("area", 0)))
    ref_area = float(largest["area"])
    n_detected = len(detections)

    if n_detected == 1:
        confidence = "alta"
        message = (
            f"Calibracao concluida com alta confianca. "
            f"Area de referencia = {ref_area:.0f} px2. Pronto para contar!"
        )
    elif n_detected <= 3:
        confidence = "media"
        message = (
            f"Detectei {n_detected} objetos; usei o maior ({ref_area:.0f} px2). "
            f"Para maior precisao, remova outros objetos do fundo."
        )
    else:
        confidence = "baixa"
        ref_area = float(np.median([float(d.get("area", 0)) for d in detections]))
        message = (
            f"Detectei {n_detected} objetos — muitos para calibracao confiavel. "
            f"Fotografe apenas 1 parafuso isolado no centro da imagem."
        )

    debug_bgr = draw_detections(img_bgr, {"count": 1, "detections": [largest]})
    cv2.putText(
        debug_bgr,
        f"ref_area calibrada = {ref_area:.0f} px2",
        (10, debug_bgr.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 0), 2, cv2.LINE_AA,
    )
    debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)

    return {
        "ref_area": ref_area,
        "confidence": confidence,
        "message": message,
        "debug_image": debug_rgb,
        "n_detected": n_detected,
    }


# Compatibilidade com versoes antigas.
def segment_adaptive(img_bgr: np.ndarray, min_area: int = 300):
    result = count_screws(img_bgr, config={"min_area": min_area, "ensemble": False}, return_debug=True)
    return result["mask"], result["segmentation_mode"], result["noise_stats"]


def preprocess(img_bgr: np.ndarray, min_area: int = 300):
    enhanced = _enhance_gray(img_bgr)
    mask, _, _ = segment_adaptive(img_bgr, min_area=min_area)
    return enhanced, mask


def filter_components(mask: np.ndarray, image_area: int, min_area: int = 300) -> list[Component]:
    """
    Compatibilidade com o notebook v3.

    A versao nova usa `_extract_components`; esta funcao preserva o nome antigo
    esperado pelo notebook.
    """
    cfg = DEFAULT_CONFIG.copy()
    cfg["min_area"] = int(min_area)
    return _extract_components(mask, cfg, "compat_filter")


def count_with_watershed(comp: Component, ref_area: float) -> int:
    """Compatibilidade: estima multiplicidade de um componente via DT + peaks."""
    _, _, n_peaks = _distance_peaks(comp.mask, ref_area)
    area_est = max(1, int(round(comp.area / max(ref_area, 1.0))))
    if 1 <= n_peaks <= area_est + 4:
        return int(n_peaks)
    return int(area_est)


def get_watershed_debug(comp: Component, ref_area: float):
    """Compatibilidade: retorna distance transform, mascara de peaks e contagem."""
    dist, peaks, _ = _distance_peaks(comp.mask, ref_area)
    return dist, peaks, count_with_watershed(comp, ref_area)


def handle_fragmented_object(valid_components: list[Component], ref_area: float) -> list[Component]:
    """
    Compatibilidade com o agrupamento antigo.

    Para manter o notebook funcionando, agrupa fragmentos pequenos por DBSCAN
    quando ha muitos componentes menores que a area de referencia.
    """
    if not valid_components:
        return valid_components
    small = [c for c in valid_components if c.area < ref_area * 0.7]
    if len(small) < max(4, len(valid_components) * 0.6):
        return valid_components
    h, w = valid_components[0].mask.shape[:2]
    pts = np.array([[c.cx, c.cy] for c in small], dtype=np.float32)
    eps = max(np.sqrt(ref_area) * 2.5, min(h, w) * 0.10)
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(pts) if HAS_SKLEARN else np.zeros(len(pts), dtype=int)
    grouped: list[Component] = []
    for cluster_id in sorted(set(labels)):
        members = [c for c, label in zip(small, labels) if label == cluster_id]
        union = np.zeros((h, w), dtype=np.uint8)
        for member in members:
            union = cv2.bitwise_or(union, member.mask)
        x, y, bw, bh = cv2.boundingRect(union)
        area = float(sum(m.area for m in members))
        grouped.append(
            Component(
                label=int(cluster_id),
                area=area,
                bbox=(x, y, bw, bh),
                cx=float(np.mean([m.cx for m in members])),
                cy=float(np.mean([m.cy for m in members])),
                aspect_ratio=float(max(bw, bh) / max(min(bw, bh), 1)),
                solidity=float(np.mean([m.solidity for m in members])),
                extent=float(area / max(bw * bh, 1)),
                mask=union,
                source="compat_dbscan",
                n_frags=len(members),
                gradient_variance=float(np.mean([m.gradient_variance for m in members])),
            )
        )
    return grouped


def classify_as_screw(comp: Component, threshold_gv: float = 0.0, image_area: int | None = None, min_area: int = 300) -> bool:
    """Compatibilidade: classificador geometrico simples para celulas antigas."""
    if comp.area < min_area:
        return False
    if image_area is not None and comp.area > image_area * DEFAULT_CONFIG["max_area_ratio"]:
        return False
    if not (1.0 <= comp.aspect_ratio <= 14.0):
        return False
    if comp.solidity < 0.08 or comp.extent < 0.03:
        return False
    return True


def draw_dt_peaks(comp: Component, ref_area: float) -> np.ndarray:
    """Compatibilidade: imagem colorida do distance transform com peaks."""
    dist, peak_mask, _ = get_watershed_debug(comp, ref_area)
    dist_color = cv2.applyColorMap(_normalize_u8(dist), cv2.COLORMAP_HOT)
    n_peaks, _, _, centroids = cv2.connectedComponentsWithStats(peak_mask, 8)
    for i in range(1, n_peaks):
        px, py = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(dist_color, (px, py), 5, (0, 255, 0), -1)
        cv2.circle(dist_color, (px, py), 6, (0, 0, 0), 1)
    return dist_color


def count_screws_for_app(
    image_bgr: np.ndarray,
    ref_area_override: float | None = None,
) -> dict[str, Any]:
    """
    Interface limpa para o aplicativo Streamlit operacional.

    Esconde parametros tecnicos do usuario final e devolve somente os artefatos
    necessarios para exibicao, auditoria e area tecnica.
    """
    h, w = image_bgr.shape[:2]
    target_w = 1100
    scale = 1.0
    image_proc = image_bgr
    if w < 500:
        scale = target_w / float(w)
        image_proc = cv2.resize(image_bgr, (target_w, int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
    elif w > 1800:
        scale = 1400 / float(w)
        image_proc = cv2.resize(image_bgr, (1400, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    base_ref = float(ref_area_override) if ref_area_override is not None else 4021.0

    candidate_configs = [
        {
            "mode": "auto",
            "ensemble": True,
            "ref_area": base_ref * scale * scale,
            "min_area": int(max(120, 300 * scale * scale)),
            "use_ref_auto": True,
            "watershed": True,
            "blur": True,
            "close_kernel": 9,
            "open_kernel": 3,
        },
        {
            "mode": "modo_multiescala",
            "ensemble": False,
            "ref_area": base_ref * scale * scale,
            "min_area": int(max(90, 220 * scale * scale)),
            "use_ref_auto": True,
            "watershed": True,
            "blur": True,
            "close_kernel": 7,
            "open_kernel": 1,
        },
        {
            "mode": "modo_cinza",
            "ensemble": False,
            "ref_area": base_ref * scale * scale,
            "min_area": int(max(100, 250 * scale * scale)),
            "use_ref_auto": True,
            "watershed": True,
            "blur": True,
            "close_kernel": 11,
            "open_kernel": 3,
        },
    ]

    results = [count_screws(image_proc, config=candidate_configs[0], return_debug=True)]
    if int(results[0].get("count", 0)) == 0:
        results.extend(
            count_screws(image_proc, config=cfg, return_debug=True)
            for cfg in candidate_configs[1:]
        )
    result = _select_app_result(results)
    quality = _estimate_capture_quality(image_proc, result)

    if scale != 1.0:
        result["debug_images"]["original"] = image_proc.copy()

    result = result
    app_original = image_proc

    # Recalcula a imagem anotada sobre a imagem processada, preservando caixas e
    # contornos coerentes com a escala usada internamente.
    annotated = draw_detections(app_original, result)

    debug = result.get("debug_images", {})
    debug["result"] = annotated
    debug["original"] = app_original

    metadata = {
        "mode": result.get("segmentation_mode"),
        "detector": result.get("mode_used"),
        "noise_ratio": result.get("noise_stats", {}).get("noise_ratio"),
        "n_big": result.get("noise_stats", {}).get("n_big"),
        "watershed": result.get("watershed_used"),
        "fragmentation": result.get("fragmentacao_detectada"),
        "processing_time_ms": result.get("processing_time_ms"),
        "low_confidence": result.get("metrics_internal", {}).get("low_confidence", False),
        "candidate_counts": result.get("metrics_internal", {}).get("candidate_counts", {}),
        "app_scale": scale,
        "quality": quality,
    }
    return {
        "count": int(result["count"]),
        "original": app_original,
        "mask": debug.get("mask", result.get("mask")),
        "enhanced": debug.get("enhanced", debug.get("preprocessed")),
        "annotated": annotated,
        "debug_images": debug,
        "detections": result.get("detections", []),
        "metadata": metadata,
        "raw_result": result,
    }


def _select_app_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Escolhe o resultado operacional mais estavel entre configuracoes leves."""
    positive = [r for r in results if int(r.get("count", 0)) > 0]
    if not positive:
        return results[0]
    counts = [int(r["count"]) for r in positive]
    median = float(np.median(counts))

    def score(result: dict[str, Any]) -> float:
        detections = result.get("detections", [])
        conf = float(np.mean([d.get("confidence_proxy", 0.4) for d in detections])) if detections else 0.2
        disagreement = abs(int(result["count"]) - median)
        low_conf = bool(result.get("metrics_internal", {}).get("low_confidence", False))
        border_penalty = _border_touch_ratio(result)
        return conf - disagreement * 0.22 - border_penalty * 0.35 - (0.2 if low_conf else 0.0)

    return max(positive, key=score)


def _border_touch_ratio(result: dict[str, Any]) -> float:
    mask = result.get("mask")
    if not isinstance(mask, np.ndarray):
        return 0.0
    h, w = mask.shape[:2]
    margin = max(6, int(min(h, w) * 0.02))
    total = max(float(np.count_nonzero(mask)), 1.0)
    border = (
        np.count_nonzero(mask[:margin, :])
        + np.count_nonzero(mask[-margin:, :])
        + np.count_nonzero(mask[:, :margin])
        + np.count_nonzero(mask[:, -margin:])
    )
    return float(border / total)


def _estimate_capture_quality(image_bgr: np.ndarray, result: dict[str, Any]) -> dict[str, Any]:
    """Gera alertas simples sobre distancia, corte e qualidade da foto."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    detections = [d for d in result.get("detections", []) if int(d.get("count", 0)) > 0]
    count = max(int(result.get("count", 0)), 1)
    image_area = h * w
    area_ratio = float(sum(d.get("area", 0.0) for d in detections) / max(image_area, 1))
    border_ratio = _border_touch_ratio(result)
    warnings: list[str] = []
    if blur_score < 45:
        warnings.append("A foto parece tremida. Se possivel, tire novamente com o celular firme.")
    if area_ratio < 0.015 and count > 1:
        warnings.append("Os parafusos parecem pequenos na imagem. Aproxime um pouco a camera.")
    if area_ratio > 0.42:
        warnings.append("A foto parece muito aproximada. Afaste um pouco a camera para incluir todos os parafusos.")
    if border_ratio > 0.10:
        warnings.append("Ha parafusos encostando ou saindo da borda. Centralize melhor as pecas.")
    if bool(result.get("metrics_internal", {}).get("low_confidence", False)):
        warnings.append("A deteccao teve baixa confianca. Confira a contagem antes de salvar.")
    return {
        "blur_score": blur_score,
        "object_area_ratio": area_ratio,
        "border_touch_ratio": border_ratio,
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }
