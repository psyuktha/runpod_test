#!/usr/bin/env python3
"""
Annotate IC pins, count per side, pick the most uniform side (spacing),
and estimate total pins assuming symmetry.

Defaults are tuned for the provided canny-style outlines:
- min_area: 120
- max_area: 800
- min_side: 3
- threshold: 128
- side strip: 7% of the smaller package dimension (at least 3px)

Usage (from repo root):
  python pin_uniform_annotate.py bi_full/bilateral_canny_anu4.jpg \
      --out-image bi_full/anu4_uniform_annotated.png
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_mask(path: Path, threshold: int) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > threshold).astype(np.uint8)


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int, List[int], List[Tuple[int, int, int, int]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    areas: List[int] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or labels[y, x] != 0:
                continue
            label += 1
            q = deque([(y, x)])
            labels[y, x] = label
            area = 0
            min_x = max_x = x
            min_y = max_y = y
            while q:
                cy, cx = q.popleft()
                area += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx] == 1
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = label
                        q.append((ny, nx))
            areas.append(area)
            bboxes.append((min_x, min_y, max_x, max_y))
    return labels, label, areas, bboxes


def package_bbox_from_components(areas: List[int], bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if not areas:
        return (0, 0, 0, 0)
    # Use overall extent of larger components (top 80% by area)
    sorted_idx = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    keep_n = max(3, int(len(sorted_idx) * 0.8))
    sel = sorted_idx[:keep_n]
    xs_min = [bboxes[i][0] for i in sel]
    ys_min = [bboxes[i][1] for i in sel]
    xs_max = [bboxes[i][2] for i in sel]
    ys_max = [bboxes[i][3] for i in sel]
    margin = 3
    return (min(xs_min) - margin, min(ys_min) - margin, max(xs_max) + margin, max(ys_max) + margin)


def side_for_pin(pin_bbox: Tuple[int, int, int, int], pkg_bbox: Tuple[int, int, int, int], strip_ratio: float = 0.07) -> str:
    px_min, py_min, px_max, py_max = pkg_bbox
    pw = px_max - px_min + 1
    ph = py_max - py_min + 1
    strip = max(3, int(strip_ratio * min(pw, ph)))
    min_x, min_y, max_x, max_y = pin_bbox
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    if cy <= py_min + strip:
        return "top"
    if cy >= py_max - strip:
        return "bottom"
    if cx <= px_min + strip:
        return "left"
    if cx >= px_max - strip:
        return "right"
    # fallback: nearest edge
    dists = {
        "top": abs(cy - py_min),
        "bottom": abs(py_max - cy),
        "left": abs(cx - px_min),
        "right": abs(px_max - cx),
    }
    return min(dists.items(), key=lambda kv: (kv[1], kv[0]))[0]


def spacing_score(centers: List[Tuple[float, float]], side: str) -> float:
    if len(centers) < 2:
        return float("inf")
    if side in ("top", "bottom"):
        coords = sorted(c[0] for c in centers)
    else:
        coords = sorted(c[1] for c in centers)
    spacings = np.diff(coords)
    m = np.mean(spacings)
    if m <= 0:
        return float("inf")
    return float(np.std(spacings) / m)


def analyze(
    path: Path,
    min_area: int,
    max_area: int,
    min_side: int,
    threshold: int,
) -> Tuple[Dict[str, int], Dict[str, List[int]], Dict[str, float], str, int, List[Tuple[int, int, int, int]], List[int], Tuple[int, int, int, int]]:
    mask = load_mask(path, threshold)
    _, num_labels, areas, bboxes = label_components(mask)

    pkg_bbox = package_bbox_from_components(areas, bboxes)

    # filter pin-like components
    pins_labels: List[int] = []
    pins_bboxes: List[Tuple[int, int, int, int]] = []
    for idx, area in enumerate(areas, start=1):
        bb = bboxes[idx - 1]
        w = bb[2] - bb[0] + 1
        h = bb[3] - bb[1] + 1
        if min_area <= area <= max_area and w >= min_side and h >= min_side:
            pins_labels.append(idx)
            pins_bboxes.append(bb)

    side_counts: Dict[str, int] = {"top": 0, "bottom": 0, "left": 0, "right": 0, "unknown": 0}
    side_labels: Dict[str, List[int]] = {k: [] for k in side_counts}
    side_centers: Dict[str, List[Tuple[float, float]]] = {k: [] for k in side_counts}

    for lbl, bb in zip(pins_labels, pins_bboxes):
        side = side_for_pin(bb, pkg_bbox)
        side_counts[side] += 1
        side_labels[side].append(lbl)
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        side_centers[side].append((cx, cy))

    side_scores: Dict[str, float] = {}
    for s in ("top", "bottom", "left", "right"):
        side_scores[s] = spacing_score(side_centers[s], s)

    valid_sides = [s for s in ("top", "bottom", "left", "right") if side_counts[s] > 0]
    best_side = min(valid_sides, key=lambda s: (side_scores[s], -side_counts[s], s)) if valid_sides else None
    symmetric_estimate = side_counts.get(best_side, 0) * len(valid_sides) if best_side else 0

    return side_counts, side_labels, side_scores, best_side or "unknown", symmetric_estimate, pins_bboxes, pins_labels, pkg_bbox


def draw_overlay(
    image_path: Path,
    out_path: Path,
    pins_bboxes: List[Tuple[int, int, int, int]],
    pins_labels: List[int],
    pkg_bbox: Tuple[int, int, int, int],
):
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    side_colors = {
        "top": (0, 255, 0),
        "bottom": (0, 128, 255),
        "left": (255, 165, 0),
        "right": (255, 0, 255),
        "unknown": (180, 180, 180),
    }
    # package outline
    draw.rectangle([(pkg_bbox[0], pkg_bbox[1]), (pkg_bbox[2], pkg_bbox[3])], outline=(0, 255, 255), width=3)

    for lbl, bb in zip(pins_labels, pins_bboxes):
        side = side_for_pin(bb, pkg_bbox)
        color = side_colors.get(side, (255, 255, 0))
        draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], outline=color, width=3)
        draw.text((bb[0], max(bb[1] - 10, 0)), str(lbl), fill=color, font=font)

    base.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate pins, count per side, choose best side by spacing.")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("--min-area", type=int, default=120, help="Minimum area for a pin component")
    parser.add_argument("--max-area", type=int, default=800, help="Maximum area for a pin component")
    parser.add_argument("--min-side", type=int, default=3, help="Minimum side length (pixels) for a pin component")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold (white > threshold)")
    parser.add_argument("--out-image", type=Path, help="Optional path to save annotated image")
    args = parser.parse_args()

    side_counts, side_labels, side_scores, best_side, symmetric_estimate, pins_bboxes, pins_labels, pkg_bbox = analyze(
        args.image, args.min_area, args.max_area, args.min_side, args.threshold
    )

    print(f"Per-side counts: {side_counts}")
    print(f"Per-side labels: {side_labels}")
    print(f"Per-side spacing scores (CV): {side_scores}")
    print(f"Best side by spacing: {best_side}")
    print(f"Symmetric estimate: {symmetric_estimate}")

    if args.out_image:
        draw_overlay(args.image, args.out_image, pins_bboxes, pins_labels, pkg_bbox)
        print(f"Saved annotated image to: {args.out_image}")


if __name__ == "__main__":
    main()


