"""
Projection-based pin counter (no rotation):
- load Canny-edge image (grayscale)
- find body box from nonzero percentiles (largest rectangle)
- sample thin bands outside each side; sum projection per band
- detect peaks in projections as pins
- pick active sides by aspect ratio (2 if long, else 4)
- total pins = max pins per active side * number of active sides
- save annotated image to bi_full_annotated_proj and JSON to pin_counts_proj.json
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class SideResult:
    name: str
    count: int
    energy: float
    boxes: List[Tuple[int, int, int, int]]


@dataclass
class ImageResult:
    filename: str
    sides_with_pins: int
    pins_per_side: List[int]
    max_pins_per_side: int
    total_pins: int
    side_names: List[str]
    steps: List[str]
    side_energy: List[float]


def load_edges(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def smooth_signal(signal: np.ndarray) -> np.ndarray:
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)
    kernel = kernel / kernel.sum()
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def simple_peaks(signal: np.ndarray, min_dist: int, threshold_rel: float) -> List[int]:
    if signal.size == 0:
        return []
    smoothed = smooth_signal(signal)
    thr = threshold_rel * float(smoothed.max() if smoothed.max() > 0 else 1.0)
    peaks = []
    last = -min_dist
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > thr and smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if i - last >= min_dist:
                peaks.append(i)
                last = i
    return peaks


def annotate(base_gray: np.ndarray, rect, side_results: List[SideResult]) -> np.ndarray:
    color = np.stack([base_gray] * 3, axis=2)
    img = Image.fromarray(color)
    draw = ImageDraw.Draw(img)
    (cx, cy), (w, h) = rect
    box = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]
    draw.line(box + [box[0]], fill=(0, 255, 0), width=2)
    side_colors = {
        "top": (0, 255, 255),
        "bottom": (255, 0, 0),
        "left": (255, 0, 255),
        "right": (0, 128, 255),
    }
    for side_res in side_results:
        color_side = side_colors.get(side_res.name, (255, 255, 0))
        for (x1, y1, x2, y2) in side_res.boxes:
            draw.rectangle((x1, y1, x2, y2), outline=color_side, width=1)
    return np.array(img)


def process_image(path: str, out_dir: str) -> ImageResult:
    steps = []
    steps.append("Load Canny-edge image.")
    edges = load_edges(path)

    steps.append("Locate non-zero pixels to find package body (no rotation).")
    nz = np.argwhere(edges > 0)
    if nz.shape[0] == 0:
        raise ValueError("No edges found")
    y_coords = nz[:, 0]
    x_coords = nz[:, 1]
    min_y = np.percentile(y_coords, 2)
    max_y = np.percentile(y_coords, 98)
    min_x = np.percentile(x_coords, 2)
    max_x = np.percentile(x_coords, 98)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    w = max_x - min_x
    h = max_y - min_y
    pad = 4
    body_box = ((cx, cy), (w + pad, h + pad))

    steps.append("Define side bands and compute projections.")
    # Bands extend just outside the body box
    band = int(max(body_box[1][0], body_box[1][1]) * 0.12)
    x1 = int(max(0, cx - (w + pad) / 2))
    x2 = int(min(edges.shape[1], cx + (w + pad) / 2))
    y1 = int(max(0, cy - (h + pad) / 2))
    y2 = int(min(edges.shape[0], cy + (h + pad) / 2))

    def band_and_projection(side: str):
        if side == "top":
            y_start = max(0, y1 - band)
            y_end = max(0, y1)
            band_img = edges[y_start:y_end, x1:x2]
            projection = band_img.sum(axis=0)
        elif side == "bottom":
            y_start = min(edges.shape[0], y2)
            y_end = min(edges.shape[0], y2 + band)
            band_img = edges[y_start:y_end, x1:x2]
            projection = band_img.sum(axis=0)
        elif side == "left":
            x_start = max(0, x1 - band)
            x_end = max(0, x1)
            band_img = edges[y1:y2, x_start:x_end]
            projection = band_img.sum(axis=1)
        else:  # right
            x_start = min(edges.shape[1], x2)
            x_end = min(edges.shape[1], x2 + band)
            band_img = edges[y1:y2, x_start:x_end]
            projection = band_img.sum(axis=1)
        return band_img, projection

    side_order = ["top", "right", "bottom", "left"]
    side_results: List[SideResult] = []
    energies: List[float] = []

    for side in side_order:
        band_img, projection = band_and_projection(side)
        projection = projection.astype(np.float32)
        if projection.size > 0:
            projection = projection - projection.min()
        energy = float(projection.sum())
        norm = projection / (projection.max() + 1e-6) if projection.size else projection
        mean = float(norm.mean()) if projection.size else 0.0
        std = float(norm.std()) if projection.size else 0.0
        thr_rel = max(0.18, mean + 0.35 * std)
        peaks = simple_peaks(
            projection,
            min_dist=max(4, int(len(projection) * 0.02)),
            threshold_rel=thr_rel,
        )
        boxes: List[Tuple[int, int, int, int]] = []
        if side in ("top", "bottom"):
            for p in peaks:
                x0 = max(0, p - 2)
                x1b = min(band_img.shape[1] - 1, p + 2)
                boxes.append((x0 + x1, 0 if side == "top" else band_img.shape[0] + y1, x1b + x1, (band_img.shape[0] - 1) + (0 if side == "top" else y1)))
        else:
            for p in peaks:
                y0 = max(0, p - 2)
                y1b = min(band_img.shape[0] - 1, p + 2)
                boxes.append((0 if side == "left" else band_img.shape[1] + x1, y0 + y1, (band_img.shape[1] - 1) + (0 if side == "left" else x1), y1b + y1))

        energies.append(energy)
        side_results.append(
            SideResult(
                name=side,
                count=len(peaks),
                energy=energy,
                boxes=boxes,
            )
        )

    steps.append("Determine active sides by aspect ratio (2 if long, else 4).")
    aspect = (max(body_box[1][0], body_box[1][1]) + 1e-6) / (
        min(body_box[1][0], body_box[1][1]) + 1e-6
    )
    if aspect >= 1.6:
        if body_box[1][0] >= body_box[1][1]:
            active_sides = ["top", "bottom"]
        else:
            active_sides = ["left", "right"]
    else:
        active_sides = side_order

    pins_per_side = [sr.count for sr in side_results]
    max_pins = max([sr.count for sr in side_results if sr.name in active_sides]) if active_sides else 0
    total_pins = max_pins * len(active_sides)
    sides_with_pins = len(active_sides)

    steps.append("Compute totals: total = max pins on active side * number of active sides.")

    annotated = annotate(edges, body_box, side_results)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(path))
    Image.fromarray(annotated).save(out_path)
    steps.append(f"Saved annotated image to {out_path}")

    return ImageResult(
        filename=os.path.basename(path),
        sides_with_pins=sides_with_pins,
        pins_per_side=pins_per_side,
        max_pins_per_side=max_pins,
        total_pins=total_pins,
        side_names=side_order,
        steps=steps,
        side_energy=energies,
    )


def run_batch(input_dir: str, output_dir: str) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    for fname in files:
        path = os.path.join(input_dir, fname)
        try:
            res = process_image(path, output_dir)
            results[fname] = asdict(res)
            print(f"\n=== {fname} ===")
            print(f"Sides with pins: {res.sides_with_pins}")
            for n, c, e in zip(res.side_names, res.pins_per_side, res.side_energy):
                print(f"  {n:6s}: count={c:2d} energy={e:.1f}")
            print(f"Max pins per side: {res.max_pins_per_side}")
            print(f"Total pins (max * sides): {res.total_pins}")
            print("Steps:")
            for s in res.steps:
                print(f" - {s}")
        except Exception as exc:
            print(f"[ERROR] {fname}: {exc}")
    return results


if __name__ == "__main__":
    import json

    INPUT_DIR = "bi_full"
    OUTPUT_DIR = "bi_full_annotated_proj"
    res = run_batch(INPUT_DIR, OUTPUT_DIR)
    with open("pin_counts_proj.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved pin_counts_proj.json")

