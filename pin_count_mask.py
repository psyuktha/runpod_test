"""
Pin counting without rotation:
- load Canny-edge image
- find largest rectangle (chip body) via nonzero percentile bbox
- mask out the body; remaining edges are pins
- connected components to count pins
- assign blobs to sides relative to the body; choose active sides by aspect ratio (2 or 4)
- total pins = max pins per active side * number of active sides
- save annotated image with body box and pin boxes
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


def connected_components(bin_img: np.ndarray, min_area: int = 5):
    """Simple two-pass 4-connected components; returns list of dicts."""
    h, w = bin_img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]  # 1-based
    next_label = 1
    for y in range(h):
        for x in range(w):
            if bin_img[y, x] == 0:
                continue
            neighbors = []
            if x > 0 and labels[y, x - 1] > 0:
                neighbors.append(labels[y, x - 1])
            if y > 0 and labels[y - 1, x] > 0:
                neighbors.append(labels[y - 1, x])
            if not neighbors:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                m = min(neighbors)
                labels[y, x] = m
                for n in neighbors:
                    if parent[n] != m:
                        parent[n] = m

    # Flatten parents
    for i in range(len(parent)):
        while parent[i] != parent[parent[i]]:
            parent[i] = parent[parent[i]]

    # Compact mapping
    mapping = {}
    compact = 1
    for y in range(h):
        for x in range(w):
            if labels[y, x] == 0:
                continue
            root = parent[labels[y, x]]
            if root not in mapping:
                mapping[root] = compact
                compact += 1
            labels[y, x] = mapping[root]

    components = []
    if compact == 1:
        return components
    for lab in range(1, compact):
        ys, xs = np.where(labels == lab)
        area = len(xs)
        if area < min_area:
            continue
        x1c, x2c = xs.min(), xs.max()
        y1c, y2c = ys.min(), ys.max()
        cx_c = float(xs.mean())
        cy_c = float(ys.mean())
        components.append(
            {
                "label": lab,
                "area": area,
                "bbox": (x1c, y1c, x2c, y2c),
                "centroid": (cx_c, cy_c),
            }
        )
    return components


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

    steps.append("Mask out the largest rectangle (chip body).")
    mask = np.ones_like(edges, dtype=np.uint8)
    x1 = int(max(0, cx - (w + pad) / 2))
    x2 = int(min(edges.shape[1], cx + (w + pad) / 2))
    y1 = int(max(0, cy - (h + pad) / 2))
    y2 = int(min(edges.shape[0], cy + (h + pad) / 2))
    mask[y1:y2, x1:x2] = 0
    pins_only = (edges * mask).astype(np.uint8)

    steps.append("Detect connected pin blobs after body removal.")
    pins_bin = (pins_only > 0).astype(np.uint8)
    comps = connected_components(pins_bin, min_area=5)

    steps.append("Assign pin blobs to sides.")
    side_order = ["top", "right", "bottom", "left"]
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

    band = int(max(body_box[1][0], body_box[1][1]) * 0.12)
    pins_per_side_map = {s: 0 for s in side_order}
    boxes_by_side: Dict[str, List[Tuple[int, int, int, int]]] = {s: [] for s in side_order}
    for comp in comps:
        cx_c, cy_c = comp["centroid"]
        x1c, y1c, x2c, y2c = comp["bbox"]
        side = None
        if cy_c < y1 - band:
            side = "top"
        elif cy_c > y2 + band:
            side = "bottom"
        elif cx_c < x1 - band:
            side = "left"
        elif cx_c > x2 + band:
            side = "right"
        else:
            dist_top = abs(cy_c - y1)
            dist_bot = abs(cy_c - y2)
            dist_left = abs(cx_c - x1)
            dist_right = abs(cx_c - x2)
            side = ["top", "bottom", "left", "right"][
                np.argmin([dist_top, dist_bot, dist_left, dist_right])
            ]
        if side in active_sides:
            pins_per_side_map[side] += 1
            boxes_by_side[side].append((int(x1c), int(y1c), int(x2c), int(y2c)))

    pins_per_side = [pins_per_side_map[s] for s in side_order]
    max_pins = max([pins_per_side_map[s] for s in active_sides]) if active_sides else 0
    total_pins = max_pins * len(active_sides)
    sides_with_pins = len(active_sides)
    side_energy = [float(pins_per_side_map[s]) for s in side_order]

    steps.append("Compute totals using max-per-active-side rule.")

    side_results: List[SideResult] = []
    for s in side_order:
        side_results.append(
            SideResult(
                name=s,
                count=pins_per_side_map[s],
                energy=pins_per_side_map[s],
                boxes=boxes_by_side[s],
            )
        )

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
        side_energy=side_energy,
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
    OUTPUT_DIR = "bi_full_annotated_mask"
    res = run_batch(INPUT_DIR, OUTPUT_DIR)
    with open("pin_counts_mask.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved pin_counts_mask.json")

