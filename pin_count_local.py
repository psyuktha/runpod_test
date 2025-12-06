"""
Local pin counter for Canny-edge IC images (no OpenCV runtime needed).

Pipeline per image:
- load Canny edge image (grayscale)
- find package bounding box using non-zero pixels + PCA for orientation
- rotate image to align package with axes (via Pillow)
- sample thin bands outside each side, detect peaks = pins
- decide which sides contain pins by band energy
- total pins = max(pins_per_side) * number_of_pin_sides
- save annotated image with body box and per-pin bounding boxes
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
    boxes: List[Tuple[int, int, int, int]]  # x1, y1, x2, y2 in rotated coords


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


def pca_angle(coords: np.ndarray) -> float:
    """Return principal axis angle in degrees."""
    # coords shape (N,2)
    coords_centered = coords - coords.mean(axis=0)
    cov = np.cov(coords_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = math.degrees(math.atan2(major[1], major[0]))
    return angle


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate using Pillow; fill empty with black."""
    pil = Image.fromarray(img)
    rotated = pil.rotate(-angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return np.array(rotated)


def simple_peaks(signal: np.ndarray, min_dist: int = 4, threshold_rel: float = 0.2) -> List[int]:
    """Lightweight peak finder without SciPy."""
    if signal.size == 0:
        return []
    # Smooth with 1D Gaussian
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(signal.astype(np.float32), kernel, mode="same")
    thr = threshold_rel * float(smoothed.max() if smoothed.max() > 0 else 1.0)
    peaks = []
    last = -min_dist
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > thr and smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if i - last >= min_dist:
                peaks.append(i)
                last = i
    return peaks


def count_side(rotated: np.ndarray, rect, side: str, band: int = 18, margin: int = 12) -> SideResult:
    """Count pins on one side using projection + peak detection."""
    (cx, cy), (w, h) = rect
    x1 = int(cx - w / 2 - margin)
    x2 = int(cx + w / 2 + margin)
    y1 = int(cy - h / 2 - margin)
    y2 = int(cy + h / 2 + margin)

    if side == "top":
        y_start = max(0, y1 - band)
        y_end = max(0, y1 + band)
        band_img = rotated[y_start:y_end, max(0, x1) : min(rotated.shape[1], x2)]
        projection = band_img.sum(axis=0)
    elif side == "bottom":
        y_start = min(rotated.shape[0], y2 - band)
        y_end = min(rotated.shape[0], y2 + band)
        band_img = rotated[y_start:y_end, max(0, x1) : min(rotated.shape[1], x2)]
        projection = band_img.sum(axis=0)
    elif side == "left":
        x_start = max(0, x1 - band)
        x_end = max(0, x1 + band)
        band_img = rotated[max(0, y1) : min(rotated.shape[0], y2), x_start:x_end]
        projection = band_img.sum(axis=1)
    elif side == "right":
        x_start = min(rotated.shape[1], x2 - band)
        x_end = min(rotated.shape[1], x2 + band)
        band_img = rotated[max(0, y1) : min(rotated.shape[0], y2), x_start:x_end]
        projection = band_img.sum(axis=1)
    else:
        raise ValueError("Unknown side")

    # Normalize projection to make thresholding more stable across images
    projection = projection.astype(np.float32)
    if projection.size > 0:
        projection = projection - projection.min()
    energy = float(projection.sum())
    # Dynamic threshold: mean + 0.35*std, with a floor at 18% of max
    if projection.size == 0:
        peaks = []
    else:
        norm = projection / (projection.max() + 1e-6)
        mean = float(norm.mean())
        std = float(norm.std())
        thr_rel = max(0.18, mean + 0.35 * std)
        peaks = simple_peaks(
            projection,
            min_dist=max(4, int(len(projection) * 0.02)),
            threshold_rel=thr_rel,
        )

    # Build bounding boxes from peaks (fixed-size boxes within the band)
    boxes: List[Tuple[int, int, int, int]] = []
    if side in ("top", "bottom"):
        for p in peaks:
            x0 = max(0, p - 2)
            x1b = min(band_img.shape[1] - 1, p + 2)
            boxes.append((x0, 0, x1b, band_img.shape[0] - 1))
    else:
        for p in peaks:
            y0 = max(0, p - 2)
            y1b = min(band_img.shape[0] - 1, p + 2)
            boxes.append((0, y0, band_img.shape[1] - 1, y1b))

    return SideResult(name=side, count=len(peaks), energy=energy, boxes=boxes)


def annotate(rotated_color: np.ndarray, rect, side_results: List[SideResult]) -> np.ndarray:
    img = Image.fromarray(rotated_color)
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
    # Draw pin boxes translated to global coords
    for side_res in side_results:
        color = side_colors.get(side_res.name, (255, 255, 0))
        for (x1, y1, x2, y2) in side_res.boxes:
            # Offset boxes depending on side to place near the package edge
            if side_res.name == "top":
                y_off = cy - h / 2 - 18
                x_off = cx - w / 2 - 12
            elif side_res.name == "bottom":
                y_off = cy + h / 2 + 12
                x_off = cx - w / 2 - 12
            elif side_res.name == "left":
                x_off = cx - w / 2 - 18
                y_off = cy - h / 2 - 12
            else:
                x_off = cx + w / 2 + 12
                y_off = cy - h / 2 - 12
            draw.rectangle(
                (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off),
                outline=color,
                width=1,
            )
    return np.array(img)


def process_image(path: str, out_dir: str) -> ImageResult:
    steps = []
    steps.append("Load Canny-edge image.")
    edges = load_edges(path)

    steps.append("Locate non-zero pixels to find package body (no rotation).")
    nz = np.argwhere(edges > 0)
    if nz.shape[0] == 0:
        raise ValueError("No edges found")
    # Use percentiles to be robust to speckles
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
    pad = 4  # small pad to include the body fully
    body_box = ((cx, cy), (w + pad, h + pad))

    steps.append("Mask out the largest rectangle (chip body).")
    mask = np.ones_like(edges, dtype=np.uint8)
    x1 = int(max(0, cx - (w + pad) / 2))
    x2 = int(min(edges.shape[1], cx + (w + pad) / 2))
    y1 = int(max(0, cy - (h + pad) / 2))
    y2 = int(min(edges.shape[0], cy + (h + pad) / 2))
    mask[y1:y2, x1:x2] = 0
    pins_only = (edges * mask).astype(np.uint8)
    rect_aligned = body_box

    steps.append("Detect connected pin blobs after body removal.")

    def connected_components(bin_img: np.ndarray, min_area: int = 5):
        """Simple two-pass 4-connected components."""
        h, w_img = bin_img.shape
        labels = np.zeros((h, w_img), dtype=np.int32)
        parent = [0]  # 1-based
        next_label = 1
        for y in range(h):
            for x in range(w_img):
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

        # Relabel compact
        mapping = {}
        compact = 1
        for y in range(h):
            for x in range(w_img):
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

    pins_bin = (pins_only > 0).astype(np.uint8)
    comps = connected_components(pins_bin, min_area=5)

    steps.append("Assign pin blobs to sides.")
    side_order = ["top", "right", "bottom", "left"]
    aspect = (max(body_box[1][0], body_box[1][1]) + 1e-6) / (
        min(body_box[1][0], body_box[1][1]) + 1e-6
    )
    if aspect >= 1.6:
        # Two-sided; choose bands on the long dimension
        if body_box[1][0] >= body_box[1][1]:
            active_sides = ["top", "bottom"]
        else:
            active_sides = ["left", "right"]
    else:
        active_sides = side_order

    band = int(max(body_box[1][0], body_box[1][1]) * 0.12)
    pins_per_side_map = {s: 0 for s in side_order}
    boxes_by_side: Dict[str, List[Tuple[int, int, int, int]]] = {s: [] for s in side_order}
    side_energy = []
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
            # near the body edges; decide based on proximity
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

    # Build side results for annotation
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

    # Annotate and save (no rotation)
    color = np.stack([edges] * 3, axis=2)
    annotated = annotate(color, rect_aligned, side_results)
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
    OUTPUT_DIR = "bi_full_annotated"
    res = run_batch(INPUT_DIR, OUTPUT_DIR)
    with open("pin_counts.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved pin_counts.json")

