"""
Count rectangles in an edge image (e.g., square/chetan_07_edges.png).

Example:
    python count_rectangles.py \
        --input square/chetan_07_edges.png \
        --output debug/chetan_07_edges_rects.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]  # x, y, w, h


def find_rectangles(
    img: np.ndarray,
    min_area: float = 20.0,
    max_area_ratio: float = 0.35,
    min_aspect: float = 0.5,
    max_aspect: float = 4.0,
    band_height_frac: float = 0.35,
    dedupe_dist: float = 8.0,
) -> List[Rect]:
    """Detect pin-like rectangles using bounding boxes and band/area filtering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # Close small gaps but avoid merging neighboring pins
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    h, w = gray.shape
    band_h = int(h * band_height_frac)
    band_h = max(1, min(band_h, h // 2))

    img_area = h * w
    max_area = img_area * max_area_ratio

    candidates: List[Rect] = []

    # Process top band
    top_roi = binary[:band_h, :]
    contours, _ = cv2.findContours(top_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        aspect = bw / max(bh, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        candidates.append((x, y, bw, bh))

    # Process bottom band
    bottom_roi = binary[h - band_h :, :]
    contours, _ = cv2.findContours(bottom_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        aspect = bw / max(bh, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        candidates.append((x, y + h - band_h, bw, bh))

    if not candidates:
        return []

    # De-duplicate overlapping / double-counted rectangles.
    candidates.sort(key=lambda r: (r[1], r[0]))
    deduped: List[Rect] = []
    for r in candidates:
        rx, ry, rw, rh = r
        rxc, ryc = rx + rw / 2.0, ry + rh / 2.0
        if any(((rxc - (dx + dw / 2.0)) ** 2 + (ryc - (dy + dh / 2.0)) ** 2) ** 0.5 < dedupe_dist for dx, dy, dw, dh in deduped):
            continue
        deduped.append(r)

    return deduped


def annotate(img: np.ndarray, rects: List[Rect]) -> np.ndarray:
    out = img.copy()
    color = (255, 0, 255)
    ordered = sorted(rects, key=lambda r: (r[1], r[0]))  # top-to-bottom, left-to-right
    for idx, (x, y, w, h) in enumerate(ordered, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(idx),
            (x + 4, y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(idx),
            (x + 4, y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def run(
    input_path: Path,
    output_path: Path,
    min_area: float,
    max_area_ratio: float,
    band_height_frac: float,
) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {input_path}")

    rects = find_rectangles(
        img,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        band_height_frac=band_height_frac,
    )

    annotated = annotate(img, rects)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    print(f"Rectangles detected: {len(rects)}")
    for idx, (x, y, w, h) in enumerate(rects, start=1):
        print(f"  {idx}: x={x}, y={y}, w={w}, h={h}, area={w*h}")
    print(f"Saved annotated image to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count rectangles in an edge image.")
    parser.add_argument("--input", type=Path, default=Path("square/chetan_07_edges.png"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug/chetan_07_edges_rects.png"),
        help="Where to write an annotated image.",
    )
    parser.add_argument("--min-area", type=float, default=20.0, help="Minimum rectangle area.")
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.25,
        help="Max rectangle area as a fraction of image area.",
    )
    parser.add_argument(
        "--band-height-frac",
        type=float,
        default=0.35,
        help="Fraction of image height to use for each of top/bottom bands.",
    )
    args = parser.parse_args()

    if args.min_area <= 0:
        raise SystemExit("--min-area must be positive")
    if not (0 < args.max_area_ratio <= 1.0):
        raise SystemExit("--max-area-ratio must be in (0, 1]")

    run(
        args.input,
        args.output,
        args.min_area,
        args.max_area_ratio,
        band_height_frac=args.band_height_frac,
    )


if __name__ == "__main__":
    main()

