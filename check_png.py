"""
check_png.py
----------------
Inspect a segmentation class-map PNG that looks black to the naked eye.
Tells you: what class indices are present, their counts, and saves a
colorized version so you can actually see the segments.

Usage:
    python check_seg_png.py path/to/class_map.png
    python check_seg_png.py path/to/class_map.png --classes class.txt
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def load_classes(class_file: str) -> dict:
    """Load class names from a text file (one name per line, index = line number)."""
    path = Path(class_file)
    if not path.exists():
        return {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return {i: name for i, name in enumerate(lines)}


def colorize(arr: np.ndarray, num_classes: int) -> Image.Image:
    """Map class indices to distinct RGB colors using a fixed palette."""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(max(num_classes, 256), 3), dtype=np.uint8)
    palette[0] = [30, 30, 30]   # class 0 = dark gray (often background)
    rgb = palette[arr]
    return Image.fromarray(rgb, mode="RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("png", help="Path to the class-map PNG")
    parser.add_argument("--classes", default=None, help="Path to class.txt (optional)")
    args = parser.parse_args()

    png_path = Path(args.png)
    class_names = load_classes(args.classes) if args.classes else {}

    # ---- Load ---------------------------------------------------------------
    img = Image.open(png_path)
    print(f"\nFile      : {png_path.name}")
    print(f"PIL mode  : {img.mode}   (should be 'L' for class-index map)")
    print(f"Size      : {img.size[0]} x {img.size[1]}  (width x height)")

    arr = np.array(img)
    print(f"Array     : dtype={arr.dtype}, shape={arr.shape}")
    print(f"Min value : {arr.min()}   (lowest class index in this image)")
    print(f"Max value : {arr.max()}   (highest class index in this image)")

    # ---- Unique classes present ----------------------------------------------
    unique, counts = np.unique(arr, return_counts=True)
    total_pixels = arr.size

    print(f"\n{'='*52}")
    print(f"  {len(unique)} unique class(es) found in this image:")
    print(f"{'='*52}")
    print(f"  {'Index':>6}  {'Pixels':>8}  {'%':>6}  Name")
    print(f"  {'-'*6}  {'-'*8}  {'-'*6}  ----")
    for idx, cnt in zip(unique, counts):
        pct = cnt / total_pixels * 100
        name = class_names.get(int(idx), "?")
        print(f"  {idx:>6}  {cnt:>8}  {pct:>5.1f}%  {name}")

    # ---- Verdict ------------------------------------------------------------
    print(f"\n{'='*52}")
    if img.mode != "L":
        print("  WARNING: mode is not 'L'. This might not be a raw class-index map.")
        print("  The training pipeline expects mode='L' (8-bit grayscale = class ids).")
    elif arr.max() > 255:
        print("  WARNING: values exceed 255. Cannot store as uint8 PNG correctly.")
    else:
        print("  OK: mode is 'L' and values are in [0, 255].")
        if class_names:
            n = len(class_names)
            print(f"  Your class.txt has {n} classes → num_classes={n} in config.")
            if arr.max() >= n:
                print(f"  WARNING: max index {arr.max()} >= num_classes {n}. Check your mapping.")
            else:
                print(f"  Max index {arr.max()} < {n}. Indices look valid.")

    # ---- Save colorized version ---------------------------------------------
    num_classes = len(class_names) if class_names else int(arr.max()) + 1
    color_img = colorize(arr, num_classes)
    out_path = png_path.parent / (png_path.stem + "_colorized.png")
    color_img.save(out_path)
    print(f"\n  Colorized preview saved → {out_path}")
    print(f"  Open it to visually confirm the segments look correct.")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
