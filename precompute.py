"""
precompute_depth.py
-------------------
Pre-compute and CACHE depth maps for training images using Intel DPT-Hybrid MiDaS.

WHY WE NEED THIS:
  The depth estimator (DPT transformer) is slow to run at every training step.
  Since training images don't change, we compute depths ONCE and save them as PNGs.
  Training then loads PNGs directly — no DPT overhead per step.

TWO INPUT MODES:
  1. DIRECTORY MODE: scan --input_dir, save depths to --output_dir
     Then auto-updates any *.json files found beside the image folder.

  2. JSON MODE: read --json_file (each entry has "raw_image_path" + "prompt"),
     compute depths, ADD "depth_path" field to each entry, save updated JSON.

JSON MANIFEST FORMAT:
  [
    {"raw_image_path": "data/images/cat.jpg",  "prompt": "a cute cat on a chair"},
    {"raw_image_path": "data/images/dog.jpg",  "prompt": "a brown dog on grass"}
  ]
  After running: "depth_path" key is added to each entry automatically.

TYPICAL WORKFLOW (directory mode — recommended):
  # Step 1: compute depths for all images, auto-updates train/val/test.json
  python precompute_depth.py --input_dir data/images --output_dir data/depths
  # Scans data/images/, saves depth PNGs to data/depths/,
  # then finds data/train.json + data/test.json and fills "depth_path" fields.

  # Step 2: train
  python train_depth.py experiment=train_depth_12gb

ALTERNATIVE (JSON mode — when you already have a JSON file):
  python precompute_depth.py --json_file data/train.json --output_dir data/depths
"""

import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.annotators.midas import DepthEstimator


def _depth_out_path(src_path: str, output_dir: Path, input_dir: Path | None) -> Path:
    """
    Compute where to save the depth PNG for src_path.

    When input_dir is given the relative folder structure is preserved:
        input_dir/A/scene_001/img.jpg  →  output_dir/A/scene_001/img.png
    This avoids stem collisions when many images share the same filename
    (e.g. all named 'original_sample_img.jpg' in different sub-folders).
    """
    src = Path(src_path)
    if input_dir is not None:
        try:
            rel = src.relative_to(input_dir)
            return output_dir / rel.parent / (rel.stem + ".png")
        except ValueError:
            pass
    return output_dir / (src.stem + ".png")


def precompute_depths_from_paths(
    image_paths: list,
    output_dir: Path,
    size: int = 512,
    batch_size: int = 4,
    model_name: str = "Intel/dpt-hybrid-midas",
    device: str = "cuda",
    skip_existing: bool = True,
    input_dir: Path = None,
) -> dict:
    """
    Core function: compute depth maps for a list of image paths, save as grayscale PNGs.

    input_dir: when provided, depth PNGs are saved preserving the relative folder
               structure under output_dir (avoids stem collisions with nested datasets).

    Returns:
        dict mapping original image path → saved depth PNG path
        (only includes entries that were actually processed)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load DPT model -------------------------------------------------- #
    print(f"\nLoading depth estimator: {model_name}")
    depth_estimator = DepthEstimator(size=size, model=model_name, local_files_only=False)
    depth_estimator = depth_estimator.to(device).eval()
    print("Depth estimator ready.\n")

    # ---- Image preprocessing --------------------------------------------- #
    # DepthEstimator expects [B, 3, H, W] in [-1, 1]
    # Resize to exact square — avoids cropping non-square images (e.g. 1280x800).
    # Mild aspect ratio stretch is acceptable for depth estimation.
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),                         # [0,255] → [0,1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # [0,1] → [-1,1]
    ])

    path_to_depth = {}   # result mapping: image path (str) → depth path (str)
    processed = skipped = errors = 0

    pbar = tqdm(range(0, len(image_paths), batch_size), desc="Computing depth maps")

    for batch_start in pbar:
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        to_process = []
        for p in batch_paths:
            out = _depth_out_path(p, output_dir, input_dir)
            if skip_existing and out.exists():
                path_to_depth[str(p)] = str(out)
                skipped += 1
            else:
                to_process.append(p)

        if not to_process:
            continue

        tensors, valid_paths = [], []
        for p in to_process:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))     # [3, H, W] in [-1, 1]
                valid_paths.append(p)
            except Exception as e:
                print(f"\n[WARN] Could not load {Path(p).name}: {e}")
                errors += 1

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)   # [B, 3, H, W]
        with torch.no_grad():
            # depth_maps: [B, 3, H, W] in [0, 1]  (all 3 channels identical)
            depth_maps = depth_estimator(batch_tensor)

        for i, src_path in enumerate(valid_paths):
            depth_single = depth_maps[i, 0]    # [H, W] in [0, 1]
            depth_np = (depth_single.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            out = _depth_out_path(src_path, output_dir, input_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(depth_np, mode="L").save(out)   # "L" = 8-bit grayscale

            path_to_depth[str(src_path)] = str(out)
            processed += 1

        pbar.set_postfix(processed=processed, skipped=skipped, errors=errors)

    print(f"\n{'='*52}")
    print(f"  Processed : {processed} images")
    print(f"  Skipped   : {skipped}   (already existed)")
    print(f"  Errors    : {errors}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*52}\n")

    return path_to_depth


def run_json_mode(args):
    """
    JSON MODE:
      Read --json_file (list of {raw_image_path, prompt} entries).
      Compute depth for each image.
      Add / update "depth_path" field in each entry.
      Save updated JSON back to the same file.
    """
    json_path = Path(args.json_file).resolve()
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from: {json_path}")

    # Resolve each image path relative to cwd or json file's directory
    image_paths = []
    for entry in entries:
        p = Path(entry["raw_image_path"])
        if not p.is_absolute():
            # Try relative to cwd first (project root), then relative to JSON location
            abs_p = Path.cwd() / p
            if abs_p.exists():
                p = abs_p
            else:
                p = json_path.parent / p
        image_paths.append(p)
        if not p.exists():
            print(f"[WARN] Image not found: {p}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else json_path.parent / "depth"

    path_to_depth = precompute_depths_from_paths(
        image_paths=[str(p) for p in image_paths],
        output_dir=output_dir,
        size=args.size,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        skip_existing=not args.no_skip,
    )

    # Update each entry with its depth path
    updated = 0
    for entry, img_path in zip(entries, image_paths):
        depth_p = path_to_depth.get(str(img_path))
        if depth_p is not None:
            try:
                rel = str(Path(depth_p).relative_to(Path.cwd()))
            except ValueError:
                rel = str(depth_p)
            entry["depth_path"] = rel.replace("\\", "/")
            updated += 1
        elif "depth_path" not in entry:
            print(f"[WARN] No depth for: {img_path}")

    # Write updated JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated}/{len(entries)} entries with depth paths.")
    print(f"Saved updated JSON → {json_path}")
    print("\nNext step:")
    print(f"  python train_depth.py experiment=train_depth_12gb data.json_file={json_path}")


def update_json_files(json_dir: Path, path_to_depth_rel: dict, input_dir: Path = None) -> None:
    """
    Scan json_dir for *.json files (train.json, val.json, test.json, ...).
    For each entry, fill the "depth_path" field by matching the source image path.

    Args:
        json_dir          : directory to scan for JSON files (e.g. data/)
        path_to_depth_rel : {relative_src_path: relative_depth_path}
                            Keys are relative to input_dir
                            (e.g. "A/scene_001/original_sample_img.jpg")
        input_dir         : absolute path to the image root used during precompute.
                            Required for correct matching with nested folder layouts.
    """
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"  No JSON files found in: {json_dir}")
        return

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not read {json_path.name}: {e}")
            continue

        if not isinstance(entries, list):
            continue   # not a manifest file, skip

        updated = 0
        for entry in entries:
            if "raw_image_path" not in entry:
                continue

            raw_path = Path(entry["raw_image_path"])

            # Derive the lookup key: path relative to input_dir.
            # JSON entries store paths relative to the project root
            # (e.g. "data/images/A/scene_001/original_sample_img.jpg").
            # input_dir is e.g. /abs/path/to/data/images, so the key is
            # "A/scene_001/original_sample_img.jpg".
            if input_dir is not None:
                abs_raw = (Path.cwd() / raw_path) if not raw_path.is_absolute() else raw_path
                try:
                    rel_key = str(abs_raw.relative_to(input_dir)).replace("\\", "/")
                except ValueError:
                    rel_key = raw_path.name
            else:
                rel_key = raw_path.stem   # legacy flat-layout fallback

            depth_rel = path_to_depth_rel.get(rel_key)
            if depth_rel:
                entry["depth_path"] = depth_rel
                updated += 1
            elif "depth_path" not in entry:
                print(f"  [WARN] No depth for: {raw_path}")

        if updated > 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            print(f"  Updated {updated}/{len(entries)} entries in {json_path.name}")
        else:
            print(f"  No matching images found in {json_path.name} (skipped)")


def run_directory_mode(args):
    """
    DIRECTORY MODE:
      Scan --input_dir for images.
      Compute depths, save to --output_dir.
      Auto-update any *.json files in the data directory with depth paths.
    """
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir.parent / "depths"   # default: data/depths/
    )

    # Recursive scan — supports nested folder layouts like
    # data/images/A/scene_001/original_sample_img.jpg
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    if not image_paths:
        print(f"[ERROR] No images found under: {input_dir}")
        return

    print(f"Found {len(image_paths)} images under: {input_dir}")

    path_to_depth = precompute_depths_from_paths(
        image_paths=[str(p) for p in image_paths],
        output_dir=output_dir,
        size=args.size,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
        skip_existing=not args.no_skip,
        input_dir=input_dir,  # preserve relative path structure → no stem collisions
    )

    # Build relative-source-path → relative-depth-path dict for JSON update.
    # Keys are relative to input_dir (e.g. "A/scene_001/original_sample_img.jpg")
    # so multiple images with the same filename but different parent folders are
    # disambiguated correctly.
    path_to_depth_rel = {}
    for img_p, depth_p in path_to_depth.items():
        try:
            rel_src = str(Path(img_p).relative_to(input_dir)).replace("\\", "/")
        except ValueError:
            rel_src = Path(img_p).name
        try:
            rel_depth = str(Path(depth_p).relative_to(Path.cwd())).replace("\\", "/")
        except ValueError:
            rel_depth = str(depth_p).replace("\\", "/")
        path_to_depth_rel[rel_src] = rel_depth

    # Auto-update JSON files in the data directory with depth paths
    json_dir = Path(args.json_dir).resolve() if args.json_dir else input_dir.parent
    print(f"\nUpdating JSON files in: {json_dir}")
    update_json_files(json_dir, path_to_depth_rel, input_dir)

    print("\nNext step — train:")
    print("  python train_depth.py experiment=train_depth_12gb")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute depth maps from a directory of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Primary arguments ----------------------------------------------- #
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing images (.jpg/.jpeg/.png). Required unless --json_file is used.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save depth PNGs. Default: data/depths/ beside --input_dir.",
    )

    # ---- Optional: JSON mode (only needed if you already have a JSON file) #
    parser.add_argument(
        "--json_file", type=str, default=None,
        help="(Optional) Path to an existing JSON manifest. If given, reads image "
             "paths from the JSON and updates depth_path fields in it.",
    )
    parser.add_argument(
        "--json_dir", type=str, default=None,
        help="(Optional) Folder to scan for *.json files to update after computing depths. "
             "Defaults to the parent of --input_dir (e.g. data/).",
    )

    # ---- Processing options ---------------------------------------------- #
    parser.add_argument("--size",       type=int, default=512,
                        help="Square spatial size for depth maps. Default: 512")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Images per GPU batch. Default: 4")
    parser.add_argument("--model",      type=str, default="Intel/dpt-hybrid-midas",
                        help="HuggingFace model ID. Default: Intel/dpt-hybrid-midas")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: 'cuda' or 'cpu'. Default: cuda if available")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-compute even if depth PNG already exists.")

    args = parser.parse_args()

    # Validate: need at least one input
    if not args.input_dir and not args.json_file:
        parser.error("Provide --input_dir (recommended) or --json_file.\n"
                     "  Example: python precompute_depth.py --input_dir data/images --output_dir data/depths")

    print(f"Device : {args.device}")
    print(f"Size   : {args.size}x{args.size}")
    print(f"Batch  : {args.batch_size}")

    if args.json_file and not args.input_dir:
        # JSON-only mode: image paths come from the JSON file
        run_json_mode(args)
    else:
        # Directory mode: scan input_dir for images (primary usage)
        run_directory_mode(args)


if __name__ == "__main__":
    main()
