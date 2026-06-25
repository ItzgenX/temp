"""
inference_depth.py
------------------
Run inference with a trained depth-conditioned LoRAdapter.

This script:
  1. Loads the SD 1.5 base model + trained LoRA/mapper from a checkpoint.
  2. For each input image, runs the DepthEstimator (Intel DPT-Hybrid-MiDaS)
     to get a depth map, then generates a new image conditioned on that depth.
  3. Saves a 4-panel grid (ORIGINAL | DEPTH MAP | PREDICTED | RAW DEPTH GEN) per
     image so you can visually evaluate quality and pick the best checkpoint.

INPUT OPTIONS:
  a) JSON manifest file (recommended) — each entry has "image" + "prompt":
       inference.json_file=dataset.json
  b) Direct list of image paths:
       "inference.images=[data/cat.jpg,data/dog.jpg]"
       "inference.prompts=['a cat','a dog']"

OUTPUT MODES:

  Default (save_generated_only=false) — saves 4 files per image:
    cat_grid.jpg        ← 4 panels side by side: ORIGINAL | DEPTH MAP | PREDICTED | RAW DEPTH GEN
    cat_original.jpg    ← the input image rescaled
    cat_depth.jpg       ← the depth map computed by DPT
    cat_predicted.jpg   ← the generated image

  Batch eval (save_generated_only=true, json_file required):
    Saves ONLY the generated image, mirroring the folder structure from the JSON.
    Example: raw_image_path="data/images/dog.jpg" → output_dir/data/images/dog.jpg
    Use this after training to evaluate test.json with a clean folder of results.

USAGE:
  # Standard single/multi image inference:
  python inference_depth.py \\
      ckpt_path=outputs/train/runs/2024-01-01/00-00-00/checkpoint-3000 \\
      inference.json_file=dataset.json \\
      inference.output_dir=results

  # Batch evaluate test.json — only generated images, mirrored folder structure:
  python inference_depth.py \\
      ckpt_path=outputs/train/runs/2024-01-01/00-00-00/checkpoint-3000 \\
      inference.json_file=data/test.json \\
      inference.output_dir=results/test_eval \\
      inference.save_generated_only=true

  python inference_depth.py \\
      ckpt_path=outputs/train/runs/2024-01-01/00-00-00/checkpoint-3000 \\
      "inference.images=[data/cat.jpg]" \\
      "inference.prompts=['a realistic photo of a cat']"
"""

import hydra
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from src.model import ModelBase
from src.utils import add_lora_from_config

torch.set_float32_matmul_precision("high")


# ===================================================================== #
#  VISUALIZATION HELPERS                                                  #
# ===================================================================== #

def _label_bar(width: int, text: str, bar_h: int = 28) -> np.ndarray:
    """Dark banner bar with centered text. Returns [bar_h, width, 3] uint8."""
    bar  = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    tx   = max(0, (width - (bbox[2] - bbox[0])) // 2)
    draw.text((tx, 5), text, fill=(255, 230, 80))
    return np.asarray(bar)


def make_inference_grid(
    orig_pil:     Image.Image,
    depth_pil:    Image.Image,
    pred_pil:     Image.Image,
    raw_pred_pil: Image.Image,
    size: int,
) -> Image.Image:
    """
    Create a single image with 4 panels SIDE BY SIDE:

        ┌──────────┬──────────┬──────────┬──────────────┐
        │ ORIGINAL │DEPTH MAP │PREDICTED │RAW DEPTH GEN │  ← label bars (28 px)
        ├──────────┼──────────┼──────────┼──────────────┤
        │          │          │          │              │
        │  size×   │  size×   │  size×   │   size×      │  ← image panels
        │  size    │  size    │  size    │   size       │
        └──────────┴──────────┴──────────┴──────────────┘

    RAW DEPTH GEN: same depth conditioning but empty prompt — shows pure
    depth adherence with no text influence. Matches training validation grid.
    Total image: (4*size) wide, (size + 28) tall.
    """
    orig     = orig_pil.resize((size, size)).convert("RGB")
    depth    = depth_pil.resize((size, size)).convert("RGB")
    pred     = pred_pil.resize((size, size)).convert("RGB")
    raw_pred = raw_pred_pil.resize((size, size)).convert("RGB")

    imgs_row  = np.concatenate([np.asarray(orig), np.asarray(depth),
                                 np.asarray(pred), np.asarray(raw_pred)], axis=1)

    label_row = np.concatenate([
        _label_bar(size, "ORIGINAL"),
        _label_bar(size, "DEPTH MAP"),
        _label_bar(size, "PREDICTED"),
        _label_bar(size, "RAW DEPTH GEN"),
    ], axis=1)

    grid = np.concatenate([label_row, imgs_row], axis=0)
    return Image.fromarray(grid)


# ===================================================================== #
#  MAIN                                                                   #
# ===================================================================== #

@hydra.main(config_path="configs", config_name="inference_depth")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  inference_depth.py")
    print(f"  Device     : {device}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Build model from Hydra config (SD15 + LoRA structure)               #
    # ------------------------------------------------------------------ #
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    model = model.to(device)
    model.pipe.to(device)
    model.unet.requires_grad_(False)
    model.unet.eval()

    # ------------------------------------------------------------------ #
    # Load trained LoRA + mapper weights from checkpoint                  #
    # add_lora_from_config reads cfg.ckpt_path and loads:                 #
    #   <ckpt_path>/struct/lora-checkpoint.pt                             #
    #   <ckpt_path>/struct/mapper-checkpoint.pt                           #
    # ------------------------------------------------------------------ #
    cfg_mask = add_lora_from_config(model, cfg, device, dtype=torch.float32)
    print(f"Loaded checkpoint. cfg_mask = {cfg_mask}\n")

    for e in model.encoders: e.eval()
    for m in model.mappers:  m.eval()

    # ------------------------------------------------------------------ #
    # Collect input images + prompts from JSON or direct list             #
    # ------------------------------------------------------------------ #
    entries = []

    if cfg.inference.get("json_file") and cfg.inference.json_file:
        json_path = Path(cfg.inference.json_file)
        if not json_path.is_absolute():
            json_path = Path.cwd() / json_path
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            entries.append({
                "image_path": item["raw_image_path"],
                "prompt":     item.get("prompt", ""),
            })
        print(f"Loaded {len(entries)} entries from JSON: {json_path}")

    elif cfg.inference.get("images") and cfg.inference.images:
        images  = cfg.inference.images
        prompts = list(cfg.inference.get("prompts") or [])
        for i, img_path in enumerate(images):
            entries.append({
                "image_path": img_path,
                "prompt":     prompts[i] if i < len(prompts) else "",
            })
        print(f"Processing {len(entries)} images from config.")

    if not entries:
        print("[ERROR] No input images provided.")
        print("  Set inference.json_file=dataset.json")
        print("  or  \"inference.images=[data/img.jpg]\"")
        return

    # ------------------------------------------------------------------ #
    # Image preprocessing (same as training)                               #
    # Output: [1, 3, H, W] in [-1, 1] — required by DepthEstimator       #
    # ------------------------------------------------------------------ #
    size = cfg.size
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),               # exact square — no crop for non-square images
        transforms.ToTensor(),                          # [0,255] → [0,1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # [0,1] → [-1,1]
    ])

    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    # ------------------------------------------------------------------ #
    # Inference loop                                                       #
    # ------------------------------------------------------------------ #
    for entry in tqdm(entries, desc="Generating"):
        img_path = Path(entry["image_path"])
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
        prompt = entry["prompt"]
        stem   = img_path.stem

        if not img_path.exists():
            print(f"[WARN] Not found: {img_path} — skipping.")
            continue

        print(f"\n  Image : {img_path.name}")
        print(f"  Prompt: {prompt!r}")

        # Load and preprocess to [-1, 1] tensor
        orig_pil   = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(orig_pil).unsqueeze(0).to(device)   # [1, 3, H, W]

        with torch.no_grad():

            # ---- Step 1: Depth map for visualization ----
            # model.encoders[0] is DepthEstimator (Intel DPT-Hybrid-MiDaS).
            # Input: [-1, 1]  Output: [0, 1] 3-channel (all channels identical)
            depth_tensor = model.encoders[0](img_tensor)   # [1, 3, H, W] in [0,1]
            depth_pil    = TF.to_pil_image(depth_tensor[0].cpu().float().clamp(0, 1))

            # ---- Step 2: Generate image (prompt-conditioned) ----
            # cs=[img_tensor]: sample_easy() calls encoder(c) unconditionally,
            # so DPT runs on img_tensor again to produce the conditioning for the mapper.
            # depth_tensor above is only for visualization — it is the same result.
            preds = model.sample(
                prompt=[prompt],
                num_images_per_prompt=cfg.inference.n_samples,
                cs=[img_tensor],
                generator=generator,
                cfg_mask=cfg_mask,
                num_inference_steps=cfg.inference.get("num_inference_steps", 50),
                guidance_scale=cfg.inference.get("guidance_scale", 7.5),
            )

            # ---- Step 2b: RAW DEPTH GEN (empty prompt) ----
            # Same depth conditioning, empty text — shows pure depth adherence
            # with zero text influence. Matches the 4th row of the training grid.
            raw_preds = model.sample(
                prompt=[""],
                num_images_per_prompt=cfg.inference.n_samples,
                cs=[img_tensor],
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
                cfg_mask=cfg_mask,
                num_inference_steps=cfg.inference.get("num_inference_steps", 50),
                guidance_scale=cfg.inference.get("guidance_scale", 7.5),
            )

        # ---- Step 3: Save outputs ----------------------------------------
        orig_display = TF.to_pil_image(((img_tensor[0].cpu().float() + 1) / 2).clamp(0, 1))

        save_generated_only = cfg.inference.get("save_generated_only", False)

        for k, (pred_pil, raw_pred_pil) in enumerate(zip(preds, raw_preds)):
            suffix = f"_{k}" if len(preds) > 1 else ""

            if save_generated_only:
                # Mirror the exact path from the JSON so folder structure is preserved.
                # e.g. raw_image_path="data/images/dog.jpg" →
                #      output_dir/data/images/dog.jpg  (or dog_1.jpg for n_samples>1)
                rel = Path(entry["image_path"])
                out_path = output_dir / rel.parent / f"{rel.stem}{suffix}{rel.suffix}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pred_pil.resize((size, size)).save(out_path, quality=95)
                print(f"  -> {out_path}")
            else:
                # Full debug output: 4-panel grid + individual panels
                grid = make_inference_grid(orig_display, depth_pil, pred_pil, raw_pred_pil, size)
                grid_path = output_dir / f"{stem}{suffix}_grid.jpg"
                grid.save(grid_path, quality=95)

                orig_display.resize((size, size)).save(
                    output_dir / f"{stem}_original.jpg"
                )
                depth_pil.resize((size, size)).convert("RGB").save(
                    output_dir / f"{stem}_depth.jpg"
                )
                pred_pil.resize((size, size)).save(
                    output_dir / f"{stem}{suffix}_predicted.jpg", quality=95
                )
                raw_pred_pil.resize((size, size)).save(
                    output_dir / f"{stem}{suffix}_raw_depth_gen.jpg", quality=95
                )
                print(f"  -> {grid_path}")

    print(f"\nDone. All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
