# LoRAdapter — Depth Workflow Complete Guide

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Flow: End-to-End](#2-data-flow-end-to-end)
3. [Step 0 — Pre-compute Depth Maps](#3-step-0--pre-compute-depth-maps)
4. [Step 1 — Data Loading](#4-step-1--data-loading)
5. [Step 2 — Training Forward Pass](#5-step-2--training-forward-pass)
6. [Step 3 — Validation & Grid Generation](#6-step-3--validation--grid-generation)
7. [Step 4 — Inference](#7-step-4--inference)
8. [Model Architecture Deep Dive](#8-model-architecture-deep-dive)
9. [Config System (Hydra)](#9-config-system-hydra)
10. [Bugs Fixed — Complete Log](#10-bugs-fixed--complete-log)
11. [Dead Config Fields](#11-dead-config-fields)

---

## 1. Architecture Overview

```
Input Image (RGB)
      │
      ▼
DepthEstimator (Intel DPT-Hybrid-MiDaS)    ← runs ONCE at precompute time
      │                                        NEVER during training forward pass
      ▼
Depth Map PNG   [H, W, 1]  uint8              ← saved to disk
      │
      ▼  (at training time)
DepthJsonDataset reads depth PNG
      │
      ▼
depth tensor  [B, 3, H, W]  float [0, 1]   ← 3-channel (grayscale replicated)
      │
      ▼
FixedStructureMapper15                       ← the ONLY trainable encoder bridge
      │
      ▼
4 feature maps  (out0, out1, out2, out3)
at 64×64 / 32×32 / 16×16 / 8×8             ← one per U-Net depth level
      │
      ▼  (via DataProvider)
NewStructLoRAConv layers (in U-Net)         ← trained LoRA weights
      │
      ▼
Modified U-Net noise prediction
      │
      ▼
Generated Image
```

### Key Design Choices

**`cfg: false` on the depth LoRA** — depth conditioning is ALWAYS applied (no null
conditioning). CFG at inference operates only on the text prompt, not on depth.
The depth is always there, providing structural guidance.

**`skip_encode=True` during training** — DPT never runs during the training forward
pass. Pre-computed depth maps are passed directly to the mapper, saving ~50% of
forward-pass compute.

**`sample_custom(skip_encode=True)` during validation** — validation sampling also
bypasses DPT and uses the pre-computed depth directly, exactly matching the training
conditioning path.

---

## 2. Data Flow: End-to-End

```
precompute_depth.py          train.json (before)
    reads ─────────────────► {"raw_image_path": "data/images/cat.jpg",
                               "prompt": "a cute cat"}

    writes depth PNG ──────► data/depths/cat.png

    updates JSON ──────────► {"raw_image_path": "data/images/cat.jpg",
                               "prompt": "a cute cat",
                               "depth_path": "data/depths/cat.png"}    ← ADDED
                              ▲
                              │  train_depth.py reads this JSON
                              │
                    DepthJsonDataset.__getitem__
                              │
                    returns {
                      "jpg":     tensor [3,512,512] in [-1,1]
                      "depth":   tensor [3,512,512] in [0,1]
                      "caption": "a cute cat"
                    }
                              │
                    train_depth.py training loop
                              │
                    model.forward_easy(imgs, prompts, cs=[depth], skip_encode=True)
```

---

## 3. Step 0 — Pre-compute Depth Maps

**File:** `precompute_depth.py`

### Why pre-compute?

DPT (Intel DPT-Hybrid-MiDaS) is a transformer — slow to run at every training step.
Since training images don't change, we run DPT once and cache the outputs as grayscale PNGs.

### JSON manifest format (required)

```json
[
  {
    "raw_image_path": "data/images/cat.jpg",
    "prompt": "a cute cat on a chair"
  }
]
```

After running `precompute_depth.py`, the `"depth_path"` key is automatically added:

```json
[
  {
    "raw_image_path": "data/images/cat.jpg",
    "prompt": "a cute cat on a chair",
    "depth_path": "data/depths/cat.png"
  }
]
```

**Key names are fixed.** If you rename any key, update these locations:
- `src/data/local.py` — `DepthJsonDataset.__getitem__` (lines ~440, ~449)
- `inference_depth.py` — entries loop (~line 176)
- `precompute_depth.py` — `run_json_mode` and `update_json_files`

### Depth PNG format

```python
# Saved as 8-bit grayscale ("L" mode):
depth_single = depth_maps[i, 0]                          # [H, W] float [0,1]
depth_np = (depth_single.cpu().numpy() * 255.0)          # [0,255]
              .clip(0, 255).astype(np.uint8)
Image.fromarray(depth_np, mode="L").save(out)            # grayscale PNG
```

Loaded back during training:

```python
depth = Image.open(depth_path).convert("L")   # 8-bit grayscale
depth = self.depth_transform(depth)            # [1, H, W] float [0,1] via ToTensor
depth = depth.repeat(3, 1, 1)                 # [3, H, W] — 3-channel for mapper input
```

### Image preprocessing for DPT

DPT requires input in **[-1, 1]** range:

```python
preprocess = transforms.Compose([
    transforms.Resize((size, size)),               # exact square resize
    transforms.ToTensor(),                          # [0,255] → [0,1]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # [0,1] → [-1,1]
])
```

### DepthEstimator internals (`src/annotators/midas.py`)

```python
class DepthEstimator(nn.Module):
    def forward(self, imgs):
        assert imgs.min() >= -1.0    # MUST be [-1,1] — assertion will catch wrong range
        assert imgs.max() <= 1.0

        imgs = (imgs + 1.0) / 2.0   # [-1,1] → [0,1] for DPT
        imgs = better_resize(imgs, self.model_size)   # resize to 384×384 for DPT

        depth_map = self.depth_estimator(pixel_values=imgs).predicted_depth
        depth_map = F.interpolate(depth_map.unsqueeze(1),
                                  size=(self.size, self.size), ...)
        # Per-image normalization to [0,1]:
        depth_min = torch.amin(depth_map, dim=[1,2,3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1,2,3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        depth_map = torch.cat([depth_map] * 3, dim=1)   # → [B, 3, H, W]
        return depth_map   # [0,1] float
```

**WARNING:** The `assert` checks `>= -1.0` and `<= 1.0`, so passing depth maps in [0,1]
to DPT will NOT assert — but gives wrong results because `(0-1+1)/2 = [0.5, 1.0]`
(uniformly bright gray image instead of RGB). Always pass the raw image in [-1,1].

---

## 4. Step 1 — Data Loading

**File:** `src/data/local.py`

### `DepthJsonDataModule` → `DepthJsonDataset`

```python
class DepthJsonDataModule:
    def __init__(self, json_file, transform, size=512, val_json_file=None, ...):
        project_root = Path(os.path.abspath(__file__)).parent.parent.parent
        image_tfm = transforms.Compose(transform)

        self.train_dataset = DepthJsonDataset(
            json_file=Path(project_root, json_file),
            image_transform=image_tfm,
            depth_size=size,
            project_root=project_root,
        )

        if val_json_file:
            self.val_dataset = DepthJsonDataset(...)
        else:
            self.val_dataset = self.train_dataset   # reuse train if no val JSON
```

### Image transform vs depth transform — different normalisation

```python
# RGB image: normalized to [-1, 1] for SD 1.5 VAE input
self.image_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),                          # → [0,1]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # → [-1,1]
])

# Depth: normalized to [0, 1] ONLY — NO Normalize step
# The mapper network expects depth in [0,1], matching DepthEstimator output
self.depth_transform = transforms.Compose([
    transforms.Resize((depth_size, depth_size)),    # ← tuple required for exact square
    transforms.ToTensor(),                          # uint8 [0,255] → float [0,1]
    # NO Normalize here — depth must stay in [0,1]
])
```

**Why `Resize((N, N))` not `Resize(N)`?**
`Resize(N)` with a single int resizes the shortest side to N and scales the other
proportionally — does NOT force a square. `Resize((N, N))` forces exact W×H.

### `__getitem__` return dict

```python
def __getitem__(self, idx):
    item = self.items[idx]
    caption = item.get("prompt", "")   # falls back to "" if key missing

    image = Image.open(self._resolve(item["raw_image_path"])).convert("RGB")
    image = self.image_transform(image)       # [3, H, W] in [-1, 1]

    depth = Image.open(self._resolve(item["depth_path"])).convert("L")   # grayscale
    depth = self.depth_transform(depth)       # [1, H, W] in [0, 1]
    depth = depth.repeat(3, 1, 1)            # [3, H, W] — replicate to 3 channels

    return {"jpg": image, "depth": depth, "caption": caption}
```

---

## 5. Step 2 — Training Forward Pass

**File:** `train_depth.py`

### Training loop core

```python
for step, batch in enumerate(train_dataloader):
    imgs      = batch["jpg"].to(device).clip(-1.0, 1.0)   # [B, 3, 512, 512] in [-1,1]
    depth_maps = batch["depth"].to(device)                 # [B, 3, 512, 512] in [0,1]

    cs = [depth_maps] * n_loras   # list of conditionings, one per LoRA

    prompts = batch["caption"]   # list of strings

    model_pred, loss, x0, _ = model.forward_easy(
        imgs,
        prompts,
        cs,
        cfg_mask=[True for _ in cfg_mask],
        skip_encode=True,    # ← bypass DPT; depth_maps go directly to mapper
        batch=batch,
    )
```

### `skip_encode=True` path in `SD15.forward()`

```python
def forward(self, latents, c, cs, timesteps, noise, cfg_mask, skip_encode=False):
    for i, (encoder, dp, mapper, lora_c) in enumerate(
            zip(encoders, self.dps, mappers, cs)):

        # cfg_mask[i]=True → apply dropout (zeros out some batch items for CFG training)
        if cfg_mask is None or cfg_mask[i]:
            dropout_mask = torch.rand(bsz, device=lora_c.device) < self.c_dropout
            lora_c[dropout_mask] = torch.zeros_like(lora_c[dropout_mask])

        if skip_encode:
            cond = lora_c          # depth → mapper directly (no DPT)
        else:
            cond = encoder(lora_c) # DPT runs here (only at inference)

        mapped_cond = mapper(cond)   # FixedStructureMapper15
        dp.set_batch(mapped_cond)    # stored for LoRA layers to read
```

### Why `cfg_mask=[True for _ in cfg_mask]` during training?

The experiment config sets `cfg: false` on the depth LoRA, so `cfg_mask = [False]`
from `add_lora_from_config`. But `[True for _ in cfg_mask]` creates `[True]` —
overriding to apply CFG dropout during training. This is intentional: training
randomly zeros out depth conditioning so the model learns the "null depth" state,
which is used as the negative conditioning at inference.

### CFG at inference vs training consistency

```
Training:   depth[dropout_mask] = zeros   → model sees both conditioned + null
Inference:  neg_c = zeros_like(depth)     → negative conditioning = zero tensor
            c = cat([neg_c, pos_depth])   → CFG: steer away from null toward depth
```

Both use the same "null" representation (zero tensor), so CFG generalises correctly.

### Multi-GPU checkpoint guard

Every `save_checkpoint` call is guarded by `accelerator.is_main_process`:

```python
# Step-level checkpoint (inside finally block):
if accelerator.is_main_process:
    save_checkpoint(
        model.get_lora_state_dict(accelerator.unwrap_model(unet)),
        [accelerator.unwrap_model(m).state_dict() for m in mappers],
        None,
        ckpt_dir,
    )

# End-of-training checkpoint:
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    save_checkpoint(...)
```

Without this guard, all N processes write to the same file simultaneously → corruption.

### Checkpoint directory structure

```
outputs/train/depth_12gb/runs/YYYY-MM-DD/HH-MM-SS/
  checkpoint-1000/
    struct/
      lora-checkpoint.pt      ← U-Net LoRA weights
      mapper-checkpoint.pt    ← FixedStructureMapper15 weights
  checkpoint-epoch-1/
    struct/
      lora-checkpoint.pt
      mapper-checkpoint.pt
  best_model/
    struct/
      lora-checkpoint.pt
      mapper-checkpoint.pt
    preview_grid.jpg
    info.txt
  image_grid/
    checkpoint-1000.jpg       ← 4-panel validation grid
    checkpoint-epoch-1.jpg
```

---

## 6. Step 3 — Validation & Grid Generation

**File:** `train_depth.py`

### Why `sample_custom` not `sample` / `sample_easy`

`model.sample()` → `sample_easy()` always calls `encoder(c)` (runs DPT). This is
correct at inference (raw image as input), but wrong during validation where we
already have pre-computed depth and want to bypass DPT exactly as training does.

`sample_custom(skip_encode=True)` passes depth directly to the mapper, exactly
mirroring the training forward pass.

```
Training forward:       depth_maps → [skip_encode=True] → mapper → LoRA layers
Validation sampling:    depth_maps → [skip_encode=True via sample_custom] → mapper → LoRA layers
Inference:              raw_image  → DPT → depth → mapper → LoRA layers
```

### Validation sampling code

```python
depth_maps = val_batch["depth"]   # [1, 3, 512, 512] in [0,1] — pre-computed
cs = [depth_maps] * n_loras

# Prompt-conditioned generation:
all_preds = model.sample_custom(
    prompt=val_prompts, num_images_per_prompt=1, cs=cs,
    generator=generator, cfg_mask=cfg_mask, skip_encode=True,
)

# Empty-prompt generation (pure depth adherence test):
all_raw_preds = model.sample_custom(
    prompt=[""], num_images_per_prompt=1, cs=cs,
    generator=torch.Generator(device=device).manual_seed(cfg.seed),
    cfg_mask=cfg_mask, skip_encode=True,
)
```

### 4-panel validation grid

```python
def build_grid(orig_11, depth_01, pred_pils, size, raw_depth_pils=None):
    """
    ┌──────────┬──────────┬──────────┬──────────────┐
    │ ORIGINAL │DEPTH MAP │PREDICTED │RAW DEPTH GEN │  ← label bars
    ├──────────┼──────────┼──────────┼──────────────┤
    │  image   │  depth   │ w/ text  │  "" prompt   │  ← size×size pixels
    └──────────┴──────────┴──────────┴──────────────┘
    """
    orig_np  = TF.to_pil_image(...).resize((size, size)).convert("RGB")
    depth_np = TF.to_pil_image(depth_01[0]).resize((size, size)).convert("RGB")
    pred_np  = pred_pils[0].resize((size, size)).convert("RGB")
    raw_np   = raw_depth_pils[0].resize((size, size)).convert("RGB")

    imgs_row  = np.concatenate([orig_np, depth_np, pred_np, raw_np], axis=1)
    label_row = np.concatenate([
        _label_bar(size, "ORIGINAL"),
        _label_bar(size, "DEPTH MAP"),
        _label_bar(size, "PREDICTED"),
        _label_bar(size, "RAW DEPTH GEN"),
    ], axis=1)
    return Image.fromarray(np.concatenate([label_row, imgs_row], axis=0))
```

**Panel meanings:**
- `ORIGINAL` — the validation image, displayed in [-1,1]→[0,1]
- `DEPTH MAP` — pre-computed depth from disk, displayed as grayscale
- `PREDICTED` — model output with the per-image text prompt
- `RAW DEPTH GEN` — same depth but empty string prompt — shows pure depth adherence
  with zero text influence; should match the general scene structure without any
  semantic content from text

### Validation index selection (reproducible but non-repeating)

```python
# Step-level validation: different image each step, same image if re-run same step
_val_idx = random.Random(cfg.seed + global_step).randint(0, len(dm.val_dataset) - 1)

# Epoch-level validation: different image each epoch
_val_idx = random.Random(cfg.seed + epoch).randint(0, len(dm.val_dataset) - 1)
```

---

## 7. Step 4 — Inference

**File:** `inference_depth.py`

### What happens at inference

At inference, there are no pre-computed depth maps. The raw image is passed to
`sample_easy()`, which calls `encoder(c)` — i.e., DPT runs on the image.

```python
# Load and preprocess image
orig_pil   = Image.open(img_path).convert("RGB")
img_tensor = preprocess(orig_pil).unsqueeze(0).to(device)   # [1,3,512,512] in [-1,1]

with torch.no_grad():
    # Step 1: Run DPT explicitly for visualization
    depth_tensor = model.encoders[0](img_tensor)   # [1,3,512,512] in [0,1]
    depth_pil    = TF.to_pil_image(depth_tensor[0].cpu().float().clamp(0,1))

    # Step 2: Generate with text prompt
    # sample_easy() calls encoder(img_tensor) AGAIN internally — DPT runs twice.
    # depth_tensor above is for display only; not passed to the model.
    preds = model.sample(
        prompt=[prompt],
        cs=[img_tensor],                   # ← raw image, not depth!
        num_inference_steps=50,
        guidance_scale=cfg.inference.get("guidance_scale", 7.5),
        ...
    )
```

**DPT runs twice at inference** — once for the display depth map, once inside
`sample_easy()` for the actual conditioning. Both calls produce the same result
because the input is the same. This is by design (visualization vs conditioning
are kept independent).

### `sample_easy()` CFG path with `cfg: false`

With `cfg_mask = [False]` (from `cfg: false` in the LoRA config):

```python
# In sample_easy():
neg_c = torch.zeros_like(c)            # zeros [1,3,512,512]
if cfg_mask is not None and not cfg_mask[i]:   # True — no CFG for depth
    c = torch.cat([c, c])              # [img, img] — same for neg and pos
else:
    c = torch.cat([neg_c, c])          # this branch NOT taken for depth

cond = encoder(c)   # DPT([img, img]) → same depth for both neg and pos paths
```

CFG operates only on the text: `noise_uncond + scale * (noise_text - noise_uncond)`.
Depth is identical for both paths → cancels out in CFG → text-only steering.

### Output files

```
results/
  cat_grid.jpg          ← 4-panel: ORIGINAL | DEPTH MAP | PREDICTED | RAW DEPTH GEN
  cat_original.jpg
  cat_depth.jpg
  cat_predicted.jpg
  cat_raw_depth_gen.jpg ← same depth, empty prompt (pure depth adherence)
```

Batch eval mode (`save_generated_only=true`) mirrors the JSON folder structure:
```
raw_image_path="data/images/A/scene/img.jpg"
→ results/data/images/A/scene/img.jpg
```

---

## 8. Model Architecture Deep Dive

### `FixedStructureMapper15` — the depth bridge

Takes a depth tensor and outputs 4 feature maps at different scales, one per U-Net depth level.

```python
class FixedStructureMapper15(nn.Module):
    def __init__(self, c_dim: int = 128):
        # Shared encoder: 512×512 → 64×64
        self.down = nn.Sequential(
            Conv2d(3→16), SiLU, Conv2d(16→16), SiLU,
            Conv2d(16→32, stride=2),  # → 256×256
            Conv2d(32→32), SiLU,
            Conv2d(32→64, stride=2),  # → 128×128
            Conv2d(64→64), SiLU,
            Conv2d(64→128, stride=2), # → 64×64
            Conv2d(128→128), SiLU,
        )
        # Per-depth-level downsampling:
        self.block0 = Identity()                           # 64×64 (depth 0)
        self.block1 = Conv2d(stride=2) → 32×32            # (depth 1)
        self.block2 = Conv2d(stride=2) → 16×16            # (depth 2)
        self.block3 = Conv2d(stride=2) → 8×8              # (depth 3)

        # Output projections: 128 → c_dim channels
        self.out0 = Conv2d(128, c_dim, 1)   # [B, c_dim, 64, 64]
        self.out1 = Conv2d(128, c_dim, 1)   # [B, c_dim, 32, 32]
        self.out2 = Conv2d(128, c_dim, 1)   # [B, c_dim, 16, 16]
        self.out3 = Conv2d(128, c_dim, 1)   # [B, c_dim, 8,  8 ]

    def forward(self, x):
        base = self.down(x)
        return (self.out0(base), self.out1(self.block1(base)),
                self.out2(self.block2(...)), self.out3(self.block3(...)))
```

### `NewStructLoRAConv` — the depth-conditioned LoRA layer

Replaces `conv1` layers in U-Net ResNet blocks. Each layer reads the feature map
at its depth level from the DataProvider.

```python
class NewStructLoRAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 c_dim, rank, depth, data_provider, lora_scale=1.0):
        self.W = Conv2d(in_ch, out_ch, ...)   # original weights (frozen)
        self.A = Conv2d(in_ch, rank, ...)     # LoRA down projection
        self.B = Conv2d(rank, out_ch, 1)      # LoRA up projection (init zeros)
        self.gamma = Conv2d(c_dim, rank, 1)   # FiLM scale from depth feature
        self.beta  = Conv2d(c_dim, rank, 1)   # FiLM shift from depth feature

    def forward(self, x):
        w_out = self.W(x)                     # original conv output (frozen)

        if self.lora_scale == 0.0:
            return w_out                      # disabled: pass through original only

        cs = self.data_provider.get_batch()   # tuple of 4 feature maps
        c  = cs[self.depth]                   # pick the map for this depth level

        element_scale = self.gamma(c) + 1.0   # FiLM scale  [B, rank, H, W]
        element_shift = self.beta(c)           # FiLM shift  [B, rank, H, W]

        a_out  = self.A(x)                    # down-project
        a_cond = a_out * element_scale + element_shift   # depth-modulated features
        b_out  = self.B(a_cond)               # up-project

        return w_out + b_out * self.lora_scale   # residual addition
```

### Depth-to-U-Net spatial alignment

The mapper outputs and U-Net spatial sizes must match. For SD 1.5 at 512×512:

| U-Net location | depth index | spatial size | mapper output |
|---|---|---|---|
| `down_blocks.0` + `up_blocks.3` | 0 | 64×64 | `out0` 64×64 |
| `down_blocks.1` + `up_blocks.2` | 1 | 32×32 | `out1` 32×32 |
| `down_blocks.2` + `up_blocks.1` | 2 | 16×16 | `out2` 16×16 |
| `down_blocks.3` + `up_blocks.0` + `mid_block` | 3 | 8×8 | `out3` 8×8 |

`adaption_mode: only_res_conv` targets `resnets.0.conv1` and `resnets.1.conv1`
in each block — the first conv in each ResNet block at every depth level.

### DataProvider — conditioning pipeline

```
mapper(depth) → (out0, out1, out2, out3)   ← computed ONCE per forward pass
      │
      ▼
dp.set_batch(tuple)                         ← stored in DataProvider
      │
      ▼ (later, inside U-Net forward)
LoRA layer at depth D: dp.get_batch()[D]   ← reads its specific feature map
```

The mapper runs once; all LoRA layers read the same pre-computed result.

---

## 9. Config System (Hydra)

### Composition chain for training

```
configs/train_depth.yaml          ← base: required fields (size, epochs, lr...)
    └─ configs/data/local_depth.yaml      ← DepthJsonDataModule config
    └─ configs/model/sd15.yaml            ← SD15 model config
    └─ configs/lora/struct.yaml           ← LoRA architecture
           └─ configs/lora/mapper_network/fsm15.yaml   ← FixedStructureMapper15
           └─ configs/lora/encoder/midas.yaml          ← DepthEstimator
    └─ configs/experiment/train_depth_12gb.yaml        ← overrides everything
```

### Key interpolations

```yaml
# configs/lora/encoder/midas.yaml
size: ${size}   # resolves to the global size (512)
                # single int is CORRECT — DepthEstimator takes size: int
                # and internally uses (self.size, self.size) in F.interpolate

# configs/lora/struct.yaml
mapper_network:
  c_dim: ${..config.c_dim}   # relative: go up 2 levels to lora.struct,
                               # then .config.c_dim = 128

# configs/lora/mapper_network/fsm15.yaml
c_dim: 128   # hardcoded — cannot use ${c_dim} (no global c_dim key exists)
```

### `cfg: false` — no CFG for depth LoRA

```yaml
# configs/lora/struct.yaml  and  experiment config
lora:
  struct:
    cfg: false
```

Effect in code:
```python
cfg_mask.append(l.get("cfg", True))   # → cfg_mask = [False]

# During training forward (cfg_mask[i]=False → skip dropout):
if cfg_mask is None or cfg_mask[i]:   # False → block skipped
    lora_c[dropout_mask] = zeros      # NOT applied

# During sampling (cfg_mask[i]=False → duplicate instead of null):
if cfg_mask is not None and not cfg_mask[i]:   # True
    c = torch.cat([c, c])   # same depth for both neg and pos CFG paths
```

### Full experiment config override (12 GB)

```yaml
# configs/experiment/train_depth_12gb.yaml
# @package _global_  ← merges into global config

gradient_checkpointing: true
gradient_accumulation_steps: 4
size: 512
learning_rate: 1.0e-4
lr_warmup_steps: 500
lr_scheduler: cosine
epochs: 1
val_steps: 100

model:
  local_files_only: false
  guidance_scale: 7.5

lora:
  struct:
    optimize: true
    cfg: false
    encoder:
      size: ${size}
      local_files_only: false
    config:
      rank: 128
      c_dim: 128
      lora_scale: 1
```

---

## 10. Bugs Fixed — Complete Log

All bugs found and fixed across the depth workflow codebase.

### Bug 1 — Multi-GPU race condition (train_depth.py)

**Location:** End-of-training checkpoint (~line 466)

**Problem:** Final `save_checkpoint` call was missing `accelerator.is_main_process`
guard. All N GPU processes write to the same file simultaneously → corruption.

```python
# BEFORE (broken):
accelerator.wait_for_everyone()
save_checkpoint(...)   # all processes write simultaneously

# AFTER (fixed):
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    save_checkpoint(...)
```

### Bug 2 — Validation used wrong sampling path (train_depth.py)

**Location:** Both step-level (~line 286) and epoch-level (~line 390) validation

**Problem:** Validation called `model.sample()` → `sample_easy()` which ALWAYS
runs DPT. The conditioning input was the pre-computed depth tensor in [0,1], but
`DepthEstimator` expects raw images in [-1,1]. The assertion passes silently:
`assert imgs.min() >= -1.0` (0 ≥ -1 ✓), but then `(imgs + 1.0) / 2.0` maps [0,1]
to [0.5, 1.0] — a uniformly bright gray image — giving garbage depth output.

```python
# BEFORE (broken):
cs = [depth_maps] * n_loras         # depth in [0,1]
all_preds = model.sample(           # → sample_easy → DPT(depth_in_[0,1]) ← WRONG
    prompt=val_prompts, cs=cs, ...)

# AFTER (fixed):
cs = [depth_maps] * n_loras
all_preds = model.sample_custom(    # skip_encode=True: depth → mapper directly
    prompt=val_prompts, cs=cs, skip_encode=True, ...)
```

This aligns validation with the training forward pass (both use pre-computed depth,
both bypass DPT).

### Bug 3 — `bf16` config ignored (train_depth.py)

**Location:** Accelerator initialization (~line 104)

**Problem:** `mixed_precision="bf16"` was hardcoded. The `cfg.bf16` config field
had zero effect — always bf16 regardless of setting.

```python
# BEFORE:
accelerator = Accelerator(..., mixed_precision="bf16")

# AFTER:
accelerator = Accelerator(..., mixed_precision="bf16" if cfg.get("bf16", True) else "no")
```

### Bug 4 — `guidance_scale` config ignored (inference_depth.py)

**Location:** Both `model.sample()` calls (~lines 243, 255)

**Problem:** `inference.guidance_scale: 7.5` was defined in the config but never
passed to `model.sample()`. Changing the config value had no effect; the diffusers
pipeline always used its default (7.5 — happened to match, so no visible effect).

```python
# BEFORE:
preds = model.sample(prompt=[prompt], cs=[img_tensor], num_inference_steps=50)

# AFTER:
preds = model.sample(
    prompt=[prompt], cs=[img_tensor],
    num_inference_steps=cfg.inference.get("num_inference_steps", 50),
    guidance_scale=cfg.inference.get("guidance_scale", 7.5),
)
```

### Bug 5 — Wrong JSON keys in `DepthJsonDataset` docstring (src/data/local.py)

**Problem:** Docstring showed `"image"` and `"depth"` as JSON keys; actual code
reads `"raw_image_path"` and `"depth_path"`.

### Bug 6 — `Resize(N)` instead of `Resize((N, N))` (src/data/local.py)

**Location:** `depth_transform` in both `DepthImageFolderDataset` and `DepthJsonDataset`

**Problem:** `Resize(512)` resizes shortest side to 512, not both sides.
Non-square depth PNGs would not be resized to exact 512×512.
(In practice, all depth PNGs are pre-computed as 512×512 squares, so no visible
effect — but now consistent with the image transform.)

```python
# BEFORE:
transforms.Resize(depth_size),
transforms.CenterCrop(depth_size),   # extra crop step also removed

# AFTER:
transforms.Resize((depth_size, depth_size)),   # exact square, no crop needed
```

### Bug 7 — `fsm15.yaml` broken interpolation (configs/lora/mapper_network/)

**Problem:** `c_dim: ${c_dim}` referenced a non-existent global key `c_dim`.
Only worked because `struct.yaml`'s `${..config.c_dim}` overrode it at merge time.

```yaml
# BEFORE:
c_dim: ${c_dim}   # broken — no global c_dim key

# AFTER:
c_dim: 128        # hardcoded — fsm15 is always used through struct.yaml
```

### Bug 8 — `SquarePad` operator precedence (src/data/transforms.py)

**Problem:** `h - w // 2` computes `h - (w//2)` instead of `(h - w) // 2`.
For `h=200, w=100`: padding = `200 - 50 = 150` per side instead of `50` per side.
Result would be 400px wide instead of 200px.

```python
# BEFORE:
padding = [h - w // 2, 0, h - w // 2, 0]   # operator precedence bug

# AFTER:
padding = [(h - w) // 2, 0, (h - w) // 2, 0]
```

*(Not used in the depth workflow, but fixed in the shared codebase.)*

### Bug 9 — Stray `tqdm` in `sample_custom` (src/model.py)

**Problem:** `for i, t in tqdm(enumerate(timesteps))` prints a 50-step progress bar
every validation call. With 4 GPUs, 4 bars print per validation step. Noisy and
not guarded by `is_main_process`.

```python
# BEFORE:
for i, t in tqdm(enumerate(timesteps)):

# AFTER:
for i, t in enumerate(timesteps):
```

Import of `tqdm` also removed as it became unused.

### Bug 10 — Wrong comment about `skip_encode` (inference_depth.py)

**Problem:** Comment claimed DPT could be suppressed at inference via `skip_encode`.
`sample_easy()` has no `skip_encode` parameter — DPT always runs unconditionally.

### Bug 11 — `inference_depth.yaml` wrong panel count in comment

**Problem:** Comment said "3-panel" grid; actual output is 4-panel including
`RAW DEPTH GEN`.

---

## 11. Dead Config Fields

These fields are defined in `configs/train_depth.yaml` and both experiment configs
but are **never read** by `train_depth.py`. Changing them has no effect.

| Field | Defined value | Actual behaviour |
|---|---|---|
| `ckpt_steps` | 100 / 500 | Ignored — checkpoints saved at `val_steps` intervals |
| `n_samples` | 1 | Ignored — validation hardcodes `num_images_per_prompt=1` |
| `val_batches` | 8 | Ignored — validation always uses 1 random image |
| `save_grid` | true | Ignored — grid always saved |
| `use_empty_prompt_eval` | false | Ignored — empty-prompt generation always runs |
| `log_cond` | false | Ignored entirely |

`ckpt_steps` is the most confusing — users may think they can decouple validation
frequency from checkpoint frequency. To actually use it, add to `train_depth.py`:

```python
if global_step % cfg.ckpt_steps == 0 and accelerator.is_main_process:
    save_checkpoint(...)   # separate from val_steps
```
