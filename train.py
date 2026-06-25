import hydra
import math
from src.model import ModelBase
from diffusers.optimization import get_scheduler
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torchvision.transforms.functional as TF
from accelerate.logging import get_logger
import signal
import os
import traceback
import random
from functools import reduce
from PIL import Image, ImageDraw

from src.utils import add_lora_from_config, save_checkpoint


torch.set_float32_matmul_precision("high")


stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("got stop signal")


# ── Validation grid helpers ───────────────────────────────────────────────────
# Produces a 4-panel horizontal grid logged to TensorBoard and saved to disk
# after every val_steps interval and at the end of every epoch.
#
#   Col 0  ORIGINAL      — the raw validation image
#   Col 1  DEPTH MAP     — pre-computed depth used as conditioning
#   Col 2  PREDICTED     — model output with the per-image text prompt
#   Col 3  RAW DEPTH GEN — same depth, empty prompt (pure depth adherence test)
#
# JSON key locations — if you rename keys in train.json / test.json, update:
#   src/data/local.py  DepthJsonDataset.__getitem__
#       item["raw_image_path"]  ← source image path
#       item["depth_path"]      ← pre-computed depth PNG path
#       item.get("prompt", "")  ← text caption
#   inference_depth.py  (entries loop, ~line 175)
#       item["raw_image_path"], item.get("prompt", "")

def _label_bar(width, text, bar_h=24):
    """Dark bar with centred text label. Returns [bar_h, width, 3] uint8 array."""
    bar  = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    draw.text(((width - bbox[2]) // 2, 4), text, fill=(255, 220, 60))
    return np.asarray(bar)


def build_grid(orig_11, depth_01, pred_pils, size, raw_depth_pils=None):
    """
    4-panel horizontal grid for one validation image:
        ORIGINAL | DEPTH MAP | PREDICTED | RAW DEPTH GEN
    Takes the first element of each list. Returns HWC uint8 numpy array.
    """
    orig_np  = np.asarray(TF.to_pil_image(((orig_11[0].float()+1)/2).clamp(0,1).cpu()).resize((size,size)).convert("RGB"))
    depth_np = np.asarray(TF.to_pil_image(depth_01[0].float().clamp(0,1).cpu()).resize((size,size)).convert("RGB"))
    pred_np  = np.asarray(pred_pils[0].resize((size,size)).convert("RGB"))
    raw_np   = (np.asarray(raw_depth_pils[0].resize((size,size)).convert("RGB"))
                if raw_depth_pils else np.zeros((size, size, 3), dtype=np.uint8))

    label_row = np.concatenate([
        _label_bar(size, "ORIGINAL"),
        _label_bar(size, "DEPTH MAP"),
        _label_bar(size, "PREDICTED"),
        _label_bar(size, "RAW DEPTH GEN"),
    ], axis=1)
    imgs_row = np.concatenate([orig_np, depth_np, pred_np, raw_np], axis=1)
    return np.concatenate([label_row, imgs_row], axis=0)
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", config_name="train_depth", version_base=None)
def main(cfg):
    if hasattr(signal, "SIGUSR1"):   # Linux/Mac only
        signal.signal(signal.SIGUSR1, signal_handler)

    # Use only locally cached HuggingFace models — no network calls during training
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Suppress expected-but-noisy warnings:
    # 1. transformers LOAD REPORT: "position_ids UNEXPECTED" — this key exists in the
    #    SD 1.5 CLIP checkpoint but was removed from newer transformers CLIP architecture.
    #    It is harmless (the model works correctly without it).
    # 2. diffusers safety checker: we intentionally disable it for training/research.
    import logging
    logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
    logging.getLogger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion").setLevel(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    accelerator = Accelerator(
        project_dir=output_path / "logs",
        log_with="tensorboard",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.get("bf16", True) else "no",
    )

    logger = get_logger(__name__)

    logger.info("==================================")
    logger.info(cfg)
    logger.info(output_path)

    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model

    model = model.to(accelerator.device)
    model.pipe.to(accelerator.device)
    n_loras = len(cfg.lora.keys())

    cfg_mask = add_lora_from_config(model, cfg, accelerator.device)

    if cfg.get("gradient_checkpointing", False):
        model.unet.enable_gradient_checkpointing()

    dm = cfg.data

    train_dataloader = dm.train_dataloader()
    val_dataloader   = dm.val_dataloader()

    mappers_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, []))
    )
    encoder_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, []))
    )

    optimizer = torch.optim.AdamW(
        model.params_to_optimize + mappers_params + encoder_params,
        lr=cfg.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.get("lr_warmup_steps", 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    logger.info(f"Number params Mapper Network(s) {sum(p.numel() for p in mappers_params):,}")
    logger.info(f"Number params Encoder Network(s) {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"Number params all LoRAs(s) {sum(p.numel() for p in model.params_to_optimize):,}")

    logger.info("init trackers")
    if accelerator.is_main_process:
        accelerator.init_trackers("tensorboard")

    logger.info("prepare network")

    prepared = accelerator.prepare(
        *model.mappers,
        *model.encoders,
        model.unet,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    mappers  = prepared[: len(model.mappers)]
    encoders = prepared[len(model.mappers) : len(model.mappers) + len(model.encoders)]
    (unet, optimizer, train_dataloader, val_dataloader, lr_scheduler) = prepared[
        len(model.mappers) + len(model.encoders) :
    ]
    model.unet     = unet
    model.mappers  = mappers
    model.encoders = encoders

    try:
        if cfg.get("max_train_steps", None) is not None:
            max_train_steps = cfg.max_train_steps
    except:
        pass

    global_step = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
    )
    progress_bar.set_description("Steps")

    best_loss = float("inf")

    logger.info("start training")
    for epoch in range(cfg.epochs):
        logger.info("new epoch")
        unet.train()
        for m in mappers:  m.train()
        for e in encoders: e.train()

        epoch_loss_sum   = 0.0
        epoch_loss_count = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch["jpg"].to(accelerator.device).clip(-1.0, 1.0)
                B = imgs.shape[0]

                # ── DEPTH CHANGE: use pre-computed depth maps instead of images ──
                # batch["depth"] is loaded from the cached PNG (precompute_depth.py).
                # skip_encode=True means the DepthEstimator is NOT called here —
                # the depth tensor goes directly to the mapper network.
                depth_maps = batch["depth"].to(accelerator.device)
                cs = [depth_maps] * n_loras
                # ─────────────────────────────────────────────────────────────────

                if cfg.get("prompt", None) is not None:
                    prompts = [cfg.prompt] * B
                else:
                    prompts = batch["caption"]

                model_pred, loss, x0, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    cfg_mask=[True for _ in cfg_mask],
                    skip_encode=True,   # depth already computed — bypass DPT
                    batch=batch,
                )

                accelerator.backward(loss)

                # Clip gradients to prevent exploding gradients during training.
                # max_norm=1.0 is the standard value for diffusion fine-tuning.
                if accelerator.sync_gradients:
                    all_params = [p for group in optimizer.param_groups for p in group["params"]]
                    accelerator.clip_grad_norm_(all_params, max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            epoch_loss_sum   += loss.detach().item()
            epoch_loss_count += 1
            progress_bar.set_postfix(**logs, refresh=False)
            accelerator.log(logs, step=global_step)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % cfg.val_steps != 0 and not stop_training:
                    continue

                # VALIDATION
                with torch.no_grad():
                    _preview_grid = None  # set inside try, used in finally to save to disk
                    try:
                        unet.eval()
                        for m in mappers:  m.eval()
                        for e in encoders: e.eval()

                        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

                        # Pick 1 random image from val dataset — vertical 4-panel grid
                        _val_idx  = random.Random(cfg.seed + global_step).randint(0, len(dm.val_dataset) - 1)
                        _raw_item = dm.val_dataset[_val_idx]
                        val_batch = {
                            "jpg"    : _raw_item["jpg"].unsqueeze(0).to(accelerator.device),
                            "depth"  : _raw_item["depth"].unsqueeze(0).to(accelerator.device),
                            "caption": [_raw_item["caption"]],
                        }
                        imgs = val_batch["jpg"].clip(-1.0, 1.0)
                        val_prompts = [cfg.prompt] if cfg.get("prompt") else val_batch["caption"]
                        depth_maps = val_batch["depth"]   # [1, 3, H, W] in [0, 1]
                        cs = [depth_maps] * n_loras
                        # sample_custom with skip_encode=True mirrors the training forward pass:
                        # pre-computed depth goes directly to the mapper, no DPT call.
                        all_preds = model.sample_custom(
                            prompt=val_prompts, num_images_per_prompt=1, cs=cs,
                            generator=generator, cfg_mask=cfg_mask, skip_encode=True,
                        )
                        all_raw_preds = model.sample_custom(
                            prompt=[""], num_images_per_prompt=1, cs=cs,
                            generator=torch.Generator(device=accelerator.device).manual_seed(cfg.seed),
                            cfg_mask=cfg_mask, skip_encode=True,
                        )
                        all_val_imgs = [imgs[0].cpu()]

                        if accelerator.is_main_process:
                            # Use pre-computed depth for visualization (matches conditioning above).
                            depth_viz = depth_maps.cpu()  # [1, 3, H, W] in [0, 1]

                            grid = build_grid(
                                orig_11=all_val_imgs,
                                depth_01=[depth_viz[j] for j in range(len(all_val_imgs))],
                                pred_pils=all_preds,
                                size=cfg.size,
                                raw_depth_pils=all_raw_preds,
                            )
                            # ─────────────────────────────────────────────────────────────

                            # Store as PIL so finally block can save it to the checkpoint folder
                            _preview_grid = Image.fromarray(grid)

                            for tracker in accelerator.trackers:
                                if tracker.name == "tensorboard":
                                    tracker.writer.add_image(
                                        "validation",
                                        grid,
                                        global_step,
                                        dataformats="HWC",
                                    )
                                    tracker.writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], global_step)
                                    tracker.writer.add_scalar("loss", loss.detach().item(), global_step)
                                    tracker.writer.add_text(
                                        "prompts",
                                        "------------".join(val_prompts),
                                        global_step,
                                    )

                    except Exception as e:
                        print("!!!!!!!!!!!!!!!!!!!")
                        print("ERROR IN VALIDATION")
                        print(e)
                        print(traceback.format_exc())
                        print("!!!!!!!!!!!!!!!!!!!")

                    finally:
                        if accelerator.is_main_process:
                            ckpt_dir = output_path / f"checkpoint-{global_step}"
                            save_checkpoint(
                                model.get_lora_state_dict(accelerator.unwrap_model(unet)),
                                [accelerator.unwrap_model(m).state_dict() for m in mappers],
                                None,
                                ckpt_dir,
                            )
                            if _preview_grid is not None:
                                stem     = f"checkpoint-{global_step}"
                                grid_dir = output_path / "image_grid"
                                grid_dir.mkdir(parents=True, exist_ok=True)
                                _preview_grid.save(grid_dir / f"{stem}.jpg", quality=95)
                                (grid_dir / f"{stem}.txt").write_text(
                                    "\n".join(f"[{i}] {p}" for i, p in enumerate(val_prompts)),
                                    encoding="utf-8",
                                )
                                logger.info(f"Preview grid -> {grid_dir / stem}.jpg")

                        unet.train()
                        for m in mappers:  m.train()
                        for e in encoders: e.train()

            if stop_training:
                break

        # ── END-OF-EPOCH checkpoint + preview grid ─────────────────────────────
        # Runs after every epoch regardless of val_steps cadence so you can
        # compare model quality epoch-by-epoch from checkpoint-epoch-N folders.
        with torch.no_grad():
            _epoch_grid   = None
            epoch_prompts = []
            try:
                unet.eval()
                for m in mappers:  m.eval()
                for e in encoders: e.eval()

                generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

                # Pick 1 random image from val dataset — different each epoch, reproducible
                _val_idx  = random.Random(cfg.seed + epoch).randint(0, len(dm.val_dataset) - 1)
                _raw_item = dm.val_dataset[_val_idx]
                val_batch = {
                    "jpg"    : _raw_item["jpg"].unsqueeze(0).to(accelerator.device),
                    "depth"  : _raw_item["depth"].unsqueeze(0).to(accelerator.device),
                    "caption": [_raw_item["caption"]],
                }
                imgs = val_batch["jpg"].clip(-1.0, 1.0)
                epoch_prompts = [cfg.prompt] if cfg.get("prompt") else val_batch["caption"]
                depth_maps = val_batch["depth"]   # [1, 3, H, W] in [0, 1]
                cs = [depth_maps] * n_loras
                # sample_custom with skip_encode=True mirrors the training forward pass:
                # pre-computed depth goes directly to the mapper, no DPT call.
                all_preds = model.sample_custom(
                    prompt=epoch_prompts, num_images_per_prompt=1, cs=cs,
                    generator=generator, cfg_mask=cfg_mask, skip_encode=True,
                )
                all_raw_preds = model.sample_custom(
                    prompt=[""], num_images_per_prompt=1, cs=cs,
                    generator=torch.Generator(device=accelerator.device).manual_seed(cfg.seed),
                    cfg_mask=cfg_mask, skip_encode=True,
                )
                all_val_imgs = [imgs[0].cpu()]

                if accelerator.is_main_process and all_val_imgs:
                    depth_viz = depth_maps.cpu()  # [1, 3, H, W] in [0, 1]
                    grid = build_grid(
                        orig_11=all_val_imgs,
                        depth_01=[depth_viz[j] for j in range(len(all_val_imgs))],
                        pred_pils=all_preds,
                        size=cfg.size,
                        raw_depth_pils=all_raw_preds,
                    )
                    _epoch_grid = Image.fromarray(grid)

            except Exception as e:
                print(f"ERROR IN EPOCH {epoch + 1} END VALIDATION: {e}")
                print(traceback.format_exc())

            finally:
                if accelerator.is_main_process:
                    # ── Checkpoint: model weights only ─────────────────────────
                    ckpt_dir = output_path / f"checkpoint-epoch-{epoch + 1}"
                    save_checkpoint(
                        model.get_lora_state_dict(accelerator.unwrap_model(unet)),
                        [accelerator.unwrap_model(m).state_dict() for m in mappers],
                        None,
                        ckpt_dir,
                    )

                    # ── image_grid: named to match checkpoint ──────────────────
                    if _epoch_grid is not None:
                        stem     = f"checkpoint-epoch-{epoch + 1}"
                        grid_dir = output_path / "image_grid"
                        grid_dir.mkdir(parents=True, exist_ok=True)
                        _epoch_grid.save(grid_dir / f"{stem}.jpg", quality=95)
                        (grid_dir / f"{stem}.txt").write_text(
                            "\n".join(f"[{i}] {p}" for i, p in enumerate(epoch_prompts)),
                            encoding="utf-8",
                        )
                        logger.info(f"Epoch {epoch + 1} grid -> {grid_dir / stem}.jpg")

                    # ── Best model: update when epoch-average loss improves ────
                    epoch_avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
                    if epoch_avg_loss < best_loss:
                        best_loss = epoch_avg_loss
                        best_dir  = output_path / "best_model"
                        save_checkpoint(
                            model.get_lora_state_dict(accelerator.unwrap_model(unet)),
                            [accelerator.unwrap_model(m).state_dict() for m in mappers],
                            None,
                            best_dir,
                        )
                        best_dir.mkdir(parents=True, exist_ok=True)
                        if _epoch_grid is not None:
                            _epoch_grid.save(best_dir / "preview_grid.jpg", quality=95)
                        (best_dir / "info.txt").write_text(
                            f"epoch: {epoch + 1}\n"
                            f"loss:  {epoch_avg_loss:.6f}\n\n"
                            + "\n".join(f"[{i}] {p}" for i, p in enumerate(epoch_prompts)),
                            encoding="utf-8",
                        )
                        logger.info(f"New best model — epoch {epoch + 1}, loss={epoch_avg_loss:.4f} -> {best_dir}")

                unet.train()
                for m in mappers:  m.train()
                for e in encoders: e.train()
        # ───────────────────────────────────────────────────────────────────────

        if stop_training:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            model.get_lora_state_dict(accelerator.unwrap_model(unet)),
            [accelerator.unwrap_model(m).state_dict() for m in mappers],
            None,
            output_path / f"checkpoint-{global_step}",
        )


if __name__ == "__main__":
    main()
