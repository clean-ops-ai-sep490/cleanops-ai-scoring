from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        config: dict[str, Any] = {}
        stack: list[tuple[int, dict[str, Any]]] = [(-1, config)]
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.strip().startswith("#") or ":" not in raw:
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            key, value = raw.strip().split(":", 1)
            value = value.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value == "":
                parent[key] = {}
                stack.append((indent, parent[key]))
            else:
                if value.isdigit():
                    parent[key] = int(value)
                else:
                    try:
                        parent[key] = float(value)
                    except ValueError:
                        parent[key] = value
        return config


def paired_files(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    if not image_dir.exists() or not mask_dir.exists():
        return []
    pairs = []
    masks = {p.stem: p for p in mask_dir.glob("*.png")}
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        mask = masks.get(image.stem)
        if mask is not None:
            pairs.append((image, mask))
    return pairs


def compute_miou_12(confusion: Any, classes: int) -> float:
    """Compute mean IoU for foreground classes 1 and 2 (excludes background=0)."""
    import numpy as np

    ious = []
    for c in range(1, classes):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - tp)
        fn = float(confusion[c, :].sum() - tp)
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)
    return float(np.mean(ious)) if ious else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline U-Net training entrypoint (config-based standalone).")
    parser.add_argument("--config", default="configs/unet_config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50, help="Print training loss every N batches.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--checkpoint-init", default=None, help="Override model.checkpoint_init from the config.")
    parser.add_argument("--output-name", default=None, help="Override output.name from the config.")
    # production overrides
    parser.add_argument("--data-root", default=None, help="Override data.root from config (for pipeline use).")
    parser.add_argument("--valid-split", default=None, help="Validation split dir name override (e.g. 'valid' for bridge dataset).")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs from config.")
    parser.add_argument("--batch", type=int, default=None, help="Override training.batch_size from config.")
    parser.add_argument("--save-path", default=None, help="Copy final best checkpoint here with production metadata (encoder, img_size).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Missing U-Net config: {args.config}")
        return 1

    try:
        import numpy as np
        import torch
        import segmentation_models_pytorch as smp
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except Exception:
        print("U-Net training dependencies are not installed. Install with: pip install pillow torch segmentation-models-pytorch pyyaml")
        return 1

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    root = Path(args.data_root) if args.data_root else Path(data_cfg.get("root", "./data/processed/unet"))
    val_split = args.valid_split or str(data_cfg.get("val_split", "val"))
    image_size = int(data_cfg.get("image_size", 384))
    classes = int(model_cfg.get("classes", 3))
    batch_size = args.batch if args.batch is not None else int(train_cfg.get("batch_size", 4))
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("learning_rate", 0.0003))
    out_base = Path(output_cfg.get("dir", "./outputs/unet/runs"))
    run_name = args.output_name or output_cfg.get("name")
    if run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_base / f"{run_name}_{timestamp}"
    else:
        out_dir = out_base
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_dir / config_path.name)
    log_file = (out_dir / "train.log").open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    train_pairs = paired_files(root, "train")
    val_pairs = paired_files(root, val_split)
    if not train_pairs:
        print("U-Net train split is empty. Run scripts/prepare_unet.py after adding raw segmentation datasets.")
        return 0

    class SegDataset(Dataset):
        def __init__(self, pairs: list[tuple[Path, Path]], augment: bool = False) -> None:
            self.pairs = pairs
            self.augment = augment

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int):
            from PIL import ImageEnhance

            image_path, mask_path = self.pairs[idx]
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
            if self.augment:
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                angle = random.uniform(-8.0, 8.0)
                image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                if random.random() < 0.8:
                    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            mask_array = np.asarray(mask, dtype=np.int64)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = torch.clamp(mask_tensor, 0, classes - 1)
            return image_tensor, mask_tensor

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = bool(args.amp and device.type == "cuda")
    encoder_name = str(model_cfg.get("encoder", "resnet34"))
    encoder_weights = model_cfg.get("encoder_weights") or None
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
    ).to(device)

    checkpoint_init = args.checkpoint_init or model_cfg.get("checkpoint_init")
    if checkpoint_init:
        checkpoint_path = Path(str(checkpoint_init))
        if not checkpoint_path.exists():
            print(f"Missing checkpoint_init: {checkpoint_path}")
            return 1
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # handle both raw state_dict and production dict format {model_state: ...}
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"loaded_checkpoint_init={checkpoint_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    class_weights_cfg = train_cfg.get("class_weights", {})
    class_names = ["background", "dirty_area", "wet_surface"]
    class_weights = []
    for idx in range(classes):
        name = class_names[idx] if idx < len(class_names) else str(idx)
        class_weights.append(float(class_weights_cfg.get(name, class_weights_cfg.get(str(idx), 1.0))))
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

    def dice_loss(logits, target, eps: float = 1e-6):
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(target, classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + eps) / (cardinality + eps)
        weights = class_weight_tensor / torch.clamp(class_weight_tensor.sum(), min=eps)
        return 1.0 - torch.sum(dice * weights)

    augment = bool(data_cfg.get("augment", False))
    train_loader = DataLoader(SegDataset(train_pairs, augment=augment), batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(SegDataset(val_pairs, augment=False), batch_size=batch_size, shuffle=False, num_workers=args.num_workers) if val_pairs else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print("CleanOps U-Net standalone training (config-based)")
    print(f"config={config_path}")
    print(f"data_root={root}")
    print(f"val_split={val_split}")
    print(f"output_dir={out_dir}")
    print(f"device={device} amp={amp_enabled}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"model=Unet encoder={encoder_name} encoder_weights={encoder_weights} classes={classes}")
    print(f"image_size={image_size} epochs={epochs} batch_size={batch_size} lr={lr} workers={args.num_workers}")
    print(f"augment={augment} class_weights={class_weights}")
    print(f"train_samples={len(train_pairs)} train_batches={len(train_loader)}")
    print(f"val_samples={len(val_pairs)} val_batches={len(val_loader) if val_loader is not None else 0}")
    print("-" * 80, flush=True)

    best_miou = 0.0
    best_state: dict | None = None
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        print(f"epoch {epoch}/{epochs} started", flush=True)
        for batch_idx, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            if args.log_interval > 0 and (batch_idx == 1 or batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)):
                running = train_loss / batch_idx
                print(
                    f"epoch {epoch}/{epochs} train batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f} running_loss={running:.4f}",
                    flush=True,
                )
        train_loss /= max(1, len(train_loader))

        val_miou: float | None = None
        if val_loader is not None:
            model.eval()
            confusion = np.zeros((classes, classes), dtype=np.int64)
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    gt_np = masks.cpu().numpy().astype(np.int64).ravel()
                    pr_np = preds.cpu().numpy().astype(np.int64).ravel()
                    valid = (gt_np >= 0) & (gt_np < classes) & (pr_np >= 0) & (pr_np < classes)
                    np.add.at(confusion, (gt_np[valid], pr_np[valid]), 1)

            val_miou = compute_miou_12(confusion, classes)
            if val_miou > best_miou:
                best_miou = val_miou
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), out_dir / "best.pt")
                print(f"epoch {epoch}/{epochs} new best mIoU_12={val_miou:.6f}", flush=True)

        torch.save(model.state_dict(), out_dir / "last.pt")
        elapsed = time.perf_counter() - epoch_start
        gpu_mem = ""
        if device.type == "cuda":
            allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            gpu_mem = f" gpu_mem_allocated={allocated_gb:.2f}GB gpu_mem_reserved={reserved_gb:.2f}GB"
            torch.cuda.reset_peak_memory_stats(device)
        if val_miou is None:
            print(f"epoch {epoch}/{epochs} done train_loss={train_loss:.4f} time_sec={elapsed:.1f}{gpu_mem}", flush=True)
        else:
            print(
                f"epoch {epoch}/{epochs} done train_loss={train_loss:.4f} "
                f"val_mIoU_12={val_miou:.6f} best_mIoU_12={best_miou:.6f} "
                f"time_sec={elapsed:.1f}{gpu_mem}",
                flush=True,
            )
        print("-" * 80, flush=True)

    print(f"Best mIoU_12: {best_miou:.6f}", flush=True)

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "encoder": encoder_name,
            "img_size": image_size,
            "best_miou": best_miou,
            "class_mapping": {0: "background", 1: "stain_or_water", 2: "wet_surface"},
        }
        torch.save(payload, save_path)
        print(f"Saved production checkpoint: {save_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
