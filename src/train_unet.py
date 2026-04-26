import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import settings

try:
    from src.models.unet_segmenter import UNetSegmenter
except ModuleNotFoundError:
    # Allow direct execution: python src/train_unet.py
    from models.unet_segmenter import UNetSegmenter


class MulticlassSegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, augment=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.augment = augment

        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            self.image_paths.extend(sorted(self.images_dir.glob(ext)))

        # Keep only pairs that really have masks.
        valid = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                valid.append(img_path)
        self.image_paths = valid

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Failed to load pair: {img_path} | {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment is not None:
            transformed = self.augment(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask.astype(np.int64), dtype=torch.long)
        return image, mask


def get_train_augment(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.RandomBrightnessContrast(p=0.4),
        ]
    )


def get_valid_augment(img_size: int):
    return A.Compose([A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR)])


def compute_iou(pred: torch.Tensor, target: torch.Tensor, class_id: int):
    pred_cls = pred == class_id
    target_cls = target == class_id
    intersection = (pred_cls & target_cls).sum().item()
    union = (pred_cls | target_cls).sum().item()
    if union == 0:
        return 0.0
    return intersection / (union + 1e-6)


def build_loaders(data_root: Path, img_size: int, batch_size: int, workers: int):
    train_ds = MulticlassSegmentationDataset(
        images_dir=data_root / "images" / "train",
        masks_dir=data_root / "masks" / "train",
        augment=get_train_augment(img_size),
    )
    valid_ds = MulticlassSegmentationDataset(
        images_dir=data_root / "images" / "valid",
        masks_dir=data_root / "masks" / "valid",
        augment=get_valid_augment(img_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader


def main():
    parser = argparse.ArgumentParser(description="Train U-Net multiclass (0:bg,1:stain,2:wet)")
    parser.add_argument("--data-root", default=str(settings.unet_processed_root))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=settings.unet_img_size)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-path", default=settings.unet_model_path)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print("[ERROR] Processed dataset chưa tồn tại.")
        print("        Hãy chạy: python src/preprocess_unet_data.py")
        return

    train_images_dir = data_root / "images" / "train"
    valid_images_dir = data_root / "images" / "valid"
    if not train_images_dir.exists() or not valid_images_dir.exists():
        print("[ERROR] Thiếu split train/valid trong processed dataset.")
        print("        Hãy chạy lại: python src/preprocess_unet_data.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 72)
    print("U-NET MULTICLASS TRAINING")
    print("=" * 72)
    print(f"device   : {device}")
    print(f"encoder  : {args.encoder}")
    print(f"weights  : {args.encoder_weights}")
    print(f"init_ckpt: {args.init_checkpoint or '(none)'}")
    print(f"img_size : {args.img_size}")
    print(f"epochs   : {args.epochs}")
    print(f"batch    : {args.batch}")
    print(f"data_root: {data_root.resolve()}")

    train_loader, valid_loader = build_loaders(
        data_root=data_root,
        img_size=args.img_size,
        batch_size=args.batch,
        workers=args.workers,
    )
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"valid samples: {len(valid_loader.dataset)}")

    encoder_weights = None if str(args.encoder_weights).strip().lower() in {"", "none", "null"} else args.encoder_weights
    model = UNetSegmenter(encoder_name=args.encoder, encoder_weights=encoder_weights, classes=3).to(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"U-Net init checkpoint not found: {init_path}")

        ckpt = torch.load(init_path, map_location=device)
        state = ckpt.get("model_state") if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Loaded U-Net init checkpoint: {init_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=use_amp)

    dice_loss = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode=smp.losses.MULTICLASS_MODE)

    best_miou = -1.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = 0.7 * dice_loss(logits, masks) + 0.3 * focal_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        iou1_scores = []
        iou2_scores = []
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Valid {epoch}/{args.epochs}"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)
                loss = 0.7 * dice_loss(logits, masks) + 0.3 * focal_loss(logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                iou1_scores.append(compute_iou(preds, masks, class_id=1))
                iou2_scores.append(compute_iou(preds, masks, class_id=2))

        val_loss /= max(1, len(valid_loader))
        iou1 = float(np.mean(iou1_scores)) if iou1_scores else 0.0
        iou2 = float(np.mean(iou2_scores)) if iou2_scores else 0.0
        miou = (iou1 + iou2) / 2.0

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | IoU_stain={iou1:.4f} | "
            f"IoU_wet={iou2:.4f} | mIoU_12={miou:.4f}"
        )

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "encoder": args.encoder,
                    "img_size": args.img_size,
                    "class_mapping": {0: "background", 1: "stain_or_water", 2: "wet_surface"},
                },
                save_path,
            )
            print(f"  -> saved best checkpoint to {save_path} (mIoU_12={best_miou:.4f})")

    print("\nTraining done.")
    print(f"Best mIoU_12: {best_miou:.4f}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
