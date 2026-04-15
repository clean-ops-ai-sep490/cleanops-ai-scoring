"""
Training Script: CleaningQualityNet
=====================================
Fine-tune ResNet50 trên cleaning dataset.

Cách chạy:
    python src/train.py
    python src/train.py --epochs 30 --batch_size 32 --lr 1e-4
    python src/train.py --resume models/checkpoint_best.pth

Output:
    models/checkpoint_best.pth   <- best model theo val loss
    models/checkpoint_last.pth   <- model sau epoch cuối
    outputs/training_curves.png  <- loss/accuracy curves
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CleaningQualityNet, build_model

# ─── Cấu hình mặc định ───────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_dir": "data/processed",
    "model_dir": "models",
    "output_dir": "outputs",
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 25,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "unfreeze_epoch": 10,      # Epoch để unfreeze toàn bộ backbone
    "patience": 7,              # Early stopping patience
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ─── Dataset ─────────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224):
    """
    Data augmentation:
    - Train: flip, rotation, color jitter, random erasing (giả lập vết bẩn)
    - Val: chỉ resize + normalize
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]
    )
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # Giả lập vết bẩn bị che
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


class CleaningDataset(Dataset):
    """
    Wrapper quanh ImageFolder để:
    1. Load ảnh clean/dirty
    2. Tạo pseudo-labels cho 4 tham số (từ binary label clean/dirty)
    
    Pseudo-label logic:
    - clean → stains=0-2, coverage=0-10%, objects=0
    - dirty → stains=3-15, coverage=20-80%, objects=1-5
    
    Khi có dataset real với annotation thật, thay thế phần này.
    """
    
    def __init__(self, root: str, split: str, img_size: int = 224):
        self.transform = get_transforms(split, img_size)
        self.img_folder = ImageFolder(root)
        # class_to_idx: {'clean': 0, 'dirty': 1}
        self.class_to_idx = self.img_folder.class_to_idx
        self.samples = self.img_folder.samples
        
        # Seed cho reproducibility nhưng vẫn có variation
        self.rng = np.random.RandomState(42)
    
    def __len__(self):
        return len(self.samples)
    
    def _generate_pseudo_labels(self, is_dirty: bool, idx: int):
        """
        Sinh pseudo-labels từ binary clean/dirty.
        Dùng idx để tạo variation nhất quán.
        """
        rng = np.random.RandomState(idx)
        
        if not is_dirty:
            stains = rng.randint(0, 3)              # 0-2 vết
            coverage = rng.uniform(0, 10)            # 0-10%
            objects = rng.randint(0, 2)             # 0-1 dị vật
            quality = rng.uniform(75, 100)           # Score cao
        else:
            stains = rng.randint(3, 16)             # 3-15 vết
            coverage = rng.uniform(20, 85)           # 20-85%
            objects = rng.randint(1, 6)             # 1-5 dị vật
            quality = rng.uniform(0, 45)            # Score thấp
        
        return {
            "stains_norm": torch.tensor(stains / 20.0, dtype=torch.float32),
            "coverage_norm": torch.tensor(coverage / 100.0, dtype=torch.float32),
            "objects_norm": torch.tensor(objects / 10.0, dtype=torch.float32),
            "quality_norm": torch.tensor(quality / 100.0, dtype=torch.float32),
            "is_dirty": torch.tensor(1 if is_dirty else 0, dtype=torch.long),
        }
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load và transform ảnh
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        
        # Class: tìm 'dirty' trong class_to_idx
        dirty_idx = self.class_to_idx.get("dirty", 1)
        is_dirty = (label == dirty_idx)
        
        labels = self._generate_pseudo_labels(is_dirty, idx)
        
        return img_tensor, labels


# ─── Loss Function ────────────────────────────────────────────────────────────

class CleaningLoss(nn.Module):
    """
    Multi-task loss:
    - Regression losses: MSE cho stains, coverage, objects, quality
    
    Weighted sum theo độ quan trọng của từng task.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
        # Weights cho từng task
        self.w_stains = 1.0
        self.w_coverage = 1.5    # dirt_coverage quan trọng nhất
        self.w_objects = 0.8
        self.w_quality = 2.0     # quality score quan trọng nhất
    
    def forward(self, preds: dict, targets: dict) -> dict:
        loss_stains = self.mse(preds["stains_norm"], targets["stains_norm"])
        loss_coverage = self.mse(preds["coverage_norm"], targets["coverage_norm"])
        loss_objects = self.mse(preds["objects_norm"], targets["objects_norm"])
        loss_quality = self.mse(preds["quality_norm"], targets["quality_norm"])
        
        total = (
            self.w_stains * loss_stains +
            self.w_coverage * loss_coverage +
            self.w_objects * loss_objects +
            self.w_quality * loss_quality
        )
        
        return {
            "total": total,
            "stains": loss_stains.item(),
            "coverage": loss_coverage.item(),
            "objects": loss_objects.item(),
            "quality": loss_quality.item(),
        }


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(config["device"])
        
        # Tạo thư mục output
        Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model = build_model(mode="single", freeze=True).to(self.device)
        
        # Loss
        self.criterion = CleaningLoss()
        
        # Optimizer: chỉ update params không bị freeze
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
        
        # Scheduler: cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["num_epochs"], eta_min=1e-6
        )
        
        # History để vẽ curves
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_quality_mae": [], "val_quality_mae": [],
            "lr": []
        }
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def load_data(self):
        """Load train/val datasets."""
        data_dir = self.cfg["data_dir"]
        img_size = self.cfg["img_size"]
        
        train_dataset = CleaningDataset(f"{data_dir}/train", "train", img_size)
        val_dataset = CleaningDataset(f"{data_dir}/val", "val", img_size)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=True if self.device.type == "cuda" else False,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=True if self.device.type == "cuda" else False,
        )
        
        print(f"[OK] Train: {len(train_dataset)} ảnh | Val: {len(val_dataset)} ảnh")
        print(f"     Classes: {train_dataset.class_to_idx}")
    
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        losses = []
        quality_maes = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for imgs, targets in pbar:
            imgs = imgs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss_dict = self.criterion(preds, targets)
            
            loss_dict["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            losses.append(loss_dict["total"].item())
            mae = (preds["quality_norm"].detach() - targets["quality_norm"]).abs().mean().item()
            quality_maes.append(mae)
            
            pbar.set_postfix({"loss": f"{np.mean(losses[-20:]):.4f}", "q_mae": f"{mae:.3f}"})
        
        return {"loss": np.mean(losses), "quality_mae": np.mean(quality_maes)}
    
    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        losses = []
        quality_maes = []
        
        for imgs, targets in tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False):
            imgs = imgs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            preds = self.model(imgs)
            loss_dict = self.criterion(preds, targets)
            
            losses.append(loss_dict["total"].item())
            mae = (preds["quality_norm"] - targets["quality_norm"]).abs().mean().item()
            quality_maes.append(mae)
            
        return {
            "loss": np.mean(losses),
            "quality_mae": np.mean(quality_maes),
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }
        
        last_path = f"{self.cfg['model_dir']}/checkpoint_last.pth"
        torch.save(ckpt, last_path)
        
        if is_best:
            best_path = f"{self.cfg['model_dir']}/checkpoint_best.pth"
            torch.save(ckpt, best_path)
            print(f"  [BEST] Đã lưu best model → {best_path}")
    
    def resume(self, path: str):
        print(f"[*] Resume từ {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        return ckpt["epoch"] + 1
    
    def plot_curves(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Training Curves — CleaningQualityNet", fontsize=14, y=1.02)
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"], label="Val", linewidth=2)
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Quality MAE
        axes[1].plot(epochs, self.history["train_quality_mae"], label="Train", linewidth=2)
        axes[1].plot(epochs, self.history["val_quality_mae"], label="Val", linewidth=2)
        axes[1].set_title("Quality Score MAE (0-1 scale)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # LR
        axes[2].plot(epochs, self.history["lr"], color="gray", linewidth=2)
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.cfg['output_dir']}/training_curves.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[OK] Đã lưu training curves → {save_path}")
    
    def train(self, resume_path: str = None):
        start_epoch = 0
        if resume_path:
            start_epoch = self.resume(resume_path)
        
        print(f"\n{'='*60}")
        print(f"Training CleaningQualityNet")
        print(f"Device: {self.device} | Epochs: {self.cfg['num_epochs']}")
        print(f"Trainable params: {self.model.get_trainable_params():,}")
        print(f"{'='*60}\n")
        
        self.load_data()
        
        for epoch in range(start_epoch, self.cfg["num_epochs"]):
            t0 = time.time()
            
            # Unfreeze backbone sau N epochs
            if epoch == self.cfg["unfreeze_epoch"]:
                self.model.unfreeze_backbone()
                # Reset optimizer với LR nhỏ hơn cho backbone
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.cfg["lr"] * 0.1,
                    weight_decay=self.cfg["weight_decay"]
                )
                print(f"\n[Epoch {epoch+1}] Backbone unfrozen, LR → {self.cfg['lr'] * 0.1:.2e}")
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            self.scheduler.step()
            
            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Logging
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_quality_mae"].append(train_metrics["quality_mae"])
            self.history["val_quality_mae"].append(val_metrics["quality_mae"])
            self.history["lr"].append(current_lr)
            
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            # Print summary
            print(
                f"Epoch {epoch+1:3d}/{self.cfg['num_epochs']} | "
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Val loss: {val_metrics['loss']:.4f} | "
                f"Val Q-MAE: {val_metrics['quality_mae']*100:.1f}% | "
                f"LR: {current_lr:.1e} | "
                f"{elapsed:.1f}s"
                + (" ← BEST" if is_best else "")
            )
            
            # Early stopping
            if self.patience_counter >= self.cfg["patience"]:
                print(f"\n[STOP] Early stopping tại epoch {epoch+1}")
                break
        
        self.plot_curves()
        
        # Lưu final metrics
        with open(f"{self.cfg['output_dir']}/training_metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n[DONE] Training xong!")
        print(f"       Best val loss: {self.best_val_loss:.4f}")
        print(f"       Model: {self.cfg['model_dir']}/checkpoint_best.pth")
        print(f"       Chạy tiếp: python src/infer.py")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train CleaningQualityNet")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--unfreeze_epoch", type=int, default=DEFAULT_CONFIG["unfreeze_epoch"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = {**DEFAULT_CONFIG}
    config.update({
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "data_dir": args.data_dir,
        "unfreeze_epoch": args.unfreeze_epoch,
    })
    
    trainer = Trainer(config)
    trainer.train(resume_path=args.resume)
