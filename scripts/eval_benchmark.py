from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import ensure_dir, iter_images


CLASS_NAMES = {0: "background", 1: "dirty_area", 2: "wet_surface"}
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 76, 76),
    2: (48, 140, 255),
}

_unet_model = None
_unet_img_size = 384
_unet_device = None
_weights_path = None


def load_unet_model(weights: str | None = None, default_img_size: int = 384) -> None:
    global _unet_model, _unet_img_size, _unet_device, _weights_path
    import torch
    from src.api.inference_utils import load_unet_model as _load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = _load(weights or "", device, default_img_size)
    if model is None:
        raise RuntimeError(f"Failed to load U-Net model from: {weights}")
    _unet_model = model
    _unet_img_size = img_size
    _unet_device = device
    _weights_path = weights or "auto"


def predict_mask(image_path: Path) -> np.ndarray:
    import torch

    assert _unet_model is not None, "Call load_unet_model() first"
    with Image.open(image_path) as src:
        image = src.convert("RGB")
    original_width, original_height = image.size
    import cv2
    rgb = np.array(image)
    resized = cv2.resize(rgb, (_unet_img_size, _unet_img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(_unet_device)
    with torch.no_grad():
        logits = _unet_model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_original = cv2.resize(pred, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return pred_original


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_gt_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        if mask.size != size:
            raise ValueError(f"{mask_path} size {mask.size[0]}x{mask.size[1]} != image size {size[0]}x{size[1]}")
        arr = np.asarray(mask, dtype=np.uint8)
    bad_values = sorted(int(v) for v in np.unique(arr) if int(v) not in CLASS_NAMES)
    if bad_values:
        raise ValueError(f"{mask_path} contains invalid class values: {bad_values}")
    return arr


def update_confusion(confusion: np.ndarray, gt: np.ndarray, pred: np.ndarray, classes: int) -> None:
    valid = (gt >= 0) & (gt < classes) & (pred >= 0) & (pred < classes)
    bins = classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    confusion += np.bincount(bins, minlength=classes * classes).reshape(classes, classes)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        out[mask == class_id] = color
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize(mask)
    foreground = mask > 0
    blended = (image.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    result = image.copy()
    result[foreground] = blended[foreground]
    return result


def error_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    result = image.copy()
    missed = (gt > 0) & (pred == 0)
    extra = (gt == 0) & (pred > 0)
    wrong_class = (gt > 0) & (pred > 0) & (gt != pred)
    result[missed] = (255, 217, 61)
    result[extra] = (255, 76, 76)
    result[wrong_class] = (48, 140, 255)
    return result


def save_overlay_grid(image_path: Path, gt: np.ndarray, pred: np.ndarray, output_path: Path) -> None:
    with Image.open(image_path) as src:
        image = np.asarray(src.convert("RGB"), dtype=np.uint8)
    panels = [
        image,
        overlay(image, gt),
        overlay(image, pred),
        error_overlay(image, gt, pred),
    ]
    pil_panels = [Image.fromarray(panel) for panel in panels]
    width, height = pil_panels[0].size
    grid = Image.new("RGB", (width * len(pil_panels), height), (0, 0, 0))
    for idx, panel in enumerate(pil_panels):
        grid.paste(panel, (idx * width, 0))
    ensure_dir(output_path.parent)
    grid.save(output_path, quality=92)


def class_metrics(confusion: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    ious = []
    dices = []
    for class_id, class_name in CLASS_NAMES.items():
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())
        union = tp + fp + fn
        iou = tp / union if union else None
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        if iou is not None:
            ious.append(iou)
        if dice is not None:
            dices.append(dice)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support_pixels": support,
                "iou": None if iou is None else round(iou, 6),
                "dice_f1": None if dice is None else round(dice, 6),
            }
        )
    total = float(confusion.sum())
    summary = {
        "pixel_accuracy": round(float(np.trace(confusion)) / total, 6) if total else 0.0,
        "mean_iou": round(float(np.mean(ious)), 6) if ious else 0.0,
        "mean_dice_f1": round(float(np.mean(dices)), 6) if dices else 0.0,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on the held-out benchmark masks.")
    parser.add_argument("--benchmark", default="data/processed/benchmark")
    parser.add_argument("--unet-root", default=None, help="Evaluate a prepared U-Net root with images/<split> and masks/<split>.")
    parser.add_argument("--split", default="test", help="Split name when --unet-root is used.")
    parser.add_argument("--output", default="outputs/benchmark")
    parser.add_argument("--weights", default=None, help="Optional U-Net weights path. Defaults to latest best.pt.")
    parser.add_argument("--allow-partial", action="store_true", help="Evaluate images that have masks and report missing masks.")
    parser.add_argument("--max-overlays", type=int, default=46)
    args = parser.parse_args()

    benchmark = Path(args.benchmark)
    if args.unet_root:
        benchmark = Path(args.unet_root)
        image_dir = benchmark / "images" / args.split
        mask_dir = benchmark / "masks" / args.split
    else:
        image_dir = benchmark / "images"
        mask_dir = benchmark / "masks"
    output_dir = Path(args.output)
    overlay_dir = output_dir / "overlays"
    images = iter_images(image_dir)
    if not benchmark.exists() or not images:
        print("Benchmark is empty. Add benchmark images/masks first.")
        return 1

    missing = [image.name for image in images if not (mask_dir / f"{image.stem}.png").exists()]
    if missing and not args.allow_partial:
        print(f"Missing ground-truth masks for {len(missing)}/{len(images)} images.")
        print(f"Mask directory: {mask_dir}")
        return 1

    eval_images = [image for image in images if (mask_dir / f"{image.stem}.png").exists()]
    if not eval_images:
        print("No labeled benchmark masks found.")
        return 1

    load_unet_model(args.weights)
    classes = len(CLASS_NAMES)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    per_image_rows: list[dict[str, Any]] = []

    for idx, image_path in enumerate(eval_images, start=1):
        with Image.open(image_path) as image:
            size = image.size
        gt = validate_gt_mask(mask_dir / f"{image_path.stem}.png", size)
        pred = predict_mask(image_path)
        update_confusion(confusion, gt, pred, classes)
        image_confusion = np.zeros((classes, classes), dtype=np.int64)
        update_confusion(image_confusion, gt, pred, classes)
        _, image_summary = class_metrics(image_confusion)
        per_image_rows.append(
            {
                "image": image_path.name,
                "pixel_accuracy": image_summary["pixel_accuracy"],
                "mean_iou": image_summary["mean_iou"],
                "mean_dice_f1": image_summary["mean_dice_f1"],
            }
        )
        if idx <= args.max_overlays:
            save_overlay_grid(image_path, gt, pred, overlay_dir / f"{image_path.stem}_benchmark_overlay.jpg")

    per_class_rows, summary = class_metrics(confusion)
    metrics = {
        "benchmark": str(benchmark),
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "split": args.split if args.unet_root else None,
        "weights": _weights_path,
        "images_total": len(images),
        "images_evaluated": len(eval_images),
        "missing_masks": missing,
        "summary": summary,
        "per_class": per_class_rows,
        "per_image": per_image_rows,
        "confusion_matrix": confusion.tolist(),
    }
    write_json(output_dir / "benchmark_metrics.json", metrics)
    write_csv(
        output_dir / "benchmark_metrics.csv",
        per_class_rows,
        ["class_id", "class_name", "support_pixels", "iou", "dice_f1"],
    )
    write_csv(
        output_dir / "benchmark_per_image.csv",
        per_image_rows,
        ["image", "pixel_accuracy", "mean_iou", "mean_dice_f1"],
    )
    report = [
        "# U-Net Benchmark Summary",
        "",
        f"- Images evaluated: `{len(eval_images)}/{len(images)}`",
        f"- Weights: `{_weights_path}`",
        f"- Pixel accuracy: `{summary['pixel_accuracy']:.6f}`",
        f"- Mean IoU: `{summary['mean_iou']:.6f}`",
        f"- Mean Dice/F1: `{summary['mean_dice_f1']:.6f}`",
        "",
        "## Per Class",
        "",
        "| class | support_pixels | IoU | Dice/F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in per_class_rows:
        report.append(
            f"| {row['class_name']} | {row['support_pixels']} | {row['iou']} | {row['dice_f1']} |"
        )
    if missing:
        report.extend(["", "## Missing Masks", "", *[f"- `{name}`" for name in missing]])
    ensure_dir(output_dir)
    (output_dir / "report_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Evaluated images: {len(eval_images)}/{len(images)}")
    print(f"Pixel accuracy: {summary['pixel_accuracy']:.6f}")
    print(f"Mean IoU: {summary['mean_iou']:.6f}")
    print(f"Mean Dice/F1: {summary['mean_dice_f1']:.6f}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
