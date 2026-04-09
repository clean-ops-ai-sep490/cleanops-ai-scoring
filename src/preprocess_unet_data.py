import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import settings

BG_CLASS = 0
STAIN_CLASS = 1
WET_SURFACE_CLASS = 2


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    split: str
    source: str
    sample_type: str  # hd10k_mask | stagnant_yolo


def is_metadata_file(path: Path) -> bool:
    return path.name.startswith("._")


def iter_images(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_metadata_file(p):
            continue
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield p


def map_hd10k_mask_to_classes(mask_bgr: np.ndarray) -> np.ndarray:
    """
    HD10K liquid mask is RGB/BGR-like mask where non-black means liquid dirt.
    Convert to multiclass mask:
      - 0: background
      - 1: stains/liquid
      - 2: wet surface (not present in HD10K liquid branch)
    """
    if mask_bgr is None:
        raise ValueError("Mask image cannot be None")

    out = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    non_zero = np.any(mask_bgr != 0, axis=2)
    out[non_zero] = STAIN_CLASS
    return out


def yolo_to_mask(image_shape, yolo_txt: Path) -> np.ndarray:
    """
    Convert Stagnant Water YOLO txt to multiclass mask.
      class 0 (water) -> 1 (stain)
      class 1 (wet surface) -> 2 (wet surface)

    Water region has higher priority if overlapping with wet-surface boxes.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not yolo_txt.exists():
        return mask

    lines = yolo_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        if cls_id == 0:
            # Water overrides wet-surface when overlap happens.
            mask[y1:y2, x1:x2] = STAIN_CLASS
        elif cls_id == 1:
            region = mask[y1:y2, x1:x2]
            region[region == BG_CLASS] = WET_SURFACE_CLASS
            mask[y1:y2, x1:x2] = region

    return mask


def split_samples(items, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def collect_hd10k_samples(hd10k_root: Path, seed: int):
    dataset_root = hd10k_root / "IROS2022_Dataset"

    train_img_root = dataset_root / "train" / "liquid_dirts" / "images"
    train_mask_root = dataset_root / "train" / "liquid_dirts" / "liquid_dirts_masks"

    test_img_root = dataset_root / "test" / "images"
    test_mask_root = dataset_root / "test" / "liquid_dirts_masks"

    if not train_img_root.exists() or not train_mask_root.exists():
        raise FileNotFoundError(f"HD10K train liquid dirs not found under {dataset_root}")
    if not test_img_root.exists() or not test_mask_root.exists():
        raise FileNotFoundError(f"HD10K test liquid dirs not found under {dataset_root}")

    train_pairs = []
    for img_path in iter_images(train_img_root):
        rel = img_path.relative_to(train_img_root)
        mask_path = (train_mask_root / rel).with_suffix(".png")
        if mask_path.exists() and not is_metadata_file(mask_path):
            train_pairs.append((img_path, mask_path))

    # Keep official test split as test, carve validation from train split.
    train_pairs_split, val_pairs_split, _ = split_samples(train_pairs, train_ratio=0.9, val_ratio=0.1, seed=seed)

    samples = []
    for img_path, mask_path in train_pairs_split:
        samples.append(Sample(img_path, mask_path, "train", "hd10k", "hd10k_mask"))
    for img_path, mask_path in val_pairs_split:
        samples.append(Sample(img_path, mask_path, "valid", "hd10k", "hd10k_mask"))

    for img_path in iter_images(test_img_root):
        rel = img_path.relative_to(test_img_root)
        mask_path = (test_mask_root / rel).with_suffix(".png")
        if mask_path.exists() and not is_metadata_file(mask_path):
            samples.append(Sample(img_path, mask_path, "test", "hd10k", "hd10k_mask"))

    return samples


def find_image_for_label(txt_path: Path):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = txt_path.with_suffix(ext)
        if p.exists() and not is_metadata_file(p):
            return p
    return None


def collect_stagnant_samples(stagnant_root: Path, seed: int):
    if not stagnant_root.exists():
        raise FileNotFoundError(f"Stagnant Water dataset not found: {stagnant_root}")

    pairs = []
    for txt_path in stagnant_root.rglob("*.txt"):
        if txt_path.name.lower() == "classes.txt" or is_metadata_file(txt_path):
            continue
        img_path = find_image_for_label(txt_path)
        if img_path is None:
            continue
        pairs.append((img_path, txt_path))

    tr, va, te = split_samples(pairs, train_ratio=0.8, val_ratio=0.1, seed=seed)

    samples = []
    for img_path, txt_path in tr:
        samples.append(Sample(img_path, txt_path, "train", "stagnant", "stagnant_yolo"))
    for img_path, txt_path in va:
        samples.append(Sample(img_path, txt_path, "valid", "stagnant", "stagnant_yolo"))
    for img_path, txt_path in te:
        samples.append(Sample(img_path, txt_path, "test", "stagnant", "stagnant_yolo"))

    return samples


def ensure_layout(out_root: Path):
    for split in ["train", "valid", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "masks" / split).mkdir(parents=True, exist_ok=True)


def clear_existing_outputs(out_root: Path):
    for split in ["train", "valid", "test"]:
        for kind in ["images", "masks"]:
            folder = out_root / kind / split
            if not folder.exists():
                continue
            for file_path in folder.glob("*"):
                if file_path.is_file():
                    file_path.unlink(missing_ok=True)


def build_output_name(sample: Sample, idx: int):
    stem = sample.image_path.stem.replace(" ", "_")
    return f"{sample.source}_{idx:06d}_{stem}"


def process_and_export(samples, out_root: Path):
    ensure_layout(out_root)

    stats = {
        "train": {"samples": 0, "pixel_counts": {0: 0, 1: 0, 2: 0}},
        "valid": {"samples": 0, "pixel_counts": {0: 0, 1: 0, 2: 0}},
        "test": {"samples": 0, "pixel_counts": {0: 0, 1: 0, 2: 0}},
    }

    for idx, sample in enumerate(samples):
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        if sample.sample_type == "hd10k_mask":
            raw_mask = cv2.imread(str(sample.label_path), cv2.IMREAD_COLOR)
            if raw_mask is None:
                continue
            mask = map_hd10k_mask_to_classes(raw_mask)
        elif sample.sample_type == "stagnant_yolo":
            mask = yolo_to_mask(image.shape, sample.label_path)
        else:
            continue

        out_name = build_output_name(sample, idx)
        out_img = out_root / "images" / sample.split / f"{out_name}.jpg"
        out_mask = out_root / "masks" / sample.split / f"{out_name}.png"

        cv2.imwrite(str(out_img), image)
        cv2.imwrite(str(out_mask), mask)

        stats[sample.split]["samples"] += 1
        unique, counts = np.unique(mask, return_counts=True)
        for cls_id, cls_count in zip(unique.tolist(), counts.tolist()):
            if cls_id in stats[sample.split]["pixel_counts"]:
                stats[sample.split]["pixel_counts"][cls_id] += int(cls_count)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare multiclass U-Net dataset from HD10K + Stagnant Water")
    parser.add_argument(
        "--hd10k-root",
        default=str(settings.hd10k_root),
        help="Path to HD10K_IROS2022 root",
    )
    parser.add_argument(
        "--stagnant-root",
        default=str(settings.stagnant_root),
        help="Path to Stagnant Water dataset root",
    )
    parser.add_argument(
        "--out-root",
        default=str(settings.unet_processed_root),
        help="Output folder for merged multiclass segmentation dataset",
    )
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    args = parser.parse_args()

    hd10k_root = Path(args.hd10k_root)
    stagnant_root = Path(args.stagnant_root)
    out_root = Path(args.out_root)

    print("[1/4] Collecting HD10K samples...")
    hd10k_samples = collect_hd10k_samples(hd10k_root, seed=args.seed)
    print(f"      HD10K samples: {len(hd10k_samples)}")

    print("[2/4] Collecting Stagnant Water samples...")
    stagnant_samples = collect_stagnant_samples(stagnant_root, seed=args.seed)
    print(f"      Stagnant samples: {len(stagnant_samples)}")

    all_samples = hd10k_samples + stagnant_samples
    print(f"[3/4] Total samples before export: {len(all_samples)}")

    if out_root.exists():
        print(f"      Clearing old processed dataset files at {out_root}")

    ensure_layout(out_root)
    clear_existing_outputs(out_root)

    print("[4/4] Exporting images/masks...")
    stats = process_and_export(all_samples, out_root)

    print("\n=== DONE: U-Net dataset prepared ===")
    print(f"Output root: {out_root.resolve()}")
    for split in ["train", "valid", "test"]:
        s = stats[split]
        px = s["pixel_counts"]
        print(
            f"{split:>5} | samples={s['samples']} | pixels(bg/stain/wet)=({px[0]}, {px[1]}, {px[2]})"
        )


if __name__ == "__main__":
    main()
