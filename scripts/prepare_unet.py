from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_utils import (
    ensure_dir,
    find_nearby_label,
    get_image_size,
    iter_images,
    iter_masks,
    load_schema,
    parse_simple_yaml_names,
    safe_stem,
    train_val_split,
    write_json,
)


DATASET_CLASS_MAP = {
    "dirty_floor_safa": {
        "stained_floor": "dirty_area",
        "stain": "dirty_area",
        "wet_floor": "wet_surface",
    },
    "dirty_floor_safa_full": {
        "stained_floor": "dirty_area",
        "stain": "dirty_area",
        "wet_floor": "wet_surface",
    },
    "dirty_floor_aistudy": {
        "stain": "dirty_area",
        "muddy_floor": "dirty_area",
        "muddy-floor": "dirty_area",
    },
    "synspill": {
        "spill": "wet_surface",
        "spills": "wet_surface",
        "liquid_spill": "wet_surface",
        "annotation_masks": "wet_surface",
        "wet_surface": "wet_surface",
    },
    "hd10k": {
        "liquid_dirt": "wet_surface",
        "liquid_dirts": "wet_surface",
        "liquid_dirts_masks": "wet_surface",
        "wet_surface": "wet_surface",
    },
    "wet_surface_stranger": {
        "wet_floor": "wet_surface",
        "wet floor": "wet_surface",
    },
    "wet_surface_stranger_2": {
        "wet-floor": "wet_surface",
        "wetfloor": "wet_surface",
        "wet_floor": "wet_surface",
    },
    "water_puddle": {
        "puddle": "wet_surface",
    },
}

WET_SURFACE_MASK_DATASETS = {
    "hd10k",
    "synspill",
    "water_puddle",
    "wet_surface_stranger",
    "wet_surface_stranger_2",
}


@dataclass
class UnetSample:
    dataset: str
    image: Path
    mask: Path | None
    polygons: list[tuple[int, list[float]]]
    source: str
    rles: list[tuple[int, dict[str, Any]]] = field(default_factory=list)


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def class_to_target(dataset: str, source_name: str, segmentation_ids: dict[str, int]) -> int | None:
    mapped = DATASET_CLASS_MAP.get(dataset, {}).get(normalize_name(source_name))
    if mapped is None:
        return None
    return segmentation_ids.get(mapped)


def require_pillow():
    try:
        from PIL import Image, ImageDraw

        return Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to render or validate U-Net masks. Install with: pip install pillow") from exc


def discover_yolo_names(dataset_root: Path) -> dict[int, str]:
    for candidate in dataset_root.rglob("*.yaml"):
        names = parse_simple_yaml_names(candidate)
        if names:
            return names
    return {}


def find_mask_for_image(dataset_root: Path, image: Path) -> Path | None:
    mask_roots = [
        dataset_root / "annotation_masks",
        dataset_root / "masks",
        dataset_root / "mask",
        dataset_root / "Masks",
        dataset_root / "labels",
        dataset_root / "SegmentationClass",
        dataset_root / "release" / "annotation_masks",
        dataset_root / "samples" / "annotation_masks",
        dataset_root / "liquid_dirts_masks",
    ]
    for root in mask_roots:
        if not root.exists():
            continue
        for ext in [".png", ".bmp", ".tif", ".tiff"]:
            direct = root / f"{image.stem}{ext}"
            if direct.exists():
                return direct
        matches = [p for p in iter_masks(root) if p.stem == image.stem]
        if matches:
            return matches[0]
    return None


def collect_mask_samples(dataset: str, dataset_root: Path) -> list[UnetSample]:
    samples: list[UnetSample] = []
    image_roots = [
        dataset_root / "generated_images",
        dataset_root / "release" / "generated_images",
        dataset_root / "samples" / "generated_images",
        dataset_root / "images",
        dataset_root / "Images",
        dataset_root,
    ]
    seen = set()
    for root in image_roots:
        for image in iter_images(root):
            if image.resolve() in seen:
                continue
            lowered = str(image).lower().replace("/", "\\")
            if "\\annotation_masks\\" in lowered or "\\masks\\" in lowered or "\\mask\\" in lowered:
                continue
            mask = find_mask_for_image(dataset_root, image)
            if mask is not None:
                samples.append(UnetSample(dataset, image, mask, [], "mask"))
                seen.add(image.resolve())
    return samples


def find_same_stem_image(root: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        direct = root / f"{stem}{ext}"
        if direct.exists():
            return direct
    matches = [p for p in iter_images(root) if p.stem == stem]
    return matches[0] if matches else None


def collect_hd10k_liquid_samples(dataset_root: Path) -> tuple[list[UnetSample], Counter]:
    stats = Counter()
    samples: list[UnetSample] = []
    seen = set()
    mask_roots = [root for root in dataset_root.rglob("liquid_dirts_masks") if root.is_dir()]
    for mask_root in mask_roots:
        if mask_root.parent.name == "liquid_dirts":
            image_root = mask_root.parent / "images"
        else:
            image_root = mask_root.parent / "images"
        for mask in iter_masks(mask_root):
            rel = mask.relative_to(mask_root)
            image = find_same_stem_image(image_root / rel.parent, mask.stem)
            if image is None:
                stats["missing_image"] += 1
                continue
            if image.resolve() in seen:
                continue
            samples.append(UnetSample("hd10k", image, mask, [], "hd10k_liquid_mask"))
            seen.add(image.resolve())
    stats["hd10k_liquid_samples"] += len(samples)
    return samples, stats


def coco_image_path(dataset_root: Path, file_name: str, annotation_dir: Path | None = None) -> Path | None:
    candidates = [
        dataset_root / file_name,
        dataset_root / "data" / file_name,
    ]
    if annotation_dir is not None:
        candidates.extend(
            [
                annotation_dir / file_name,
                annotation_dir / "images" / file_name,
            ]
        )
    for direct in candidates:
        if direct.exists():
            return direct
    matches = list(dataset_root.rglob(Path(file_name).name))
    return matches[0] if matches else None


def collect_coco_samples(dataset: str, dataset_root: Path, segmentation_ids: dict[str, int]) -> tuple[list[UnetSample], Counter]:
    stats = Counter()
    samples: list[UnetSample] = []
    for json_path in sorted(dataset_root.rglob("*.json")):
        samples_by_image: dict[int, UnetSample] = {}
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            stats["bad_json"] += 1
            continue
        if not {"images", "annotations", "categories"}.issubset(data):
            continue
        categories = {int(c["id"]): c["name"] for c in data.get("categories", []) if "id" in c and "name" in c}
        images = {int(img["id"]): img for img in data.get("images", []) if "id" in img and "file_name" in img}
        for image_id, image in images.items():
            path = coco_image_path(dataset_root, image["file_name"], json_path.parent)
            if path is not None:
                samples_by_image[image_id] = UnetSample(dataset, path, None, [], "coco")
        for ann in data.get("annotations", []):
            image_id = int(ann.get("image_id", -1))
            sample = samples_by_image.get(image_id)
            if sample is None:
                stats["missing_image"] += 1
                continue
            source_name = categories.get(int(ann.get("category_id", -1)), "")
            target_id = class_to_target(dataset, source_name, segmentation_ids)
            if target_id is None:
                stats["ignored_class"] += 1
                continue
            segmentation = ann.get("segmentation")
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    if isinstance(polygon, list) and len(polygon) >= 6:
                        sample.polygons.append((target_id, [float(v) for v in polygon]))
                        stats["polygons_kept"] += 1
            elif isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
                sample.rles.append((target_id, segmentation))
                stats["rles_kept"] += 1
            else:
                stats["ignored_rle"] += 1
        stats["coco_files"] += 1
        samples.extend(s for s in samples_by_image.values() if s.polygons or s.rles)
    return samples, stats


def collect_yolo_seg_samples(dataset: str, dataset_root: Path, segmentation_ids: dict[str, int]) -> tuple[list[UnetSample], Counter]:
    stats = Counter()
    samples: list[UnetSample] = []
    names = discover_yolo_names(dataset_root)
    label_roots = [dataset_root / "labels", dataset_root / "Annotations", dataset_root / "annotations"]
    for image in iter_images(dataset_root):
        label_path = find_nearby_label(image, label_roots)
        if label_path is None:
            continue
        size = get_image_size(image)
        if not size:
            stats["missing_image_size"] += 1
            continue
        width, height = size
        polygons: list[tuple[int, list[float]]] = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                src_id = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                stats["bad_label_lines"] += 1
                continue
            source_name = names.get(src_id)
            if source_name is None:
                stats["unknown_class"] += 1
                continue
            target_id = class_to_target(dataset, source_name, segmentation_ids)
            if target_id is None:
                stats["ignored_class"] += 1
                continue
            if max(coords, default=0.0) <= 1.0:
                points = []
                for idx, value in enumerate(coords):
                    points.append(value * (width if idx % 2 == 0 else height))
            else:
                points = coords
            polygons.append((target_id, points))
            stats["polygons_kept"] += 1
        if polygons:
            samples.append(UnetSample(dataset, image, None, polygons, "yolo_seg"))
    return samples, stats


def collect_dataset(dataset: str, dataset_root: Path, segmentation_ids: dict[str, int]) -> tuple[list[UnetSample], Counter]:
    stats = Counter()
    if not dataset_root.exists():
        stats["missing_dataset_dir"] += 1
        return [], stats
    if not iter_images(dataset_root):
        stats["empty_dataset_dir"] += 1
        return [], stats

    samples = collect_mask_samples(dataset, dataset_root)
    seen = {s.image.resolve() for s in samples}
    if dataset == "hd10k":
        hd10k_samples, hd10k_stats = collect_hd10k_liquid_samples(dataset_root)
        stats.update(hd10k_stats)
        for sample in hd10k_samples:
            if sample.image.resolve() not in seen:
                samples.append(sample)
                seen.add(sample.image.resolve())
    coco_samples, coco_stats = collect_coco_samples(dataset, dataset_root, segmentation_ids)
    stats.update(coco_stats)
    for sample in coco_samples:
        if sample.image.resolve() not in seen:
            samples.append(sample)
            seen.add(sample.image.resolve())
    yolo_samples, yolo_stats = collect_yolo_seg_samples(dataset, dataset_root, segmentation_ids)
    stats.update(yolo_stats)
    for sample in yolo_samples:
        if sample.image.resolve() not in seen:
            samples.append(sample)
            seen.add(sample.image.resolve())

    stats["images_seen"] += len(iter_images(dataset_root))
    stats["samples_kept"] += len(samples)
    stats["polygons_kept"] += sum(len(s.polygons) for s in samples)
    stats["rles_kept"] += sum(len(s.rles) for s in samples)
    stats["mask_pairs_kept"] += sum(1 for s in samples if s.mask is not None)
    return samples, stats


def clean_output(out: Path) -> None:
    for rel in ["images/train", "images/val", "masks/train", "masks/val"]:
        target = out / rel
        if target.exists():
            shutil.rmtree(target)
        ensure_dir(target)


def convert_existing_mask(mask_path: Path, out_path: Path, allowed_values: set[int], foreground_class: int = 1) -> Counter:
    Image, _ = require_pillow()
    stats = Counter()
    with Image.open(mask_path) as src:
        gray = src.convert("L")
        values = set(gray.getdata())
        if values.issubset(allowed_values):
            gray.save(out_path)
            for value in values:
                stats[f"mask_value_{value}"] += 1
            return stats
        binary = gray.point(lambda p: foreground_class if p > 0 else 0)
        binary.save(out_path)
        stats[f"mask_reindexed_binary_to_{foreground_class}"] += 1
    return stats


def decode_compressed_rle_counts(value: str) -> list[int]:
    counts: list[int] = []
    index = 0
    value_len = len(value)
    while index < value_len:
        x = 0
        k = 0
        while True:
            c = ord(value[index]) - 48
            index += 1
            x |= (c & 0x1F) << (5 * k)
            k += 1
            if not (c & 0x20):
                break
        if c & 0x10:
            x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_coco_rle(segmentation: dict[str, Any], width: int, height: int):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required to decode COCO RLE masks. Install with: pip install numpy") from exc

    size = segmentation.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("COCO RLE is missing a valid size field.")
    rle_height, rle_width = int(size[0]), int(size[1])
    counts_value = segmentation.get("counts")
    if isinstance(counts_value, str):
        counts = decode_compressed_rle_counts(counts_value)
    elif isinstance(counts_value, list):
        counts = [int(v) for v in counts_value]
    else:
        raise ValueError("COCO RLE counts must be a string or list.")

    flat = np.zeros(rle_height * rle_width, dtype=np.uint8)
    offset = 0
    value = 0
    for run_length in counts:
        if run_length < 0:
            raise ValueError("COCO RLE contains a negative run length.")
        next_offset = min(offset + run_length, flat.size)
        if value:
            flat[offset:next_offset] = 1
        offset = next_offset
        value = 1 - value

    mask = flat.reshape((rle_width, rle_height)).T
    if (rle_width, rle_height) != (width, height):
        Image, _ = require_pillow()
        pil_mask = Image.fromarray(mask * 255, mode="L").resize((width, height), Image.NEAREST)
        mask = np.array(pil_mask, dtype=np.uint8) > 0
    return mask


def render_annotation_mask(sample: UnetSample, out_path: Path) -> Counter:
    Image, ImageDraw = require_pillow()
    size = get_image_size(sample.image)
    if not size:
        raise RuntimeError(f"Cannot read image size: {sample.image}")
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    stats = Counter()
    for class_id, polygon in sample.polygons:
        points = list(zip(polygon[0::2], polygon[1::2]))
        if len(points) >= 3:
            draw.polygon(points, fill=int(class_id))
            stats[f"polygon_class_{class_id}"] += 1
    if sample.rles:
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required to render COCO RLE masks. Install with: pip install numpy") from exc
        mask_array = np.array(mask, dtype=np.uint8)
        for class_id, rle in sample.rles:
            rle_mask = decode_coco_rle(rle, width, height)
            mask_array[rle_mask.astype(bool)] = int(class_id)
            stats[f"rle_class_{class_id}"] += 1
        mask = Image.fromarray(mask_array, mode="L")
    mask.save(out_path)
    return stats


def write_unet_sample(sample: UnetSample, split: str, out: Path, allowed_values: set[int]) -> Counter:
    stats = Counter()
    stem = safe_stem(sample.dataset, sample.image)
    image_dst = out / "images" / split / f"{stem}{sample.image.suffix.lower()}"
    mask_dst = out / "masks" / split / f"{stem}.png"
    shutil.copy2(sample.image, image_dst)
    if sample.mask is not None:
        foreground_class = 2 if sample.dataset in WET_SURFACE_MASK_DATASETS else 1
        stats.update(convert_existing_mask(sample.mask, mask_dst, allowed_values, foreground_class))
    else:
        stats.update(render_annotation_mask(sample, mask_dst))
    stats["images_written"] += 1
    stats["masks_written"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare U-Net segmentation dataset for CleanOps AI.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/unet")
    parser.add_argument("--schema", default="configs/label_schema.json")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-synspill", action="store_true", help="Exclude SynSpill wet_surface masks from the default U-Net dataset.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    _, segmentation_ids = load_schema(Path(args.schema))
    allowed_values = set(segmentation_ids.values())

    all_samples: list[UnetSample] = []
    included_datasets = [
        "dirty_floor_safa",
        "dirty_floor_safa_full",
        "dirty_floor_aistudy",
        "hd10k",
        "wet_surface_stranger",
        "wet_surface_stranger_2",
        "water_puddle",
    ]
    excluded_datasets = []
    if args.exclude_synspill:
        excluded_datasets.append({"dataset": "synspill", "reason": "excluded by --exclude-synspill"})
    else:
        included_datasets.append("synspill")

    report: dict[str, Any] = {
        "options": {
            "include_synspill": not args.exclude_synspill,
            "exclude_synspill": args.exclude_synspill,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "included_datasets": included_datasets,
        "excluded_datasets": excluded_datasets,
        "datasets": {},
        "splits": {},
        "warnings": [],
    }
    for dataset in included_datasets:
        samples, stats = collect_dataset(dataset, raw_dir / dataset, segmentation_ids)
        report["datasets"][dataset] = dict(stats)
        all_samples.extend(samples)

    clean_output(out)
    if not all_samples:
        report["warnings"].append("No U-Net samples were exported. Add datasets under data/raw and rerun.")
        write_json(out / "prepare_unet_report.json", report)
        print("No U-Net samples found. Expected raw datasets in data/raw/dirty_floor_safa and data/raw/dirty_floor_aistudy.")
        return 0

    train_samples, val_samples = train_val_split(all_samples, args.val_ratio, args.seed)
    write_stats = Counter()
    for sample in train_samples:
        write_stats.update(write_unet_sample(sample, "train", out, allowed_values))
    for sample in val_samples:
        write_stats.update(write_unet_sample(sample, "val", out, allowed_values))

    report["splits"] = {"train": len(train_samples), "val": len(val_samples)}
    report["written"] = dict(write_stats)
    write_json(out / "prepare_unet_report.json", report)
    print(f"U-Net export complete: {len(train_samples)} train, {len(val_samples)} val samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
