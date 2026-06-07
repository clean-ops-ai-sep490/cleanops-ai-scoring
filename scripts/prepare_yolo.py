from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_utils import (
    ensure_dir,
    find_nearby_label,
    get_image_size,
    iter_images,
    load_schema,
    parse_simple_yaml_names,
    safe_stem,
    train_val_split,
    write_json,
)


DATASET_CLASS_MAP = {
    "floor_damage": {
        "stain": "stain_object",
        "damage": "stain_object",
        "floor_damage": "stain_object",
    },
    "dirty_floor_safa": {
        "garbage": "garbage_object",
        "garbage_object": "garbage_object",
    },
    "dirty_floor_safa_full": {
        "garbage": "garbage_object",
        "garbage_object": "garbage_object",
        "stain": "stain_object",
        "stain_object": "stain_object",
    },
    "saafai4": {
        "dirty_floor": "stain_object",
        "garbage": "garbage_object",
        "partial_dirty_floor": "stain_object",
    },
    "hd10k": {
        "solid_dirt": "garbage_object",
        "solid_dirts": "garbage_object",
        "garbage": "garbage_object",
        "0": "garbage_object",
    },
    "synspill": {
        "spill": "stain_object",
        "spills": "stain_object",
        "liquid_spill": "stain_object",
        "annotation_masks": "stain_object",
        "wet_surface": "stain_object",
    },
}

NEGATIVE_CLASS_NAMES = {"clean_floor", "clean", "background"}


@dataclass
class YoloSample:
    dataset: str
    image: Path
    labels: list[tuple[int, float, float, float, float]]
    source: str


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def class_to_target(dataset: str, source_name: str, detection_ids: dict[str, int], include_safa_stain: bool = False) -> int | None:
    if dataset in {"taco", "mju_waste"} and normalize_name(source_name) not in NEGATIVE_CLASS_NAMES:
        return detection_ids.get("garbage_object")
    if dataset == "dirty_floor_safa" and include_safa_stain and normalize_name(source_name) in {"stain", "stain_object"}:
        return detection_ids.get("stain_object")
    mapped = DATASET_CLASS_MAP.get(dataset, {}).get(normalize_name(source_name))
    if mapped is None:
        return None
    return detection_ids.get(mapped)


def valid_yolo_box(box: tuple[float, float, float, float]) -> bool:
    x, y, w, h = box
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def clamp_yolo_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    x, y, w, h = box
    x1 = max(0.0, x - w / 2)
    y1 = max(0.0, y - h / 2)
    x2 = min(1.0, x + w / 2)
    y2 = min(1.0, y + h / 2)
    w2 = x2 - x1
    h2 = y2 - y1
    if w2 <= 0.001 or h2 <= 0.001:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, w2, h2)


def bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    yolo = ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
    return clamp_yolo_box(yolo)


def polygon_to_yolo(points: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(points) < 6 or width <= 0 or height <= 0:
        return None
    xs = points[0::2]
    ys = points[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return bbox_to_yolo([x1, y1, x2 - x1, y2 - y1], width, height)


def normalized_polygon_to_yolo(points: list[float]) -> tuple[float, float, float, float] | None:
    if len(points) < 6:
        return None
    xs = points[0::2]
    ys = points[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    w = x2 - x1
    h = y2 - y1
    if w <= 0.001 or h <= 0.001:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, w, h)


def discover_yolo_names(dataset_root: Path) -> dict[int, str]:
    for candidate in dataset_root.rglob("*.yaml"):
        names = parse_simple_yaml_names(candidate)
        if names:
            return names
    return {}


def require_pillow():
    try:
        from PIL import Image

        return Image
    except Exception:
        return None


def find_synspill_mask(dataset_root: Path, image_path: Path) -> Path | None:
    mask_roots = [
        dataset_root / "annotation_masks",
        dataset_root / "masks",
        dataset_root / "release" / "annotation_masks",
        dataset_root / "samples" / "annotation_masks",
    ]
    for root in mask_roots:
        if not root.exists():
            continue
        for ext in [".png", ".bmp", ".tif", ".tiff"]:
            direct = root / f"{image_path.stem}{ext}"
            if direct.exists():
                return direct
        matches = [p for p in root.rglob("*") if p.is_file() and p.stem == image_path.stem and p.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}]
        if matches:
            return matches[0]
    return None


def mask_to_yolo_bbox(mask_path: Path, width: int, height: int) -> tuple[float, float, float, float] | None:
    Image = require_pillow()
    if Image is None:
        return None
    with Image.open(mask_path) as src:
        gray = src.convert("L")
        if gray.size != (width, height):
            gray = gray.resize((width, height), Image.NEAREST)
        bbox = gray.point(lambda p: 255 if p > 0 else 0).getbbox()
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return bbox_to_yolo([x1, y1, x2 - x1, y2 - y1], width, height)


def collect_synspill_mask_boxes(dataset_root: Path, detection_ids: dict[str, int]) -> tuple[list[YoloSample], Counter]:
    stats = Counter()
    target_id = detection_ids.get("stain_object")
    if target_id is None:
        return [], stats
    image_roots = [
        dataset_root / "generated_images",
        dataset_root / "release" / "generated_images",
        dataset_root / "samples" / "generated_images",
        dataset_root / "images",
    ]
    candidate_images: list[Path] = []
    seen = set()
    for root in image_roots:
        for image in iter_images(root):
            if image.resolve() not in seen:
                candidate_images.append(image)
                seen.add(image.resolve())
    if not candidate_images:
        candidate_images = [
            image
            for image in iter_images(dataset_root)
            if "annotation_mask" not in str(image).lower() and "\\masks\\" not in str(image).lower().replace("/", "\\")
        ]
    samples: list[YoloSample] = []
    for image in candidate_images:
        mask = find_synspill_mask(dataset_root, image)
        if mask is None:
            stats["missing_mask"] += 1
            continue
        size = get_image_size(image)
        if not size:
            stats["missing_image_size"] += 1
            continue
        box = mask_to_yolo_bbox(mask, size[0], size[1])
        if box is None:
            stats["empty_mask_or_missing_pillow"] += 1
            continue
        samples.append(YoloSample("synspill", image, [(target_id, *box)], "mask_bbox"))
    stats["mask_bbox_samples"] += len(samples)
    return samples, stats


def find_same_stem_image(root: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        direct = root / f"{stem}{ext}"
        if direct.exists():
            return direct
    matches = [p for p in iter_images(root) if p.stem == stem]
    return matches[0] if matches else None


def collect_hd10k_solid_samples(dataset_root: Path, detection_ids: dict[str, int]) -> tuple[list[YoloSample], Counter]:
    stats = Counter()
    target_id = detection_ids.get("garbage_object")
    if target_id is None:
        return [], stats
    samples: list[YoloSample] = []
    seen = set()
    bbox_roots = [root for root in dataset_root.rglob("solid_dirts_bboxes") if root.is_dir()]
    for bbox_root in bbox_roots:
        if bbox_root.parent.name == "solid_dirts":
            image_root = bbox_root.parent / "images"
        else:
            image_root = bbox_root.parent / "images"
        for label_path in sorted(bbox_root.rglob("*.txt")):
            rel = label_path.relative_to(bbox_root)
            image = find_same_stem_image(image_root / rel.parent, label_path.stem)
            if image is None:
                stats["missing_image"] += 1
                continue
            if image.resolve() in seen:
                continue
            labels: list[tuple[int, float, float, float, float]] = []
            for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    stats["bad_label_lines"] += 1
                    continue
                try:
                    x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                except ValueError:
                    stats["bad_label_lines"] += 1
                    continue
                box = clamp_yolo_box((x, y, w, h))
                if box is None:
                    stats["bad_bbox"] += 1
                    continue
                labels.append((target_id, *box))
            if labels:
                samples.append(YoloSample("hd10k", image, labels, "hd10k_solid_bbox"))
                seen.add(image.resolve())
    stats["hd10k_solid_samples"] += len(samples)
    stats["hd10k_solid_labels"] += sum(len(sample.labels) for sample in samples)
    return samples, stats


def collect_mju_waste_mask_boxes(dataset_root: Path, detection_ids: dict[str, int]) -> tuple[list[YoloSample], Counter]:
    stats = Counter()
    target_id = detection_ids.get("garbage_object")
    if target_id is None:
        return [], stats
    Image = require_pillow()
    if Image is None:
        stats["missing_pillow"] += 1
        return [], stats
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        stats["missing_cv2_or_numpy"] += 1
        return [], stats

    image_root = dataset_root / "JPEGImages"
    mask_root = dataset_root / "SegmentationClass"
    if not image_root.exists() or not mask_root.exists():
        stats["missing_voc_segmentation_dirs"] += 1
        return [], stats

    samples: list[YoloSample] = []
    for mask_path in sorted(mask_root.glob("*.png")):
        stats["mju_masks_seen"] += 1
        image = find_same_stem_image(image_root, mask_path.stem)
        if image is None:
            stats["missing_image"] += 1
            continue
        try:
            with Image.open(mask_path) as src:
                mask_array = np.array(src)
        except Exception:
            stats["bad_mask"] += 1
            continue
        if mask_array.ndim != 2:
            stats["bad_mask_shape"] += 1
            continue

        # MJU-Waste uses palette index 0 for waste foreground and 1 for background.
        foreground = (mask_array == 0).astype("uint8")
        foreground_ratio = float(foreground.mean())
        if foreground_ratio > 0.40:
            stats["mju_skipped_high_foreground_ratio"] += 1
            continue
        if foreground_ratio < 0.0005:
            stats["mju_skipped_low_foreground_ratio"] += 1
            continue

        mask_height, mask_width = foreground.shape
        image_size = get_image_size(image)
        if image_size and image_size != (mask_width, mask_height):
            stats["mju_image_mask_size_mismatch"] += 1

        labels: list[tuple[int, float, float, float, float]] = []
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        for component_idx in range(1, component_count):
            x, y, w, h, area = [int(v) for v in component_stats[component_idx]]
            if area < 25:
                stats["mju_components_too_small"] += 1
                continue
            box = bbox_to_yolo([x, y, w, h], mask_width, mask_height)
            if box is None:
                stats["bad_bbox"] += 1
                continue
            labels.append((target_id, *box))

        if labels:
            stats["mju_components_kept"] += len(labels)
            samples.append(YoloSample("mju_waste", image, labels, "mju_segmentation_bbox"))
        else:
            stats["mju_masks_without_valid_components"] += 1

    stats["mju_mask_bbox_samples"] += len(samples)
    stats["mju_mask_bbox_labels"] += sum(len(sample.labels) for sample in samples)
    return samples, stats


def load_yolo_labels(
    dataset: str,
    image_path: Path,
    dataset_root: Path,
    detection_ids: dict[str, int],
    names: dict[int, str],
    include_safa_stain: bool = False,
) -> tuple[list[tuple[int, float, float, float, float]], Counter]:
    stats = Counter()
    label_roots = [dataset_root / "labels", dataset_root / "Annotations", dataset_root / "annotations"]
    label_path = find_nearby_label(image_path, label_roots)
    if label_path is None:
        return [], stats

    labels: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            stats["bad_label_lines"] += 1
            continue
        try:
            src_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            stats["bad_label_lines"] += 1
            continue
        source_name = names.get(src_id)
        if source_name is None and dataset == "floor_damage":
            source_name = "stain"
        if source_name is None:
            stats["unknown_class"] += 1
            continue
        target_id = class_to_target(dataset, source_name, detection_ids, include_safa_stain)
        if target_id is None:
            stats["ignored_class"] += 1
            continue
        if len(coords) == 4:
            box = tuple(coords)  # type: ignore[assignment]
        elif max(coords, default=0.0) <= 1.0:
            box = normalized_polygon_to_yolo(coords)
        else:
            size = get_image_size(image_path)
            if not size:
                stats["missing_image_size"] += 1
                continue
            box = polygon_to_yolo(coords, size[0], size[1])
            if box is None:
                stats["bad_polygon"] += 1
                continue
        if not valid_yolo_box(box):
            box = clamp_yolo_box(box)
        if box is None:
            stats["bad_bbox"] += 1
            continue
        labels.append((target_id, *box))
    return labels, stats


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


def load_coco_samples(dataset: str, dataset_root: Path, detection_ids: dict[str, int], include_safa_stain: bool = False) -> tuple[list[YoloSample], Counter]:
    stats = Counter()
    json_files = sorted(p for p in dataset_root.rglob("*.json") if p.is_file())
    if dataset == "taco":
        json_files = [p for p in json_files if p.name == "annotations.json"]
    samples: list[YoloSample] = []
    for json_path in json_files:
        samples_by_image: dict[int, YoloSample] = {}
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
                samples_by_image[image_id] = YoloSample(dataset, path, [], "coco")
        for ann in data.get("annotations", []):
            image_id = int(ann.get("image_id", -1))
            sample = samples_by_image.get(image_id)
            if sample is None:
                stats["missing_image"] += 1
                continue
            source_name = categories.get(int(ann.get("category_id", -1)), "")
            target_id = class_to_target(dataset, source_name, detection_ids, include_safa_stain)
            if target_id is None:
                stats["ignored_class"] += 1
                continue
            width = int(images[image_id].get("width") or 0)
            height = int(images[image_id].get("height") or 0)
            if not width or not height:
                size = get_image_size(sample.image)
                if not size:
                    stats["missing_image_size"] += 1
                    continue
                width, height = size
            box = None
            if "bbox" in ann:
                box = bbox_to_yolo(ann["bbox"], width, height)
            elif isinstance(ann.get("segmentation"), list) and ann["segmentation"]:
                box = polygon_to_yolo(ann["segmentation"][0], width, height)
            if box is None:
                stats["bad_bbox"] += 1
                continue
            sample.labels.append((target_id, *box))
        stats["coco_files"] += 1
        samples.extend(sample for sample in samples_by_image.values() if sample.labels)
    stats["coco_samples_with_labels"] += len(samples)
    return samples, stats


def find_xml_image(dataset_root: Path, filename: str, xml_path: Path) -> Path | None:
    candidates = [
        xml_path.parent / filename,
        xml_path.parent.parent / "JPEGImages" / filename,
        xml_path.parent.parent / "images" / filename,
        dataset_root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(dataset_root.rglob(filename))
    return matches[0] if matches else None


def load_voc_samples(dataset: str, dataset_root: Path, detection_ids: dict[str, int]) -> tuple[list[YoloSample], Counter]:
    stats = Counter()
    samples: list[YoloSample] = []
    for xml_path in sorted(dataset_root.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            stats["bad_xml"] += 1
            continue
        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        image = find_xml_image(dataset_root, filename, xml_path)
        if image is None:
            stats["missing_image"] += 1
            continue
        size = get_image_size(image)
        if not size:
            stats["missing_image_size"] += 1
            continue
        width, height = size
        labels: list[tuple[int, float, float, float, float]] = []
        for obj in root.findall("object"):
            source_name = obj.findtext("name") or "waste"
            target_id = class_to_target(dataset, source_name, detection_ids)
            if target_id is None:
                stats["ignored_class"] += 1
                continue
            box_node = obj.find("bndbox")
            if box_node is None:
                stats["missing_bbox"] += 1
                continue
            try:
                xmin = float(box_node.findtext("xmin") or 0)
                ymin = float(box_node.findtext("ymin") or 0)
                xmax = float(box_node.findtext("xmax") or 0)
                ymax = float(box_node.findtext("ymax") or 0)
            except ValueError:
                stats["bad_bbox"] += 1
                continue
            box = bbox_to_yolo([xmin, ymin, xmax - xmin, ymax - ymin], width, height)
            if box is None:
                stats["bad_bbox"] += 1
                continue
            labels.append((target_id, *box))
        if labels:
            samples.append(YoloSample(dataset, image, labels, "voc"))
    stats["voc_samples_with_labels"] += len(samples)
    return samples, stats


def include_negative_images(dataset: str, dataset_root: Path) -> list[YoloSample]:
    samples: list[YoloSample] = []
    for image in iter_images(dataset_root):
        parent_names = {normalize_name(p.name) for p in image.parents if dataset_root in p.parents or p == dataset_root}
        if parent_names & NEGATIVE_CLASS_NAMES:
            samples.append(YoloSample(dataset, image, [], "negative_folder"))
    return samples


def collect_dataset(dataset: str, dataset_root: Path, detection_ids: dict[str, int], include_safa_stain: bool = False) -> tuple[list[YoloSample], Counter]:
    stats = Counter()
    if not dataset_root.exists():
        stats["missing_dataset_dir"] += 1
        return [], stats
    if not iter_images(dataset_root):
        stats["empty_dataset_dir"] += 1
        return [], stats

    samples, coco_stats = load_coco_samples(dataset, dataset_root, detection_ids, include_safa_stain)
    stats.update(coco_stats)
    seen_images = {s.image.resolve() for s in samples}

    if dataset == "hd10k":
        hd10k_samples, hd10k_stats = collect_hd10k_solid_samples(dataset_root, detection_ids)
        stats.update(hd10k_stats)
        for sample in hd10k_samples:
            if sample.image.resolve() not in seen_images:
                samples.append(sample)
                seen_images.add(sample.image.resolve())

    if dataset == "mju_waste":
        mask_samples, mask_stats = collect_mju_waste_mask_boxes(dataset_root, detection_ids)
        stats.update(mask_stats)
        for sample in mask_samples:
            if sample.image.resolve() not in seen_images:
                samples.append(sample)
                seen_images.add(sample.image.resolve())

        voc_samples, voc_stats = load_voc_samples(dataset, dataset_root, detection_ids)
        stats.update(voc_stats)
        for sample in voc_samples:
            if sample.image.resolve() not in seen_images:
                samples.append(sample)
                seen_images.add(sample.image.resolve())

    if dataset == "synspill":
        mask_samples, mask_stats = collect_synspill_mask_boxes(dataset_root, detection_ids)
        stats.update(mask_stats)
        for sample in mask_samples:
            if sample.image.resolve() not in seen_images:
                samples.append(sample)
                seen_images.add(sample.image.resolve())

    names = discover_yolo_names(dataset_root)
    yolo_images = [p for p in iter_images(dataset_root) if p.resolve() not in seen_images]
    for image in yolo_images:
        labels, label_stats = load_yolo_labels(dataset, image, dataset_root, detection_ids, names, include_safa_stain)
        stats.update(label_stats)
        if labels:
            samples.append(YoloSample(dataset, image, labels, "yolo"))
            seen_images.add(image.resolve())

    if dataset == "saafai4":
        for sample in include_negative_images(dataset, dataset_root):
            if sample.image.resolve() not in seen_images:
                samples.append(sample)
                seen_images.add(sample.image.resolve())
                stats["negative_images"] += 1

    stats["images_seen"] += len(iter_images(dataset_root))
    stats["samples_kept"] += len(samples)
    stats["labels_kept"] += sum(len(s.labels) for s in samples)
    return samples, stats


def clean_output(out: Path) -> None:
    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        target = out / rel
        if target.exists():
            shutil.rmtree(target)
        ensure_dir(target)


def write_yolo_sample(sample: YoloSample, split: str, out: Path) -> Counter:
    stats = Counter()
    stem = safe_stem(sample.dataset, sample.image)
    image_dst = out / "images" / split / f"{stem}{sample.image.suffix.lower()}"
    label_dst = out / "labels" / split / f"{stem}.txt"
    shutil.copy2(sample.image, image_dst)
    lines = [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in sample.labels]
    label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    stats["images_written"] += 1
    stats["labels_written"] += len(lines)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLOv8 dataset for CleanOps AI.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/yolo")
    parser.add_argument("--schema", default="configs/label_schema.json")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-synspill", action="store_true", help="Opt in to derived SynSpill bbox samples for YOLO experiments.")
    parser.add_argument("--include-safa-stain", action="store_true", help="Opt in to Safa Stain annotations as stain_object. Default only uses Safa Garbage.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    detection_ids, _ = load_schema(Path(args.schema))

    all_samples: list[YoloSample] = []
    included_datasets = [
        "floor_damage",
        "saafai4",
        "dirty_floor_safa",
        "dirty_floor_safa_full",
        "hd10k",
        "taco",
        "mju_waste",
    ]
    excluded_datasets = []
    if args.include_synspill:
        included_datasets.append("synspill")
    else:
        excluded_datasets.append({"dataset": "synspill", "reason": "excluded by default; use --include-synspill to derive bbox from synthetic masks"})

    report: dict[str, Any] = {
        "options": {
            "include_synspill": args.include_synspill,
            "include_safa_stain": args.include_safa_stain,
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
        samples, stats = collect_dataset(dataset, raw_dir / dataset, detection_ids, args.include_safa_stain)
        report["datasets"][dataset] = dict(stats)
        all_samples.extend(samples)
        if dataset == "dirty_floor_safa" and stats.get("labels_kept", 0) == 0:
            report["warnings"].append("dirty_floor_safa produced no YOLO labels. Current raw data may not contain Garbage; use --include-safa-stain only if you want Safa Stain as stain_object.")

    clean_output(out)
    if not all_samples:
        report["warnings"].append("No YOLO samples were exported. Add datasets under data/raw and rerun.")
        write_json(out / "prepare_yolo_report.json", report)
        print("No YOLO samples found. Expected raw datasets in data/raw/floor_damage, data/raw/saafai4, data/raw/dirty_floor_safa.")
        return 0

    train_samples, val_samples = train_val_split(all_samples, args.val_ratio, args.seed)
    write_stats = Counter()
    for sample in train_samples:
        write_stats.update(write_yolo_sample(sample, "train", out))
    for sample in val_samples:
        write_stats.update(write_yolo_sample(sample, "val", out))

    report["splits"] = {"train": len(train_samples), "val": len(val_samples)}
    report["written"] = dict(write_stats)
    class_counts = Counter(cls for s in all_samples for cls, *_ in s.labels)
    class_names = {idx: name for name, idx in detection_ids.items()}
    report["class_counts"] = {
        str(cls): {
            "name": class_names.get(cls, "unknown"),
            "count": count,
        }
        for cls, count in sorted(class_counts.items())
    }
    write_json(out / "prepare_yolo_report.json", report)
    print(f"YOLO export complete: {len(train_samples)} train, {len(val_samples)} val samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
