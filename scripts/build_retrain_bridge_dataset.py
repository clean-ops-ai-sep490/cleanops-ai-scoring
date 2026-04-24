from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient

if TYPE_CHECKING:
    from azure.storage.blob import BlobClient, BlobContainerClient


@dataclass
class ApprovedAnnotationItem:
    candidate_id: str
    annotation_id: str
    result_id: str
    job_id: str
    request_id: str
    environment_key: str
    approved_at_utc: Optional[str]
    snapshot_key: str
    metadata_key: str
    annotation_key: str


YOLO_CLASS_MAP = {
    "stain_or_water": 0,
    "stain": 0,
    "water": 0,
    "liquid": 0,
    "wet_surface": 1,
    "wet": 1,
}

MASK_CLASS_MAP = {
    "stain_or_water": 1,
    "stain": 1,
    "water": 1,
    "liquid": 1,
    "wet_surface": 2,
    "wet": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build retrain bridge dataset from approved annotation manifests. "
            "Only human-approved annotations are exported into YOLO/U-Net training artifacts."
        )
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("MODEL_STORAGE_CONNECTION_STRING") or os.getenv("SCORING_RETRAIN_STORAGE_CONNECTION_STRING", ""),
        help="Azure Blob connection string (or set MODEL_STORAGE_CONNECTION_STRING)",
    )
    parser.add_argument("--container", default="retrain-samples", help="Blob container containing reviewed snapshots and approved annotations")
    parser.add_argument(
        "--prefix",
        default="scoring/retrain-samples",
        help="Blob prefix root for reviewed snapshots and approved annotation manifests",
    )
    parser.add_argument(
        "--output-root",
        default="data/retrain_bridge",
        help="Local output root for bridge datasets",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="Stop after processing N approved annotations (0 = no limit)")
    parser.add_argument(
        "--only-after-date",
        default="",
        help="Only include annotations with approvedAtUtc >= YYYY-MM-DD",
    )
    return parser.parse_args()


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def choose_split(seed_text: str, train_ratio: float, valid_ratio: float) -> str:
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < (train_ratio + valid_ratio):
        return "valid"
    return "test"


def safe_stem(text: str) -> str:
    trimmed = text.strip()
    if not trimmed:
        return "item"
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in trimmed)


def parse_iso_datetime(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None

    candidate = raw.strip()
    if not candidate:
        return None

    try:
        return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_json_blob(container: BlobContainerClient, key: str) -> Dict[str, Any]:
    blob = container.get_blob_client(key)
    payload = blob.download_blob().readall()
    return json.loads(payload.decode("utf-8"))


def load_blob_bytes(blob: BlobClient) -> bytes:
    return blob.download_blob().readall()


def list_manifest_keys(container: BlobContainerClient, root_prefix: str) -> Iterable[str]:
    manifests_prefix = f"{root_prefix}/manifests/annotations/approved/"
    for blob in container.list_blobs(name_starts_with=manifests_prefix):
        if blob.name.endswith("/manifest.json"):
            yield blob.name


def to_approved_item(raw: Dict[str, Any]) -> Optional[ApprovedAnnotationItem]:
    candidate_id = str(raw.get("candidateId") or "").strip()
    annotation_id = str(raw.get("annotationId") or "").strip()
    result_id = str(raw.get("resultId") or "").strip()
    job_id = str(raw.get("jobId") or "").strip()
    snapshot_key = str(raw.get("snapshotKey") or "").strip()
    metadata_key = str(raw.get("metadataKey") or "").strip()
    annotation_key = str(raw.get("annotationKey") or "").strip()

    if not candidate_id or not annotation_id or not result_id or not job_id or not snapshot_key or not metadata_key or not annotation_key:
        return None

    return ApprovedAnnotationItem(
        candidate_id=candidate_id,
        annotation_id=annotation_id,
        result_id=result_id,
        job_id=job_id,
        request_id=str(raw.get("requestId") or "").strip(),
        environment_key=str(raw.get("environmentKey") or "").strip(),
        approved_at_utc=str(raw.get("approvedAtUtc") or "").strip() or None,
        snapshot_key=snapshot_key,
        metadata_key=metadata_key,
        annotation_key=annotation_key,
    )


def ensure_layout(root: Path) -> None:
    for split in ["train", "valid", "test"]:
        (root / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "unet" / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "unet" / "masks" / split).mkdir(parents=True, exist_ok=True)

    (root / "reports").mkdir(parents=True, exist_ok=True)


def guess_image_extension(snapshot_key: str) -> str:
    ext = Path(snapshot_key).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return ext
    return ".jpg"


def make_base_name(item: ApprovedAnnotationItem, approved_at: Optional[datetime]) -> str:
    stamp = approved_at.strftime("%Y%m%dT%H%M%SZ") if approved_at else "unknown"
    result_part = safe_stem(item.result_id)[:16]
    candidate_part = safe_stem(item.candidate_id)[:12]
    return f"ann_{stamp}_{candidate_part}_{result_part}"


def write_binary(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def decode_image_shape(image_bytes: bytes) -> tuple[int, int]:
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode snapshot image bytes")
    h, w = decoded.shape[:2]
    return h, w


def normalize_label(raw: Any) -> Optional[str]:
    if raw is None:
        return None

    label = str(raw).strip().lower()
    return label if label in YOLO_CLASS_MAP else None


def normalize_shape_type(raw: Any) -> str:
    value = str(raw or "rectangle").strip().lower()
    if value in {"rectangle", "rect", "bbox", "box"}:
        return "rectangle"
    if value in {"polygon", "region", "polygon-region"}:
        return "polygon"
    return "rectangle"


def parse_points(raw_points: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(raw_points, list):
        return points

    for item in raw_points:
        if not isinstance(item, list) or len(item) < 2:
            continue

        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue

        points.append((x, y))

    return points


def clamp_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    px = int(round(max(0.0, min(float(width - 1), x))))
    py = int(round(max(0.0, min(float(height - 1), y))))
    return px, py


def points_to_bbox(points: list[tuple[float, float]], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if len(points) < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = clamp_point(min(xs), min(ys), width, height)
    x2, y2 = clamp_point(max(xs), max(ys), width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    cx = ((x1 + x2) / 2.0) / max(1.0, float(width))
    cy = ((y1 + y2) / 2.0) / max(1.0, float(height))
    bw = (x2 - x1) / max(1.0, float(width))
    bh = (y2 - y1) / max(1.0, float(height))

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def annotation_to_yolo_lines(labels: list[dict[str, Any]], width: int, height: int) -> list[str]:
    lines: list[str] = []
    for label_item in labels:
        normalized_label = normalize_label(label_item.get("label"))
        if normalized_label is None:
            continue

        points = parse_points(label_item.get("points"))
        bbox = points_to_bbox(points, width, height)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        lines.append(to_yolo_line(YOLO_CLASS_MAP[normalized_label], x1, y1, x2, y2, width, height))

    return lines


def draw_shape(mask: np.ndarray, shape_type: str, points: list[tuple[float, float]], class_id: int) -> None:
    height, width = mask.shape[:2]
    if shape_type == "rectangle":
        bbox = points_to_bbox(points, width, height)
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = class_id
        return

    if len(points) < 3:
        return

    polygon = np.array([clamp_point(x, y, width, height) for x, y in points], dtype=np.int32)
    if polygon.size == 0:
        return
    cv2.fillPoly(mask, [polygon], color=class_id)


def annotation_to_mask(labels: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    wet_items: list[tuple[str, list[tuple[float, float]], int]] = []
    stain_items: list[tuple[str, list[tuple[float, float]], int]] = []

    for label_item in labels:
        normalized_label = normalize_label(label_item.get("label"))
        if normalized_label is None:
            continue

        points = parse_points(label_item.get("points"))
        shape_type = normalize_shape_type(label_item.get("shapeType"))
        class_id = MASK_CLASS_MAP[normalized_label]
        target = stain_items if class_id == 1 else wet_items
        target.append((shape_type, points, class_id))

    for shape_type, points, class_id in wet_items:
        draw_shape(mask, shape_type, points, class_id)
    for shape_type, points, class_id in stain_items:
        draw_shape(mask, shape_type, points, class_id)

    return mask


def export_approved_annotation(
    root: Path,
    split: str,
    base_name: str,
    ext: str,
    image_bytes: bytes,
    yolo_lines: list[str],
    mask: np.ndarray,
) -> dict[str, str]:
    yolo_image_path = root / "yolo" / "images" / split / f"{base_name}{ext}"
    yolo_label_path = root / "yolo" / "labels" / split / f"{base_name}.txt"
    unet_image_path = root / "unet" / "images" / split / f"{base_name}{ext}"
    unet_mask_path = root / "unet" / "masks" / split / f"{base_name}.png"

    write_binary(yolo_image_path, image_bytes)
    write_text(yolo_label_path, "\n".join(yolo_lines))
    write_binary(unet_image_path, image_bytes)

    ok = cv2.imwrite(str(unet_mask_path), mask)
    if not ok:
        raise ValueError(f"Failed to write mask file: {unet_mask_path}")

    return {
        "yoloImage": str(yolo_image_path),
        "yoloLabel": str(yolo_label_path),
        "unetImage": str(unet_image_path),
        "unetMask": str(unet_mask_path),
    }


def main() -> None:
    args = parse_args()

    if not args.connection_string:
        raise ValueError("Azure Blob connection string is required (or set MODEL_STORAGE_CONNECTION_STRING)")

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise ValueError("train-ratio must be in (0, 1)")
    if args.valid_ratio < 0 or (args.train_ratio + args.valid_ratio) >= 1:
        raise ValueError("valid-ratio must be >= 0 and train-ratio + valid-ratio must be < 1")

    only_after = None
    if args.only_after_date:
        try:
            only_after = datetime.strptime(args.only_after_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("only-after-date must follow YYYY-MM-DD") from exc

    output_root = Path(args.output_root).resolve()
    ensure_layout(output_root)

    service = BlobServiceClient.from_connection_string(args.connection_string)
    container = service.get_container_client(args.container)
    root_prefix = normalize_prefix(args.prefix)

    processed = 0
    exported = 0
    skipped = 0
    errors = 0
    empty_label_items = 0
    dedupe_keys: set[str] = set()
    index_lines = []

    for manifest_key in list_manifest_keys(container, root_prefix):
        manifest = read_json_blob(container, manifest_key)
        items = manifest.get("items", [])
        if not isinstance(items, list):
            continue

        for raw_item in items:
            if not isinstance(raw_item, dict):
                skipped += 1
                continue

            item = to_approved_item(raw_item)
            if item is None:
                skipped += 1
                continue

            if item.annotation_key in dedupe_keys:
                skipped += 1
                continue
            dedupe_keys.add(item.annotation_key)

            approved_at_dt = parse_iso_datetime(item.approved_at_utc)
            if only_after is not None and approved_at_dt is not None:
                if approved_at_dt.replace(tzinfo=None) < only_after:
                    skipped += 1
                    continue

            if args.limit > 0 and processed >= args.limit:
                break

            processed += 1

            try:
                annotation_json = read_json_blob(container, item.annotation_key)
                labels = annotation_json.get("annotation", {}).get("labels", [])
                if not isinstance(labels, list):
                    raise ValueError("annotation.labels must be a JSON array")

                metadata_json = read_json_blob(container, item.metadata_key)
                snapshot_blob = container.get_blob_client(item.snapshot_key)
                snapshot_bytes = load_blob_bytes(snapshot_blob)
                height, width = decode_image_shape(snapshot_bytes)

                ext = guess_image_extension(item.snapshot_key)
                base_name = make_base_name(item, approved_at_dt)
                split_seed = item.candidate_id or item.annotation_id or item.result_id
                split = choose_split(split_seed, args.train_ratio, args.valid_ratio)

                yolo_lines = annotation_to_yolo_lines(labels, width, height)
                if not yolo_lines:
                    empty_label_items += 1
                mask = annotation_to_mask(labels, width, height)

                exported_paths = export_approved_annotation(
                    output_root,
                    split,
                    base_name,
                    ext,
                    snapshot_bytes,
                    yolo_lines,
                    mask,
                )
                exported += 1

                index_lines.append(
                    {
                        "candidateId": item.candidate_id,
                        "annotationId": item.annotation_id,
                        "resultId": item.result_id,
                        "jobId": item.job_id,
                        "requestId": item.request_id,
                        "environmentKey": item.environment_key,
                        "split": split,
                        "snapshotKey": item.snapshot_key,
                        "metadataKey": item.metadata_key,
                        "annotationKey": item.annotation_key,
                        "approvedAtUtc": item.approved_at_utc,
                        "reviewedVerdict": metadata_json.get("reviewedVerdict"),
                        "exportedLabelCount": len(yolo_lines),
                        **exported_paths,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                index_lines.append(
                    {
                        "candidateId": item.candidate_id,
                        "annotationId": item.annotation_id,
                        "resultId": item.result_id,
                        "annotationKey": item.annotation_key,
                        "snapshotKey": item.snapshot_key,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if args.limit > 0 and processed >= args.limit:
            break

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "container": args.container,
        "prefix": root_prefix,
        "output_root": str(output_root),
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "only_after_date": args.only_after_date or None,
        "processed_items": processed,
        "exported_annotation_items": exported,
        "empty_label_items": empty_label_items,
        "skipped_items": skipped,
        "error_items": errors,
        "notes": [
            "Only approved human annotations are exported to the deep retrain dataset.",
            "Reviewed PASS/FAIL items without approved annotations are not treated as perception ground truth here.",
            "YOLO labels are derived from annotation regions using bounding rectangles.",
            "U-Net masks are rasterized from the same approved annotation regions.",
        ],
    }

    report_path = output_root / "reports" / "bridge_summary.json"
    index_path = output_root / "reports" / "bridge_index.jsonl"

    write_text(report_path, json.dumps(report, ensure_ascii=True, indent=2))
    write_text(index_path, "\n".join(json.dumps(x, ensure_ascii=True) for x in index_lines) + "\n")

    print("[DONE] Retrain bridge dataset export completed")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"[REPORT] {report_path}")
    print(f"[INDEX]  {index_path}")


if __name__ == "__main__":
    main()
