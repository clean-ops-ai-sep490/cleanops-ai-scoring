from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
import torch
from azure.storage.blob import BlobClient, BlobContainerClient, BlobServiceClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.unet_segmenter import UNetSegmenter

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass
class ReviewedItem:
    request_id: str
    result_id: str
    job_id: str
    reviewed_verdict: str
    reviewed_at: Optional[str]
    snapshot_key: str
    metadata_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build retrain bridge dataset from reviewed snapshot manifests. "
            "PASS items become YOLO negative samples and U-Net background masks; FAIL items can be pseudo-labeled."
        )
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("MODEL_STORAGE_CONNECTION_STRING") or os.getenv("SCORING_RETRAIN_STORAGE_CONNECTION_STRING", ""),
        help="Azure Blob connection string (or set MODEL_STORAGE_CONNECTION_STRING)",
    )
    parser.add_argument("--container", default="retrain-samples", help="Blob container containing reviewed snapshots")
    parser.add_argument(
        "--prefix",
        default="scoring/retrain-samples",
        help="Blob prefix root for reviewed snapshots and manifests",
    )
    parser.add_argument(
        "--output-root",
        default="data/retrain_bridge",
        help="Local output root for bridge datasets",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="Stop after processing N reviewed items (0 = no limit)")
    parser.add_argument(
        "--include-fail-unlabeled",
        action="store_true",
        help="Export FAIL images into unlabeled buckets when pseudo-labeling is disabled or rejected",
    )
    parser.add_argument(
        "--only-after-date",
        default="",
        help="Only include items with reviewedAt >= YYYY-MM-DD",
    )

    parser.add_argument(
        "--fail-mode",
        choices=["pseudo", "unlabeled", "skip"],
        default="pseudo",
        help="How FAIL samples are handled",
    )
    parser.add_argument("--yolo-weights", default=os.getenv("MODEL_PATH") or os.getenv("YOLO_WEIGHTS_PATH", "yolov8s.pt"))
    parser.add_argument("--yolo-conf-threshold", type=float, default=0.35)
    parser.add_argument("--unet-weights", default=os.getenv("UNET_MODEL_PATH", "models/unet_multiclass_best.pth"))
    parser.add_argument("--unet-img-size", type=int, default=int(os.getenv("UNET_IMG_SIZE", "384")))
    parser.add_argument("--unet-prob-threshold", type=float, default=0.60)
    parser.add_argument("--disable-yolo-pseudo", action="store_true")
    parser.add_argument("--disable-unet-pseudo", action="store_true")
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
    return re.sub(r"[^a-zA-Z0-9_-]", "_", trimmed)


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


def list_manifest_keys(container: BlobContainerClient, root_prefix: str) -> Iterable[str]:
    manifests_prefix = f"{root_prefix}/manifests/reviewed/"
    for blob in container.list_blobs(name_starts_with=manifests_prefix):
        if blob.name.endswith("/manifest.json"):
            yield blob.name


def to_reviewed_item(raw: Dict[str, Any]) -> Optional[ReviewedItem]:
    metadata_key = str(raw.get("metadataKey") or "").strip()
    snapshot_key = str(raw.get("snapshotKey") or "").strip()
    result_id = str(raw.get("resultId") or "").strip()
    job_id = str(raw.get("jobId") or "").strip()

    if not metadata_key or not snapshot_key or not result_id or not job_id:
        return None

    return ReviewedItem(
        request_id=str(raw.get("requestId") or "").strip(),
        result_id=result_id,
        job_id=job_id,
        reviewed_verdict=str(raw.get("reviewedVerdict") or "").strip().upper(),
        reviewed_at=str(raw.get("reviewedAt") or "").strip() or None,
        snapshot_key=snapshot_key,
        metadata_key=metadata_key,
    )


def ensure_layout(root: Path) -> None:
    for split in ["train", "valid", "test"]:
        (root / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "unet" / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "unet" / "masks" / split).mkdir(parents=True, exist_ok=True)
        (root / "unlabeled" / "fail" / split).mkdir(parents=True, exist_ok=True)

    (root / "reports").mkdir(parents=True, exist_ok=True)


def load_blob_bytes(blob: BlobClient) -> bytes:
    return blob.download_blob().readall()


def guess_image_extension(snapshot_key: str) -> str:
    ext = Path(snapshot_key).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return ext
    return ".jpg"


def make_base_name(item: ReviewedItem, reviewed_at: Optional[datetime]) -> str:
    stamp = reviewed_at.strftime("%Y%m%dT%H%M%SZ") if reviewed_at else "unknown"
    result_part = safe_stem(item.result_id)[:16]
    job_part = safe_stem(item.job_id)[:12]
    return f"rv_{stamp}_{job_part}_{result_part}"


def write_binary(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed in this environment")
    return YOLO(weights_path)


def load_unet_model(weights_path: str, device: torch.device) -> UNetSegmenter:
    ckpt_path = Path(weights_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"U-Net checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = "resnet50"
    if isinstance(ckpt, dict):
        encoder = str(ckpt.get("encoder", "resnet50"))

    model = UNetSegmenter(encoder_name=encoder, classes=3).to(device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def pil_from_bytes(image_bytes: bytes) -> Image.Image:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image bytes")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


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


def pseudo_yolo_label(model, image: Image.Image, conf_threshold: float) -> str:
    results = model.predict(source=image, conf=conf_threshold, save=False, verbose=False)
    lines: list[str] = []

    width, height = image.size
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0].item())
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            lines.append(to_yolo_line(cls_id, x1, y1, x2, y2, width, height))

    return "\n".join(lines)


def pseudo_unet_mask(
    model: UNetSegmenter,
    image: Image.Image,
    img_size: int,
    prob_threshold: float,
    device: torch.device,
) -> Optional[np.ndarray]:
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        max_probs, pred = torch.max(probs, dim=1)

    mean_prob = float(max_probs.mean().item())
    if mean_prob < prob_threshold:
        return None

    pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    pred_original = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred_original


def export_pass_item(root: Path, split: str, base_name: str, ext: str, image_bytes: bytes) -> tuple[Path, Path, Path, Path]:
    yolo_image_path = root / "yolo" / "images" / split / f"{base_name}{ext}"
    yolo_label_path = root / "yolo" / "labels" / split / f"{base_name}.txt"

    unet_image_path = root / "unet" / "images" / split / f"{base_name}{ext}"
    unet_mask_path = root / "unet" / "masks" / split / f"{base_name}.png"

    write_binary(yolo_image_path, image_bytes)
    write_text(yolo_label_path, "")

    write_binary(unet_image_path, image_bytes)

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode snapshot image bytes")

    h, w = decoded.shape[:2]
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    ok = cv2.imwrite(str(unet_mask_path), bg_mask)
    if not ok:
        raise ValueError(f"Failed to write mask file: {unet_mask_path}")

    return yolo_image_path, yolo_label_path, unet_image_path, unet_mask_path


def export_fail_item(root: Path, split: str, base_name: str, ext: str, image_bytes: bytes, metadata: Dict[str, Any]) -> tuple[Path, Path]:
    fail_image_path = root / "unlabeled" / "fail" / split / f"{base_name}{ext}"
    fail_meta_path = root / "unlabeled" / "fail" / split / f"{base_name}.metadata.json"

    write_binary(fail_image_path, image_bytes)
    write_text(fail_meta_path, json.dumps(metadata, ensure_ascii=True, indent=2))

    return fail_image_path, fail_meta_path


def export_fail_pseudo_item(
    root: Path,
    split: str,
    base_name: str,
    ext: str,
    image_bytes: bytes,
    yolo_label_text: str,
    unet_mask: Optional[np.ndarray],
) -> Dict[str, str]:
    yolo_image_path = root / "yolo" / "images" / split / f"{base_name}{ext}"
    yolo_label_path = root / "yolo" / "labels" / split / f"{base_name}.txt"

    write_binary(yolo_image_path, image_bytes)
    write_text(yolo_label_path, yolo_label_text)

    out = {
        "yoloImage": str(yolo_image_path),
        "yoloLabel": str(yolo_label_path),
    }

    if unet_mask is not None:
        unet_image_path = root / "unet" / "images" / split / f"{base_name}{ext}"
        unet_mask_path = root / "unet" / "masks" / split / f"{base_name}.png"

        write_binary(unet_image_path, image_bytes)
        ok = cv2.imwrite(str(unet_mask_path), unet_mask)
        if not ok:
            raise ValueError(f"Failed to write pseudo U-Net mask: {unet_mask_path}")

        out["unetImage"] = str(unet_image_path)
        out["unetMask"] = str(unet_mask_path)

    return out


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
    skipped = 0
    exported_pass = 0
    exported_fail = 0
    exported_fail_pseudo = 0
    skipped_hard_fail = 0
    errors = 0
    dedupe_keys: set[str] = set()

    yolo_model = None
    unet_model = None
    unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pseudo_enabled = args.fail_mode == "pseudo"
    yolo_pseudo_enabled = pseudo_enabled and not args.disable_yolo_pseudo
    unet_pseudo_enabled = pseudo_enabled and not args.disable_unet_pseudo

    if yolo_pseudo_enabled:
        yolo_model = load_yolo_model(args.yolo_weights)

    if unet_pseudo_enabled:
        try:
            unet_model = load_unet_model(args.unet_weights, unet_device)
        except FileNotFoundError:
            # Keep FAIL pseudo-labeling active for YOLO even if U-Net checkpoint is missing.
            unet_model = None
            unet_pseudo_enabled = False

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

            item = to_reviewed_item(raw_item)
            if item is None:
                skipped += 1
                continue

            if item.metadata_key in dedupe_keys:
                skipped += 1
                continue
            dedupe_keys.add(item.metadata_key)

            reviewed_at_dt = parse_iso_datetime(item.reviewed_at)
            if only_after is not None and reviewed_at_dt is not None:
                if reviewed_at_dt.replace(tzinfo=None) < only_after:
                    skipped += 1
                    continue

            if args.limit > 0 and processed >= args.limit:
                break

            processed += 1

            try:
                metadata_json = read_json_blob(container, item.metadata_key)
                verdict = str(
                    metadata_json.get("reviewedVerdict")
                    or item.reviewed_verdict
                    or ""
                ).strip().upper()

                split_seed = item.result_id or item.metadata_key
                split = choose_split(split_seed, args.train_ratio, args.valid_ratio)

                snapshot_blob = container.get_blob_client(item.snapshot_key)
                snapshot_bytes = load_blob_bytes(snapshot_blob)

                ext = guess_image_extension(item.snapshot_key)
                base_name = make_base_name(item, reviewed_at_dt)

                if verdict == "PASS":
                    yolo_image_path, yolo_label_path, unet_image_path, unet_mask_path = export_pass_item(
                        output_root,
                        split,
                        base_name,
                        ext,
                        snapshot_bytes,
                    )
                    exported_pass += 1
                    index_lines.append(
                        {
                            "resultId": item.result_id,
                            "jobId": item.job_id,
                            "requestId": item.request_id,
                            "verdict": verdict,
                            "split": split,
                            "snapshotKey": item.snapshot_key,
                            "metadataKey": item.metadata_key,
                            "yoloImage": str(yolo_image_path),
                            "yoloLabel": str(yolo_label_path),
                            "unetImage": str(unet_image_path),
                            "unetMask": str(unet_mask_path),
                        }
                    )
                elif verdict == "FAIL":
                    if args.fail_mode == "unlabeled":
                        fail_image_path, fail_meta_path = export_fail_item(
                            output_root,
                            split,
                            base_name,
                            ext,
                            snapshot_bytes,
                            metadata_json,
                        )
                        exported_fail += 1
                        index_lines.append(
                            {
                                "resultId": item.result_id,
                                "jobId": item.job_id,
                                "requestId": item.request_id,
                                "verdict": verdict,
                                "split": split,
                                "snapshotKey": item.snapshot_key,
                                "metadataKey": item.metadata_key,
                                "unlabeledFailImage": str(fail_image_path),
                                "unlabeledFailMetadata": str(fail_meta_path),
                            }
                        )
                    elif args.fail_mode == "pseudo":
                        image = pil_from_bytes(snapshot_bytes)

                        yolo_label_text = ""
                        if yolo_model is not None:
                            yolo_label_text = pseudo_yolo_label(
                                yolo_model,
                                image,
                                args.yolo_conf_threshold,
                            )

                        unet_mask = None
                        if unet_model is not None and unet_pseudo_enabled:
                            unet_mask = pseudo_unet_mask(
                                unet_model,
                                image,
                                args.unet_img_size,
                                args.unet_prob_threshold,
                                unet_device,
                            )

                        if yolo_model is None and unet_mask is None:
                            if args.include_fail_unlabeled:
                                fail_image_path, fail_meta_path = export_fail_item(
                                    output_root,
                                    split,
                                    base_name,
                                    ext,
                                    snapshot_bytes,
                                    metadata_json,
                                )
                                exported_fail += 1
                                index_lines.append(
                                    {
                                        "resultId": item.result_id,
                                        "jobId": item.job_id,
                                        "requestId": item.request_id,
                                        "verdict": verdict,
                                        "split": split,
                                        "snapshotKey": item.snapshot_key,
                                        "metadataKey": item.metadata_key,
                                        "unlabeledFailImage": str(fail_image_path),
                                        "unlabeledFailMetadata": str(fail_meta_path),
                                        "reason": "pseudo-label-not-available",
                                    }
                                )
                            else:
                                skipped_hard_fail += 1
                        else:
                            exported = export_fail_pseudo_item(
                                output_root,
                                split,
                                base_name,
                                ext,
                                snapshot_bytes,
                                yolo_label_text,
                                unet_mask,
                            )
                            exported_fail_pseudo += 1
                            index_lines.append(
                                {
                                    "resultId": item.result_id,
                                    "jobId": item.job_id,
                                    "requestId": item.request_id,
                                    "verdict": verdict,
                                    "split": split,
                                    "snapshotKey": item.snapshot_key,
                                    "metadataKey": item.metadata_key,
                                    "mode": "pseudo",
                                    **exported,
                                }
                            )
                    else:
                        skipped_hard_fail += 1
                else:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                index_lines.append(
                    {
                        "resultId": item.result_id,
                        "metadataKey": item.metadata_key,
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
        "exported_pass_items": exported_pass,
        "exported_fail_unlabeled_items": exported_fail,
        "exported_fail_pseudo_items": exported_fail_pseudo,
        "skipped_hard_fail_items": skipped_hard_fail,
        "skipped_items": skipped,
        "error_items": errors,
        "fail_mode": args.fail_mode,
        "notes": [
            "PASS items are exported as YOLO negative samples (empty label files).",
            "PASS items are exported as U-Net background masks (all pixels = class 0).",
            "FAIL items can be pseudo-labeled by current YOLO/U-Net models when fail-mode=pseudo.",
            "Hard FAIL samples can be skipped or exported unlabeled based on settings.",
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
