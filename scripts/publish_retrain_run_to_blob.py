from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish local retrain artifacts to Blob run namespace")
    parser.add_argument(
        "--connection-string",
        default=os.getenv("MODEL_STORAGE_CONNECTION_STRING", ""),
        help="Azure Blob connection string",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("SCORING_BLOB_RETRAIN_CONTAINER", "retrain"),
        help="Retrain container name",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("SCORING_BLOB_NAMESPACE", "scoring"),
        help="Top-level namespace for scoring blobs",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"),
        help="Run identifier under namespace/runs/{run-id}",
    )
    parser.add_argument("--yolo", required=True, help="Path to local YOLO model (.pt)")
    parser.add_argument("--unet", required=True, help="Path to local U-Net model (.pth)")
    parser.add_argument("--metrics", required=True, help="Path to local metrics JSON")
    parser.add_argument("--log", help="Optional path to train log")
    return parser.parse_args()


def object_key(prefix: str, suffix: str) -> str:
    left = (prefix or "").strip("/")
    right = (suffix or "").strip("/")
    if not left:
        return right
    if not right:
        return left
    return f"{left}/{right}"


def validate_metrics(metrics_path: Path) -> None:
    data = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("metrics JSON must be an object")

    yolo = data.get("yolo", {}) if isinstance(data.get("yolo"), dict) else {}
    unet = data.get("unet", {}) if isinstance(data.get("unet"), dict) else {}
    if "map" not in yolo or "miou" not in unet:
        raise ValueError("metrics JSON must include yolo.map and unet.miou")


def upload_file(container_client, blob_name: str, file_path: Path, content_type: str) -> None:
    with file_path.open("rb") as stream:
        container_client.upload_blob(
            name=blob_name,
            data=stream,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def upload_text(container_client, blob_name: str, text: str, content_type: str) -> None:
    container_client.upload_blob(
        name=blob_name,
        data=text.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )


def main() -> None:
    args = parse_args()
    if not args.connection_string:
        raise ValueError("Azure Blob connection string is required")

    yolo_path = Path(args.yolo).expanduser().resolve()
    unet_path = Path(args.unet).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve() if args.log else None

    for path in [yolo_path, unet_path, metrics_path]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    if log_path and (not log_path.exists() or not log_path.is_file()):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    validate_metrics(metrics_path)

    blob_service = BlobServiceClient.from_connection_string(args.connection_string)
    container_client = blob_service.get_container_client(args.container)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    run_prefix = object_key(args.namespace, f"runs/{args.run_id}")
    yolo_key = object_key(run_prefix, "artifacts/yolo/model.pt")
    unet_key = object_key(run_prefix, "artifacts/unet/model.pth")
    metrics_key = object_key(run_prefix, "metrics/metrics.json")
    log_key = object_key(run_prefix, "logs/train.log")
    manifest_key = object_key(run_prefix, "manifests/run.json")

    upload_file(container_client, yolo_key, yolo_path, "application/octet-stream")
    upload_file(container_client, unet_key, unet_path, "application/octet-stream")
    upload_file(container_client, metrics_key, metrics_path, "application/json")
    if log_path:
        upload_file(container_client, log_key, log_path, "text/plain")

    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "container": args.container,
        "namespace": args.namespace,
        "artifacts": {
            "yolo": yolo_key,
            "unet": unet_key,
            "metrics": metrics_key,
            "log": log_key if log_path else None,
        },
    }
    upload_text(container_client, manifest_key, json.dumps(manifest, indent=2), "application/json")

    print("Published retrain run artifacts:")
    print(f"  container={args.container}")
    print(f"  run_id={args.run_id}")
    print(f"  yolo={yolo_key}")
    print(f"  unet={unet_key}")
    print(f"  metrics={metrics_key}")
    if log_path:
        print(f"  log={log_key}")
    print(f"  manifest={manifest_key}")


if __name__ == "__main__":
    main()
