from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local retrain candidate artifacts to Azure Blob")
    parser.add_argument(
        "--connection-string",
        default=os.getenv("MODEL_STORAGE_CONNECTION_STRING", ""),
        help="Azure Blob connection string (or set MODEL_STORAGE_CONNECTION_STRING)",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("SCORING_BLOB_RETRAIN_CONTAINER", os.getenv("SCORING_RETRAIN_CONTAINER", "retrain")),
        help="Blob container name",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("SCORING_BLOB_EXTERNAL_PREFIX", "scoring/external/latest"),
        help="Candidate prefix in container",
    )
    parser.add_argument("--yolo", required=True, help="Path to local YOLO candidate model (.pt)")
    parser.add_argument("--unet", required=True, help="Path to local U-Net candidate model (.pth)")
    parser.add_argument("--metrics", required=True, help="Path to local candidate metrics JSON")
    return parser.parse_args()


def object_key(prefix: str, suffix: str) -> str:
    left = prefix.strip("/")
    right = suffix.strip("/")
    if not left:
        return right
    return f"{left}/{right}"


def validate_metrics(metrics_path: Path) -> None:
    data = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("metrics JSON must be an object")

    yolo = data.get("yolo", {}) if isinstance(data.get("yolo"), dict) else {}
    unet = data.get("unet", {}) if isinstance(data.get("unet"), dict) else {}
    if "map" not in yolo or "miou" not in unet:
        raise ValueError("metrics JSON must include yolo.map and unet.miou")


def upload_file(container_client, blob_name: str, path: Path, content_type: str) -> None:
    with path.open("rb") as handle:
        container_client.upload_blob(
            name=blob_name,
            data=handle,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def main() -> None:
    args = parse_args()
    if not args.connection_string:
        raise ValueError("Azure Blob connection string is required.")

    yolo_path = Path(args.yolo).expanduser().resolve()
    unet_path = Path(args.unet).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve()

    for path in [yolo_path, unet_path, metrics_path]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    validate_metrics(metrics_path)

    blob_service = BlobServiceClient.from_connection_string(args.connection_string)
    container_client = blob_service.get_container_client(args.container)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    yolo_key = object_key(args.prefix, "yolo/model.pt")
    unet_key = object_key(args.prefix, "unet/model.pth")
    metrics_key = object_key(args.prefix, "metrics/metrics.json")

    upload_file(container_client, yolo_key, yolo_path, "application/octet-stream")
    upload_file(container_client, unet_key, unet_path, "application/octet-stream")
    upload_file(container_client, metrics_key, metrics_path, "application/json")

    print("Uploaded candidate artifacts:")
    print(f"  container={args.container}")
    print(f"  prefix={args.prefix}")
    print(f"  yolo={yolo_key}")
    print(f"  unet={unet_key}")
    print(f"  metrics={metrics_key}")


if __name__ == "__main__":
    main()