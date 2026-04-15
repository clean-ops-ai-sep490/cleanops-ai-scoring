from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote retrain run or external candidate into active scoring model keys"
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("MODEL_STORAGE_CONNECTION_STRING", ""),
        help="Azure Blob connection string",
    )
    parser.add_argument(
        "--source-container",
        default=os.getenv("SCORING_BLOB_RETRAIN_CONTAINER", "retrain"),
        help="Source retrain container",
    )
    parser.add_argument(
        "--target-container",
        default=os.getenv("SCORING_BLOB_MODELS_CONTAINER", "models"),
        help="Target models container",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("SCORING_BLOB_NAMESPACE", "scoring"),
        help="Top-level namespace for scoring artifacts",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID to promote from scoring/runs/{run-id}",
    )
    parser.add_argument(
        "--external-prefix",
        default=os.getenv("SCORING_BLOB_EXTERNAL_PREFIX", "scoring/external/latest"),
        help="External candidate prefix used when --run-id is omitted",
    )
    parser.add_argument(
        "--manifest-key",
        default=os.getenv("SCORING_BLOB_ACTIVE_MANIFEST_KEY", "scoring/manifests/active.json"),
        help="Manifest key to update in target container",
    )
    return parser.parse_args()


def object_key(prefix: str, suffix: str) -> str:
    left = (prefix or "").strip("/")
    right = (suffix or "").strip("/")
    if not left:
        return right
    if not right:
        return left
    return f"{left}/{right}"


def exists(container_client, blob_name: str) -> bool:
    try:
        container_client.get_blob_client(blob_name).get_blob_properties()
        return True
    except ResourceNotFoundError:
        return False


def wait_copy_complete(blob_client, max_attempts: int = 120, sleep_sec: float = 0.5) -> None:
    for _ in range(max_attempts):
        props = blob_client.get_blob_properties()
        status = (props.copy.status if props.copy else "") or ""
        normalized = str(status).lower()
        if normalized == "success":
            return
        if normalized == "pending":
            time.sleep(sleep_sec)
            continue
        raise RuntimeError(f"Copy failed for {blob_client.blob_name}. status={status}")
    raise TimeoutError(f"Timed out waiting copy for {blob_client.blob_name}")


def copy_blob(source_container, source_key: str, target_container, target_key: str) -> None:
    source_blob = source_container.get_blob_client(source_key)
    target_blob = target_container.get_blob_client(target_key)

    try:
        target_blob.delete_blob(delete_snapshots="include")
    except ResourceNotFoundError:
        pass

    try:
        target_blob.start_copy_from_url(source_blob.url)
    except ResourceNotFoundError:
        raise FileNotFoundError(f"Source blob not found: {source_key}")

    wait_copy_complete(target_blob)


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

    blob_service = BlobServiceClient.from_connection_string(args.connection_string)
    source_container = blob_service.get_container_client(args.source_container)
    target_container = blob_service.get_container_client(args.target_container)

    try:
        target_container.create_container()
    except ResourceExistsError:
        pass

    namespace = args.namespace.strip("/")
    active_prefix = object_key(namespace, "active")
    archive_prefix = object_key(namespace, f"archive/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

    if args.run_id:
        source_prefix = object_key(namespace, f"runs/{args.run_id}")
        source_yolo = object_key(source_prefix, "artifacts/yolo/model.pt")
        source_unet = object_key(source_prefix, "artifacts/unet/model.pth")
        source_metrics = object_key(source_prefix, "metrics/metrics.json")
        source_ref = {"mode": "run", "run_id": args.run_id, "prefix": source_prefix}
    else:
        source_prefix = args.external_prefix.strip("/")
        source_yolo = object_key(source_prefix, "yolo/model.pt")
        source_unet = object_key(source_prefix, "unet/model.pth")
        source_metrics = object_key(source_prefix, "metrics/metrics.json")
        source_ref = {"mode": "external", "prefix": source_prefix}

    target_yolo = object_key(active_prefix, "yolo/model.pt")
    target_unet = object_key(active_prefix, "unet/model.pth")
    target_metrics = object_key(active_prefix, "metrics/metrics.json")

    for key in [source_yolo, source_unet, source_metrics]:
        if not exists(source_container, key):
            raise FileNotFoundError(f"Required source blob not found: {key}")

    archive_yolo = object_key(archive_prefix, "yolo/model.pt")
    archive_unet = object_key(archive_prefix, "unet/model.pth")
    archive_metrics = object_key(archive_prefix, "metrics/metrics.json")

    if exists(target_container, target_yolo):
        copy_blob(target_container, target_yolo, target_container, archive_yolo)
    if exists(target_container, target_unet):
        copy_blob(target_container, target_unet, target_container, archive_unet)
    if exists(target_container, target_metrics):
        copy_blob(target_container, target_metrics, target_container, archive_metrics)

    copy_blob(source_container, source_yolo, target_container, target_yolo)
    copy_blob(source_container, source_unet, target_container, target_unet)
    copy_blob(source_container, source_metrics, target_container, target_metrics)

    manifest = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_ref,
        "target": {
            "container": args.target_container,
            "yolo": target_yolo,
            "unet": target_unet,
            "metrics": target_metrics,
        },
        "archive_prefix": archive_prefix,
    }
    upload_text(
        target_container,
        args.manifest_key,
        json.dumps(manifest, indent=2),
        "application/json",
    )

    print("Promoted scoring active model set:")
    print(f"  source_container={args.source_container}")
    print(f"  source_prefix={source_prefix}")
    print(f"  target_container={args.target_container}")
    print(f"  target_yolo={target_yolo}")
    print(f"  target_unet={target_unet}")
    print(f"  target_metrics={target_metrics}")
    print(f"  manifest={args.manifest_key}")


if __name__ == "__main__":
    main()
