from __future__ import annotations

import json
import os
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(raw: str, default: str) -> Path:
    text = (raw or "").strip() or default
    path = Path(text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _object_key(prefix: str, suffix: str) -> str:
    left = (prefix or "").strip("/")
    right = (suffix or "").strip("/")
    if not left:
        return right
    if not right:
        return left
    return f"{left}/{right}"


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _upload_blob(
    container_client,
    blob_name: str,
    file_path: Path,
    content_type: str,
) -> None:
    with file_path.open("rb") as stream:
        container_client.upload_blob(
            name=blob_name,
            data=stream,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def main() -> int:
    yolo_path = _resolve_path(
        os.getenv("RETRAIN_CANDIDATE_YOLO_FILE", ""),
        "outputs/retrain/candidate/yolo_best.pt",
    )
    unet_path = _resolve_path(
        os.getenv("RETRAIN_CANDIDATE_UNET_FILE", ""),
        "outputs/retrain/candidate/unet_best.pth",
    )
    metrics_path = _resolve_path(
        os.getenv("RETRAIN_CANDIDATE_METRICS_FILE", ""),
        "outputs/retrain/candidate_metrics.json",
    )

    metrics_payload = {
        "trainer": {
            "mode": "smoke",
            "job_id": os.getenv("TRAINER_JOB_ID", ""),
            "batch_id": os.getenv("TRAINER_BATCH_ID", ""),
        },
        "yolo": {
            "map": 0.999,
        },
        "unet": {
            "miou": 0.999,
        },
    }

    _write_file(
        yolo_path,
        b"cleanops smoke yolo artifact\n",
    )
    _write_file(
        unet_path,
        b"cleanops smoke unet artifact\n",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    connection_string = (
        os.getenv("RETRAIN_STORAGE_CONNECTION_STRING")
        or os.getenv("MODEL_STORAGE_CONNECTION_STRING")
        or ""
    ).strip()
    container_name = (os.getenv("RETRAIN_CONTAINER") or "retrain").strip()
    external_prefix = (os.getenv("RETRAIN_EXTERNAL_PREFIX") or "scoring/external/latest").strip("/")

    if connection_string:
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client(container_name)
        try:
            container_client.create_container()
        except ResourceExistsError:
            pass

        _upload_blob(
            container_client,
            _object_key(external_prefix, "yolo/model.pt"),
            yolo_path,
            "application/octet-stream",
        )
        _upload_blob(
            container_client,
            _object_key(external_prefix, "unet/model.pth"),
            unet_path,
            "application/octet-stream",
        )
        _upload_blob(
            container_client,
            _object_key(external_prefix, "metrics/metrics.json"),
            metrics_path,
            "application/json",
        )

    print(
        json.dumps(
            {
                "status": "completed",
                "mode": "smoke",
                "yolo_path": str(yolo_path),
                "unet_path": str(unet_path),
                "metrics_path": str(metrics_path),
                "uploaded_to_blob": bool(connection_string),
                "container": container_name,
                "external_prefix": external_prefix,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
