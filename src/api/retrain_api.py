from __future__ import annotations

import json
import logging
import os
import requests
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _object_key(prefix: str, suffix: str) -> str:
    left = (prefix or "").strip("/")
    right = (suffix or "").strip("/")
    if not left:
        return right
    if not right:
        return left
    return f"{left}/{right}"


def _resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _read_metric(node: Dict[str, Any], key: str) -> Optional[float]:
    current: Any = node
    for segment in key.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]

    if isinstance(current, (int, float)):
        return float(current)

    return None


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Metrics JSON must be an object")

    yolo_map = _read_metric(payload, "yolo.map")
    unet_miou = _read_metric(payload, "unet.miou")
    if yolo_map is None or unet_miou is None:
        raise ValueError("Metrics JSON must include yolo.map and unet.miou")

    return payload


def _upload_blob_file(container_client, blob_name: str, file_path: Path, content_type: str) -> None:
    with file_path.open("rb") as stream:
        container_client.upload_blob(
            name=blob_name,
            data=stream,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def _create_blob_container(connection_string: str, container_name: str):
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass
    return container_client


def _download_existing_metrics(
    connection_string: str,
    container_name: str,
    candidate_prefix: str,
) -> Optional[Dict[str, Any]]:
    if not connection_string.strip():
        return None

    container_client = _create_blob_container(connection_string, container_name)
    metrics_key = _object_key(candidate_prefix, "metrics/metrics.json")
    blob_client = container_client.get_blob_client(metrics_key)

    try:
        payload = blob_client.download_blob().readall().decode("utf-8")
    except ResourceNotFoundError:
        return None

    try:
        metrics = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid existing candidate metrics JSON: {exc}") from exc

    if not isinstance(metrics, dict):
        raise ValueError("Existing candidate metrics JSON must be an object")

    if _read_metric(metrics, "yolo.map") is None or _read_metric(metrics, "unet.miou") is None:
        raise ValueError("Existing candidate metrics must include yolo.map and unet.miou")

    return metrics


def _truncate(text: str, max_length: int = 2000) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length]


RETRAIN_API_ENABLED = _as_bool("RETRAIN_API_ENABLED", True)
RETRAIN_API_KEY = os.getenv("RETRAIN_API_KEY", "").strip()
RETRAIN_USE_REMOTE_TRAINER = _as_bool("RETRAIN_USE_REMOTE_TRAINER", False)
RETRAIN_TRAINER_BASE_URL = os.getenv("RETRAIN_TRAINER_BASE_URL", "http://cleanops-ai-scoring-trainer:8001").strip()
RETRAIN_TRAINER_SUBMIT_PATH = os.getenv("RETRAIN_TRAINER_SUBMIT_PATH", "/trainer/jobs").strip()
RETRAIN_TRAINER_API_KEY = os.getenv("RETRAIN_TRAINER_API_KEY", "").strip()
RETRAIN_TRAINER_TIMEOUT_SEC = max(30, _as_int("RETRAIN_TRAINER_TIMEOUT_SEC", 7200))
RETRAIN_COMMAND = os.getenv("RETRAIN_COMMAND", "").strip()
RETRAIN_COMMAND_TIMEOUT_SEC = max(30, _as_int("RETRAIN_COMMAND_TIMEOUT_SEC", 7200))
RETRAIN_WORKDIR = _resolve_path(os.getenv("RETRAIN_WORKDIR", str(PROJECT_ROOT)))
RETRAIN_STORAGE_CONNECTION_STRING = (
    os.getenv("RETRAIN_STORAGE_CONNECTION_STRING")
    or os.getenv("MODEL_STORAGE_CONNECTION_STRING")
    or ""
).strip()
RETRAIN_CONTAINER = (
    os.getenv("RETRAIN_CONTAINER")
    or os.getenv("SCORING_BLOB_RETRAIN_CONTAINER")
    or "retrain"
).strip()
RETRAIN_EXTERNAL_PREFIX = (
    os.getenv("RETRAIN_EXTERNAL_PREFIX")
    or os.getenv("SCORING_BLOB_EXTERNAL_PREFIX")
    or "scoring/external/latest"
).strip("/")
RETRAIN_CANDIDATE_YOLO_FILE = _resolve_path(
    os.getenv("RETRAIN_CANDIDATE_YOLO_FILE", "outputs/retrain/candidate/yolo_best.pt")
)
RETRAIN_CANDIDATE_UNET_FILE = _resolve_path(
    os.getenv("RETRAIN_CANDIDATE_UNET_FILE", "outputs/retrain/candidate/unet_best.pth")
)
RETRAIN_CANDIDATE_METRICS_FILE = _resolve_path(
    os.getenv("RETRAIN_CANDIDATE_METRICS_FILE", "outputs/retrain/candidate_metrics.json")
)
RETRAIN_ALLOW_EXISTING_BLOB_CANDIDATE = _as_bool("RETRAIN_ALLOW_EXISTING_BLOB_CANDIDATE", True)


class RetrainSample(BaseModel):
    resultId: str
    jobId: str
    requestId: str
    environmentKey: str
    sourceType: str
    source: str
    reviewedVerdict: str
    reviewedAtUtc: str
    reviewedByEmail: Optional[str] = None


class RetrainJobCreateRequest(BaseModel):
    batchId: str = Field(..., min_length=1)
    sourceWindowFromUtc: Optional[str] = None
    reviewedSampleCount: int = 0
    samples: List[RetrainSample] = Field(default_factory=list)


retrain_router = APIRouter(tags=["production"])
_job_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _authorize_or_raise(x_retrain_api_key: Optional[str]) -> None:
    if not RETRAIN_API_ENABLED:
        raise HTTPException(status_code=404, detail="Retrain API is disabled")

    if RETRAIN_API_KEY and x_retrain_api_key != RETRAIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid retrain API key")


def _set_job(job_id: str, **updates: Any) -> None:
    with _job_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(updates)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        return dict(job)


def _as_model_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _build_remote_url(base_url: str, path: str) -> str:
    normalized_base = (base_url or "").strip().rstrip("/") + "/"
    normalized_path = (path or "").strip().lstrip("/")
    return urljoin(normalized_base, normalized_path)


def _invoke_remote_trainer(job_id: str, payload: RetrainJobCreateRequest) -> None:
    if not RETRAIN_TRAINER_BASE_URL:
        raise RuntimeError("RETRAIN_TRAINER_BASE_URL is empty while RETRAIN_USE_REMOTE_TRAINER=true")

    request_payload = {
        "jobId": job_id,
        "batchId": payload.batchId,
        "sourceWindowFromUtc": payload.sourceWindowFromUtc,
        "reviewedSampleCount": payload.reviewedSampleCount,
        "samples": [_as_model_dict(item) for item in payload.samples],
    }
    submit_url = _build_remote_url(RETRAIN_TRAINER_BASE_URL, RETRAIN_TRAINER_SUBMIT_PATH)

    headers: Dict[str, str] = {}
    if RETRAIN_TRAINER_API_KEY:
        headers["X-Trainer-Api-Key"] = RETRAIN_TRAINER_API_KEY

    try:
        response = requests.post(
            submit_url,
            json=request_payload,
            headers=headers,
            timeout=RETRAIN_TRAINER_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Remote trainer call failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Remote trainer returned {response.status_code}: {_truncate(response.text.strip())}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Remote trainer response is not valid JSON") from exc

    status = str(data.get("status", "")).strip().lower()
    exit_code = int(data.get("exitCode", 0)) if str(data.get("exitCode", "")).strip() else 0
    if status not in {"completed", "succeeded", "success"} or exit_code != 0:
        message = str(data.get("message") or "Remote trainer reported failed status")
        raise RuntimeError(message)


def _run_retrain_job(job_id: str, payload: RetrainJobCreateRequest) -> None:
    _set_job(job_id, status="running", startedAtUtc=_utc_now_iso())

    try:
        if RETRAIN_USE_REMOTE_TRAINER:
            _invoke_remote_trainer(job_id, payload)
        elif RETRAIN_COMMAND:
            logger.info("Running retrain command for job %s: %s", job_id, RETRAIN_COMMAND)
            proc = subprocess.run(
                RETRAIN_COMMAND,
                shell=True,
                cwd=str(RETRAIN_WORKDIR),
                capture_output=True,
                text=True,
                timeout=RETRAIN_COMMAND_TIMEOUT_SEC,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    "Retrain command failed "
                    f"(exit={proc.returncode}). "
                    f"stderr={_truncate(proc.stderr.strip())}"
                )
        else:
            raise RuntimeError(
                "No retrain execution path configured. "
                "Set RETRAIN_USE_REMOTE_TRAINER=true or provide RETRAIN_COMMAND."
            )

        local_candidate_exists = (
            RETRAIN_CANDIDATE_YOLO_FILE.is_file()
            and RETRAIN_CANDIDATE_UNET_FILE.is_file()
            and RETRAIN_CANDIDATE_METRICS_FILE.is_file()
        )

        metrics: Optional[Dict[str, Any]] = None
        candidate_yolo_key = _object_key(RETRAIN_EXTERNAL_PREFIX, "yolo/model.pt")
        candidate_unet_key = _object_key(RETRAIN_EXTERNAL_PREFIX, "unet/model.pth")
        candidate_metrics_key = _object_key(RETRAIN_EXTERNAL_PREFIX, "metrics/metrics.json")

        if local_candidate_exists:
            metrics = _load_metrics_json(RETRAIN_CANDIDATE_METRICS_FILE)
            if not RETRAIN_STORAGE_CONNECTION_STRING:
                raise RuntimeError(
                    "RETRAIN_STORAGE_CONNECTION_STRING (or MODEL_STORAGE_CONNECTION_STRING) is required "
                    "to publish retrain candidate artifacts."
                )

            container_client = _create_blob_container(RETRAIN_STORAGE_CONNECTION_STRING, RETRAIN_CONTAINER)
            _upload_blob_file(container_client, candidate_yolo_key, RETRAIN_CANDIDATE_YOLO_FILE, "application/octet-stream")
            _upload_blob_file(container_client, candidate_unet_key, RETRAIN_CANDIDATE_UNET_FILE, "application/octet-stream")
            _upload_blob_file(container_client, candidate_metrics_key, RETRAIN_CANDIDATE_METRICS_FILE, "application/json")
        elif RETRAIN_ALLOW_EXISTING_BLOB_CANDIDATE:
            metrics = _download_existing_metrics(
                RETRAIN_STORAGE_CONNECTION_STRING,
                RETRAIN_CONTAINER,
                RETRAIN_EXTERNAL_PREFIX,
            )
            if metrics is None:
                raise RuntimeError(
                    "No local retrain candidate files found and existing blob candidate is missing."
                )
        else:
            raise RuntimeError(
                "No local retrain candidate files found. "
                "Expected paths: "
                f"{RETRAIN_CANDIDATE_YOLO_FILE}, "
                f"{RETRAIN_CANDIDATE_UNET_FILE}, "
                f"{RETRAIN_CANDIDATE_METRICS_FILE}"
            )

        yolo_map = _read_metric(metrics, "yolo.map") if metrics else None
        unet_miou = _read_metric(metrics, "unet.miou") if metrics else None

        _set_job(
            job_id,
            status="completed",
            completedAtUtc=_utc_now_iso(),
            message="Retrain job completed and candidate artifacts are available.",
            candidatePrefix=RETRAIN_EXTERNAL_PREFIX,
            candidateYoloKey=candidate_yolo_key,
            candidateUnetKey=candidate_unet_key,
            candidateMetricsKey=candidate_metrics_key,
            yoloMap=yolo_map,
            unetMiou=unet_miou,
        )
    except subprocess.TimeoutExpired:
        _set_job(
            job_id,
            status="failed",
            completedAtUtc=_utc_now_iso(),
            message=f"Retrain command timed out after {RETRAIN_COMMAND_TIMEOUT_SEC} seconds.",
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Retrain job %s failed", job_id)
        _set_job(
            job_id,
            status="failed",
            completedAtUtc=_utc_now_iso(),
            message=str(exc),
        )


@retrain_router.post("/retrain/jobs")
def create_retrain_job(
    payload: RetrainJobCreateRequest,
    x_retrain_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _authorize_or_raise(x_retrain_api_key)

    job_id = uuid.uuid4().hex
    job = {
        "jobId": job_id,
        "status": "queued",
        "submittedAtUtc": _utc_now_iso(),
        "batchId": payload.batchId,
        "reviewedSampleCount": payload.reviewedSampleCount,
        "message": "Retrain job queued.",
    }

    with _job_lock:
        _jobs[job_id] = job

    worker = threading.Thread(target=_run_retrain_job, args=(job_id, payload), daemon=True)
    worker.start()

    return {
        "jobId": job_id,
        "status": "queued",
        "submittedAtUtc": job["submittedAtUtc"],
        "message": "Retrain job queued.",
    }


@retrain_router.get("/retrain/jobs/{job_id}")
def get_retrain_job_status(
    job_id: str,
    x_retrain_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _authorize_or_raise(x_retrain_api_key)

    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Retrain job not found")

    return job
