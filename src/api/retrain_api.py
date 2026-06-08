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


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
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

    if isinstance(current, bool):
        return None

    if isinstance(current, (int, float)):
        return float(current)

    return None


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Metrics JSON must be an object")

    unet_miou = _read_metric(payload, "unet.miou")
    if unet_miou is None:
        raise ValueError("Metrics JSON must include unet.miou")

    return payload


def _download_metrics_blob(container_client, blob_name: str) -> Optional[Dict[str, Any]]:
    blob_client = container_client.get_blob_client(blob_name)

    try:
        payload = blob_client.download_blob().readall().decode("utf-8")
    except ResourceNotFoundError:
        return None

    try:
        metrics = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid candidate metrics blob JSON: {exc}") from exc

    if not isinstance(metrics, dict):
        raise ValueError("Candidate metrics blob JSON must be an object")

    return metrics


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

    if _read_metric(metrics, "unet.miou") is None:
        raise ValueError("Existing candidate metrics must include unet.miou")

    return metrics


def _metric_matches(left: Optional[float], right: Optional[float], tolerance: float = 1e-9) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= tolerance


def _read_benchmark_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if metrics is None:
        return {
            "candidate": None,
            "baseline": None,
            "required": None,
            "passed": None,
        }

    return {
        "candidate": _read_metric(metrics, "benchmark.candidate.mean_iou"),
        "baseline": _read_metric(metrics, "benchmark.baseline.mean_iou"),
        "required": _read_metric(metrics, "benchmark.gate.required_mean_iou"),
        "passed": _read_benchmark_gate_passed(metrics),
    }


def _read_benchmark_gate_passed(metrics: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not metrics:
        return None

    current: Any = metrics
    for segment in "benchmark.gate.passed".split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]

    return current if isinstance(current, bool) else None


def _verify_candidate_metrics_blob(
    local_metrics: Dict[str, Any],
    blob_metrics: Optional[Dict[str, Any]],
    metrics_key: str,
) -> Dict[str, Any]:
    local_benchmark = _read_benchmark_metrics(local_metrics)
    local_has_benchmark = local_benchmark["candidate"] is not None

    if not local_has_benchmark:
        return blob_metrics or {}

    if blob_metrics is None:
        raise RuntimeError(
            f"Candidate metrics blob verification failed: metrics blob '{metrics_key}' was not found."
        )

    blob_benchmark = _read_benchmark_metrics(blob_metrics)
    missing = [
        key
        for key in ("candidate", "baseline", "required", "passed")
        if blob_benchmark[key] is None
    ]
    if missing:
        raise RuntimeError(
            "Candidate metrics blob verification failed: "
            f"metrics blob '{metrics_key}' is missing benchmark fields: {', '.join(missing)}."
        )

    if not (
        _metric_matches(local_benchmark["candidate"], blob_benchmark["candidate"])
        and _metric_matches(local_benchmark["baseline"], blob_benchmark["baseline"])
        and _metric_matches(local_benchmark["required"], blob_benchmark["required"])
        and local_benchmark["passed"] == blob_benchmark["passed"]
    ):
        raise RuntimeError(
            "Candidate metrics blob verification failed: "
            f"metrics blob '{metrics_key}' benchmark values do not match local metrics."
        )

    return blob_metrics


def _logs_indicate_benchmark_evaluated(logs: Optional[str]) -> bool:
    if not logs:
        return False

    return "[BENCHMARK_GATE]" in logs or "[BENCHMARK] candidate mIoU=" in logs


def _validate_remote_metrics_blob(metrics: Optional[Dict[str, Any]], metrics_key: str, logs: Optional[str]) -> Dict[str, Any]:
    if metrics is None:
        raise RuntimeError(
            f"Remote trainer completed but candidate metrics blob '{metrics_key}' was not found."
        )

    if _logs_indicate_benchmark_evaluated(logs):
        benchmark = _read_benchmark_metrics(metrics)
        missing = [
            key
            for key in ("candidate", "baseline", "required", "passed")
            if benchmark[key] is None
        ]
        if missing:
            raise RuntimeError(
                "Candidate metrics blob verification failed: "
                f"remote trainer evaluated benchmark but metrics blob '{metrics_key}' "
                f"is missing benchmark fields: {', '.join(missing)}."
            )

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
RETRAIN_MIN_BENCHMARK_MIOU_IMPROVEMENT = _as_float("RETRAIN_MIN_BENCHMARK_MIOU_IMPROVEMENT", 0.005)


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


def _invoke_remote_trainer(job_id: str, payload: RetrainJobCreateRequest) -> Optional[str]:
    """Call the remote trainer and return combined stdout logs on success, or raise on failure."""
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

    stdout_raw = _truncate(str(data.get("stdoutTail") or "").strip(), 8000)
    stderr_raw = _truncate(str(data.get("stderrTail") or "").strip(), 2000)
    combined_logs = "\n".join(filter(None, [stdout_raw, stderr_raw]))

    status = str(data.get("status", "")).strip().lower()
    exit_code = int(data.get("exitCode", 0)) if str(data.get("exitCode", "")).strip() else 0
    if status not in {"completed", "succeeded", "success"} or exit_code != 0:
        message = str(data.get("message") or "Remote trainer reported failed status")
        # Store logs so backend can read them even on failure
        if combined_logs:
            log_lines = [line for line in combined_logs.splitlines() if line.strip()]
            _set_job(job_id, logs=log_lines)
        stderr_tail = _truncate(stderr_raw, 1500)
        stdout_tail = _truncate(stdout_raw, 500)
        parts = [message]
        if stderr_tail:
            parts.append(f"[stderr] {stderr_tail}")
        if stdout_tail:
            parts.append(f"[stdout] {stdout_tail}")
        raise RuntimeError("\n".join(parts))

    return combined_logs or None


def _run_retrain_job(job_id: str, payload: RetrainJobCreateRequest) -> None:
    _set_job(job_id, status="running", startedAtUtc=_utc_now_iso())

    try:
        remote_trainer_used = RETRAIN_USE_REMOTE_TRAINER
        trainer_logs: Optional[str] = None
        if RETRAIN_USE_REMOTE_TRAINER:
            trainer_logs = _invoke_remote_trainer(job_id, payload)
            if trainer_logs:
                log_lines = [line for line in trainer_logs.splitlines() if line.strip()]
                _set_job(job_id, logs=log_lines)
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

        if remote_trainer_used:
            metrics = _download_existing_metrics(
                RETRAIN_STORAGE_CONNECTION_STRING,
                RETRAIN_CONTAINER,
                RETRAIN_EXTERNAL_PREFIX,
            )
            metrics = _validate_remote_metrics_blob(metrics, candidate_metrics_key, trainer_logs)
        elif local_candidate_exists:
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
            blob_metrics = _download_metrics_blob(container_client, candidate_metrics_key)
            _verify_candidate_metrics_blob(metrics, blob_metrics, candidate_metrics_key)
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
        benchmark_candidate_miou = _read_metric(metrics, "benchmark.candidate.mean_iou") if metrics else None
        benchmark_baseline_miou = _read_metric(metrics, "benchmark.baseline.mean_iou") if metrics else None
        benchmark_required_miou = _read_metric(metrics, "benchmark.gate.required_mean_iou") if metrics else None
        benchmark_gate_passed: Optional[bool] = None
        if metrics:
            benchmark_gate_passed = _read_benchmark_gate_passed(metrics)
        if benchmark_gate_passed is None and benchmark_candidate_miou is not None and benchmark_baseline_miou is not None:
            benchmark_required_miou = benchmark_baseline_miou + RETRAIN_MIN_BENCHMARK_MIOU_IMPROVEMENT
            benchmark_gate_passed = benchmark_candidate_miou >= benchmark_required_miou

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
            benchmarkCandidateMiou=benchmark_candidate_miou,
            benchmarkBaselineMiou=benchmark_baseline_miou,
            benchmarkRequiredMiou=benchmark_required_miou,
            benchmarkGatePassed=benchmark_gate_passed,
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
        "logs": [],
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
