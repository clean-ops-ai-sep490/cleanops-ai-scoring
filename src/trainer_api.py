from __future__ import annotations

import logging
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


def _truncate(text: str, max_length: int = 2000) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length]


TRAINER_API_ENABLED = _as_bool("TRAINER_API_ENABLED", True)
TRAINER_API_KEY = os.getenv("TRAINER_API_KEY", "").strip()
RETRAIN_API_ENABLED = _as_bool("RETRAIN_API_ENABLED", True)
RETRAIN_API_KEY = (os.getenv("RETRAIN_API_KEY") or TRAINER_API_KEY).strip()
TRAINER_COMMAND = (
    os.getenv("TRAINER_COMMAND")
    or os.getenv("SCORING_TRAIN_COMMAND")
    or ""
).strip()
TRAINER_TIMEOUT_SEC = max(30, _as_int("TRAINER_TIMEOUT_SEC", 7200))
TRAINER_WORKDIR = os.getenv("TRAINER_WORKDIR", "/app").strip() or "/app"
TRAINER_LOG_DIR = os.getenv("TRAINER_LOG_DIR", "outputs/retrain/logs").strip() or "outputs/retrain/logs"
TRAINER_LOG_TAIL_LINES = max(50, _as_int("TRAINER_LOG_TAIL_LINES", 500))


class TrainerSample(BaseModel):
    resultId: str
    jobId: str
    requestId: str
    environmentKey: str
    sourceType: str
    source: str
    reviewedVerdict: str
    reviewedAtUtc: str
    reviewedByEmail: Optional[str] = None


class TrainerJobRequest(BaseModel):
    jobId: str = Field(..., min_length=1)
    batchId: str = Field(..., min_length=1)
    sourceWindowFromUtc: Optional[str] = None
    reviewedSampleCount: int = 0
    approvedAnnotationCount: int = 0
    minApprovedAnnotations: int = 100
    maxSamplesPerBatch: int = 500
    samples: List[TrainerSample] = Field(default_factory=list)


class RetrainJobRequest(BaseModel):
    batchId: str = Field(..., min_length=1)
    sourceWindowFromUtc: Optional[str] = None
    reviewedSampleCount: int = 0
    approvedAnnotationCount: int = 0
    minApprovedAnnotations: int = 100
    maxSamplesPerBatch: int = 500
    samples: List[TrainerSample] = Field(default_factory=list)


class TrainerJobResponse(BaseModel):
    jobId: str
    status: str
    exitCode: int
    message: str
    stdoutTail: Optional[str] = None
    stderrTail: Optional[str] = None
    logs: Optional[List[str]] = Field(default_factory=list)

app = FastAPI(
    title="CleanOps Scoring Trainer API",
    version="1.0.0",
    docs_url="/trainer/docs",
    openapi_url="/trainer/openapi.json",
)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _authorize_or_raise(x_trainer_api_key: Optional[str]) -> None:
    if not TRAINER_API_ENABLED:
        raise HTTPException(status_code=404, detail="Trainer API is disabled")

    if TRAINER_API_KEY and x_trainer_api_key != TRAINER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid trainer API key")


def _authorize_retrain_or_raise(x_retrain_api_key: Optional[str]) -> None:
    if not RETRAIN_API_ENABLED:
        raise HTTPException(status_code=404, detail="Retrain API is disabled")

    if RETRAIN_API_KEY and x_retrain_api_key != RETRAIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid retrain API key")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _set_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _bounded_tail(lines: List[str], max_lines: int = TRAINER_LOG_TAIL_LINES) -> List[str]:
    if len(lines) <= max_lines:
        return lines.copy()
    return lines[-max_lines:].copy()


def _resolve_job_log_path(workdir_path: Path, job_id: str) -> Path:
    log_dir = Path(TRAINER_LOG_DIR)
    if not log_dir.is_absolute():
        log_dir = workdir_path / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{job_id}.log"


def _append_log(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _append_and_publish_log(job_id: str, log_path: Path, all_lines: List[str], line: str) -> None:
    all_lines.append(line)
    _append_log(log_path, line)
    _set_job(job_id, logs=_bounded_tail(all_lines), logPath=str(log_path))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "cleanops-ai-scoring-trainer",
    }


def _execute_trainer_job(payload: TrainerJobRequest) -> TrainerJobResponse:
    if not TRAINER_COMMAND:
        raise HTTPException(status_code=500, detail="TRAINER_COMMAND is empty")

    workdir_path = Path(TRAINER_WORKDIR)
    if not workdir_path.exists() or not workdir_path.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"TRAINER_WORKDIR does not exist or is not a directory: {TRAINER_WORKDIR}",
        )

    logger.info("Trainer job started. job_id=%s batch_id=%s", payload.jobId, payload.batchId)
    log_path = _resolve_job_log_path(workdir_path, payload.jobId)
    all_lines: List[str] = []
    header_lines = [
        "[JOB] CleanOps retrain trainer started",
        f"[JOB] job_id={payload.jobId}",
        f"[JOB] batch_id={payload.batchId}",
        f"[JOB] source_window_from_utc={payload.sourceWindowFromUtc or '(not set)'}",
        f"[JOB] reviewed_sample_count={payload.reviewedSampleCount}",
        f"[JOB] approved_annotation_count={payload.approvedAnnotationCount}",
        f"[JOB] min_approved_annotations={payload.minApprovedAnnotations}",
        f"[JOB] max_samples_per_batch={payload.maxSamplesPerBatch}",
        f"[JOB] trainer_command={TRAINER_COMMAND}",
        f"[JOB] trainer_workdir={workdir_path}",
        f"[JOB] log_path={log_path}",
    ]
    for header_line in header_lines:
        _append_and_publish_log(payload.jobId, log_path, all_lines, header_line)

    command_env = os.environ.copy()
    command_env.update(
        {
            "TRAINER_JOB_ID": payload.jobId,
            "TRAINER_BATCH_ID": payload.batchId,
            "TRAINER_SOURCE_WINDOW_FROM_UTC": payload.sourceWindowFromUtc or "",
            "TRAINER_REVIEWED_SAMPLE_COUNT": str(payload.reviewedSampleCount),
            "TRAINER_APPROVED_ANNOTATION_COUNT": str(payload.approvedAnnotationCount),
            "TRAINER_MIN_APPROVED_ANNOTATIONS": str(payload.minApprovedAnnotations),
            "TRAINER_MAX_SAMPLES_PER_BATCH": str(payload.maxSamplesPerBatch),
        }
    )

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    
    try:
        proc = subprocess.Popen(
            TRAINER_COMMAND,
            shell=True,
            cwd=str(workdir_path),
            env=command_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        def read_stream(stream, lines_list, stream_type):
            try:
                for line in iter(stream.readline, ''):
                    if line:
                        line = line.rstrip('\n')
                        lines_list.append(line)
                        _append_and_publish_log(payload.jobId, log_path, all_lines, f"[{stream_type}] {line}")
                        logger.debug(f"[{stream_type}] {line}")
            except Exception as e:
                logger.exception(f"Error reading {stream_type}")
        
        stdout_thread = threading.Thread(target=read_stream, args=(proc.stdout, stdout_lines, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=read_stream, args=(proc.stderr, stderr_lines, "stderr"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        try:
            proc.wait(timeout=TRAINER_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            logger.warning("Trainer job timed out. job_id=%s timeout=%s", payload.jobId, TRAINER_TIMEOUT_SEC)
            timeout_message = f"Trainer command timed out after {TRAINER_TIMEOUT_SEC} seconds."
            _append_and_publish_log(payload.jobId, log_path, all_lines, f"[ERROR] {timeout_message}")
            return TrainerJobResponse(
                jobId=payload.jobId,
                status="failed",
                exitCode=-1,
                message=timeout_message,
                logs=_bounded_tail(all_lines),
            )
        
        # Wait for threads to finish reading
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        exit_code = proc.returncode
        
    except Exception as exc:
        logger.exception("Trainer command failed unexpectedly. job_id=%s", payload.jobId)
        _append_and_publish_log(payload.jobId, log_path, all_lines, f"[ERROR] {exc}")
        return TrainerJobResponse(
            jobId=payload.jobId,
            status="failed",
            exitCode=1,
            message=str(exc),
            logs=_bounded_tail(all_lines),
        )

    if exit_code != 0:
        stderr_str = "\n".join(stderr_lines).strip() if stderr_lines else ""
        stdout_str = "\n".join(stdout_lines).strip() if stdout_lines else ""
        logger.error("Trainer job failed. job_id=%s exit_code=%s", payload.jobId, exit_code)
        failure_message = f"Trainer command failed with exit code {exit_code}."
        _append_and_publish_log(payload.jobId, log_path, all_lines, f"[ERROR] {failure_message}")
        return TrainerJobResponse(
            jobId=payload.jobId,
            status="failed",
            exitCode=exit_code,
            message=failure_message,
            stdoutTail=_truncate(stdout_str) or None,
            stderrTail=_truncate(stderr_str) or None,
            logs=_bounded_tail(all_lines),
        )

    logger.info("Trainer job completed. job_id=%s", payload.jobId)
    stdout_str = "\n".join(stdout_lines).strip() if stdout_lines else ""
    stderr_str = "\n".join(stderr_lines).strip() if stderr_lines else ""
    _append_and_publish_log(payload.jobId, log_path, all_lines, "[DONE] Trainer command completed successfully.")
    return TrainerJobResponse(
        jobId=payload.jobId,
        status="completed",
        exitCode=0,
        message="Trainer command completed successfully.",
        stdoutTail=_truncate(stdout_str) or None,
        stderrTail=_truncate(stderr_str) or None,
        logs=_bounded_tail(all_lines),
    )

@app.post("/trainer/jobs", response_model=TrainerJobResponse)
def run_trainer_job(
    payload: TrainerJobRequest,
    x_trainer_api_key: Optional[str] = Header(default=None),
) -> TrainerJobResponse:
    _authorize_or_raise(x_trainer_api_key)
    return _execute_trainer_job(payload)


@app.post("/retrain/jobs")
def create_retrain_job(
    payload: RetrainJobRequest,
    x_retrain_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _authorize_retrain_or_raise(x_retrain_api_key)

    job_id = uuid.uuid4().hex
    retrain_payload = TrainerJobRequest(jobId=job_id, **_model_to_dict(payload))
    submitted_at_utc = _utc_now_iso()

    with _jobs_lock:
        _jobs[job_id] = {
            "jobId": job_id,
            "batchId": retrain_payload.batchId,
            "status": "queued",
            "submittedAtUtc": submitted_at_utc,
            "message": "Retrain job queued.",
        }

    def worker() -> None:
        _set_job(job_id, status="running", startedAtUtc=_utc_now_iso(), message="Trainer job running.")
        try:
            result = _execute_trainer_job(retrain_payload)
            _set_job(
                job_id,
                status=result.status,
                completedAtUtc=_utc_now_iso(),
                exitCode=result.exitCode,
                message=result.message,
                stdoutTail=result.stdoutTail,
                stderrTail=result.stderrTail,
                logs=result.logs,
            )
        except HTTPException as exc:
            _set_job(
                job_id,
                status="failed",
                completedAtUtc=_utc_now_iso(),
                exitCode=1,
                message=str(exc.detail),
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Retrain job failed unexpectedly. job_id=%s", job_id)
            _set_job(
                job_id,
                status="failed",
                completedAtUtc=_utc_now_iso(),
                exitCode=1,
                message=str(exc),
            )

    threading.Thread(target=worker, daemon=True).start()

    return {
        "jobId": job_id,
        "status": "queued",
        "submittedAtUtc": submitted_at_utc,
        "message": "Retrain job queued.",
    }


@app.get("/retrain/jobs/{job_id}")
def get_retrain_job_status(
    job_id: str,
    x_retrain_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _authorize_retrain_or_raise(x_retrain_api_key)

    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Retrain job not found")

    return job
