from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from dotenv import dotenv_values as _dotenv_values
    _root = Path(__file__).resolve().parents[1]
    _ENV_FILE = _root / ".env.trainer.local" if (_root / ".env.trainer.local").exists() else _root / ".env"
    _DOTENV: dict = {k: v for k, v in _dotenv_values(_ENV_FILE).items() if v is not None} if _ENV_FILE.exists() else {}
except ImportError:
    _DOTENV = {}

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


def _tail(text: str, max_length: int = 20000) -> str:
    """Keep the last N characters — epoch progress and metrics are at the end."""
    if len(text) <= max_length:
        return text
    return "...[truncated]\n" + text[-max_length:]


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or _DOTENV.get(name) or default).strip()


TRAINER_API_ENABLED = _as_bool("TRAINER_API_ENABLED", True)
TRAINER_API_KEY = _env("TRAINER_API_KEY")
TRAINER_COMMAND = (
    _env("TRAINER_COMMAND")
    or _env("SCORING_TRAIN_COMMAND")
)
TRAINER_TIMEOUT_SEC = max(30, _as_int("TRAINER_TIMEOUT_SEC", 7200))
TRAINER_WORKDIR = _env("TRAINER_WORKDIR") or "/app"


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
    samples: List[TrainerSample] = Field(default_factory=list)


class TrainerJobResponse(BaseModel):
    jobId: str
    status: str
    exitCode: int
    message: str
    stdoutTail: Optional[str] = None
    stderrTail: Optional[str] = None


app = FastAPI(
    title="CleanOps Scoring Trainer API",
    version="1.0.0",
    docs_url="/trainer/docs",
    openapi_url="/trainer/openapi.json",
)


def _authorize_or_raise(x_trainer_api_key: Optional[str]) -> None:
    if not TRAINER_API_ENABLED:
        raise HTTPException(status_code=404, detail="Trainer API is disabled")

    if TRAINER_API_KEY and x_trainer_api_key != TRAINER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid trainer API key")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "cleanops-ai-scoring-trainer",
    }


@app.post("/trainer/jobs", response_model=TrainerJobResponse)
def run_trainer_job(
    payload: TrainerJobRequest,
    x_trainer_api_key: Optional[str] = Header(default=None),
) -> TrainerJobResponse:
    _authorize_or_raise(x_trainer_api_key)

    if not TRAINER_COMMAND:
        raise HTTPException(status_code=500, detail="TRAINER_COMMAND is empty")

    workdir_path = Path(TRAINER_WORKDIR)
    if not workdir_path.exists() or not workdir_path.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"TRAINER_WORKDIR does not exist or is not a directory: {TRAINER_WORKDIR}",
        )

    logger.info("Trainer job started. job_id=%s batch_id=%s", payload.jobId, payload.batchId)
    command_env = {**_DOTENV, **os.environ}
    command_env.update(
        {
            "TRAINER_JOB_ID": payload.jobId,
            "TRAINER_BATCH_ID": payload.batchId,
            "TRAINER_SOURCE_WINDOW_FROM_UTC": payload.sourceWindowFromUtc or "",
            "TRAINER_REVIEWED_SAMPLE_COUNT": str(payload.reviewedSampleCount),
        }
    )

    try:
        proc = subprocess.run(
            TRAINER_COMMAND,
            shell=True,
            cwd=str(workdir_path),
            env=command_env,
            capture_output=True,
            text=True,
            timeout=TRAINER_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Trainer job timed out. job_id=%s timeout=%s", payload.jobId, TRAINER_TIMEOUT_SEC)
        return TrainerJobResponse(
            jobId=payload.jobId,
            status="failed",
            exitCode=-1,
            message=f"Trainer command timed out after {TRAINER_TIMEOUT_SEC} seconds.",
        )
    except Exception as exc:
        logger.exception("Trainer command failed unexpectedly. job_id=%s", payload.jobId)
        return TrainerJobResponse(
            jobId=payload.jobId,
            status="failed",
            exitCode=1,
            message=str(exc),
        )

    if proc.returncode != 0:
        stderr = _truncate((proc.stderr or "").strip())
        stdout = _truncate((proc.stdout or "").strip())
        logger.error(
            "Trainer job failed. job_id=%s exit_code=%s\nSTDOUT:\n%s\nSTDERR:\n%s",
            payload.jobId, proc.returncode, stdout or "(empty)", stderr or "(empty)",
        )
        return TrainerJobResponse(
            jobId=payload.jobId,
            status="failed",
            exitCode=proc.returncode,
            message=f"Trainer command failed with exit code {proc.returncode}.",
            stdoutTail=stdout or None,
            stderrTail=stderr or None,
        )

    logger.info("Trainer job completed. job_id=%s", payload.jobId)
    return TrainerJobResponse(
        jobId=payload.jobId,
        status="completed",
        exitCode=0,
        message="Trainer command completed successfully.",
        stdoutTail=_tail((proc.stdout or "").strip()) or None,
        stderrTail=_truncate((proc.stderr or "").strip(), 3000) or None,
    )
