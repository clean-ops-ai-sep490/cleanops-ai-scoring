from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.app_state import build_health_payload, build_live_payload

router = APIRouter(tags=["health"])


@router.get("/")
def health_check():
    return build_health_payload()


@router.get("/health/live")
def health_live():
    return build_live_payload()


@router.get("/health/ready")
def health_ready():
    payload = build_health_payload()
    status_code = 200 if payload["ready"] else 503
    return JSONResponse(status_code=status_code, content=payload)

