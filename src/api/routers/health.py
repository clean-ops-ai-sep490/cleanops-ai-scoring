from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.app_state import SAM3_DETECTOR, build_health_payload, build_live_payload

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


@router.get("/health/sam3")
def health_sam3():
    payload = SAM3_DETECTOR.status_payload()
    payload["status"] = "ready" if payload["sam3_loaded"] else "disabled" if not payload["sam3_enabled"] else "degraded"
    status_code = 200 if payload["status"] in {"ready", "disabled"} else 503
    return JSONResponse(status_code=status_code, content=payload)
