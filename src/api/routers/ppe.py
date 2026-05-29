from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api import app_state
from src.api.ppe_utils import GeminiPpeConfig, evaluate_ppe_payload
from src.api.schemas import PpeEvaluateRequest

router = APIRouter(tags=["ppe"])


def _require_ppe_model():
    if not app_state.settings.ppe_enabled:
        return JSONResponse(status_code=503, content={"error": "PPE inference is disabled."})
    if app_state.PPE_MODEL is None:
        return JSONResponse(status_code=503, content={"error": "PPE model is not loaded."})
    return None


@router.get("/ppe/labels")
async def get_ppe_labels():
    unavailable = _require_ppe_model()
    if unavailable:
        return unavailable

    return {
        "labels": app_state.PPE_CLASS_LABELS,
        "count": len(app_state.PPE_CLASS_LABELS),
        "model_source": app_state.PPE_MODEL_SOURCE,
    }


@router.post("/ppe/evaluate", response_model=None)
async def evaluate_ppe(request: PpeEvaluateRequest):
    unavailable = _require_ppe_model()
    if unavailable:
        return unavailable

    return await evaluate_ppe_payload(
        image_urls=request.image_urls,
        required_objects=request.required_objects,
        model=app_state.PPE_MODEL,
        timeout_sec=app_state.REQUEST_TIMEOUT_SEC,
        min_confidence=request.min_confidence,
        batch_concurrency=app_state.settings.inference_batch_concurrency,
        gemini_config=GeminiPpeConfig(
            enabled=app_state.settings.ppe_gemini_enabled,
            mode=app_state.settings.ppe_gemini_mode,
            api_key=app_state.settings.gemini_api_key,
            model=app_state.settings.ppe_gemini_model,
            base_url=app_state.settings.gemini_base_url,
            timeout_sec=app_state.settings.ppe_gemini_timeout_sec,
        ),
    )
