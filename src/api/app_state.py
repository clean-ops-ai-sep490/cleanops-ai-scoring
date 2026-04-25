from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import requests
import torch
from PIL import Image
from ultralytics import YOLO

from src.api.inference_utils import (
    evaluate_image as evaluate_image_impl,
    evaluate_image_with_artifacts as evaluate_image_with_artifacts_impl,
    load_unet_model as load_unet_model_impl,
    unet_predict_from_pil as unet_predict_from_pil_impl,
    yolo_predict_from_pil as yolo_predict_from_pil_impl,
)
from src.api.llm_filter import GeminiFilterConfig, GeminiLLMFilter
from src.api.scoring_utils import normalize_env as normalize_env_impl
from src.api.scoring_utils import parse_url_items as parse_url_items_impl
from src.api.scoring_utils import score_image as score_image_impl
from src.api.scoring_utils import summarize_penalty_detections as summarize_penalty_detections_impl
from src.api.visualization_utils import (
    build_visualize_blob_url_payload as build_visualize_blob_url_payload_impl,
    build_visualize_json_payload as build_visualize_json_payload_impl,
    extract_dirty_region_candidates as extract_dirty_region_candidates_impl,
    render_hybrid_overlay as render_hybrid_overlay_impl,
    render_unet_overlay as render_unet_overlay_impl,
)
from src.config.settings import get_env_rules, settings
from src.storage.model_loader import ObjectStorageConfig, ObjectStorageModelLoader
from src.storage.visualization_blob_store import VisualizationBlobConfig, VisualizationBlobStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = settings.project_root
BASE_OUTPUT_DIR = str(settings.base_output_dir)
MODEL_PATH = settings.model_path
UNET_MODEL_PATH = settings.unet_model_path
PPE_MODEL_PATH = settings.ppe_model_path
UNET_IMG_SIZE = settings.unet_img_size
MODEL_CACHE_DIR = settings.model_cache_dir
UNET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAP = {
    0: "background",
    1: "stain_or_water",
    2: "wet_surface",
}

ENV_RULES = get_env_rules()
MAX_BATCH_IMAGES = settings.max_batch_images
PENDING_LOWER_BOUND = settings.pending_lower_bound
SCORING_PENALTY_LABELS = settings.scoring_penalty_labels
SCORING_OBJECT_PENALTY_PER_DETECTION = max(0.0, settings.scoring_object_penalty_per_detection)
YOLO_CONF = settings.yolo_conf
REQUEST_TIMEOUT_SEC = settings.request_timeout_sec
VISUALIZE_JPEG_QUALITY = max(20, min(100, settings.visualize_jpeg_quality))
APP_PUBLIC_BASE_URL = settings.app_public_base_url
VISUALIZE_TEMP_URL_TTL_SEC = max(30, settings.visualize_temp_url_ttl_sec)
VISUALIZE_TEMP_MAX_ITEMS = max(50, settings.visualize_temp_max_items)

MODEL_STORAGE = ObjectStorageConfig(
    enabled=settings.model_storage_enabled,
    connection_string=settings.model_storage_connection_string,
    container=settings.model_storage_container,
    force_refresh=settings.model_force_refresh,
)
MODEL_LOADER = ObjectStorageModelLoader(MODEL_STORAGE, logger)
MODEL_REQUIRE_BLOB = settings.model_require_blob

VISUALIZATION_BLOB_STORE = VisualizationBlobStore(
    VisualizationBlobConfig(
        enabled=settings.visualization_blob_enabled,
        connection_string=settings.visualization_blob_connection_string,
        container=settings.visualization_blob_container,
        prefix=settings.visualization_blob_prefix,
    ),
    logger,
)

YOLO_MODEL_SOURCE = "unknown"
UNET_MODEL_SOURCE = "unknown"
PPE_MODEL_SOURCE = "unknown"
LLM_FILTER = GeminiLLMFilter(
    GeminiFilterConfig(
        enabled=settings.llm_filter_enabled,
        mode=settings.llm_filter_mode,
        model=settings.llm_filter_model,
        timeout_sec=settings.llm_filter_timeout_sec,
        batch_concurrency=settings.llm_filter_batch_concurrency,
        queue_enabled=settings.llm_filter_queue_enabled,
        queue_mode=settings.llm_filter_queue_mode,
        deadline_sec=settings.llm_filter_deadline_sec,
        retry_429_max_retries=settings.llm_filter_429_max_retries,
        retry_5xx_max_retries=settings.llm_filter_5xx_max_retries,
        cooldown_sec=settings.llm_filter_cooldown_sec,
        enable_borderline_only=settings.llm_filter_enable_borderline_only,
        scoring_pass_window=settings.llm_filter_scoring_pass_window,
        ppe_verify_on_missing_only=settings.llm_filter_ppe_verify_on_missing_only,
        retry_initial_delay_ms=settings.llm_filter_retry_initial_delay_ms,
        retry_max_delay_ms=settings.llm_filter_retry_max_delay_ms,
        retryable_status_codes=settings.llm_filter_retryable_status_codes,
        max_image_dimension=settings.llm_filter_max_image_dimension,
        jpeg_quality=settings.llm_filter_jpeg_quality,
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
    ),
    logger,
)


def _extract_model_labels(model: YOLO | None) -> List[str]:
    if model is None:
        return []

    names = getattr(model, "names", None)
    if isinstance(names, dict):
        values = names.values()
    elif isinstance(names, list):
        values = names
    else:
        return []

    labels = {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }
    return sorted(labels)


def _extract_label_to_id_map(model: YOLO | None) -> Dict[str, int]:
    if model is None:
        return {}

    names = getattr(model, "names", None)
    if isinstance(names, dict):
        iterator = names.items()
    elif isinstance(names, list):
        iterator = enumerate(names)
    else:
        return {}

    mapping: Dict[str, int] = {}
    for class_id, value in iterator:
        label = str(value).strip().lower()
        if label:
            mapping[label] = int(class_id)
    return mapping


def _load_yolo_model() -> tuple[YOLO | None, Optional[str], str]:
    global YOLO_MODEL_SOURCE

    try:
        yolo_cache_path = Path(MODEL_CACHE_DIR) / "active" / "yolo" / "model.pt"
        yolo_fallback_paths = [] if MODEL_REQUIRE_BLOB else ([Path(MODEL_PATH)] if MODEL_PATH else [])
        yolo_resolved_path, YOLO_MODEL_SOURCE = MODEL_LOADER.resolve_model_path(
            settings.model_storage_active_yolo_key,
            yolo_cache_path,
            yolo_fallback_paths,
        )

        if not yolo_resolved_path and settings.yolo_weights_path and not MODEL_REQUIRE_BLOB:
            yolo_resolved_path = settings.yolo_weights_path
            YOLO_MODEL_SOURCE = "ultralytics-default"

        if yolo_resolved_path:
            loaded_model = YOLO(yolo_resolved_path)
            return loaded_model, yolo_resolved_path, YOLO_MODEL_SOURCE

        return None, MODEL_PATH, YOLO_MODEL_SOURCE
    except Exception as ex:  # noqa: BLE001
        YOLO_MODEL_SOURCE = f"load-error:{type(ex).__name__}"
        logger.exception("Failed to initialize scoring YOLO model.")
        return None, MODEL_PATH, YOLO_MODEL_SOURCE


def _load_unet_model() -> tuple[Any, str, int, str]:
    global UNET_MODEL_SOURCE

    resolved_path = UNET_MODEL_PATH
    image_size = UNET_IMG_SIZE
    try:
        unet_cache_path = Path(MODEL_CACHE_DIR) / "active" / "unet" / "model.pth"
        unet_fallback_paths = [] if MODEL_REQUIRE_BLOB else [Path(UNET_MODEL_PATH)]
        unet_resolved_path, UNET_MODEL_SOURCE = MODEL_LOADER.resolve_model_path(
            settings.model_storage_active_unet_key,
            unet_cache_path,
            unet_fallback_paths,
        )

        if unet_resolved_path:
            resolved_path = unet_resolved_path

        loaded_model, image_size = load_unet_model_impl(
            resolved_path,
            UNET_DEVICE,
            image_size,
        )
        return loaded_model, resolved_path, image_size, UNET_MODEL_SOURCE
    except Exception as ex:  # noqa: BLE001
        UNET_MODEL_SOURCE = f"load-error:{type(ex).__name__}"
        logger.exception("Failed to initialize U-Net model.")
        return None, resolved_path, image_size, UNET_MODEL_SOURCE


def _load_ppe_model() -> tuple[YOLO | None, str, List[str], str]:
    global PPE_MODEL_SOURCE

    if not settings.ppe_enabled:
        PPE_MODEL_SOURCE = "disabled"
        return None, PPE_MODEL_PATH, [], PPE_MODEL_SOURCE

    resolved_path = PPE_MODEL_PATH
    try:
        ppe_cache_path = Path(MODEL_CACHE_DIR) / "active" / "ppe" / "model.pt"
        ppe_fallback_paths = [] if MODEL_REQUIRE_BLOB else [Path(PPE_MODEL_PATH)]
        ppe_resolved_path, PPE_MODEL_SOURCE = MODEL_LOADER.resolve_model_path(
            settings.model_storage_active_ppe_key,
            ppe_cache_path,
            ppe_fallback_paths,
        )

        if ppe_resolved_path:
            resolved_path = ppe_resolved_path

        if not resolved_path:
            return None, PPE_MODEL_PATH, [], PPE_MODEL_SOURCE

        loaded_model = YOLO(resolved_path)
        return loaded_model, resolved_path, _extract_model_labels(loaded_model), PPE_MODEL_SOURCE
    except Exception as ex:  # noqa: BLE001
        PPE_MODEL_SOURCE = f"load-error:{type(ex).__name__}"
        logger.exception("Failed to initialize PPE model.")
        return None, resolved_path, [], PPE_MODEL_SOURCE


MODEL, MODEL_PATH, YOLO_MODEL_SOURCE = _load_yolo_model()
UNET_MODEL, UNET_MODEL_PATH, UNET_IMG_SIZE, UNET_MODEL_SOURCE = _load_unet_model()
PPE_MODEL, PPE_MODEL_PATH, PPE_CLASS_LABELS, PPE_MODEL_SOURCE = _load_ppe_model()
YOLO_CLASS_LABELS = _extract_model_labels(MODEL)
YOLO_LABEL_TO_ID = _extract_label_to_id_map(MODEL)


def _merge_reasons(*reason_groups: List[str]) -> List[str]:
    seen: set[str] = set()
    merged: List[str] = []
    for group in reason_groups:
        for raw in group or []:
            item = str(raw or "").strip()
            if not item:
                continue
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(item)
    return merged


def build_health_payload() -> Dict[str, Any]:
    readiness_reasons: List[str] = []

    if MODEL is None:
        readiness_reasons.append("yolo-model-not-loaded")
    if UNET_MODEL is None:
        readiness_reasons.append("unet-model-not-loaded")
    if settings.ppe_enabled and PPE_MODEL is None:
        readiness_reasons.append("ppe-model-not-loaded")

    if MODEL_REQUIRE_BLOB:
        if not MODEL_STORAGE.enabled:
            readiness_reasons.append("blob-storage-disabled-in-strict-mode")
        if not settings.model_storage_connection_string:
            readiness_reasons.append("blob-connection-string-missing")
        if not settings.model_storage_container:
            readiness_reasons.append("blob-container-missing")

    ready = len(readiness_reasons) == 0

    return {
        "status": "ready" if ready else "degraded",
        "live": True,
        "ready": ready,
        "readiness_reasons": readiness_reasons,
        "yolo_loaded": MODEL is not None,
        "unet_loaded": UNET_MODEL is not None,
        "ppe_loaded": PPE_MODEL is not None,
        "yolo_model_path": MODEL_PATH,
        "yolo_model_source": YOLO_MODEL_SOURCE,
        "unet_model_path": UNET_MODEL_PATH,
        "unet_model_source": UNET_MODEL_SOURCE,
        "ppe_model_path": PPE_MODEL_PATH,
        "ppe_model_source": PPE_MODEL_SOURCE,
        "ppe_labels": PPE_CLASS_LABELS,
        "ppe_enabled": settings.ppe_enabled,
        "model_storage_enabled": MODEL_STORAGE.enabled,
        "model_storage_container": MODEL_STORAGE.container,
        "model_require_blob": MODEL_REQUIRE_BLOB,
        "max_batch_images": MAX_BATCH_IMAGES,
        "pending_lower_bound": PENDING_LOWER_BOUND,
        "scoring_penalty_labels": list(SCORING_PENALTY_LABELS),
        "scoring_object_penalty_per_detection": SCORING_OBJECT_PENALTY_PER_DETECTION,
        "visualize_jpeg_quality": VISUALIZE_JPEG_QUALITY,
        "visualize_temp_url_ttl_sec": VISUALIZE_TEMP_URL_TTL_SEC,
        "visualize_temp_max_items": VISUALIZE_TEMP_MAX_ITEMS,
        "visualization_blob_enabled": settings.visualization_blob_enabled,
        "visualization_blob_container": settings.visualization_blob_container,
        "visualization_blob_prefix": settings.visualization_blob_prefix,
        "env_rules": ENV_RULES,
        "message": "Welcome to CleanOps AI Service",
        **LLM_FILTER.status_payload(),
    }


def build_live_payload() -> Dict[str, Any]:
    return {
        "status": "live",
        "live": True,
        **LLM_FILTER.status_payload(),
    }


def build_gemini_probe_payload(model: Optional[str] = None) -> Dict[str, Any]:
    selected_model = (model or settings.llm_filter_model or "").strip()
    timeout_sec = max(3, min(15, settings.llm_filter_timeout_sec))
    payload: Dict[str, Any] = {
        "status": "unknown",
        "ok": False,
        "enabled": settings.llm_filter_enabled,
        "configured": bool(settings.gemini_api_key.strip()),
        "model": selected_model,
        "timeout_sec": timeout_sec,
        "http_status": None,
        "latency_ms": None,
        "error": None,
    }

    if not settings.llm_filter_enabled:
        payload["status"] = "disabled"
        return payload

    if not settings.gemini_api_key.strip():
        payload["status"] = "not_configured"
        payload["error"] = "GEMINI_API_KEY is empty"
        return payload

    if not selected_model:
        payload["status"] = "invalid_config"
        payload["error"] = "LLM_FILTER_MODEL is empty"
        return payload

    url = f"{settings.gemini_base_url.rstrip('/')}/models/{selected_model}:generateContent"
    request_body = {
        "contents": [
            {
                "parts": [
                    {"text": "say ok"},
                ]
            }
        ]
    }

    started_at = time.perf_counter()
    try:
        response = requests.post(
            url,
            params={"key": settings.gemini_api_key},
            json=request_body,
            timeout=timeout_sec,
        )
        payload["http_status"] = response.status_code
        payload["latency_ms"] = int(round((time.perf_counter() - started_at) * 1000))
        if response.ok:
            payload["status"] = "ok"
            payload["ok"] = True
            return payload

        payload["status"] = "error"
        payload["error"] = response.text[:500]
        return payload
    except requests.RequestException as exc:
        payload["status"] = "error"
        payload["latency_ms"] = int(round((time.perf_counter() - started_at) * 1000))
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload


def normalize_env(env: Optional[str]) -> str:
    return normalize_env_impl(env, ENV_RULES)


def parse_url_items(image_urls: List[str]) -> List[str]:
    return parse_url_items_impl(image_urls)


def build_llm_filter_payload(source: str, *, kinds: List[str], route_mode: str | None = None) -> Dict[str, Any]:
    payload = {
        "llm_filter": LLM_FILTER.response_metadata(source, kinds),
    }
    if route_mode:
        payload["llm_filter"]["route_mode"] = route_mode
    return payload


def yolo_predict_from_pil(
    img: Image.Image,
    *,
    apply_llm_filter: bool = True,
    source: str = "yolo",
) -> Dict[str, Any]:
    result = yolo_predict_from_pil_impl(
        img,
        model=MODEL,
        yolo_conf=YOLO_CONF,
    )
    if apply_llm_filter:
        return LLM_FILTER.refine_yolo_result(
            img,
            result,
            allowed_labels=YOLO_CLASS_LABELS,
            label_to_id=YOLO_LABEL_TO_ID,
            source=source,
        )
    return result


def unet_predict_from_pil(
    img: Image.Image,
    *,
    apply_llm_filter: bool = True,
    source: str = "unet",
) -> Dict[str, Any]:
    result = unet_predict_from_pil_impl(
        img,
        unet_model=UNET_MODEL,
        unet_img_size=UNET_IMG_SIZE,
        unet_device=UNET_DEVICE,
        class_map=CLASS_MAP,
    )
    if apply_llm_filter:
        verified_dirty = LLM_FILTER.verify_dirty_evidence(
            img,
            result["summary"],
            dirty_region_candidates=extract_dirty_region_candidates_impl(result["mask_original_size"]),
            source=source,
        )
        result["summary"] = verified_dirty["summary"]
    return result


def evaluate_image_with_artifacts(
    img: Image.Image,
    env_key: str,
    *,
    source: str = "hybrid",
    visualize_enhanced: bool = False,
):
    raw_yolo_result, raw_unet_result, baseline_scoring = evaluate_image_with_artifacts_impl(
        img,
        env_key,
        model=MODEL,
        unet_model=UNET_MODEL,
        yolo_conf=YOLO_CONF,
        unet_img_size=UNET_IMG_SIZE,
        unet_device=UNET_DEVICE,
        class_map=CLASS_MAP,
        env_rules=ENV_RULES,
        pending_lower_bound=PENDING_LOWER_BOUND,
        scoring_penalty_labels=SCORING_PENALTY_LABELS,
        scoring_object_penalty_per_detection=SCORING_OBJECT_PENALTY_PER_DETECTION,
    )
    dirty_region_candidates = extract_dirty_region_candidates_impl(raw_unet_result["mask_original_size"])
    verified_bundle = LLM_FILTER.verify_scoring_evidence(
        img,
        env_key=env_key,
        yolo_result=raw_yolo_result,
        unet_summary=raw_unet_result["summary"],
        dirty_region_candidates=dirty_region_candidates,
        scoring=baseline_scoring,
        pending_lower_bound=PENDING_LOWER_BOUND,
        allowed_labels=YOLO_CLASS_LABELS,
        label_to_id=YOLO_LABEL_TO_ID,
        source=source,
        visualize_enhanced=visualize_enhanced,
    )
    verified_yolo = verified_bundle["yolo"]
    raw_unet_result["summary"] = verified_bundle["summary"]

    penalty_summary = summarize_penalty_detections_impl(
        verified_yolo.get("results", []),
        SCORING_PENALTY_LABELS,
    )
    recomputed_scoring = score_image_impl(
        total_dirty_coverage_pct=raw_unet_result["summary"]["total_dirty_coverage_pct"],
        detections_count=verified_yolo["detections_count"],
        env_key=env_key,
        env_rules=ENV_RULES,
        pending_lower_bound=PENDING_LOWER_BOUND,
        object_penalty_per_detection=SCORING_OBJECT_PENALTY_PER_DETECTION,
        **penalty_summary,
    )
    recomputed_scoring["reasons"] = _merge_reasons(
        recomputed_scoring.get("reasons", []),
        verified_bundle["review"].get("reasons", []),
    )
    if not recomputed_scoring["reasons"]:
        recomputed_scoring["reasons"] = baseline_scoring.get("reasons", [])

    return verified_yolo, raw_unet_result, recomputed_scoring, dirty_region_candidates, verified_bundle["review"]


def evaluate_image(
    img: Image.Image,
    env_key: str,
    *,
    source: str = "hybrid",
) -> Dict[str, Any]:
    yolo_result, unet_result, score, _, _ = evaluate_image_with_artifacts(img, env_key, source=source)
    return {
        "yolo": yolo_result,
        "unet": unet_result["summary"],
        "scoring": score,
        **build_llm_filter_payload(source, kinds=["scoring_verification"], route_mode="scoring_quota_saver"),
    }


def evaluate_image_with_visual_review(
    img: Image.Image,
    env_key: str,
    *,
    source: str = "hybrid-visual",
):
    return evaluate_image_with_artifacts(
        img,
        env_key,
        source=source,
        visualize_enhanced=True,
    )


def render_unet_overlay(rgb, pred_original_size):
    return render_unet_overlay_impl(rgb, pred_original_size)


def render_hybrid_overlay(
    rgb,
    pred_original_size,
    yolo_result: Dict[str, Any],
    scoring: Dict[str, Any],
    env_key: str,
    visual_review: Dict[str, Any] | None = None,
    dirty_region_candidates: List[Dict[str, Any]] | None = None,
):
    return render_hybrid_overlay_impl(
        rgb,
        pred_original_size,
        yolo_result=yolo_result,
        scoring=scoring,
        env_key=env_key,
        visualize_jpeg_quality=VISUALIZE_JPEG_QUALITY,
        visual_review=visual_review,
        dirty_region_candidates=dirty_region_candidates,
    )


def build_visualize_json_payload(
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    rendered: bytes,
    llm_filter: Dict[str, Any] | None = None,
):
    return build_visualize_json_payload_impl(
        source_type=source_type,
        source=source,
        env_key=env_key,
        yolo_result=yolo_result,
        unet_result=unet_result,
        scoring=scoring,
        rendered=rendered,
        llm_filter=llm_filter,
    )


def build_visualize_blob_payload(
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    rendered: bytes,
    llm_filter: Dict[str, Any] | None = None,
):
    upload_info = VISUALIZATION_BLOB_STORE.upload_visualization(
        image_bytes=rendered,
        source_type=source_type,
        source=source,
        env_key=env_key,
    )

    return build_visualize_blob_url_payload_impl(
        source_type=source_type,
        source=source,
        env_key=env_key,
        yolo_result=yolo_result,
        unet_result=unet_result,
        scoring=scoring,
        visualization_url=upload_info["url"],
        mime_type=upload_info["mime_type"],
        byte_size=upload_info["byte_size"],
        llm_filter=llm_filter,
    )
