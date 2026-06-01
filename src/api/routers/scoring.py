from __future__ import annotations

import asyncio
import io
from typing import List

import numpy as np
import requests
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.api import app_state
from src.api.scoring_utils import merge_unet_and_sam3_masks
from src.api.schemas import EvaluateVisualizeRequest

router = APIRouter(tags=["scoring"])


def _load_image_from_bytes(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content)).convert("RGB")


def _load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=app_state.REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _require_scoring_models():
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})
    return None


@router.post("/evaluate-batch", tags=["production"])
async def evaluate_batch(
    files: List[UploadFile] = File(default=[]),
    image_urls: List[str] = Form(default=[]),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    unavailable = _require_scoring_models()
    if unavailable:
        return unavailable

    upload_items = [u for u in files if (u.filename or "").strip()]
    url_items = app_state.parse_url_items(image_urls)
    total_images = len(upload_items) + len(url_items)
    if total_images == 0:
        return JSONResponse(status_code=400, content={"error": "Request must include at least 1 image."})
    if total_images > app_state.MAX_BATCH_IMAGES:
        return JSONResponse(status_code=400, content={"error": f"Maximum {app_state.MAX_BATCH_IMAGES} images per request."})

    try:
        env_key = app_state.normalize_env(env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    semaphore = asyncio.Semaphore(max(1, app_state.settings.inference_batch_concurrency))

    async def process_upload(seq: int, upload: UploadFile):
        try:
            content = await upload.read()
            if not content:
                return {"ok": False}
            async with semaphore:
                img = await asyncio.to_thread(_load_image_from_bytes, content)
                eval_result = await asyncio.to_thread(
                    app_state.evaluate_image,
                    img,
                    env_key,
                    source=f"/evaluate-batch:upload:{upload.filename or seq}",
                )
            return {
                "ok": True,
                "payload": {
                    "id": seq,
                    "source_type": "upload",
                    "source": upload.filename,
                    **eval_result,
                },
            }
        except Exception as exc:  # noqa: BLE001
            app_state.logger.warning("Skip failed upload in /evaluate-batch: filename=%s, error=%s", upload.filename, exc)
            return {"ok": False}

    async def process_url(seq: int, url: str):
        try:
            async with semaphore:
                img = await asyncio.to_thread(_load_image_from_url, url)
                eval_result = await asyncio.to_thread(
                    app_state.evaluate_image,
                    img,
                    env_key,
                    source=f"/evaluate-batch:url:{url}",
                )
            return {
                "ok": True,
                "payload": {
                    "id": seq,
                    "source_type": "url",
                    "source": url,
                    **eval_result,
                },
            }
        except Exception as exc:  # noqa: BLE001
            app_state.logger.warning("Skip failed URL in /evaluate-batch: url=%s, error=%s", url, exc)
            return {"ok": False}

    tasks = []
    seq = 1
    for upload in upload_items:
        tasks.append(process_upload(seq, upload))
        seq += 1
    for url in url_items:
        tasks.append(process_url(seq, url))
        seq += 1

    batch_results = await asyncio.gather(*tasks)
    results = []
    processed_count = 0
    skipped_count = 0
    pass_count = 0
    pending_count = 0
    fail_count = 0

    for item in batch_results:
        if not item["ok"]:
            skipped_count += 1
            continue
        payload_item = item["payload"]
        verdict = payload_item["scoring"]["verdict"]
        pass_count += int(verdict == "PASS")
        pending_count += int(verdict == "PENDING")
        fail_count += int(verdict == "FAIL")
        processed_count += 1
        results.append(payload_item)

    results.sort(key=lambda item: item["id"])
    return {
        "env": env_key,
        "env_label": app_state.ENV_RULES[env_key]["label"],
        "max_batch_images": app_state.MAX_BATCH_IMAGES,
        "pending_lower_bound": app_state.PENDING_LOWER_BOUND,
        "summary": {
            "total_requested": total_images,
            "processed": processed_count,
            "skipped": skipped_count,
            "pass": pass_count,
            "pending": pending_count,
            "fail": fail_count,
        },
        "results": results,
    }


@router.post("/evaluate-visualize-link", tags=["internal-upload"])
async def evaluate_visualize_link(
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    unavailable = _require_scoring_models()
    if unavailable:
        return unavailable

    try:
        env_key = app_state.normalize_env(env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = _load_image_from_bytes(content)
        yolo_result, unet_result, scoring, dirty_region_candidates, visual_review, sam3_result = (
            app_state.evaluate_image_with_visual_review(
                img,
                env_key,
                source=f"/evaluate-visualize-link:{file.filename or 'upload'}",
            )
        )
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
            visual_review=visual_review,
            dirty_region_candidates=dirty_region_candidates,
            sam3_result=sam3_result,
        )
        return app_state.build_visualize_blob_payload(
            source_type="upload",
            source=file.filename or "upload",
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
            sam3_result=sam3_result,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-url-visualize-link", tags=["production"])
async def evaluate_url_visualize_link(payload: EvaluateVisualizeRequest):
    unavailable = _require_scoring_models()
    if unavailable:
        return unavailable

    try:
        env_key = app_state.normalize_env(payload.env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        img = _load_image_from_url(payload.url)
        yolo_result, unet_result, scoring, dirty_region_candidates, visual_review, sam3_result = (
            app_state.evaluate_image_with_visual_review(
                img,
                env_key,
                source=f"/evaluate-url-visualize-link:{payload.url}",
            )
        )
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
            visual_review=visual_review,
            dirty_region_candidates=dirty_region_candidates,
            sam3_result=sam3_result,
        )
        return app_state.build_visualize_blob_payload(
            source_type="url",
            source=payload.url,
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
            sam3_result=sam3_result,
        )
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/check", tags=["production"])
async def check_sam3(
    image: UploadFile = File(...),
    classes: str = Form(default="Garbage, Trash, Debris, Stain, Wet_Floor"),
    threshold: float = Form(default=0.3),
    resolution: int = Form(default=1008),
):
    if not app_state.SAM3_DETECTOR.enabled:
        return JSONResponse(status_code=503, content={"error": "SAM3 inference is disabled."})
    if not app_state.SAM3_DETECTOR.loaded:
        return JSONResponse(status_code=503, content={"error": "SAM3 model is not loaded."})
    if app_state.settings.sam3_provider == "local" and resolution != app_state.settings.sam3_resolution:
        return JSONResponse(
            status_code=400,
            content={"error": f"SAM3 resolution must be {app_state.settings.sam3_resolution} for this service."},
        )

    try:
        content = await image.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = _load_image_from_bytes(content)
        sam3_result = app_state.detect_dirty_with_sam3(img, prompts=classes, threshold=threshold)
        rgb = np.array(img.convert("RGB"))
        empty_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        coverage_summary = merge_unet_and_sam3_masks(
            empty_mask,
            sam3_result,
            dirty_labels=app_state.settings.roboflow_dirty_labels,
            wet_labels=app_state.settings.roboflow_wet_labels,
        )
        merged_mask = coverage_summary.pop("merged_mask")
        scoring = {
            "verdict": "SAM3",
            "quality_score": max(0.0, 100.0 - float(coverage_summary["combined_dirty_coverage_pct"])),
            "base_clean_score": max(0.0, 100.0 - float(coverage_summary["combined_dirty_coverage_pct"])),
            "object_penalty": 0.0,
            "penalty_detections_count": 0,
            "penalty_detection_indexes": [],
            **coverage_summary,
        }
        rendered = app_state.render_hybrid_overlay(
            rgb=rgb,
            pred_original_size=merged_mask,
            yolo_result={"detections_count": 0, "results": []},
            scoring=scoring,
            env_key="SAM3_CHECK",
            visual_review={},
            dirty_region_candidates=sam3_result.get("regions", []),
            sam3_result=sam3_result,
        )
        payload = app_state.build_visualize_blob_payload(
            source_type="upload",
            source=image.filename or "sam3-check",
            env_key="SAM3_CHECK",
            yolo_result={"detections_count": 0, "results": []},
            unet_result={"summary": {"total_dirty_coverage_pct": 0.0}},
            scoring=scoring,
            rendered=rendered,
            sam3_result=sam3_result,
        )
        return {
            "predictions": sam3_result["predictions"],
            "outputs": {
                "overlay_url": payload["visualization"]["url"],
            },
            "sam3": app_state.public_sam3_result(sam3_result),
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})
