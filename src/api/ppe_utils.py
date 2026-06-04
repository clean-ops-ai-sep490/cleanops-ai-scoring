from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DetectionPayload = dict[str, Any]
BBoxPayload = dict[str, float]


@dataclass(frozen=True)
class GeminiPpeConfig:
    enabled: bool
    mode: str
    api_key: str | None
    model: str
    base_url: str
    timeout_sec: float


def normalize_confidence_threshold(min_confidence: float) -> float:
    return min_confidence * 100 if min_confidence <= 1 else min_confidence


def load_image_from_url(image_url: str, timeout_sec: int) -> Image.Image:
    response = requests.get(image_url, timeout=timeout_sec)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def serialize_bbox(box: Any) -> BBoxPayload:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return {
        "x1": round(float(x1), 1),
        "y1": round(float(y1), 1),
        "x2": round(float(x2), 1),
        "y2": round(float(y2), 1),
    }


def collect_filtered_detections(
    image: Image.Image,
    model: YOLO,
    min_confidence: float,
    image_index: int,
) -> list[DetectionPayload]:
    confidence_threshold = normalize_confidence_threshold(min_confidence)
    detections: list[DetectionPayload] = []

    results = model(image)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf) * 100
            if confidence < confidence_threshold:
                continue

            detections.append(
                {
                    "name": str(model.names[class_id]).lower(),
                    "confidence": round(confidence, 1),
                    "image_index": image_index,
                    "bbox": serialize_bbox(box),
                }
            )

    return detections


def summarize_detections(
    detections: list[DetectionPayload],
    image_index: int,
) -> tuple[dict[str, float], list[DetectionPayload]]:
    best_by_name: dict[str, DetectionPayload] = {}
    for detection in detections:
        class_name = str(detection["name"])
        confidence = float(detection["confidence"])
        current_best = best_by_name.get(class_name)
        if current_best is None or confidence > float(current_best["confidence"]):
            best_by_name[class_name] = {
                "name": class_name,
                "confidence": round(confidence, 1),
                "image_index": image_index,
                "source": "detector",
            }

    detected_dict = {
        str(item["name"]): float(item["confidence"])
        for item in best_by_name.values()
    }
    detected_list = sorted(best_by_name.values(), key=lambda item: str(item["name"]))
    return detected_dict, detected_list


def _normalize_item_name(item: object) -> str:
    return re.sub(r"[\s\-]+", "_", str(item or "").strip().lower()).strip("_")


def _image_to_inline_part(image: Image.Image) -> dict[str, Any]:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": encoded,
        }
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def verify_missing_items_with_gemini(
    images: list[Image.Image],
    missing_items: list[str],
    config: GeminiPpeConfig,
) -> dict[str, Any]:
    normalized_missing = [_normalize_item_name(item) for item in missing_items if _normalize_item_name(item)]
    if not normalized_missing:
        return {"status": "skipped", "reason": "no_missing_items", "confirmed_items": []}
    if not config.enabled:
        return {"status": "skipped", "reason": "disabled", "confirmed_items": []}
    if config.mode != "missing_only":
        return {"status": "skipped", "reason": f"unsupported_mode:{config.mode}", "confirmed_items": []}
    if not config.api_key:
        return {"status": "skipped", "reason": "api_key_missing", "confirmed_items": []}
    if not images:
        return {"status": "skipped", "reason": "no_images", "confirmed_items": []}

    prompt = (
        "You are verifying PPE compliance from images. The detector may have missed some required PPE items. "
        "Only confirm an item if it is clearly visible in at least one image. "
        "Return JSON only with keys present_items and notes. "
        f"Candidate missing items: {', '.join(normalized_missing)}."
    )
    parts: list[dict[str, Any]] = [{"text": prompt}]
    parts.extend(_image_to_inline_part(image) for image in images)
    payload = {"contents": [{"role": "user", "parts": parts}]}
    endpoint = f"{config.base_url.rstrip('/')}/models/{config.model}:generateContent"

    try:
        response = requests.post(
            endpoint,
            params={"key": config.api_key},
            json=payload,
            timeout=config.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        text_parts: list[str] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        raw_text = "\n".join(text_parts).strip()
        parsed = _extract_json_object(raw_text)
        present_items = parsed.get("present_items", [])
        if not isinstance(present_items, list):
            present_items = []

        confirmed = sorted(
            {
                _normalize_item_name(item)
                for item in present_items
                if _normalize_item_name(item) in set(normalized_missing)
            }
        )
        return {
            "status": "ok",
            "mode": config.mode,
            "confirmed_items": confirmed,
            "remaining_missing_items": [item for item in normalized_missing if item not in confirmed],
            "notes": str(parsed.get("notes", ""))[:500],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("PPE Gemini verification failed: %s", exc)
        return {
            "status": "error",
            "mode": config.mode,
            "confirmed_items": [],
            "remaining_missing_items": normalized_missing,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _gemini_deadline_exceeded_review(missing_items: list[str]) -> dict[str, Any]:
    normalized_missing = [_normalize_item_name(item) for item in missing_items if _normalize_item_name(item)]
    return {
        "status": "skipped",
        "reason": "deadline_exceeded",
        "confirmed_items": [],
        "remaining_missing_items": normalized_missing,
    }


async def evaluate_ppe_payload(
    image_urls: list[str],
    required_objects: list[str],
    model: YOLO,
    timeout_sec: int,
    min_confidence: float,
    batch_concurrency: int = 2,
    gemini_config: GeminiPpeConfig | None = None,
    gemini_deadline_sec: float = 30.0,
) -> dict[str, Any]:
    started_at = time.monotonic()
    aggregated_confidences: dict[str, float] = {}
    detected_items: list[dict[str, Any]] = []
    failed_images: list[dict[str, Any]] = []
    processed_images: list[Image.Image] = []
    normalized_required_objects = [
        _normalize_item_name(item)
        for item in required_objects
        if _normalize_item_name(item)
    ]
    semaphore = asyncio.Semaphore(max(1, batch_concurrency))

    async def process_image(image_index: int, image_url: str) -> dict[str, Any]:
        try:
            async with semaphore:
                image = await asyncio.to_thread(load_image_from_url, image_url, timeout_sec)
                detections = await asyncio.to_thread(
                    collect_filtered_detections,
                    image,
                    model,
                    min_confidence,
                    image_index,
                )
                _, per_image_items = summarize_detections(detections, image_index)
                return {
                    "image_url": image_url,
                    "image_index": image_index,
                    "detected_items": per_image_items,
                    "_image": image,
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process PPE image '%s': %s", image_url, exc)
            return {
                "image_url": image_url,
                "image_index": image_index,
                "error": str(exc),
            }

    tasks = [process_image(image_index, image_url) for image_index, image_url in enumerate(image_urls)]
    results = await asyncio.gather(*tasks)

    for result in results:
        if "error" in result:
            failed_images.append(result)
            continue

        per_image_items = result["detected_items"]
        processed_image = result.get("_image")
        if isinstance(processed_image, Image.Image):
            processed_images.append(processed_image)
        for item in per_image_items:
            label = str(item["name"])
            confidence = float(item["confidence"])
            previous_confidence = aggregated_confidences.get(label)
            if previous_confidence is None or confidence > previous_confidence:
                aggregated_confidences[label] = confidence

        detected_items.extend(per_image_items)

    missing_items = [
        required_item
        for required_item in normalized_required_objects
        if required_item not in aggregated_confidences
    ]
    gemini_review: dict[str, Any] = {"status": "skipped", "reason": "no_missing_items"}
    if missing_items:
        if gemini_config is None:
            gemini_review = {"status": "skipped", "reason": "not_configured", "confirmed_items": []}
        else:
            remaining_deadline_sec = float(gemini_deadline_sec) - (time.monotonic() - started_at)
            if remaining_deadline_sec <= 0:
                gemini_review = _gemini_deadline_exceeded_review(missing_items)
            else:
                gemini_timeout_sec = min(float(gemini_config.timeout_sec), remaining_deadline_sec)
                effective_gemini_config = replace(gemini_config, timeout_sec=gemini_timeout_sec)
                try:
                    gemini_review = await asyncio.wait_for(
                        asyncio.to_thread(
                            verify_missing_items_with_gemini,
                            processed_images,
                            missing_items,
                            effective_gemini_config,
                        ),
                        timeout=gemini_timeout_sec,
                    )
                except asyncio.TimeoutError:
                    gemini_review = _gemini_deadline_exceeded_review(missing_items)
            confirmed_items = [
                item for item in gemini_review.get("confirmed_items", []) if item in set(missing_items)
            ]
            for item in confirmed_items:
                aggregated_confidences[item] = 100.0
                detected_items.append(
                    {
                        "name": item,
                        "confidence": 100.0,
                        "image_index": None,
                        "source": "gemini",
                    }
                )
            missing_items = [
                required_item
                for required_item in normalized_required_objects
                if required_item not in aggregated_confidences
            ]
    status = "PASS" if not missing_items else "FAIL"
    message = (
        "All required PPE items detected"
        if status == "PASS"
        else f"Missing items: {', '.join(missing_items)}"
    )

    response: dict[str, Any] = {
        "status": status,
        "message": message,
        "detected_items": detected_items,
        "missing_items": missing_items,
        "gemini_review": gemini_review,
    }
    if failed_images:
        response["failed_images"] = failed_images

    return response
