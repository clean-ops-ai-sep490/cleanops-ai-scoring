from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DetectionPayload = dict[str, Any]
BBoxPayload = dict[str, float]


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


async def evaluate_ppe_payload(
    image_urls: list[str],
    required_objects: list[str],
    model: YOLO,
    timeout_sec: int,
    min_confidence: float,
    llm_filter: Any | None = None,
    batch_concurrency: int = 2,
    allowed_labels: list[str] | None = None,
) -> dict[str, Any]:
    aggregated_confidences: dict[str, float] = {}
    detected_items: list[dict[str, Any]] = []
    failed_images: list[dict[str, Any]] = []
    normalized_required_objects = [
        item.strip().lower()
        for item in required_objects
        if item and item.strip()
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
                if llm_filter is not None:
                    should_verify, skip_reason = llm_filter.should_verify_ppe(
                        required_objects=normalized_required_objects,
                        detected_items=per_image_items,
                        min_confidence=min_confidence,
                    )
                    if should_verify:
                        per_image_items = await asyncio.to_thread(
                            llm_filter.refine_ppe_detected_items,
                            image,
                            required_objects=normalized_required_objects,
                            detected_items=per_image_items,
                            allowed_labels=allowed_labels or [],
                            min_confidence=min_confidence,
                            source=f"ppe[{image_index}]",
                        )
                    else:
                        llm_filter.mark_skip("ppe_verification", f"ppe[{image_index}]", skip_reason or "cv_confident")

                return {
                    "image_url": image_url,
                    "image_index": image_index,
                    "detected_items": per_image_items,
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
    }
    if failed_images:
        response["failed_images"] = failed_images

    return response
