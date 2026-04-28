from __future__ import annotations

import ast
import base64
import io
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import requests
from PIL import Image

ADVISORY_OBJECT_LABELS = {"trash", "debris", "tool", "foreign_object"}
ADVISORY_DIRTY_LABELS = {"stain", "wet_area", "dust_patch", "dirty_zone"}

YOLO_VERIFY_SYSTEM_PROMPT = (
    "You are an image-quality observer that verifies computer-vision detections.\n"
    "Your role is to improve reliability, not to invent new labels or new scoring rules.\n"
    "You must follow the provided label whitelist exactly.\n"
    "Keep only visible trash-like cleanliness evidence such as metal, paper, plastic, trash, marks, debris, or waste.\n"
    "If a detection is actually a toilet, sink, fixture, furniture, background, floor pattern, or scene structure, reject it.\n"
    "Return strict JSON only."
)

DIRTY_VERIFY_SYSTEM_PROMPT = (
    "You are an image-quality observer that verifies dirty-area evidence.\n"
    "You must use the provided U-Net summary and dirty-region candidates as grounding.\n"
    "You may propose small guarded adjustments and approximate advisory dirty boxes only when the image clearly supports them.\n"
    "Do not invent masks or free-form shapes. Return strict JSON only."
)

PPE_VERIFY_SYSTEM_PROMPT = (
    "You are a PPE compliance observer that verifies model detections.\n"
    "You must only use labels from the provided PPE model label whitelist.\n"
    "Remove detections that are not visibly present.\n"
    "You may add clearly visible missing PPE items, but only with allowed labels.\n"
    "Return strict JSON only."
)

SCORING_VERIFY_SYSTEM_PROMPT = (
    "You are a quota-aware image-quality observer that verifies computer-vision evidence for cleanliness scoring.\n"
    "You must improve reliability without inventing new scoring rules.\n"
    "Use the provided YOLO label whitelist and dirty-region candidates as grounding.\n"
    "Keep only visible trash-like cleanliness evidence such as metal, paper, plastic, trash, marks, debris, or waste.\n"
    "Reject detections that are actually a toilet, sink, fixture, furniture, background, floor pattern, or scene structure.\n"
    "Only return the minimal JSON fields requested.\n"
    "Do not include markdown, explanations, comments, or extra keys.\n"
    "Return strict JSON only."
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_json_text(raw: str) -> str:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    raise ValueError("Gemini response did not contain a JSON object.")


def _loads_json_with_repairs(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", raw)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            literal_ready = repaired.replace("null", "None").replace("true", "True").replace("false", "False")
            parsed = ast.literal_eval(literal_ready)

    if not isinstance(parsed, dict):
        raise ValueError("Gemini response JSON root must be an object.")
    return parsed


def _preview_raw_response(raw: str | None, *, limit: int = 280) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text)
    return compact[:limit]


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(item)
    return ordered


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sanitize_bbox_norm(raw_bbox: Any) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(item) for item in raw_bbox]
    except (TypeError, ValueError):
        return None

    x1 = _clamp(x1, 0.0, 1.0)
    y1 = _clamp(y1, 0.0, 1.0)
    x2 = _clamp(x2, 0.0, 1.0)
    y2 = _clamp(y2, 0.0, 1.0)
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    area = width * height
    if area < 0.003 or area > 0.75:
        return None
    if width > 0.98 or height > 0.98:
        return None

    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _bbox_norm_to_px(bbox_norm: Sequence[float], image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    x1 = int(round(float(bbox_norm[0]) * width))
    y1 = int(round(float(bbox_norm[1]) * height))
    x2 = int(round(float(bbox_norm[2]) * width))
    y2 = int(round(float(bbox_norm[3]) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


@dataclass(frozen=True)
class GeminiFilterConfig:
    enabled: bool
    mode: str
    model: str
    timeout_sec: int
    batch_concurrency: int
    queue_enabled: bool
    queue_mode: str
    deadline_sec: int
    retry_429_max_retries: int
    retry_5xx_max_retries: int
    cooldown_sec: int
    enable_borderline_only: bool
    scoring_pass_window: float
    ppe_verify_on_missing_only: bool
    retry_initial_delay_ms: int
    retry_max_delay_ms: int
    retryable_status_codes: tuple[int, ...]
    max_image_dimension: int
    jpeg_quality: int
    api_key: str
    base_url: str


@dataclass
class _QueueJob:
    url: str
    body: dict[str, Any]
    kind: str
    source: str
    submitted_at: float
    deadline_at: float
    result_event: threading.Event = field(default_factory=threading.Event)
    result_payload: dict[str, Any] | None = None
    result_status: str = "queued"
    result_error: str | None = None
    attempts: int = 0


class GeminiLLMFilter:
    def __init__(self, config: GeminiFilterConfig, logger) -> None:
        self._config = config
        self._logger = logger
        self._last_error_lock = threading.Lock()
        self._last_error: str | None = None
        self._last_result: str | None = None
        self._trace_lock = threading.Lock()
        self._request_traces: dict[tuple[str, str], dict[str, Any]] = {}
        self._queue_depth = 0
        self._queue_depth_lock = threading.Lock()
        self._job_queue: queue.Queue[_QueueJob] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._cooldown_lock = threading.Lock()
        self._cooldown_until_monotonic = 0.0
        self._cooldown_until_epoch: float | None = None
        self._last_429_at_epoch: float | None = None
        self._calls_lock = threading.Lock()
        self._calls_saved = 0

        if self.enabled and self._queue_active:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="gemini-llm-filter-worker",
                daemon=True,
            )
            self._worker_thread.start()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self._config.api_key.strip())

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def _queue_active(self) -> bool:
        return self._config.queue_enabled and self._config.queue_mode == "global_fifo"

    def status_payload(self) -> dict[str, Any]:
        return {
            "llm_filter_enabled": self.enabled,
            "llm_filter_configured": self.configured,
            "llm_filter_mode": self._config.mode,
            "llm_filter_model": self.model,
            "llm_filter_last_error": self._get_last_error(),
            "llm_filter_last_result": self._get_last_result(),
            "llm_filter_queue_mode": self._config.queue_mode if self._queue_active else "disabled",
            "llm_filter_deadline_sec": self._config.deadline_sec,
            "llm_filter_queue_depth": self._get_queue_depth(),
            "llm_filter_cooldown_until": self._get_cooldown_until_epoch(),
            "llm_filter_last_429_at": self._get_last_429_at_epoch(),
            "llm_filter_calls_saved": self._get_calls_saved(),
        }

    def response_metadata(self, source: str, kinds: Sequence[str]) -> dict[str, Any]:
        stages = {
            kind: self._get_trace(kind, source)
            for kind in kinds
            if self._get_trace(kind, source) is not None
        }
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "mode": self._config.mode,
            "model": self.model,
            "queue_mode": self._config.queue_mode if self._queue_active else "disabled",
            "deadline_sec": self._config.deadline_sec,
            "cooldown_active": self._is_cooldown_active(),
            "stages": stages,
        }

    def _empty_scoring_review(self, *, skip_reason: str | None = None) -> dict[str, Any]:
        return {
            "keep_detection_indexes": [],
            "highlight_dirty_region_ids": [],
            "dirty_region_labels": [],
            "advisory_object_boxes": [],
            "advisory_dirty_boxes": [],
            "overlay_summary": "",
            "reasons": [],
            "confidence_note": "",
            "skip_reason": skip_reason,
        }

    def mark_skip(self, kind: str, source: str, reason: str) -> None:
        self._increment_calls_saved()
        self._set_last_result("skipped")
        self._set_last_error(reason)
        self._set_trace(kind, source, status="skipped", error=reason, attempts=0, response_preview=None)
        self._logger.info("llm_filter=skipped kind=%s source=%s reason=%s model=%s", kind, source, reason, self.model)

    def should_verify_scoring(
        self,
        *,
        yolo_result: dict[str, Any],
        unet_summary: dict[str, Any],
        scoring: dict[str, Any],
        pending_lower_bound: float,
    ) -> tuple[bool, str | None]:
        if self._is_cooldown_active():
            return False, "429_circuit_open"
        if self._config.mode != "quota_saver" or not self._config.enable_borderline_only:
            return True, None

        detections_count = _safe_int(yolo_result.get("detections_count"))
        quality_score = _safe_float(scoring.get("quality_score"))
        pass_threshold = _safe_float(scoring.get("pass_threshold"))
        total_dirty = _safe_float(unet_summary.get("total_dirty_coverage_pct"))
        window = max(1.0, float(self._config.scoring_pass_window))
        clean_skip_threshold = min(100.0, pass_threshold + max(3.0, window * 0.35))

        if detections_count > 0:
            return True, None
        if quality_score >= clean_skip_threshold and total_dirty <= 8.0:
            return False, "cv_confident"
        if quality_score <= max(0.0, pending_lower_bound - window) and total_dirty >= 25.0:
            return False, "cv_confident"
        return True, None

    def should_verify_ppe(
        self,
        *,
        required_objects: Sequence[str],
        detected_items: Sequence[dict[str, Any]],
        min_confidence: float,
    ) -> tuple[bool, str | None]:
        if self._is_cooldown_active():
            return False, "429_circuit_open"
        if self._config.mode != "quota_saver" or not self._config.ppe_verify_on_missing_only:
            return True, None

        normalized_required = {
            str(item).strip().lower()
            for item in required_objects
            if str(item).strip()
        }
        detected_names = {
            str(item.get("name", "")).strip().lower()
            for item in detected_items
            if str(item.get("name", "")).strip()
        }
        confidence_floor = _safe_float(min_confidence)
        confidence_floor = confidence_floor * 100.0 if confidence_floor <= 1 else confidence_floor
        low_confidence = any(_safe_float(item.get("confidence")) < max(60.0, confidence_floor) for item in detected_items)
        unexpected_items = any(name not in normalized_required for name in detected_names)
        missing_required = any(item not in detected_names for item in normalized_required)
        if missing_required or low_confidence or unexpected_items:
            return True, None
        return False, "cv_confident"

    def verify_scoring_evidence(
        self,
        image: Image.Image,
        *,
        env_key: str,
        yolo_result: dict[str, Any],
        unet_summary: dict[str, Any],
        dirty_region_candidates: Sequence[dict[str, Any]],
        scoring: dict[str, Any],
        pending_lower_bound: float,
        allowed_labels: Sequence[str],
        label_to_id: dict[str, int] | None,
        source: str,
        visualize_enhanced: bool = False,
    ) -> dict[str, Any]:
        should_verify, skip_reason = self.should_verify_scoring(
            yolo_result=yolo_result,
            unet_summary=unet_summary,
            scoring=scoring,
            pending_lower_bound=pending_lower_bound,
        )
        if not should_verify:
            self.mark_skip("scoring_verification", source, skip_reason or "cv_confident")
            return {
                "yolo": yolo_result,
                "summary": unet_summary,
                "review": self._empty_scoring_review(skip_reason=skip_reason or "cv_confident"),
            }

        allowed_label_list = [str(item).strip().lower() for item in allowed_labels if str(item).strip()]
        allowed_label_set = set(allowed_label_list)
        label_to_id = label_to_id or {}
        prompt_payload = {
            "environment": env_key,
            "pass_threshold": scoring.get("pass_threshold"),
            "pending_lower_bound": pending_lower_bound,
            "allowed_labels": allowed_label_list,
            "detections": [
                {
                    "index": idx,
                    "class_name": str(item.get("class_name", "")).strip().lower(),
                    "confidence": round(_safe_float(item.get("confidence")), 3),
                    "bbox": item.get("bbox", []),
                }
                for idx, item in enumerate(yolo_result.get("results", []))
            ],
            "dirty_summary": {
                "stain_or_water_coverage_pct": unet_summary.get("stain_or_water_coverage_pct"),
                "wet_surface_coverage_pct": unet_summary.get("wet_surface_coverage_pct"),
                "total_dirty_coverage_pct": unet_summary.get("total_dirty_coverage_pct"),
            },
            "dirty_region_candidates": [
                {
                    "region_id": item.get("region_id"),
                    "kind_hint": item.get("kind_hint"),
                    "bbox_norm": item.get("bbox_norm"),
                    "area_pct": item.get("area_pct"),
                    "centroid_norm": item.get("centroid_norm"),
                }
                for item in dirty_region_candidates
            ],
        }
        if visualize_enhanced:
            prompt = (
                "Verify the object and dirty-area evidence used for cleanliness scoring, and improve the final visualization.\n"
                "Return exactly one JSON object with only these keys:\n"
                '{"verified_detection_indexes":[],"highlight_dirty_region_ids":[],"stain_delta_pct":0.0,"wet_delta_pct":0.0,'
                '"advisory_dirty_boxes":[],"overlay_summary":"","reasons":[],"confidence_note":""}\n'
                "Rules:\n"
                "- Keep only detections that clearly match allowed labels.\n"
                "- Keep only visible trash-like cleanliness evidence: metal, paper, plastic, trash, marks, debris, or waste.\n"
                "- Reject toilet, sink, fixture, furniture, architecture, floor pattern, and background false positives.\n"
                "- highlight_dirty_region_ids must only reference provided region IDs.\n"
                "- stain_delta_pct and wet_delta_pct must be small guarded corrections, usually between -5 and 5.\n"
                "- You may add at most 2 advisory dirty boxes only when the image visibly supports them.\n"
                "- advisory dirty labels: stain, wet_area, dust_patch, dirty_zone.\n"
                "- advisory boxes must be approximate rectangles in normalized coordinates [x1,y1,x2,y2].\n"
                "- overlay_summary must be one short sentence for the panel note.\n"
                "- If unsure, return empty arrays and zero deltas.\n"
                f"Evidence JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
            )
        else:
            prompt = (
                "Verify both object detections and dirty-area evidence used for cleanliness scoring.\n"
                "Return exactly one JSON object with only these keys:\n"
                '{"verified_detection_indexes":[],"highlight_dirty_region_ids":[],"stain_delta_pct":0.0,"wet_delta_pct":0.0,"reasons":[],"confidence_note":""}\n'
                "Rules:\n"
                "- Keep only detections that clearly match allowed labels.\n"
                "- Keep only visible trash-like cleanliness evidence: metal, paper, plastic, trash, marks, debris, or waste.\n"
                "- Reject toilet, sink, fixture, furniture, architecture, floor pattern, and background false positives.\n"
                "- highlight_dirty_region_ids must only reference provided region IDs.\n"
                "- stain_delta_pct and wet_delta_pct must be small corrections, usually between -5 and 5.\n"
                "- If unsure, return empty arrays and zero deltas.\n"
                f"Evidence JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
            )
        parsed = self._invoke_json(
            prompt,
            image,
            kind="scoring_verification",
            source=source,
            system_instruction=SCORING_VERIFY_SYSTEM_PROMPT,
            response_schema=self._build_scoring_verification_schema(visualize_enhanced=visualize_enhanced),
        )
        if parsed is None:
            return {
                "yolo": yolo_result,
                "summary": unet_summary,
                "review": self._empty_scoring_review(),
            }

        raw_indexes = parsed.get("verified_detection_indexes")
        keep_indexes = [
            idx
            for idx in (raw_indexes if isinstance(raw_indexes, list) else [])
            if isinstance(idx, int) and 0 <= idx < len(yolo_result.get("results", []))
        ]
        refined_yolo = self._apply_yolo_verification(
            parsed,
            image=image,
            yolo_result=yolo_result,
            allowed_label_set=allowed_label_set,
            label_to_id=label_to_id,
            source=source,
            trace_kind="scoring_verification",
        )
        refined_dirty = self._apply_dirty_verification(
            parsed,
            image=image,
            summary=unet_summary,
            dirty_region_candidates=dirty_region_candidates,
            source=source,
            trace_kind="scoring_verification",
        )
        review = {
            **refined_dirty["review"],
            "keep_detection_indexes": keep_indexes,
            "advisory_object_boxes": [],
            "overlay_summary": refined_dirty["review"].get("overlay_summary", "")[:140],
            "skip_reason": None,
        }
        return {
            "yolo": refined_yolo,
            "summary": refined_dirty["summary"],
            "review": review,
        }

    def refine_yolo_result(
        self,
        image: Image.Image,
        yolo_result: dict[str, Any],
        *,
        allowed_labels: Sequence[str] | None = None,
        label_to_id: dict[str, int] | None = None,
        source: str,
    ) -> dict[str, Any]:
        detections = list(yolo_result.get("results", []))
        allowed_label_list = [str(item).strip().lower() for item in (allowed_labels or []) if str(item).strip()]
        allowed_label_set = set(allowed_label_list)
        label_to_id = label_to_id or {}
        if not detections and not allowed_label_set:
            return yolo_result

        raw_payload = {
            "allowed_labels": allowed_label_list,
            "detections": [
                {
                    "index": idx,
                    "class_name": str(item.get("class_name", "")),
                    "confidence": round(_safe_float(item.get("confidence")), 3),
                    "bbox": item.get("bbox", []),
                }
                for idx, item in enumerate(detections)
            ],
        }
        prompt = (
            "Review the image and verify object detections used for cleanliness scoring.\n"
            "Keep detections only when the object visibly matches one of the allowed model labels.\n"
            "Keep only visible trash-like cleanliness evidence: metal, paper, plastic, trash, marks, debris, or waste.\n"
            "If a detected object is actually a toilet, sink, fixture, furniture, background, wall, floor pattern, or scene structure, reject it.\n"
            "You may add at most 2 advisory object boxes only when the image clearly shows a missed object belonging to the allowed labels.\n"
            "Return JSON only with this shape:\n"
            '{"verified_detection_indexes":[0,1],"advisory_object_boxes":[{"label":"trash","confidence":0.85,"bbox_norm":[0.1,0.2,0.3,0.4],"reason":"short"}],"reasons":["short"],"confidence_note":"short"}\n'
            f"Raw detections JSON:\n{json.dumps(raw_payload, ensure_ascii=False)}"
        )
        parsed = self._invoke_json(
            prompt,
            image,
            kind="yolo_verification",
            source=source,
            system_instruction=YOLO_VERIFY_SYSTEM_PROMPT,
            response_schema=self._build_yolo_verification_schema(),
        )
        if parsed is None:
            return yolo_result
        return self._apply_yolo_verification(
            parsed,
            image=image,
            yolo_result=yolo_result,
            allowed_label_set=allowed_label_set,
            label_to_id=label_to_id,
            source=source,
            trace_kind="yolo_verification",
        )

    def verify_dirty_evidence(
        self,
        image: Image.Image,
        summary: dict[str, Any],
        *,
        dirty_region_candidates: Sequence[dict[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        raw_stain = _safe_float(summary.get("stain_or_water_coverage_pct"))
        raw_wet = _safe_float(summary.get("wet_surface_coverage_pct"))
        raw_total = _safe_float(summary.get("total_dirty_coverage_pct"))
        prompt_payload = {
            "summary": summary,
            "dirty_region_candidates": [
                {
                    "region_id": item.get("region_id"),
                    "kind_hint": item.get("kind_hint"),
                    "bbox_norm": item.get("bbox_norm"),
                    "area_pct": item.get("area_pct"),
                    "centroid_norm": item.get("centroid_norm"),
                }
                for item in dirty_region_candidates
            ],
        }
        prompt = (
            "Review the image and verify dirty-area evidence used for cleanliness scoring.\n"
            "Use the provided region candidates as grounding.\n"
            "You may highlight the credible dirty regions, propose small guarded delta adjustments, and add at most 3 advisory dirty boxes when the image clearly shows missed dirty areas.\n"
            "Return JSON only with this shape:\n"
            '{"highlight_dirty_region_ids":[1],"dirty_region_labels":[{"region_id":1,"label":"wet_area"}],"stain_delta_pct":0.0,"wet_delta_pct":0.0,"advisory_dirty_boxes":[{"label":"stain","confidence":0.8,"bbox_norm":[0.2,0.4,0.5,0.7],"reason":"short"}],"reasons":["short"],"confidence_note":"short"}\n'
            f"Raw dirty evidence JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
        )
        parsed = self._invoke_json(
            prompt,
            image,
            kind="dirty_verification",
            source=source,
            system_instruction=DIRTY_VERIFY_SYSTEM_PROMPT,
            response_schema=self._build_dirty_verification_schema(),
        )
        if parsed is None:
            return {
                "summary": summary,
                "review": {
                    "highlight_dirty_region_ids": [],
                    "dirty_region_labels": [],
                    "advisory_object_boxes": [],
                    "advisory_dirty_boxes": [],
                    "overlay_summary": "",
                    "reasons": [],
                    "confidence_note": "",
                },
            }
        return self._apply_dirty_verification(
            parsed,
            image=image,
            summary=summary,
            dirty_region_candidates=dirty_region_candidates,
            source=source,
            trace_kind="dirty_verification",
        )

    def _apply_yolo_verification(
        self,
        parsed: dict[str, Any],
        *,
        image: Image.Image,
        yolo_result: dict[str, Any],
        allowed_label_set: set[str],
        label_to_id: dict[str, int],
        source: str,
        trace_kind: str,
    ) -> dict[str, Any]:
        detections = list(yolo_result.get("results", []))
        keep_indexes_raw = parsed.get("verified_detection_indexes")
        if not isinstance(keep_indexes_raw, list):
            self._log_parse_error(trace_kind, source, "verified_detection_indexes missing or invalid")
            return yolo_result

        valid_indexes = {
            idx
            for idx in keep_indexes_raw
            if isinstance(idx, int) and 0 <= idx < len(detections)
        }
        kept_results = [dict(item) for idx, item in enumerate(detections) if idx in valid_indexes]
        advisory_items = self._sanitize_advisory_boxes(
            parsed.get("advisory_object_boxes", []),
            allowed_labels=allowed_label_set or {str(item.get("class_name", "")).strip().lower() for item in detections},
            max_items=2,
            image_size=image.size,
        )
        advisory_results = []
        for item in advisory_items:
            class_name = str(item["label"]).strip().lower()
            advisory_results.append(
                {
                    "class_name": class_name,
                    "class_id": label_to_id.get(class_name, -1),
                    "confidence": round(_safe_float(item.get("confidence")), 3),
                    "bbox": item.get("bbox_px", []),
                }
            )

        final_results = kept_results + advisory_results
        refined = {
            **yolo_result,
            "detections_count": len(final_results),
            "results": final_results,
        }
        self._logger.info(
            "llm_filter=applied kind=%s source=%s raw_count=%s refined_count=%s advisory_count=%s model=%s",
            trace_kind,
            source,
            len(detections),
            len(final_results),
            len(advisory_results),
            self.model,
        )
        self._set_last_result("success")
        self._clear_last_error()
        return refined

    def _apply_dirty_verification(
        self,
        parsed: dict[str, Any],
        *,
        image: Image.Image,
        summary: dict[str, Any],
        dirty_region_candidates: Sequence[dict[str, Any]],
        source: str,
        trace_kind: str,
    ) -> dict[str, Any]:
        raw_stain = _safe_float(summary.get("stain_or_water_coverage_pct"))
        raw_wet = _safe_float(summary.get("wet_surface_coverage_pct"))
        raw_total = _safe_float(summary.get("total_dirty_coverage_pct"))

        valid_region_ids = {
            _safe_int(item.get("region_id"))
            for item in dirty_region_candidates
            if isinstance(item, dict)
        }
        highlight_ids_raw = parsed.get("highlight_dirty_region_ids")
        highlight_dirty_region_ids = [
            idx
            for idx in (highlight_ids_raw if isinstance(highlight_ids_raw, list) else [])
            if isinstance(idx, int) and idx in valid_region_ids
        ]

        dirty_region_labels: list[dict[str, Any]] = []
        dirty_region_labels_raw = parsed.get("dirty_region_labels")
        for item in (dirty_region_labels_raw if isinstance(dirty_region_labels_raw, list) else []):
            if not isinstance(item, dict):
                continue
            region_id = _safe_int(item.get("region_id"))
            label = str(item.get("label", "")).strip().lower()
            if region_id not in valid_region_ids or label not in ADVISORY_DIRTY_LABELS:
                continue
            dirty_region_labels.append({"region_id": region_id, "label": label})

        advisory_dirty_boxes = self._sanitize_advisory_boxes(
            parsed.get("advisory_dirty_boxes", []),
            allowed_labels=ADVISORY_DIRTY_LABELS,
            max_items=3,
            image_size=image.size,
        )

        stain_delta = _clamp(_safe_float(parsed.get("stain_delta_pct")), -15.0, 15.0)
        wet_delta = _clamp(_safe_float(parsed.get("wet_delta_pct")), -15.0, 15.0)

        for item in advisory_dirty_boxes:
            bbox_norm = item.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
            area_pct = max(
                0.0,
                (float(bbox_norm[2]) - float(bbox_norm[0])) * (float(bbox_norm[3]) - float(bbox_norm[1])) * 100.0,
            )
            advisory_boost = min(6.0, round(area_pct * 0.35, 3))
            if str(item.get("label", "")).strip().lower() == "wet_area":
                wet_delta += advisory_boost
            else:
                stain_delta += advisory_boost

        adjusted_stain = _clamp(raw_stain + stain_delta, max(0.0, raw_stain - 15.0), min(100.0, raw_stain + 15.0))
        adjusted_wet = _clamp(raw_wet + wet_delta, max(0.0, raw_wet - 15.0), min(100.0, raw_wet + 15.0))
        adjusted_total = adjusted_stain + adjusted_wet
        max_total = min(100.0, raw_total + 15.0)
        min_total = max(0.0, raw_total - 15.0)

        if adjusted_total > 0 and adjusted_total > max_total:
            scale = max_total / adjusted_total
            adjusted_stain *= scale
            adjusted_wet *= scale
        elif adjusted_total < min_total:
            ratio_stain = 0.5 if adjusted_total <= 0 else adjusted_stain / adjusted_total
            adjusted_total = min_total
            adjusted_stain = adjusted_total * ratio_stain
            adjusted_wet = adjusted_total - adjusted_stain

        total_pixels = max(1, int(summary.get("model_input_size", 1)) ** 2)
        stain_pixels = int(round((adjusted_stain / 100.0) * total_pixels))
        wet_pixels = int(round((adjusted_wet / 100.0) * total_pixels))

        reasons = parsed.get("reasons")
        deduped_reasons = _dedupe_strings(reasons) if isinstance(reasons, list) else []
        confidence_note = str(parsed.get("confidence_note", "")).strip()[:120]
        overlay_summary = "; ".join([*deduped_reasons[:2], confidence_note]).strip("; ")

        refined = {
            **summary,
            "stain_or_water_pixels": stain_pixels,
            "wet_surface_pixels": wet_pixels,
            "stain_or_water_coverage_pct": round(adjusted_stain, 3),
            "wet_surface_coverage_pct": round(adjusted_wet, 3),
            "total_dirty_coverage_pct": round(adjusted_stain + adjusted_wet, 3),
        }
        review = {
            "highlight_dirty_region_ids": highlight_dirty_region_ids[:3],
            "dirty_region_labels": dirty_region_labels[:3],
            "advisory_object_boxes": [],
            "advisory_dirty_boxes": advisory_dirty_boxes,
            "overlay_summary": overlay_summary[:140],
            "reasons": deduped_reasons,
            "confidence_note": confidence_note,
        }
        self._logger.info(
            "llm_filter=applied kind=%s source=%s raw_total=%.3f refined_total=%.3f advisory_dirty=%s model=%s",
            trace_kind,
            source,
            raw_total,
            refined["total_dirty_coverage_pct"],
            len(advisory_dirty_boxes),
            self.model,
        )
        self._set_last_result("success")
        self._clear_last_error()
        return {
            "summary": refined,
            "review": review,
        }

    def refine_unet_summary(
        self,
        image: Image.Image,
        summary: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        verified = self.verify_dirty_evidence(
            image,
            summary,
            dirty_region_candidates=[],
            source=source,
        )
        return verified["summary"]

    def refine_scoring(
        self,
        image: Image.Image,
        *,
        env_key: str,
        yolo_result: dict[str, Any],
        unet_summary: dict[str, Any],
        scoring: dict[str, Any],
        pending_lower_bound: float,
        source: str,
    ) -> dict[str, Any]:
        prompt_payload = {
            "environment": env_key,
            "yolo": {
                "detections_count": yolo_result.get("detections_count", 0),
                "results": [
                    {
                        "class_name": item.get("class_name"),
                        "confidence": item.get("confidence"),
                    }
                    for item in yolo_result.get("results", [])
                ],
            },
            "unet": {
                "stain_or_water_coverage_pct": unet_summary.get("stain_or_water_coverage_pct"),
                "wet_surface_coverage_pct": unet_summary.get("wet_surface_coverage_pct"),
                "total_dirty_coverage_pct": unet_summary.get("total_dirty_coverage_pct"),
            },
            "scoring": scoring,
            "pending_lower_bound": pending_lower_bound,
        }
        prompt = (
            "You are reviewing the final cleanliness verdict for a graduation-project demo.\n"
            "Use the image plus raw CV evidence to decide whether the current score is trustworthy.\n"
            "Do not change the raw YOLO or U-Net evidence; only refine the final scoring block.\n"
            "Prefer keeping the original verdict unless the image clearly supports a different result.\n"
            "Return JSON only with this shape:\n"
            '{"verdict":"PASS|PENDING|FAIL","quality_score":0.0,"reasons":["short reason"]}\n'
            f"Raw payload JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
        )
        parsed = self._invoke_json(prompt, image, kind="scoring", source=source)
        if parsed is None:
            return scoring

        raw_quality = _safe_float(scoring.get("quality_score"))
        pass_threshold = _safe_float(scoring.get("pass_threshold"))
        verdict = str(parsed.get("verdict", scoring.get("verdict", "UNKNOWN"))).strip().upper()
        if verdict not in {"PASS", "PENDING", "FAIL"}:
            self._log_parse_error("scoring", source, "invalid verdict")
            return scoring

        adjusted_quality = _safe_float(parsed.get("quality_score"), raw_quality)
        adjusted_quality = _clamp(adjusted_quality, max(0.0, raw_quality - 25.0), min(100.0, raw_quality + 25.0))

        if verdict == "PASS":
            adjusted_quality = max(adjusted_quality, pass_threshold)
        elif verdict == "PENDING":
            upper = max(pending_lower_bound, pass_threshold - 0.001)
            adjusted_quality = _clamp(adjusted_quality, pending_lower_bound, upper)
            if adjusted_quality >= pass_threshold:
                adjusted_quality = max(pending_lower_bound, pass_threshold - 0.001)
        else:
            adjusted_quality = min(adjusted_quality, max(0.0, pending_lower_bound - 0.001))

        parsed_reasons = parsed.get("reasons")
        reasons = (
            _dedupe_strings(parsed_reasons)
            if isinstance(parsed_reasons, list)
            else _dedupe_strings(scoring.get("reasons", []))
        )
        if not reasons:
            reasons = _dedupe_strings(scoring.get("reasons", []))

        refined = {
            **scoring,
            "verdict": verdict,
            "quality_score": round(adjusted_quality, 3),
            "reasons": reasons,
        }
        self._logger.info(
            "llm_filter=applied kind=scoring source=%s raw_verdict=%s refined_verdict=%s model=%s",
            source,
            scoring.get("verdict"),
            verdict,
            self.model,
        )
        self._set_last_result("success")
        self._clear_last_error()
        return refined

    def review_visual_overlay(
        self,
        image: Image.Image,
        *,
        yolo_result: dict[str, Any],
        unet_summary: dict[str, Any],
        dirty_region_candidates: Sequence[dict[str, Any]],
        scoring: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        prompt_payload = {
            "yolo": {
                "detections": [
                    {
                        "index": idx,
                        "class_name": item.get("class_name"),
                        "confidence": item.get("confidence"),
                        "bbox": item.get("bbox"),
                    }
                    for idx, item in enumerate(yolo_result.get("results", []))
                ],
            },
            "unet": {
                "summary": unet_summary,
                "dirty_region_candidates": [
                    {
                        "region_id": item.get("region_id"),
                        "kind_hint": item.get("kind_hint"),
                        "bbox_norm": item.get("bbox_norm"),
                        "area_pct": item.get("area_pct"),
                        "centroid_norm": item.get("centroid_norm"),
                    }
                    for item in dirty_region_candidates
                ],
            },
            "scoring": scoring,
        }
        prompt = (
            "You are reviewing a cleaning-quality demo image for visualization overlay.\n"
            "Your job is to make the final visualization more trustworthy for a human reviewer.\n"
            "You must ONLY verify or reject existing computer-vision evidence, and you may add a small number of\n"
            "advisory approximate rectangles when the image clearly shows dirty areas or foreign objects that CV missed.\n"
            "Never invent polygons, never use full-image boxes, and never add more than 2 advisory object boxes or 3 advisory dirty boxes.\n"
            "Allowed advisory object labels: trash, debris, tool, foreign_object.\n"
            "Allowed advisory dirty labels: stain, wet_area, dust_patch, dirty_zone.\n"
            "Only add advisory boxes when the image visibly supports them. If unsure, return empty arrays.\n"
            "Return JSON only with this exact shape:\n"
            '{"keep_detection_indexes":[0],"highlight_dirty_region_ids":[1],"dirty_region_labels":[{"region_id":1,"label":"wet_area"}],'
            '"advisory_object_boxes":[{"label":"trash","confidence":0.82,"bbox_norm":[0.1,0.2,0.3,0.4],"reason":"short"}],'
            '"advisory_dirty_boxes":[{"label":"stain","confidence":0.8,"bbox_norm":[0.2,0.4,0.5,0.7],"reason":"short"}],'
            '"overlay_summary":"short summary"}\n'
            f"Raw payload JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
        )
        parsed = self._invoke_json(prompt, image, kind="visual", source=source)
        if parsed is None:
            return {
                "keep_detection_indexes": [],
                "highlight_dirty_region_ids": [],
                "dirty_region_labels": [],
                "advisory_object_boxes": [],
                "advisory_dirty_boxes": [],
                "overlay_summary": "",
            }

        detections = list(yolo_result.get("results", []))
        keep_indexes_raw = parsed.get("keep_detection_indexes")
        valid_detection_indexes = {
            idx
            for idx in (keep_indexes_raw if isinstance(keep_indexes_raw, list) else [])
            if isinstance(idx, int) and 0 <= idx < len(detections)
        }
        valid_region_ids = {
            _safe_int(item.get("region_id"))
            for item in dirty_region_candidates
            if isinstance(item, dict)
        }
        highlight_ids_raw = parsed.get("highlight_dirty_region_ids")
        highlight_dirty_region_ids = [
            idx
            for idx in (highlight_ids_raw if isinstance(highlight_ids_raw, list) else [])
            if isinstance(idx, int) and idx in valid_region_ids
        ]

        dirty_region_labels: list[dict[str, Any]] = []
        dirty_region_labels_raw = parsed.get("dirty_region_labels")
        for item in (dirty_region_labels_raw if isinstance(dirty_region_labels_raw, list) else []):
            if not isinstance(item, dict):
                continue
            region_id = _safe_int(item.get("region_id"))
            label = str(item.get("label", "")).strip().lower()
            if region_id not in valid_region_ids or label not in ADVISORY_DIRTY_LABELS:
                continue
            dirty_region_labels.append({"region_id": region_id, "label": label})

        image_size = image.size
        advisory_object_boxes = self._sanitize_advisory_boxes(
            parsed.get("advisory_object_boxes", []),
            allowed_labels=ADVISORY_OBJECT_LABELS,
            max_items=2,
            image_size=image_size,
        )
        advisory_dirty_boxes = self._sanitize_advisory_boxes(
            parsed.get("advisory_dirty_boxes", []),
            allowed_labels=ADVISORY_DIRTY_LABELS,
            max_items=3,
            image_size=image_size,
        )
        overlay_summary = str(parsed.get("overlay_summary", "")).strip()
        if len(overlay_summary) > 140:
            overlay_summary = overlay_summary[:140].rstrip()

        review = {
            "keep_detection_indexes": sorted(valid_detection_indexes),
            "highlight_dirty_region_ids": highlight_dirty_region_ids[:3],
            "dirty_region_labels": dirty_region_labels[:3],
            "advisory_object_boxes": advisory_object_boxes,
            "advisory_dirty_boxes": advisory_dirty_boxes,
            "overlay_summary": overlay_summary,
        }
        self._logger.info(
            "llm_filter=applied kind=visual source=%s kept_detections=%s highlight_regions=%s advisory_objects=%s advisory_dirty=%s model=%s",
            source,
            len(review["keep_detection_indexes"]),
            len(review["highlight_dirty_region_ids"]),
            len(review["advisory_object_boxes"]),
            len(review["advisory_dirty_boxes"]),
            self.model,
        )
        self._set_last_result("success")
        self._clear_last_error()
        return review

    def refine_ppe_detected_items(
        self,
        image: Image.Image,
        *,
        required_objects: Sequence[str],
        detected_items: list[dict[str, Any]],
        allowed_labels: Sequence[str] | None = None,
        min_confidence: float = 25.0,
        source: str,
    ) -> list[dict[str, Any]]:
        allowed_label_list = [str(item).strip().lower() for item in (allowed_labels or []) if str(item).strip()]
        allowed_label_set = set(allowed_label_list)
        if not detected_items and not required_objects:
            return detected_items

        prompt_payload = {
            "allowed_labels": allowed_label_list,
            "required_objects": [str(item).strip().lower() for item in required_objects],
            "detected_items": [
                {
                    "name": str(item.get("name", "")).strip().lower(),
                    "confidence": item.get("confidence"),
                }
                for item in detected_items
            ],
        }
        prompt = (
            "Review the image and verify PPE compliance detections.\n"
            "Keep objects only if they are visibly present and their labels belong to the allowed PPE labels.\n"
            "You may add clearly visible detector-missed PPE items only when they belong to the allowed label set.\n"
            "Do not list required-but-absent objects as additions.\n"
            "Return JSON only with this shape:\n"
            '{"present_objects":["helmet","gloves"],"visible_missed_objects":["face_shield"],"reasons":["short"],"confidence_note":"short"}\n'
            f"Raw payload JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
        )
        parsed = self._invoke_json(
            prompt,
            image,
            kind="ppe_verification",
            source=source,
            system_instruction=PPE_VERIFY_SYSTEM_PROMPT,
            response_schema=self._build_ppe_verification_schema(),
        )
        if parsed is None:
            return detected_items

        present_objects_raw = parsed.get("present_objects")
        if not isinstance(present_objects_raw, list):
            self._log_parse_error("ppe_verification", source, "present_objects missing or invalid")
            return detected_items

        visible_missed_raw = parsed.get("visible_missed_objects")
        visible_missed = [
            str(item).strip().lower()
            for item in (visible_missed_raw if isinstance(visible_missed_raw, list) else [])
            if str(item).strip()
        ]
        present_objects = {str(item).strip().lower() for item in present_objects_raw if str(item).strip()}
        visible_candidates = present_objects | set(visible_missed)
        refined: list[dict[str, Any]] = []
        for item in detected_items:
            label = str(item.get("name", "")).strip().lower()
            if label not in visible_candidates:
                continue
            refined_item = dict(item)
            refined_item["source"] = str(refined_item.get("source") or "detector")
            refined.append(refined_item)

        existing_names = {str(item.get("name", "")).strip().lower() for item in refined}
        normalized_required = [str(item).strip().lower() for item in required_objects if str(item).strip()]
        confidence_floor = _safe_float(min_confidence)
        confidence_floor = confidence_floor * 100.0 if confidence_floor <= 1 else confidence_floor
        synthetic_confidence = round(min(99.0, max(85.0, confidence_floor)), 1)
        image_index = _safe_int(refined[0].get("image_index")) if refined else _safe_int(detected_items[0].get("image_index")) if detected_items else 0
        for label in normalized_required:
            if label not in visible_candidates:
                continue
            if label in existing_names:
                continue
            if allowed_label_set and label not in allowed_label_set:
                continue
            refined.append(
                {
                    "name": label,
                    "confidence": synthetic_confidence,
                    "image_index": image_index,
                    "source": "filter",
                }
            )
            existing_names.add(label)
        self._logger.info(
            "llm_filter=applied kind=ppe_verification source=%s raw_count=%s refined_count=%s model=%s",
            source,
            len(detected_items),
            len(refined),
            self.model,
        )
        self._set_last_result("success")
        self._clear_last_error()
        return refined

    def _invoke_json(
        self,
        prompt: str,
        image: Image.Image,
        *,
        kind: str,
        source: str,
        system_instruction: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            self._set_trace(kind, source, status="disabled", error=None, attempts=0, response_preview=None)
            self._logger.debug("llm_filter=disabled kind=%s source=%s", kind, source)
            return None

        if not self.configured:
            message = "Gemini filter is enabled but GEMINI_API_KEY is missing."
            self._set_last_error(message)
            self._set_last_result("not_configured")
            self._set_trace(kind, source, status="not_configured", error=message, attempts=0, response_preview=None)
            self._logger.warning("llm_filter=non_retryable_fallback kind=%s source=%s reason=not_configured", kind, source)
            return None

        if self._is_cooldown_active():
            self.mark_skip(kind, source, "429_circuit_open")
            return None

        image_data = self._serialize_image(image)
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_data,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "maxOutputTokens": 384,
            },
        }
        if response_schema:
            body["generationConfig"]["responseJsonSchema"] = response_schema
        if system_instruction:
            body["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction},
                ]
            }
        self._logger.info(
            "llm_filter=prepared kind=%s source=%s prompt_chars=%s image_b64_chars=%s model=%s",
            kind,
            source,
            len(prompt),
            len(image_data),
            self.model,
        )
        url = f"{self._config.base_url.rstrip('/')}/models/{self.model}:generateContent"

        if self._queue_active:
            return self._enqueue_and_wait(url, body, kind=kind, source=source)
        return self._execute_inline(url, body, kind=kind, source=source)

    def _enqueue_and_wait(
        self,
        url: str,
        body: dict[str, Any],
        *,
        kind: str,
        source: str,
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        job = _QueueJob(
            url=url,
            body=body,
            kind=kind,
            source=source,
            submitted_at=now,
            deadline_at=now + self._config.deadline_sec,
        )
        queue_depth = self._increment_queue_depth()
        self._logger.info(
            "llm_filter=queued kind=%s source=%s queue_depth=%s deadline_sec=%s",
            kind,
            source,
            queue_depth,
            self._config.deadline_sec,
        )
        self._job_queue.put(job)
        remaining = max(0.0, job.deadline_at - time.monotonic() + 1.0)
        completed = job.result_event.wait(timeout=remaining)
        if not completed:
            self._set_last_error(f"timeout_wait:{kind}:{source}")
            self._set_last_result("expired")
            self._set_trace(kind, source, status="expired", error="caller_wait_timeout", attempts=job.attempts, response_preview=None)
            self._logger.warning("llm_filter=expired_fallback kind=%s source=%s detail=caller_wait_timeout", kind, source)
            return None
        return job.result_payload

    def _execute_inline(
        self,
        url: str,
        body: dict[str, Any],
        *,
        kind: str,
        source: str,
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        job = _QueueJob(
            url=url,
            body=body,
            kind=kind,
            source=source,
            submitted_at=now,
            deadline_at=now + self._config.deadline_sec,
        )
        result = self._process_job(job)
        job.result_event.set()
        return result

    def _worker_loop(self) -> None:
        while True:
            job = self._job_queue.get()
            try:
                job.result_payload = self._process_job(job)
            finally:
                job.result_event.set()
                self._decrement_queue_depth()
                self._job_queue.task_done()

    def _process_job(self, job: _QueueJob) -> dict[str, Any] | None:
        if time.monotonic() >= job.deadline_at:
            job.result_status = "expired"
            self._set_last_error(f"expired:{job.kind}:{job.source}")
            self._set_last_result("expired")
            self._logger.warning("llm_filter=expired_fallback kind=%s source=%s detail=deadline_before_start", job.kind, job.source)
            return None

        wait_ms = int(round((time.monotonic() - job.submitted_at) * 1000))
        self._logger.info(
            "llm_filter=started kind=%s source=%s queue_wait_ms=%s model=%s",
            job.kind,
            job.source,
            wait_ms,
            self.model,
        )

        delay_sec = max(0.1, self._config.retry_initial_delay_ms / 1000.0)
        max_delay_sec = max(delay_sec, self._config.retry_max_delay_ms / 1000.0)

        while True:
            job.attempts += 1
            raw_text: str | None = None
            try:
                payload = self._send_request(job.url, job.body)
                raw_text = self._extract_response_text(payload)
                parsed = _loads_json_with_repairs(_safe_json_text(raw_text))
                job.result_status = "success"
                self._clear_last_error()
                self._set_last_result("success")
                self._set_trace(
                    job.kind,
                    job.source,
                    status="success",
                    error=None,
                    attempts=job.attempts,
                    response_preview=self._preview_payload(parsed),
                )
                return parsed
            except requests.Timeout:
                retry_action = self._schedule_retry_or_expire(job, delay_sec, "timeout", None)
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in self._config.retryable_status_codes:
                    retry_after_sec = self._parse_retry_after(exc.response)
                    retry_action = self._schedule_retry_or_expire(job, delay_sec, f"http_{status_code}", retry_after_sec)
                else:
                    job.result_status = "non_retryable_error"
                    self._set_last_error(f"http_error:{status_code}:{job.kind}")
                    self._set_last_result("non_retryable_error")
                    self._set_trace(
                        job.kind,
                        job.source,
                        status="non_retryable_error",
                        error=f"http_{status_code}",
                        attempts=job.attempts,
                        response_preview=None,
                    )
                    self._logger.warning(
                        "llm_filter=non_retryable_fallback kind=%s source=%s status_code=%s",
                        job.kind,
                        job.source,
                        status_code,
                    )
                    return None
            except requests.RequestException as exc:
                retry_action = self._schedule_retry_or_expire(job, delay_sec, type(exc).__name__, None)
            except (ValueError, json.JSONDecodeError, SyntaxError) as exc:
                raw_preview = _preview_raw_response(raw_text)
                job.result_status = "non_retryable_error"
                self._set_last_error(f"parse_error:{job.kind}:{type(exc).__name__}")
                self._set_last_result("non_retryable_error")
                self._set_trace(
                    job.kind,
                    job.source,
                    status="non_retryable_error",
                    error=f"parse_{type(exc).__name__}",
                    attempts=job.attempts,
                    response_preview={"raw_response_preview": raw_preview} if raw_preview else None,
                )
                self._logger.warning(
                    "llm_filter=non_retryable_fallback kind=%s source=%s error=%s raw_response=%s",
                    job.kind,
                    job.source,
                    exc,
                    raw_preview,
                )
                return None
            except Exception as exc:  # noqa: BLE001
                retry_action = self._schedule_retry_or_expire(job, delay_sec, type(exc).__name__, None)

            if retry_action == "expired":
                return None

            delay_sec = min(delay_sec * 2, max_delay_sec)

    def _schedule_retry_or_expire(
        self,
        job: _QueueJob,
        delay_sec: float,
        reason: str,
        retry_after_sec: float | None,
    ) -> str:
        now = time.monotonic()
        remaining_sec = job.deadline_at - now
        max_attempts = self._max_attempts_for_reason(reason)
        if reason == "http_429":
            self._open_cooldown(max(retry_after_sec or 0.0, float(self._config.cooldown_sec)))
        if remaining_sec <= 0 or job.attempts >= max_attempts:
            job.result_status = "expired"
            self._set_last_error(f"expired:{job.kind}:{reason}")
            self._set_last_result("expired")
            self._set_trace(
                job.kind,
                job.source,
                status="expired",
                error=reason,
                attempts=job.attempts,
                response_preview=None,
            )
            self._logger.warning(
                "llm_filter=expired_fallback kind=%s source=%s attempts=%s reason=%s",
                job.kind,
                job.source,
                job.attempts,
                reason,
            )
            return "expired"

        sleep_sec = min(retry_after_sec or delay_sec, remaining_sec)
        self._logger.warning(
            "llm_filter=retrying kind=%s source=%s attempts=%s reason=%s sleep_ms=%s",
            job.kind,
            job.source,
            job.attempts,
            reason,
            int(round(sleep_sec * 1000)),
        )
        time.sleep(sleep_sec)
        return "retry"

    def _send_request(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            url,
            json=body,
            params={"key": self._config.api_key},
            timeout=self._config.timeout_sec,
        )
        response.raise_for_status()
        return response.json()

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Gemini response did not include candidates.")

        first = candidates[0]
        content = first.get("content")
        if not isinstance(content, dict):
            raise ValueError("Gemini candidate missing content.")

        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Gemini candidate missing parts.")

        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
        joined = "\n".join(text_parts).strip()
        if not joined:
            raise ValueError("Gemini candidate did not contain text.")
        return joined

    def _parse_retry_after(self, response: requests.Response | None) -> float | None:
        if response is None:
            return None
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _max_attempts_for_reason(self, reason: str) -> int:
        if reason == "http_429":
            return 1 + max(0, int(self._config.retry_429_max_retries))
        if reason.startswith("http_5") or reason == "timeout":
            return 1 + max(0, int(self._config.retry_5xx_max_retries))
        return 1

    def _open_cooldown(self, cooldown_sec: float) -> None:
        cooldown_value = max(0.0, float(cooldown_sec))
        now_epoch = time.time()
        now_monotonic = time.monotonic()
        with self._cooldown_lock:
            self._cooldown_until_monotonic = max(self._cooldown_until_monotonic, now_monotonic + cooldown_value)
            self._cooldown_until_epoch = max(self._cooldown_until_epoch or 0.0, now_epoch + cooldown_value)
            self._last_429_at_epoch = now_epoch

    def _is_cooldown_active(self) -> bool:
        with self._cooldown_lock:
            if self._cooldown_until_monotonic <= time.monotonic():
                self._cooldown_until_monotonic = 0.0
                self._cooldown_until_epoch = None
                return False
            return True

    def _get_cooldown_until_epoch(self) -> int | None:
        with self._cooldown_lock:
            if self._cooldown_until_monotonic <= time.monotonic():
                return None
            return int(round(self._cooldown_until_epoch or 0.0))

    def _get_last_429_at_epoch(self) -> int | None:
        with self._cooldown_lock:
            return int(round(self._last_429_at_epoch)) if self._last_429_at_epoch else None

    def _increment_calls_saved(self) -> None:
        with self._calls_lock:
            self._calls_saved += 1

    def _get_calls_saved(self) -> int:
        with self._calls_lock:
            return self._calls_saved

    def _serialize_image(self, image: Image.Image) -> str:
        rgb = image.convert("RGB")
        width, height = rgb.size
        max_dim = max(256, int(self._config.max_image_dimension))
        longest = max(width, height)
        if longest > max_dim:
            scale = max_dim / float(longest)
            resized = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            rgb = rgb.resize(resized)

        jpeg_quality = max(40, min(95, int(self._config.jpeg_quality)))
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _increment_queue_depth(self) -> int:
        with self._queue_depth_lock:
            self._queue_depth += 1
            return self._queue_depth

    def _decrement_queue_depth(self) -> int:
        with self._queue_depth_lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            return self._queue_depth

    def _get_queue_depth(self) -> int:
        with self._queue_depth_lock:
            return self._queue_depth

    def _get_last_error(self) -> str | None:
        with self._last_error_lock:
            return self._last_error

    def _set_last_error(self, message: str) -> None:
        with self._last_error_lock:
            self._last_error = message

    def _clear_last_error(self) -> None:
        with self._last_error_lock:
            self._last_error = None

    def _get_last_result(self) -> str | None:
        with self._last_error_lock:
            return self._last_result

    def _set_last_result(self, message: str) -> None:
        with self._last_error_lock:
            self._last_result = message

    def _log_parse_error(self, kind: str, source: str, detail: str) -> None:
        self._set_last_error(f"parse_error:{kind}:{detail}")
        self._set_last_result("non_retryable_error")
        self._set_trace(kind, source, status="non_retryable_error", error=f"parse_{detail}", attempts=0, response_preview=None)
        self._logger.warning("llm_filter=parse_error kind=%s source=%s detail=%s", kind, source, detail)

    def _set_trace(
        self,
        kind: str,
        source: str,
        *,
        status: str,
        error: str | None,
        attempts: int,
        response_preview: Any,
    ) -> None:
        payload = {
            "status": status,
            "error": error,
            "attempts": attempts,
            "model": self.model,
            "response_preview": response_preview,
        }
        with self._trace_lock:
            self._request_traces[(kind, source)] = payload

    def _get_trace(self, kind: str, source: str) -> dict[str, Any] | None:
        with self._trace_lock:
            trace = self._request_traces.get((kind, source))
            return dict(trace) if trace is not None else None

    def _preview_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        def _limit(value: Any) -> Any:
            if isinstance(value, dict):
                limited: dict[str, Any] = {}
                for idx, (key, item) in enumerate(value.items()):
                    if idx >= 8:
                        limited["_truncated"] = True
                        break
                    if item is None:
                        continue
                    limited[str(key)] = _limit(item)
                return limited
            if isinstance(value, list):
                items = [_limit(item) for item in value[:4]]
                if len(value) > 4:
                    items.append("...truncated...")
                return items
            if isinstance(value, str):
                return value[:160]
            return value

        return _limit(payload)

    def _build_yolo_verification_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "verified_detection_indexes": {"type": "array", "items": {"type": "integer"}},
                "advisory_object_boxes": self._build_advisory_box_array_schema(sorted(ADVISORY_OBJECT_LABELS)),
                "reasons": {"type": "array", "items": {"type": "string"}},
                "confidence_note": {"type": "string"},
            },
            "required": ["verified_detection_indexes", "advisory_object_boxes", "reasons", "confidence_note"],
        }

    def _build_dirty_verification_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "highlight_dirty_region_ids": {"type": "array", "items": {"type": "integer"}},
                "dirty_region_labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "region_id": {"type": "integer"},
                            "label": {"type": "string", "enum": sorted(ADVISORY_DIRTY_LABELS)},
                        },
                        "required": ["region_id", "label"],
                    },
                },
                "stain_delta_pct": {"type": "number"},
                "wet_delta_pct": {"type": "number"},
                "advisory_dirty_boxes": self._build_advisory_box_array_schema(sorted(ADVISORY_DIRTY_LABELS)),
                "reasons": {"type": "array", "items": {"type": "string"}},
                "confidence_note": {"type": "string"},
            },
            "required": [
                "highlight_dirty_region_ids",
                "dirty_region_labels",
                "stain_delta_pct",
                "wet_delta_pct",
                "advisory_dirty_boxes",
                "reasons",
                "confidence_note",
            ],
        }

    def _build_scoring_verification_schema(self, *, visualize_enhanced: bool = False) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "verified_detection_indexes": {"type": "array", "items": {"type": "integer"}},
            "highlight_dirty_region_ids": {"type": "array", "items": {"type": "integer"}},
            "stain_delta_pct": {"type": "number"},
            "wet_delta_pct": {"type": "number"},
            "reasons": {"type": "array", "items": {"type": "string"}},
            "confidence_note": {"type": "string"},
        }
        required = [
            "verified_detection_indexes",
            "highlight_dirty_region_ids",
            "stain_delta_pct",
            "wet_delta_pct",
            "reasons",
            "confidence_note",
        ]
        if visualize_enhanced:
            properties.update(
                {
                    "advisory_dirty_boxes": self._build_advisory_box_array_schema(sorted(ADVISORY_DIRTY_LABELS)),
                    "overlay_summary": {"type": "string"},
                }
            )
            required.extend(
                [
                    "advisory_dirty_boxes",
                    "overlay_summary",
                ]
            )
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _build_ppe_verification_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "present_objects": {"type": "array", "items": {"type": "string"}},
                "visible_missed_objects": {"type": "array", "items": {"type": "string"}},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "confidence_note": {"type": "string"},
            },
            "required": ["present_objects", "visible_missed_objects", "reasons", "confidence_note"],
            "additionalProperties": False,
        }

    def _build_advisory_box_array_schema(self, allowed_labels: Sequence[str]) -> dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(allowed_labels)},
                    "confidence": {"type": "number"},
                    "bbox_norm": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["label", "confidence", "bbox_norm", "reason"],
            },
        }

    def _sanitize_advisory_boxes(
        self,
        items: Any,
        *,
        allowed_labels: set[str],
        max_items: int,
        image_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []

        sanitized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().lower()
            confidence = _safe_float(item.get("confidence"))
            bbox_norm = _sanitize_bbox_norm(item.get("bbox_norm"))
            if label not in allowed_labels or confidence < 0.75 or bbox_norm is None:
                continue

            sanitized.append(
                {
                    "label": label,
                    "confidence": round(confidence, 3),
                    "bbox_norm": bbox_norm,
                    "bbox_px": _bbox_norm_to_px(bbox_norm, image_size),
                    "reason": str(item.get("reason", "")).strip()[:120],
                }
            )
            if len(sanitized) >= max_items:
                break
        return sanitized
