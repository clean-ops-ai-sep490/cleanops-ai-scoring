from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


DEFAULT_SCORING_PENALTY_LABELS = (
    "metal",
    "paper",
    "plastic",
    "trash",
    "marks",
    "garbage",
    "rubbish",
    "litter",
    "waste",
    "debris",
    "bottle",
    "plastic_bottle",
    "can",
    "cup",
    "cardboard",
    "bag",
    "trash_bag",
)

DEFAULT_ROBOFLOW_DIRTY_LABELS = (
    "garbage",
    "stain",
    "stained_floor",
    "dirty_area",
    "marks",
    "trash",
    "debris",
)

DEFAULT_ROBOFLOW_WET_LABELS = (
    "wet_floor",
    "wet_surface",
    "water",
    "spill",
    "puddle",
)


def normalize_env(env: Optional[str], env_rules: Dict[str, Dict[str, object]]) -> str:
    env_key = (env or "LOBBY_CORRIDOR").strip().upper()
    if env_key not in env_rules:
        raise ValueError(
            f"Unsupported env '{env_key}'. Allowed envs: {', '.join(sorted(env_rules.keys()))}"
        )
    return env_key


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def normalize_detection_label(raw: object) -> str:
    label = str(raw or "").strip().lower()
    label = re.sub(r"[\s\-]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def normalize_penalty_labels(labels: Sequence[object] | None) -> List[str]:
    normalized = {
        normalize_detection_label(item)
        for item in (labels or DEFAULT_SCORING_PENALTY_LABELS)
        if normalize_detection_label(item)
    }
    return sorted(normalized)


def summarize_penalty_detections(
    detections: Sequence[Dict[str, Any]],
    penalty_labels: Sequence[object] | None,
) -> Dict[str, object]:
    penalty_label_set = set(normalize_penalty_labels(penalty_labels))
    penalty_detection_labels: List[str] = []
    ignored_detection_labels: List[str] = []
    penalty_detection_indexes: List[int] = []
    ignored_detection_indexes: List[int] = []

    for idx, detection in enumerate(detections):
        class_name = normalize_detection_label(detection.get("class_name", ""))
        if not class_name:
            continue

        if class_name in penalty_label_set:
            penalty_detection_labels.append(class_name)
            penalty_detection_indexes.append(idx)
        else:
            ignored_detection_labels.append(class_name)
            ignored_detection_indexes.append(idx)

    return {
        "penalty_detections_count": len(penalty_detection_labels),
        "ignored_detections_count": len(ignored_detection_labels),
        "penalty_detection_labels": sorted(set(penalty_detection_labels)),
        "ignored_detection_labels": sorted(set(ignored_detection_labels)),
        "penalty_detection_indexes": penalty_detection_indexes,
        "ignored_detection_indexes": ignored_detection_indexes,
    }


def combine_dirty_coverage(unet_dirty_coverage_pct: float, sam3_dirty_coverage_pct: float) -> Dict[str, object]:
    unet_pct = max(0.0, float(unet_dirty_coverage_pct or 0.0))
    sam3_pct = max(0.0, float(sam3_dirty_coverage_pct or 0.0))
    combined_pct = max(unet_pct, sam3_pct)
    if sam3_pct > unet_pct:
        source = "sam3"
    elif unet_pct > sam3_pct:
        source = "unet"
    else:
        source = "equal"

    return {
        "unet_dirty_coverage_pct": round(unet_pct, 3),
        "sam3_dirty_coverage_pct": round(sam3_pct, 3),
        "combined_dirty_coverage_pct": round(combined_pct, 3),
        "dirty_coverage_source": source,
    }


def score_image(
    total_dirty_coverage_pct: float,
    detections_count: int,
    env_key: str,
    env_rules: Dict[str, Dict[str, object]],
    pending_lower_bound: float,
    penalty_detections_count: Optional[int] = None,
    object_penalty_per_detection: float = 10.0,
    penalty_detection_labels: Optional[List[str]] = None,
    ignored_detection_labels: Optional[List[str]] = None,
    ignored_detections_count: Optional[int] = None,
    penalty_detection_indexes: Optional[List[int]] = None,
    ignored_detection_indexes: Optional[List[int]] = None,
) -> Dict[str, object]:
    base_clean_score = 100.0 - float(total_dirty_coverage_pct)
    scorable_detections_count = int(
        detections_count if penalty_detections_count is None else penalty_detections_count
    )
    penalty_weight = max(0.0, float(object_penalty_per_detection))
    object_penalty = min(40.0, float(scorable_detections_count) * penalty_weight)
    quality_score = clamp(base_clean_score - object_penalty, 0.0, 100.0)

    pass_threshold = float(env_rules[env_key]["pass_threshold"])
    if quality_score >= pass_threshold:
        verdict = "PASS"
    elif quality_score >= pending_lower_bound:
        verdict = "PENDING"
    else:
        verdict = "FAIL"

    reasons: List[str] = []
    if total_dirty_coverage_pct >= 20.0:
        reasons.append("coverage high")
    if scorable_detections_count > 0:
        reasons.append("trash-like objects remain")
    if not reasons:
        reasons.append("good cleanliness")

    return {
        "base_clean_score": round(base_clean_score, 3),
        "object_penalty": round(object_penalty, 3),
        "quality_score": round(quality_score, 3),
        "pass_threshold": pass_threshold,
        "verdict": verdict,
        "reasons": reasons,
        "penalty_detections_count": scorable_detections_count,
        "ignored_detections_count": int(ignored_detections_count or 0),
        "penalty_detection_labels": penalty_detection_labels or [],
        "ignored_detection_labels": ignored_detection_labels or [],
        "penalty_detection_indexes": penalty_detection_indexes or [],
        "ignored_detection_indexes": ignored_detection_indexes or [],
    }


def merge_unet_and_sam3_masks(
    unet_mask: np.ndarray,
    sam3_result: Dict[str, Any],
    *,
    dirty_labels: Sequence[object] | None = None,
    wet_labels: Sequence[object] | None = None,
) -> Dict[str, object]:
    merged = np.zeros_like(unet_mask, dtype=np.uint8)
    unet_dirty = np.asarray(unet_mask == 1, dtype=bool)
    unet_wet = np.asarray(unet_mask == 2, dtype=bool)
    merged[unet_dirty] = 1
    merged[unet_wet] = 2

    dirty_label_set = set(normalize_penalty_labels(dirty_labels or DEFAULT_ROBOFLOW_DIRTY_LABELS))
    wet_label_set = set(normalize_penalty_labels(wet_labels or DEFAULT_ROBOFLOW_WET_LABELS))
    label_masks = sam3_result.get("_label_masks") if isinstance(sam3_result, dict) else {}
    if not isinstance(label_masks, dict):
        label_masks = {}

    roboflow_dirty_mask = np.zeros_like(merged, dtype=bool)
    roboflow_wet_mask = np.zeros_like(merged, dtype=bool)
    class_counts = {"stain_or_water": 0, "wet_surface": 0, "ignored": 0}

    for raw_label, raw_mask in label_masks.items():
        if not isinstance(raw_mask, np.ndarray):
            continue
        if raw_mask.shape[:2] != merged.shape[:2]:
            continue
        label = normalize_detection_label(raw_label)
        mask = raw_mask.astype(bool)
        if label in wet_label_set:
            roboflow_wet_mask |= mask
            class_counts["wet_surface"] += 1
        elif label in dirty_label_set:
            roboflow_dirty_mask |= mask
            class_counts["stain_or_water"] += 1
        else:
            class_counts["ignored"] += 1

    merged[roboflow_dirty_mask] = 1
    merged[roboflow_wet_mask] = 2

    total_px = max(1, int(merged.size))
    unet_px = int(np.count_nonzero(unet_dirty | unet_wet))
    sam3_px = int(np.count_nonzero(roboflow_dirty_mask | roboflow_wet_mask))
    merged_px = int(np.count_nonzero(merged > 0))
    wet_px = int(np.count_nonzero(merged == 2))
    dirty_px = int(np.count_nonzero(merged == 1))

    sources: list[str] = []
    if unet_px > 0:
        sources.append("unet")
    if sam3_px > 0:
        sources.append("sam3")
    if len(sources) == 2:
        source = "merged"
    elif sources:
        source = sources[0]
    else:
        source = "equal"

    return {
        "merged_mask": merged,
        "unet_dirty_coverage_pct": round((unet_px / total_px) * 100.0, 3),
        "sam3_dirty_coverage_pct": round((sam3_px / total_px) * 100.0, 3),
        "combined_dirty_coverage_pct": round((merged_px / total_px) * 100.0, 3),
        "merged_dirty_coverage_pct": round((merged_px / total_px) * 100.0, 3),
        "merged_stain_or_water_coverage_pct": round((dirty_px / total_px) * 100.0, 3),
        "merged_wet_surface_coverage_pct": round((wet_px / total_px) * 100.0, 3),
        "dirty_coverage_source": source,
        "merged_dirty_sources": sources,
        "roboflow_label_class_counts": class_counts,
    }


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pending_quality(raw_quality: float, pass_threshold: float, pending_lower_bound: float) -> float:
    target = min(raw_quality, pass_threshold - 0.001)
    return round(clamp(target, pending_lower_bound, pass_threshold - 0.001), 3)


def calibrate_score(
    scoring: Dict[str, object],
    *,
    env_key: str,
    pending_lower_bound: float,
    enabled: bool = True,
    high_risk_review_envs: Sequence[object] = ("RESTROOM", "HOSPITAL_OR"),
    ignored_object_review_count: int = 2,
    ignored_object_review_labels: Sequence[object] = (),
    unet_only_review_min_pct: float = 20.0,
    unet_only_review_max_pct: float = 60.0,
    single_sam3_review_max_predictions: int = 1,
    strong_dirty_coverage_pct: float = 45.0,
) -> Dict[str, object]:
    calibrated = dict(scoring)
    raw_verdict = str(calibrated.get("verdict", "")).upper()
    raw_quality = _as_float(calibrated.get("quality_score"))
    pass_threshold = _as_float(calibrated.get("pass_threshold"), 90.0)
    penalty_count = _as_int(calibrated.get("penalty_detections_count"))
    ignored_count = _as_int(calibrated.get("ignored_detections_count"))
    unet_pct = _as_float(calibrated.get("unet_dirty_coverage_pct"))
    sam3_pct = _as_float(calibrated.get("sam3_dirty_coverage_pct"))
    combined_pct = _as_float(calibrated.get("combined_dirty_coverage_pct"))
    sam3_predictions = _as_int(calibrated.get("sam3_predictions_count"))
    source = str(calibrated.get("dirty_coverage_source", "")).lower()
    ignored_labels = {
        normalize_detection_label(item)
        for item in (calibrated.get("ignored_detection_labels") or [])
        if normalize_detection_label(item)
    }

    calibrated["raw_verdict"] = raw_verdict
    calibrated["raw_quality_score"] = round(raw_quality, 3)
    calibrated["calibrated"] = False
    calibrated["calibration_rules"] = []
    calibrated["calibration_reason"] = ""

    if not enabled:
        return calibrated

    high_risk_envs = {str(item).strip().upper() for item in high_risk_review_envs if str(item).strip()}
    ignored_review_labels = {
        normalize_detection_label(item)
        for item in ignored_object_review_labels
        if normalize_detection_label(item)
    }
    env_normalized = str(env_key or "").strip().upper()
    rules: list[str] = []

    strong_sources = 0
    if unet_pct >= strong_dirty_coverage_pct:
        strong_sources += 1
    if sam3_pct >= strong_dirty_coverage_pct:
        strong_sources += 1
    if penalty_count > 0:
        strong_sources += 1

    if combined_pct >= strong_dirty_coverage_pct and strong_sources >= 2:
        if raw_verdict == "FAIL" and penalty_count == 0 and combined_pct < 85.0:
            rules.append("coverage_only_fail_review")
            calibrated["verdict"] = "PENDING"
            calibrated["quality_score"] = round(max(raw_quality, pending_lower_bound), 3)
        elif raw_verdict != "FAIL":
            rules.append("strong_multi_source_dirty")
            calibrated["verdict"] = "FAIL"
            calibrated["quality_score"] = round(
                clamp(min(raw_quality, pending_lower_bound - 0.001), 0.0, 100.0),
                3,
            )
    elif (
        raw_verdict == "PASS"
        and env_normalized in high_risk_envs
        and combined_pct < 1.0
        and penalty_count == 0
        and sam3_predictions == 0
    ):
        rules.append("high_risk_weak_evidence_review")
        calibrated["verdict"] = "PENDING"
        calibrated["quality_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "PASS"
        and ignored_count >= max(1, int(ignored_object_review_count))
        and penalty_count == 0
        and (not ignored_review_labels or bool(ignored_labels & ignored_review_labels))
    ):
        rules.append("ignored_objects_review")
        calibrated["verdict"] = "PENDING"
        calibrated["quality_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "FAIL"
        and source == "unet"
        and unet_only_review_min_pct <= unet_pct <= unet_only_review_max_pct
        and sam3_pct < 1.0
        and penalty_count == 0
    ):
        rules.append("unet_only_high_coverage_review")
        calibrated["verdict"] = "PENDING"
        calibrated["quality_score"] = round(max(raw_quality, pending_lower_bound), 3)
    elif (
        raw_verdict == "FAIL"
        and source == "sam3"
        and sam3_predictions <= max(1, int(single_sam3_review_max_predictions))
        and unet_pct < strong_dirty_coverage_pct
        and penalty_count == 0
    ):
        rules.append("single_sam3_large_mask_review")
        calibrated["verdict"] = "PENDING"
        calibrated["quality_score"] = round(max(raw_quality, pending_lower_bound), 3)
    elif (
        raw_verdict == "FAIL"
        and source in {"sam3", "merged"}
        and (sam3_pct >= strong_dirty_coverage_pct or combined_pct >= strong_dirty_coverage_pct)
        and unet_pct < strong_dirty_coverage_pct
        and penalty_count == 0
    ):
        rules.append("auxiliary_segmentation_review")
        calibrated["verdict"] = "PENDING"
        calibrated["quality_score"] = round(max(raw_quality, pending_lower_bound), 3)

    if rules:
        calibrated["calibrated"] = True
        calibrated["calibration_rules"] = rules
        calibrated["calibration_reason"] = "; ".join(rules)
        reasons = list(calibrated.get("reasons") or [])
        reasons.append("calibration review required")
        calibrated["reasons"] = reasons

    return calibrated


def parse_url_items(image_urls: List[str]) -> List[str]:
    parsed: List[str] = []
    for raw in image_urls:
        if not isinstance(raw, str):
            continue

        candidate = raw.strip()
        if not candidate:
            continue

        # Swagger/UI integrations may send many URLs in one comma-separated string.
        parts = [p.strip() for p in candidate.split(",")]
        for part in parts:
            if part:
                parsed.append(part)

    return parsed
