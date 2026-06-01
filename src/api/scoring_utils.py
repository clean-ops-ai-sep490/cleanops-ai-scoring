from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import cv2
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

STRONG_AUX_DIRTY_LABELS = {"garbage", "trash", "debris", "rubbish", "litter", "waste"}
FLOOR_LIKE_DISCOUNT_ENVS = {"LOBBY_CORRIDOR", "BASEMENT_PARKING", "GLASS_SURFACE"}
HIGH_RISK_REVIEW_ENVS = {"RESTROOM", "HOSPITAL_OR"}


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


def _unet_component_summary(unet_mask: np.ndarray) -> Dict[str, object]:
    binary = np.asarray(unet_mask > 0, dtype=np.uint8)
    height, width = binary.shape[:2]
    total_px = max(1, height * width)
    if int(binary.sum()) == 0:
        return {
            "unet_component_count": 0,
            "unet_largest_component_area_pct": 0.0,
            "unet_largest_component_bottom_touch": False,
            "unet_largest_component_bottom_area_ratio": 0.0,
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    significant = []
    min_component_px = max(8, int(total_px * 0.0025))
    for label_id in range(1, num_labels):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px >= min_component_px:
            significant.append((label_id, area_px))

    if not significant:
        return {
            "unet_component_count": 0,
            "unet_largest_component_area_pct": 0.0,
            "unet_largest_component_bottom_touch": False,
            "unet_largest_component_bottom_area_ratio": 0.0,
        }

    largest_label, largest_area_px = max(significant, key=lambda item: item[1])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    largest_component = labels == largest_label
    bottom_start = int(height * 0.60)
    bottom_area_px = int(np.count_nonzero(largest_component[bottom_start:, :]))
    bottom_area_ratio = bottom_area_px / max(1, largest_area_px)

    return {
        "unet_component_count": len(significant),
        "unet_largest_component_area_pct": round((largest_area_px / total_px) * 100.0, 3),
        "unet_largest_component_bottom_touch": (y + h) >= int(height * 0.85),
        "unet_largest_component_bottom_area_ratio": round(bottom_area_ratio, 3),
    }


def merge_unet_and_sam3_masks(
    unet_mask: np.ndarray,
    sam3_result: Dict[str, Any],
    *,
    dirty_labels: Sequence[object] | None = None,
    wet_labels: Sequence[object] | None = None,
    env_key: str | None = None,
) -> Dict[str, object]:
    merged = np.zeros_like(unet_mask, dtype=np.uint8)
    unet_dirty = np.asarray(unet_mask == 1, dtype=bool)
    unet_wet = np.asarray(unet_mask == 2, dtype=bool)
    merged[unet_dirty] = 1
    merged[unet_wet] = 2

    dirty_label_set = set(normalize_penalty_labels(dirty_labels or DEFAULT_ROBOFLOW_DIRTY_LABELS))
    wet_label_set = set(normalize_penalty_labels(wet_labels or DEFAULT_ROBOFLOW_WET_LABELS))
    total_px = max(1, int(merged.size))
    unet_px = int(np.count_nonzero(unet_dirty | unet_wet))
    unet_pct = (unet_px / total_px) * 100.0
    unet_dirty_px = int(np.count_nonzero(unet_dirty))
    unet_wet_px = int(np.count_nonzero(unet_wet))
    unet_dirty_pct = (unet_dirty_px / total_px) * 100.0
    unet_wet_pct = (unet_wet_px / total_px) * 100.0
    env_normalized = str(env_key or "").strip().upper()
    high_risk_envs = HIGH_RISK_REVIEW_ENVS

    scored_dirty_mask = np.zeros_like(merged, dtype=bool)
    scored_wet_mask = np.zeros_like(merged, dtype=bool)
    advisory_mask = np.zeros_like(merged, dtype=bool)
    raw_aux_mask = np.zeros_like(merged, dtype=bool)
    class_counts = {"stain_or_water": 0, "wet_surface": 0, "ignored": 0}
    filter_rules: list[str] = []
    label_coverage: dict[str, dict[str, object]] = {}
    scored_predictions_count = 0
    advisory_predictions_count = 0
    rejected_predictions_count = 0

    prediction_masks = sam3_result.get("_prediction_masks") if isinstance(sam3_result, dict) else []
    prediction_items: list[dict[str, Any]] = []
    if isinstance(prediction_masks, list) and prediction_masks:
        for item in prediction_masks:
            if isinstance(item, dict):
                prediction_items.append(item)
    else:
        label_masks = sam3_result.get("_label_masks") if isinstance(sam3_result, dict) else {}
        if isinstance(label_masks, dict):
            for raw_label, raw_mask in label_masks.items():
                if isinstance(raw_mask, np.ndarray):
                    prediction_items.append(
                        {
                            "label": raw_label,
                            "label_normalized": normalize_detection_label(raw_label),
                            "mask": raw_mask,
                            "mask_source": "label_union",
                        }
                    )

    total_aux_predictions = len(prediction_items)

    for item in prediction_items:
        raw_mask = item.get("mask")
        if not isinstance(raw_mask, np.ndarray):
            continue
        if raw_mask.shape[:2] != merged.shape[:2]:
            continue

        label = normalize_detection_label(item.get("label_normalized") or item.get("label"))
        mask = raw_mask.astype(bool)
        area_px = int(np.count_nonzero(mask))
        area_pct = (area_px / total_px) * 100.0
        mask_source = str(item.get("mask_source", "")).strip().lower()
        raw_aux_mask |= mask

        label_bucket = label_coverage.setdefault(
            label or "unknown",
            {"raw_area_pct": 0.0, "scored_area_pct": 0.0, "advisory_area_pct": 0.0, "count": 0},
        )
        label_bucket["raw_area_pct"] = round(float(label_bucket["raw_area_pct"]) + area_pct, 3)
        label_bucket["count"] = int(label_bucket["count"]) + 1

        if label in wet_label_set:
            class_counts["wet_surface"] += 1
            advisory_mask |= mask
            advisory_predictions_count += 1
            label_bucket["advisory_area_pct"] = round(float(label_bucket["advisory_area_pct"]) + area_pct, 3)
            filter_rules.append("wet_surface_advisory")
            continue

        if label not in dirty_label_set:
            class_counts["ignored"] += 1
            rejected_predictions_count += 1
            filter_rules.append("label_not_scored")
            continue

        class_counts["stain_or_water"] += 1
        is_strong_label = label in STRONG_AUX_DIRTY_LABELS
        is_large_bbox_fallback = mask_source == "bbox_fallback" and area_pct >= 15.0
        is_single_giant_aux = total_aux_predictions <= 1 and area_pct >= 35.0 and unet_pct < 5.0
        is_moderate_stain = area_pct <= 30.0
        is_multi_region_dirty = total_aux_predictions >= 3

        if is_large_bbox_fallback:
            advisory_mask |= mask
            advisory_predictions_count += 1
            label_bucket["advisory_area_pct"] = round(float(label_bucket["advisory_area_pct"]) + area_pct, 3)
            filter_rules.append("large_bbox_fallback_advisory")
            continue

        if is_single_giant_aux and not is_strong_label:
            advisory_mask |= mask
            advisory_predictions_count += 1
            label_bucket["advisory_area_pct"] = round(float(label_bucket["advisory_area_pct"]) + area_pct, 3)
            if env_normalized in high_risk_envs:
                filter_rules.append("single_giant_aux_high_risk_review")
            else:
                filter_rules.append("single_giant_aux_ignored")
            continue

        if is_strong_label or is_moderate_stain or is_multi_region_dirty:
            scored_dirty_mask |= mask
            scored_predictions_count += 1
            label_bucket["scored_area_pct"] = round(float(label_bucket["scored_area_pct"]) + area_pct, 3)
            filter_rules.append("aux_prediction_scored")
            continue

        advisory_mask |= mask
        advisory_predictions_count += 1
        label_bucket["advisory_area_pct"] = round(float(label_bucket["advisory_area_pct"]) + area_pct, 3)
        filter_rules.append("large_stain_advisory")

    merged[scored_dirty_mask] = 1
    merged[scored_wet_mask] = 2

    sam3_px = int(np.count_nonzero(scored_dirty_mask | scored_wet_mask))
    raw_sam3_px = int(np.count_nonzero(raw_aux_mask))
    advisory_px = int(np.count_nonzero(advisory_mask))
    raw_merged_mask = merged.copy()
    raw_merged_px = int(np.count_nonzero(raw_merged_mask > 0))
    wet_px = int(np.count_nonzero(merged == 2))
    dirty_px = int(np.count_nonzero(merged == 1))
    raw_combined_pct = (raw_merged_px / total_px) * 100.0
    effective_combined_pct = raw_combined_pct
    adjustment_reason = ""
    adjustment_factor = 1.0
    floor_like_overmask_detected = False
    component_summary = _unet_component_summary((unet_dirty | unet_wet).astype(np.uint8))
    largest_component_area_pct = _as_float(component_summary["unet_largest_component_area_pct"])
    component_count = _as_int(component_summary["unet_component_count"])
    largest_ratio = largest_component_area_pct / max(0.001, unet_pct)
    bottom_touch = bool(component_summary["unet_largest_component_bottom_touch"])
    bottom_area_ratio = _as_float(component_summary["unet_largest_component_bottom_area_ratio"])
    sam3_scored_pct = (sam3_px / total_px) * 100.0
    sam3_advisory_pct = (advisory_px / total_px) * 100.0
    has_strong_scored_aux = sam3_scored_pct >= 5.0 or scored_predictions_count >= 3
    has_aux_floor_advisory = sam3_advisory_pct >= 20.0 or advisory_predictions_count > 0
    wet_signal_is_floor_like = unet_wet_pct < 5.0 or (has_aux_floor_advisory and unet_dirty_pct >= 8.0)
    component_shape_is_floor_like = component_count <= 3 or (has_aux_floor_advisory and component_count <= 4)
    spatial_shape_is_floor_like = (
        bottom_touch
        or bottom_area_ratio >= 0.45
        or (has_aux_floor_advisory and largest_component_area_pct >= 45.0 and largest_ratio >= 0.80)
    )

    looks_floor_like = (
        unet_pct >= 18.0
        and (unet_dirty_pct >= 12.0 or (has_aux_floor_advisory and unet_dirty_pct >= 8.0))
        and wet_signal_is_floor_like
        and component_shape_is_floor_like
        and largest_component_area_pct >= 15.0
        and largest_ratio >= 0.70
        and spatial_shape_is_floor_like
    )

    if looks_floor_like and not has_strong_scored_aux:
        floor_like_overmask_detected = True
        if env_normalized in FLOOR_LIKE_DISCOUNT_ENVS:
            if raw_combined_pct <= 25.0 or has_aux_floor_advisory:
                adjustment_factor = 0.10
                adjustment_reason = "floor_like_overmask_discount"
                adjusted_unet_pct = unet_pct * adjustment_factor
                effective_combined_pct = max(adjusted_unet_pct, sam3_scored_pct)
                merged = np.zeros_like(raw_merged_mask, dtype=np.uint8)
                merged[scored_dirty_mask] = 1
                merged[scored_wet_mask] = 2
            else:
                adjustment_reason = "floor_like_overmask_review_unet_only"
        elif env_normalized in high_risk_envs:
            adjustment_factor = 0.40
            adjustment_reason = "floor_like_overmask_high_risk_review"
            adjusted_unet_pct = unet_pct * adjustment_factor
            effective_combined_pct = max(adjusted_unet_pct, sam3_scored_pct)
        else:
            adjustment_reason = "floor_like_overmask_detected_no_discount"

    merged_px = int(np.count_nonzero(merged > 0))
    if not adjustment_reason:
        effective_combined_pct = raw_combined_pct

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
        "unet_dirty_coverage_pct": round(unet_pct, 3),
        "sam3_dirty_coverage_pct": round((sam3_px / total_px) * 100.0, 3),
        "sam3_raw_coverage_pct": round((raw_sam3_px / total_px) * 100.0, 3),
        "sam3_scored_coverage_pct": round((sam3_px / total_px) * 100.0, 3),
        "sam3_advisory_coverage_pct": round((advisory_px / total_px) * 100.0, 3),
        "sam3_scored_predictions_count": scored_predictions_count,
        "sam3_advisory_predictions_count": advisory_predictions_count,
        "sam3_rejected_predictions_count": rejected_predictions_count,
        "sam3_filter_rules": sorted(set(filter_rules)),
        "sam3_label_coverage": label_coverage,
        "raw_combined_dirty_coverage_pct": round(raw_combined_pct, 3),
        "effective_dirty_coverage_pct": round(effective_combined_pct, 3),
        "combined_dirty_coverage_pct": round(effective_combined_pct, 3),
        "coverage_adjustment_reason": adjustment_reason,
        "coverage_adjustment_factor": round(adjustment_factor, 3),
        "floor_like_overmask_detected": floor_like_overmask_detected,
        **component_summary,
        "unet_stain_or_water_coverage_pct": round(unet_dirty_pct, 3),
        "unet_wet_surface_coverage_pct": round(unet_wet_pct, 3),
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


def _fail_decision_score(raw_quality: float, pending_lower_bound: float) -> float:
    return round(clamp(min(raw_quality, pending_lower_bound - 0.001), 0.0, 100.0), 3)


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
    sam3_advisory_pct = _as_float(calibrated.get("sam3_advisory_coverage_pct"))
    combined_pct = _as_float(calibrated.get("combined_dirty_coverage_pct"))
    sam3_predictions = _as_int(calibrated.get("sam3_predictions_count"))
    sam3_scored_predictions = _as_int(calibrated.get("sam3_scored_predictions_count"))
    source = str(calibrated.get("dirty_coverage_source", "")).lower()
    floor_like_overmask = bool(calibrated.get("floor_like_overmask_detected"))
    ignored_labels = {
        normalize_detection_label(item)
        for item in (calibrated.get("ignored_detection_labels") or [])
        if normalize_detection_label(item)
    }

    calibrated["raw_verdict"] = raw_verdict
    calibrated["raw_quality_score"] = round(raw_quality, 3)
    calibrated["decision_score"] = round(raw_quality, 3)
    calibrated["calibrated"] = False
    calibrated["calibration_rules"] = []
    calibrated["calibration_reason"] = ""
    calibrated["review_required"] = False
    calibrated["verdict_source"] = "score"
    calibrated["verdict_reason_code"] = raw_verdict.lower() if raw_verdict else ""
    calibrated["risk_flags"] = []

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
            calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
        elif raw_verdict != "FAIL":
            rules.append("strong_multi_source_dirty")
            calibrated["verdict"] = "FAIL"
            calibrated["decision_score"] = _fail_decision_score(raw_quality, pending_lower_bound)
    elif (
        raw_verdict == "PASS"
        and env_normalized in high_risk_envs
        and (combined_pct < 1.0 or floor_like_overmask)
        and penalty_count == 0
        and (sam3_predictions == 0 or sam3_advisory_pct >= 20.0)
    ):
        rules.append("high_risk_weak_evidence_review")
        calibrated["verdict"] = "PENDING"
        calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "PENDING"
        and env_normalized in high_risk_envs
        and floor_like_overmask
        and penalty_count == 0
    ):
        rules.append("high_risk_floor_like_review")
        calibrated["verdict"] = "PENDING"
        calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "PASS"
        and ignored_count >= max(1, int(ignored_object_review_count))
        and penalty_count == 0
        and (not ignored_review_labels or bool(ignored_labels & ignored_review_labels))
    ):
        rules.append("ignored_objects_review")
        calibrated["verdict"] = "PENDING"
        calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "PASS"
        and combined_pct >= 5.0
        and raw_quality < pass_threshold + 5.0
        and penalty_count == 0
        and not floor_like_overmask
    ):
        rules.append("near_threshold_dirty_review")
        calibrated["verdict"] = "PENDING"
        calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "PENDING"
        and env_normalized in (high_risk_envs | {"OUTDOOR_LANDSCAPE"})
        and combined_pct >= 40.0
        and (sam3_predictions >= 10 or sam3_scored_predictions >= 3)
    ):
        rules.append("dense_dirty_regions_fail")
        calibrated["verdict"] = "FAIL"
        calibrated["decision_score"] = _fail_decision_score(raw_quality, pending_lower_bound)
    elif (
        raw_verdict == "FAIL"
        and source == "unet"
        and unet_only_review_min_pct <= unet_pct <= unet_only_review_max_pct
        and sam3_pct < 1.0
        and penalty_count == 0
    ):
        rules.append("unet_only_high_coverage_review")
        calibrated["verdict"] = "PENDING"
        calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "FAIL"
        and source == "sam3"
        and sam3_predictions <= max(1, int(single_sam3_review_max_predictions))
        and unet_pct < strong_dirty_coverage_pct
        and penalty_count == 0
    ):
        rules.append("single_sam3_large_mask_review")
        calibrated["verdict"] = "PENDING"
        calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)
    elif (
        raw_verdict == "FAIL"
        and source in {"sam3", "merged"}
        and (sam3_pct >= strong_dirty_coverage_pct or combined_pct >= strong_dirty_coverage_pct)
        and unet_pct < strong_dirty_coverage_pct
        and penalty_count == 0
    ):
        rules.append("auxiliary_segmentation_review")
        calibrated["verdict"] = "PENDING"
        calibrated["decision_score"] = _pending_quality(raw_quality, pass_threshold, pending_lower_bound)

    if rules:
        review_required = calibrated["verdict"] == "PENDING"
        calibrated["calibrated"] = True
        calibrated["calibration_rules"] = rules
        calibrated["calibration_reason"] = "; ".join(rules)
        calibrated["review_required"] = review_required
        calibrated["verdict_source"] = "calibration"
        calibrated["verdict_reason_code"] = rules[0]
        calibrated["risk_flags"] = rules
        reasons = list(calibrated.get("reasons") or [])
        if review_required:
            reasons.append("calibration review required")
        else:
            reasons.append("strong dirty evidence")
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
