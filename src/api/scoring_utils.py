from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


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


def _merge_reason_values(*reason_groups: Sequence[object] | None) -> List[str]:
    seen: set[str] = set()
    merged: List[str] = []
    for group in reason_groups:
        for raw in group or []:
            item = str(raw or "").strip()
            if not item:
                continue
            lowered_text = item.lower().replace("dirty_zone", "dirty area").replace("dirty zone", "dirty area")
            item = lowered_text[0].upper() + lowered_text[1:] if lowered_text else ""
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(item)
    return merged


def apply_floor_condition_severity(scoring: Dict[str, Any], visual_review: Dict[str, Any]) -> Dict[str, Any]:
    condition = str(visual_review.get("floor_condition") or "").strip().lower()
    needs_cleaning = bool(visual_review.get("needs_cleaning"))
    if condition not in {"lightly_dirty", "dirty", "very_dirty"}:
        return scoring

    updated = dict(scoring)
    pass_threshold = float(updated.get("pass_threshold", 90.0))
    current_quality = float(updated.get("quality_score", 0.0))
    severity_reasons: List[str] = []

    if condition == "lightly_dirty":
        if needs_cleaning and str(updated.get("verdict", "")).upper() == "PASS":
            updated["verdict"] = "PENDING"
            updated["quality_score"] = round(max(70.0, min(current_quality, pass_threshold - 0.5, 89.0)), 3)
            severity_reasons.append("light visible floor marks require review")
    elif condition == "dirty":
        updated["verdict"] = "FAIL"
        updated["quality_score"] = round(min(current_quality, 45.0), 3)
        severity_reasons.append("dirty floor requires cleaning")
    else:
        updated["verdict"] = "FAIL"
        updated["quality_score"] = round(min(current_quality, 35.0), 3)
        severity_reasons.append("very dirty floor requires urgent cleaning")

    existing_reasons = [
        reason
        for reason in updated.get("reasons", [])
        if str(reason or "").strip().lower() != "good cleanliness"
    ]
    updated["reasons"] = _merge_reason_values(existing_reasons, visual_review.get("reasons", []), severity_reasons)
    return updated


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
