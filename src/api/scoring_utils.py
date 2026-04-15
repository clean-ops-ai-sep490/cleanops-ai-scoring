from __future__ import annotations

from typing import Dict, List, Optional


def normalize_env(env: Optional[str], env_rules: Dict[str, Dict[str, object]]) -> str:
    env_key = (env or "LOBBY_CORRIDOR").strip().upper()
    if env_key not in env_rules:
        raise ValueError(
            f"Unsupported env '{env_key}'. Allowed envs: {', '.join(sorted(env_rules.keys()))}"
        )
    return env_key


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def score_image(
    total_dirty_coverage_pct: float,
    detections_count: int,
    env_key: str,
    env_rules: Dict[str, Dict[str, object]],
    pending_lower_bound: float,
) -> Dict[str, object]:
    base_clean_score = 100.0 - float(total_dirty_coverage_pct)
    object_penalty = min(30.0, float(detections_count) * 5.0)
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
    if detections_count > 0:
        reasons.append("objects remain")
    if not reasons:
        reasons.append("good cleanliness")

    return {
        "base_clean_score": round(base_clean_score, 3),
        "object_penalty": round(object_penalty, 3),
        "quality_score": round(quality_score, 3),
        "pass_threshold": pass_threshold,
        "verdict": verdict,
        "reasons": reasons,
    }


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
