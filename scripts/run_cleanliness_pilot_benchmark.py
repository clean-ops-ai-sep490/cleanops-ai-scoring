from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib import error, request


DEFAULT_FIELDS = [
    "image_id",
    "image_url",
    "local_image_path",
    "environment_key",
    "expected_verdict",
    "predicted_verdict",
    "quality_score",
    "raw_verdict",
    "raw_quality_score",
    "decision_score",
    "calibrated",
    "review_required",
    "verdict_source",
    "verdict_reason_code",
    "risk_flags",
    "calibration_rules",
    "calibration_reason",
    "pass_threshold",
    "latency_ms",
    "dirty_coverage_source",
    "unet_dirty_coverage_pct",
    "sam3_dirty_coverage_pct",
    "combined_dirty_coverage_pct",
    "sam3_status",
    "sam3_predictions_count",
    "sam3_elapsed_ms",
    "dirty_level",
    "has_large_trash",
    "visualization_url",
    "notes",
]


def _post_json(url: str, payload: dict, timeout_sec: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _clean_error_message(exc: Exception) -> str:
    if isinstance(exc, error.HTTPError):
        try:
            payload = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            payload = str(exc)
        return f"HTTP {exc.code}: {payload}"
    return str(exc)


def _nested_value(payload: dict, *keys: str, default: object = "") -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def run_benchmark(
    input_csv: Path,
    output_csv: Path,
    api_base_url: str,
    timeout_sec: int,
    limit: int | None = None,
) -> None:
    endpoint = f"{api_base_url.rstrip('/')}/evaluate-url-visualize-link"
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Input CSV contains no benchmark rows: {input_csv}")
    if limit is not None:
        rows = rows[: max(0, limit)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys([*DEFAULT_FIELDS, *rows[0].keys()]))

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            image_url = (row.get("image_url") or "").strip()
            env_key = (row.get("environment_key") or "").strip()
            if not image_url or not env_key:
                raise ValueError(f"Row {row.get('image_id', '<missing>')} is missing image_url or environment_key.")

            started = time.perf_counter()
            try:
                payload = _post_json(endpoint, {"url": image_url, "env": env_key}, timeout_sec)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                scoring = payload.get("scoring") or {}
                visualization = payload.get("visualization") or {}
                sam3 = payload.get("sam3") or {}

                row["predicted_verdict"] = str(scoring.get("verdict", "")).upper()
                row["quality_score"] = scoring.get("quality_score", "")
                row["raw_verdict"] = str(scoring.get("raw_verdict", "")).upper()
                row["raw_quality_score"] = scoring.get("raw_quality_score", "")
                row["decision_score"] = scoring.get("decision_score", "")
                row["calibrated"] = scoring.get("calibrated", "")
                row["review_required"] = scoring.get("review_required", "")
                row["verdict_source"] = scoring.get("verdict_source", "")
                row["verdict_reason_code"] = scoring.get("verdict_reason_code", "")
                row["risk_flags"] = ",".join(str(item) for item in scoring.get("risk_flags", []) or [])
                row["calibration_rules"] = ",".join(str(item) for item in scoring.get("calibration_rules", []) or [])
                row["calibration_reason"] = scoring.get("calibration_reason", "")
                row["pass_threshold"] = scoring.get("pass_threshold", row.get("pass_threshold", ""))
                row["dirty_coverage_source"] = scoring.get("dirty_coverage_source", "")
                row["unet_dirty_coverage_pct"] = scoring.get("unet_dirty_coverage_pct", "")
                row["sam3_dirty_coverage_pct"] = scoring.get("sam3_dirty_coverage_pct", "")
                row["combined_dirty_coverage_pct"] = scoring.get("combined_dirty_coverage_pct", "")
                row["sam3_status"] = sam3.get("status", "")
                row["sam3_predictions_count"] = _nested_value(sam3, "summary", "predictions_count")
                row["sam3_elapsed_ms"] = _nested_value(sam3, "summary", "elapsed_ms")
                row["latency_ms"] = elapsed_ms
                row["visualization_url"] = visualization.get("url", "")
            except Exception as exc:  # noqa: BLE001
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                row["notes"] = f"{row.get('notes', '').strip()} | inference_error={_clean_error_message(exc)}".strip()

            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real cleanliness pilot benchmark against a live scoring API.")
    parser.add_argument("--input-csv", default="benchmarks/cleanliness/pilot_benchmark.csv")
    parser.add_argument("--output-csv", default="benchmarks/reports/cleanliness_pilot_evaluated.csv")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to evaluate for smoke runs.")
    args = parser.parse_args()

    run_benchmark(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        api_base_url=args.api_base_url,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
