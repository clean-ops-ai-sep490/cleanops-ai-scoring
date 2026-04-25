from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


VERDICTS = ("PASS", "PENDING", "FAIL")

EXPECTED_VERDICT_ALIASES = (
    "expected_verdict",
    "ground_truth_verdict",
    "reviewed_verdict",
    "final_verdict",
)
PREDICTED_VERDICT_ALIASES = (
    "predicted_verdict",
    "model_verdict",
    "verdict",
)
ENVIRONMENT_ALIASES = (
    "environment_key",
    "env",
    "environment",
)
LATENCY_ALIASES = (
    "latency_ms",
    "elapsed_ms",
    "duration_ms",
)
QUALITY_SCORE_ALIASES = (
    "quality_score",
    "score",
)
DIRTY_LEVEL_ALIASES = (
    "dirty_level",
    "cleanliness_bucket",
    "case_bucket",
)


def _first_value(row: dict[str, str], aliases: Iterable[str], default: str = "") -> str:
    for alias in aliases:
        value = (row.get(alias) or "").strip()
        if value:
            return value
    return default


def _normalize_verdict(raw: str) -> str:
    verdict = (raw or "").strip().upper()
    if verdict in VERDICTS:
        return verdict
    return verdict


def _parse_float(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def build_summary(rows: list[dict[str, str]]) -> dict:
    total = len(rows)
    if total == 0:
        raise ValueError("Input CSV contains no data rows.")

    confusion = {expected: {predicted: 0 for predicted in VERDICTS} for expected in VERDICTS}
    predicted_counts = Counter()
    expected_counts = Counter()
    dirty_level_totals = Counter()
    dirty_level_matches = Counter()
    by_environment: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "samples": 0,
            "matches": 0,
            "false_pass": 0,
            "false_fail": 0,
            "pending_predictions": 0,
            "latency_total_ms": 0.0,
            "latency_samples": 0,
        }
    )

    matches = 0
    false_pass = 0
    false_fail = 0
    pending_predictions = 0
    latency_total = 0.0
    latency_samples = 0
    quality_total = 0.0
    quality_samples = 0

    for row in rows:
        expected = _normalize_verdict(_first_value(row, EXPECTED_VERDICT_ALIASES))
        predicted = _normalize_verdict(_first_value(row, PREDICTED_VERDICT_ALIASES))
        environment = _first_value(row, ENVIRONMENT_ALIASES, default="UNKNOWN")
        dirty_level = _first_value(row, DIRTY_LEVEL_ALIASES, default="UNSPECIFIED")

        if expected not in VERDICTS:
            raise ValueError(f"Unsupported expected verdict '{expected}'.")
        if predicted not in VERDICTS:
            raise ValueError(f"Unsupported predicted verdict '{predicted}'.")

        expected_counts[expected] += 1
        predicted_counts[predicted] += 1
        confusion[expected][predicted] += 1
        dirty_level_totals[dirty_level] += 1

        env_bucket = by_environment[environment]
        env_bucket["samples"] += 1

        if expected == predicted:
            matches += 1
            dirty_level_matches[dirty_level] += 1
            env_bucket["matches"] += 1

        if predicted == "PENDING":
            pending_predictions += 1
            env_bucket["pending_predictions"] += 1

        if predicted == "PASS" and expected != "PASS":
            false_pass += 1
            env_bucket["false_pass"] += 1

        if predicted == "FAIL" and expected == "PASS":
            false_fail += 1
            env_bucket["false_fail"] += 1

        latency_ms = _parse_float(_first_value(row, LATENCY_ALIASES))
        if latency_ms is not None:
            latency_total += latency_ms
            latency_samples += 1
            env_bucket["latency_total_ms"] += latency_ms
            env_bucket["latency_samples"] += 1

        quality_score = _parse_float(_first_value(row, QUALITY_SCORE_ALIASES))
        if quality_score is not None:
            quality_total += quality_score
            quality_samples += 1

    summary = {
        "total_samples": total,
        "verdict_accuracy": round(_safe_rate(matches, total), 4),
        "false_pass_rate": round(_safe_rate(false_pass, total), 4),
        "false_fail_rate": round(_safe_rate(false_fail, total), 4),
        "pending_review_rate": round(_safe_rate(pending_predictions, total), 4),
        "manual_correction_rate": round(_safe_rate(total - matches, total), 4),
        "average_latency_ms": round(latency_total / latency_samples, 2) if latency_samples else None,
        "average_quality_score": round(quality_total / quality_samples, 2) if quality_samples else None,
        "counts": {
            "expected": dict(expected_counts),
            "predicted": dict(predicted_counts),
            "matches": matches,
            "mismatches": total - matches,
            "false_pass": false_pass,
            "false_fail": false_fail,
            "pending_predictions": pending_predictions,
        },
        "confusion_matrix": confusion,
        "by_environment": {},
        "by_dirty_level": {},
        "definitions": {
            "false_pass": "predicted PASS while expected verdict is not PASS",
            "false_fail": "predicted FAIL while expected verdict is PASS",
            "pending_review_rate": "share of samples predicted as PENDING",
        },
    }

    for environment, bucket in sorted(by_environment.items()):
        samples = int(bucket["samples"])
        latency_samples_env = int(bucket["latency_samples"])
        summary["by_environment"][environment] = {
            "samples": samples,
            "verdict_accuracy": round(_safe_rate(int(bucket["matches"]), samples), 4),
            "false_pass_rate": round(_safe_rate(int(bucket["false_pass"]), samples), 4),
            "false_fail_rate": round(_safe_rate(int(bucket["false_fail"]), samples), 4),
            "pending_review_rate": round(_safe_rate(int(bucket["pending_predictions"]), samples), 4),
            "average_latency_ms": round(float(bucket["latency_total_ms"]) / latency_samples_env, 2)
            if latency_samples_env
            else None,
        }

    for dirty_level, count in sorted(dirty_level_totals.items()):
        summary["by_dirty_level"][dirty_level] = {
            "samples": count,
            "verdict_accuracy": round(_safe_rate(dirty_level_matches[dirty_level], count), 4),
        }

    return summary


def render_markdown(summary: dict) -> str:
    metric_rows = [
        ["Total samples", str(summary["total_samples"])],
        ["Verdict accuracy", f'{summary["verdict_accuracy"]:.2%}'],
        ["False pass rate", f'{summary["false_pass_rate"]:.2%}'],
        ["False fail rate", f'{summary["false_fail_rate"]:.2%}'],
        ["Pending review rate", f'{summary["pending_review_rate"]:.2%}'],
        ["Manual correction rate", f'{summary["manual_correction_rate"]:.2%}'],
        ["Average latency", f'{summary["average_latency_ms"]} ms' if summary["average_latency_ms"] is not None else "N/A"],
        ["Average quality score", str(summary["average_quality_score"]) if summary["average_quality_score"] is not None else "N/A"],
    ]

    confusion_rows: list[list[str]] = []
    for expected in VERDICTS:
        row = [expected]
        for predicted in VERDICTS:
            row.append(str(summary["confusion_matrix"][expected][predicted]))
        confusion_rows.append(row)

    environment_rows: list[list[str]] = []
    for environment, bucket in summary["by_environment"].items():
        environment_rows.append(
            [
                environment,
                str(bucket["samples"]),
                f'{bucket["verdict_accuracy"]:.2%}',
                f'{bucket["false_pass_rate"]:.2%}',
                f'{bucket["false_fail_rate"]:.2%}',
                f'{bucket["pending_review_rate"]:.2%}',
                f'{bucket["average_latency_ms"]} ms' if bucket["average_latency_ms"] is not None else "N/A",
            ]
        )

    dirty_rows: list[list[str]] = []
    for dirty_level, bucket in summary["by_dirty_level"].items():
        dirty_rows.append(
            [
                dirty_level,
                str(bucket["samples"]),
                f'{bucket["verdict_accuracy"]:.2%}',
            ]
        )

    sections = [
        "# Pilot Benchmark Summary",
        "",
        "## Core Metrics",
        "",
        _markdown_table(["Metric", "Value"], metric_rows),
        "",
        "## Confusion Matrix",
        "",
        _markdown_table(["Expected \\ Predicted", *VERDICTS], confusion_rows),
        "",
        "## By Environment",
        "",
        _markdown_table(
            [
                "Environment",
                "Samples",
                "Accuracy",
                "False pass",
                "False fail",
                "Pending review",
                "Avg latency",
            ],
            environment_rows or [["N/A", "0", "0.00%", "0.00%", "0.00%", "0.00%", "N/A"]],
        ),
        "",
        "## By Dirty Level",
        "",
        _markdown_table(
            ["Dirty level", "Samples", "Accuracy"],
            dirty_rows or [["N/A", "0", "0.00%"]],
        ),
        "",
        "## Notes",
        "",
        "- `false_pass`: predicted PASS while expected verdict is not PASS.",
        "- `false_fail`: predicted FAIL while expected verdict is PASS.",
        "- `pending_review_rate`: share of samples predicted as PENDING.",
    ]
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize pilot benchmark CSV into business metrics for AI scoring report."
    )
    parser.add_argument("--input-csv", required=True, help="Path to the annotated pilot benchmark CSV.")
    parser.add_argument("--output-json", help="Optional path to write machine-readable summary JSON.")
    parser.add_argument("--output-md", help="Optional path to write markdown summary.")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    summary = build_summary(rows)
    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.output_md:
        output_md_path = Path(args.output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(render_markdown(summary), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

