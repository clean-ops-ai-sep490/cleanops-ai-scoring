from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STATUSES = ("PASS", "FAIL")


def _split_items(raw: str) -> set[str]:
    text = (raw or "").strip()
    if not text:
        return set()
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def _normalize_status(raw: str) -> str:
    status = (raw or "").strip().upper()
    if status not in STATUSES:
        raise ValueError(f"Unsupported PPE status '{status}'.")
    return status


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


def summarize(rows: list[dict[str, str]]) -> dict:
    total = len(rows)
    if total == 0:
        raise ValueError("Input CSV contains no data rows.")

    matches = 0
    false_missing = 0
    missing_gt_total = 0
    missing_gt_hit = 0
    latency_total = 0.0
    latency_samples = 0

    confusion = {expected: {predicted: 0 for predicted in STATUSES} for expected in STATUSES}

    for row in rows:
        expected_status = _normalize_status(row.get("expected_status", ""))
        predicted_status = _normalize_status(row.get("predicted_status", ""))
        expected_missing = _split_items(row.get("expected_missing_items", ""))
        predicted_missing = _split_items(row.get("predicted_missing_items", ""))

        confusion[expected_status][predicted_status] += 1
        if expected_status == predicted_status:
            matches += 1

        for item in expected_missing:
            missing_gt_total += 1
            if item in predicted_missing:
                missing_gt_hit += 1

        for item in predicted_missing:
            if item not in expected_missing:
                false_missing += 1

        latency_ms = _parse_float(row.get("latency_ms", ""))
        if latency_ms is not None:
            latency_total += latency_ms
            latency_samples += 1

    summary = {
        "total_samples": total,
        "ppe_status_accuracy": round(_safe_rate(matches, total), 4),
        "missing_item_recall": round(_safe_rate(missing_gt_hit, missing_gt_total), 4),
        "false_missing_rate": round(_safe_rate(false_missing, total), 4),
        "average_latency_ms": round(latency_total / latency_samples, 2) if latency_samples else None,
        "counts": {
            "matches": matches,
            "mismatches": total - matches,
            "missing_items_in_ground_truth": missing_gt_total,
            "missing_items_detected_correctly": missing_gt_hit,
            "false_missing_items": false_missing,
        },
        "confusion_matrix": confusion,
        "definitions": {
            "ppe_status_accuracy": "share of PASS/FAIL predictions matching ground truth",
            "missing_item_recall": "share of truly missing PPE items that the model reported missing",
            "false_missing_rate": "number of falsely reported missing items divided by total cases (items per case, not a percentage)",
        },
    }
    return summary


def render_markdown(summary: dict) -> str:
    rows = [
        ["Total samples", str(summary["total_samples"])],
        ["PPE status accuracy", f'{summary["ppe_status_accuracy"]:.2%}'],
        ["Missing item recall", f'{summary["missing_item_recall"]:.2%}'],
        ["False missing items per case", f'{summary["false_missing_rate"]:.2f}'],
        ["Average latency", f'{summary["average_latency_ms"]} ms' if summary["average_latency_ms"] is not None else "N/A"],
    ]

    metric_table = "\n".join(
        [
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {label} | {value} |" for label, value in rows],
        ]
    )

    confusion_rows = []
    for expected in STATUSES:
        confusion_rows.append(
            f"| {expected} | {summary['confusion_matrix'][expected]['PASS']} | {summary['confusion_matrix'][expected]['FAIL']} |"
        )

    confusion_table = "\n".join(
        [
            "| Expected \\ Predicted | PASS | FAIL |",
            "| --- | --- | --- |",
            *confusion_rows,
        ]
    )

    return "\n".join(
        [
            "# PPE Pilot Benchmark Summary",
            "",
            "## Core Metrics",
            "",
            metric_table,
            "",
            "## Confusion Matrix",
            "",
            confusion_table,
            "",
            "## Notes",
            "",
            "- `missing_item_recall`: share of truly missing PPE items that were reported missing.",
            "- `false_missing_rate`: falsely reported missing items divided by total cases; this is an items-per-case ratio, not a percentage.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize PPE pilot benchmark CSV into report-friendly metrics."
    )
    parser.add_argument("--input-csv", required=True, help="Path to the annotated PPE benchmark CSV.")
    parser.add_argument("--output-json", help="Optional path to write summary JSON.")
    parser.add_argument("--output-md", help="Optional path to write summary markdown.")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(summary), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
