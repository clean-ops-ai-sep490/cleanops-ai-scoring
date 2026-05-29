from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from summarize_pilot_benchmark import build_summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_verdict(value: str) -> str:
    return (value or "").strip().upper()


def _is_valid(row: dict[str, str]) -> bool:
    return _normalize_verdict(row.get("expected_verdict", "")) in {"PASS", "PENDING", "FAIL"} and _normalize_verdict(
        row.get("predicted_verdict", "")
    ) in {"PASS", "PENDING", "FAIL"}


def _safe_float(value: str) -> float | None:
    try:
        text = str(value or "").strip()
        return float(text) if text else None
    except ValueError:
        return None


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def _parse_variant(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Variant must use label=csv_path format: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Variant label is empty: {raw}")
    return label, Path(path.strip())


def build_comparison(variants: list[tuple[str, Path]]) -> dict:
    if len(variants) < 2:
        raise ValueError("At least two variants are required for A/B comparison.")

    variant_payloads: list[dict] = []
    for label, path in variants:
        rows = _read_rows(path)
        summary = build_summary(rows)
        valid_rows = {row.get("image_id", ""): row for row in rows if row.get("image_id") and _is_valid(row)}
        variant_payloads.append(
            {
                "label": label,
                "path": str(path),
                "rows": rows,
                "valid_rows": valid_rows,
                "summary": summary,
            }
        )

    baseline = variant_payloads[0]
    baseline_rows = baseline["valid_rows"]
    variant_summaries = []
    for item in variant_payloads:
        summary = item["summary"]
        variant_summaries.append(
            {
                "label": item["label"],
                "path": item["path"],
                "evaluated_samples": summary["evaluated_samples"],
                "skipped_samples": summary["skipped_samples"],
                "verdict_accuracy": summary["verdict_accuracy"],
                "false_pass_rate": summary["false_pass_rate"],
                "false_fail_rate": summary["false_fail_rate"],
                "pending_review_rate": summary["pending_review_rate"],
                "average_latency_ms": summary["average_latency_ms"],
                "average_quality_score": summary["average_quality_score"],
                "average_unet_dirty_coverage_pct": summary.get("average_unet_dirty_coverage_pct"),
                "average_sam3_dirty_coverage_pct": summary.get("average_sam3_dirty_coverage_pct"),
                "average_combined_dirty_coverage_pct": summary.get("average_combined_dirty_coverage_pct"),
                "average_sam3_elapsed_ms": summary.get("average_sam3_elapsed_ms"),
                "dirty_coverage_source_counts": summary.get("dirty_coverage_source_counts", {}),
                "sam3_status_counts": summary.get("sam3_status_counts", {}),
            }
        )

    flip_reports = []
    for item in variant_payloads[1:]:
        common_ids = sorted(set(baseline_rows) & set(item["valid_rows"]))
        flips = []
        improved = 0
        regressed = 0
        same = 0
        for image_id in common_ids:
            base_row = baseline_rows[image_id]
            candidate_row = item["valid_rows"][image_id]
            expected = _normalize_verdict(candidate_row.get("expected_verdict", ""))
            base_pred = _normalize_verdict(base_row.get("predicted_verdict", ""))
            candidate_pred = _normalize_verdict(candidate_row.get("predicted_verdict", ""))
            base_correct = base_pred == expected
            candidate_correct = candidate_pred == expected
            if base_correct == candidate_correct:
                same += 1
            elif candidate_correct:
                improved += 1
            else:
                regressed += 1

            if base_pred != candidate_pred:
                flips.append(
                    {
                        "image_id": image_id,
                        "expected": expected,
                        "baseline_predicted": base_pred,
                        "candidate_predicted": candidate_pred,
                        "baseline_quality_score": _safe_float(base_row.get("quality_score", "")),
                        "candidate_quality_score": _safe_float(candidate_row.get("quality_score", "")),
                        "candidate_dirty_coverage_source": candidate_row.get("dirty_coverage_source", ""),
                        "candidate_unet_dirty_coverage_pct": _safe_float(candidate_row.get("unet_dirty_coverage_pct", "")),
                        "candidate_sam3_dirty_coverage_pct": _safe_float(candidate_row.get("sam3_dirty_coverage_pct", "")),
                        "candidate_combined_dirty_coverage_pct": _safe_float(
                            candidate_row.get("combined_dirty_coverage_pct", "")
                        ),
                    }
                )

        flip_reports.append(
            {
                "label": item["label"],
                "common_samples": len(common_ids),
                "verdict_flip_count": len(flips),
                "improved_count": improved,
                "regressed_count": regressed,
                "same_correctness_count": same,
                "flips": flips,
            }
        )

    return {
        "baseline_label": baseline["label"],
        "variants": variant_summaries,
        "flip_reports": flip_reports,
    }


def render_markdown(comparison: dict) -> str:
    variant_rows = []
    for item in comparison["variants"]:
        variant_rows.append(
            [
                item["label"],
                str(item["evaluated_samples"]),
                str(item["skipped_samples"]),
                f'{item["verdict_accuracy"]:.2%}',
                f'{item["false_pass_rate"]:.2%}',
                f'{item["false_fail_rate"]:.2%}',
                f'{item["pending_review_rate"]:.2%}',
                f'{item["average_latency_ms"]} ms' if item["average_latency_ms"] is not None else "N/A",
                str(item["dirty_coverage_source_counts"]),
                str(item["sam3_status_counts"]),
            ]
        )

    flip_rows = []
    for report in comparison["flip_reports"]:
        flip_rows.append(
            [
                report["label"],
                str(report["common_samples"]),
                str(report["verdict_flip_count"]),
                str(report["improved_count"]),
                str(report["regressed_count"]),
                str(report["same_correctness_count"]),
            ]
        )

    sections = [
        "# Cleanliness A/B Benchmark Comparison",
        "",
        f"Baseline variant: `{comparison['baseline_label']}`",
        "",
        "## Variant Metrics",
        "",
        _markdown_table(
            [
                "Variant",
                "Evaluated",
                "Skipped",
                "Accuracy",
                "False pass",
                "False fail",
                "Pending",
                "Avg latency",
                "Coverage source counts",
                "SAM3 status counts",
            ],
            variant_rows,
        ),
        "",
        "## Verdict Stability Versus Baseline",
        "",
        _markdown_table(
            ["Variant", "Common", "Verdict flips", "Improved", "Regressed", "Same correctness"],
            flip_rows,
        ),
    ]

    for report in comparison["flip_reports"]:
        if not report["flips"]:
            continue
        rows = [
            [
                item["image_id"],
                item["expected"],
                item["baseline_predicted"],
                item["candidate_predicted"],
                str(item["candidate_dirty_coverage_source"]),
                str(item["candidate_unet_dirty_coverage_pct"]),
                str(item["candidate_sam3_dirty_coverage_pct"]),
                str(item["candidate_combined_dirty_coverage_pct"]),
            ]
            for item in report["flips"]
        ]
        sections.extend(
            [
                "",
                f"## Verdict Flips: {report['label']}",
                "",
                _markdown_table(
                    [
                        "Image",
                        "Expected",
                        "Baseline",
                        "Candidate",
                        "Coverage source",
                        "U-Net %",
                        "SAM3 %",
                        "Combined %",
                    ],
                    rows,
                ),
            ]
        )

    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multiple cleanliness benchmark CSV outputs.")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant in label=csv_path format. First variant is treated as baseline.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    variants = [_parse_variant(raw) for raw in args.variant]
    comparison = build_comparison(variants)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(comparison), encoding="utf-8")

    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
