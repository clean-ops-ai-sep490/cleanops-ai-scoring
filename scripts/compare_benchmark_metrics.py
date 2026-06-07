from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_utils import ensure_dir, write_json


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return round(after - before, 6)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned benchmark metric JSON files.")
    parser.add_argument("--baseline", default="outputs/benchmark_real/baseline/benchmark_metrics.json")
    parser.add_argument("--finetuned", default="outputs/benchmark_real/finetuned/benchmark_metrics.json")
    parser.add_argument("--output", default="outputs/benchmark_real/comparison")
    args = parser.parse_args()

    baseline = load_metrics(Path(args.baseline))
    finetuned = load_metrics(Path(args.finetuned))
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    summary_rows = []
    for key in ["pixel_accuracy", "mean_iou", "mean_dice_f1"]:
        before = baseline["summary"].get(key)
        after = finetuned["summary"].get(key)
        summary_rows.append({"metric": key, "baseline": before, "finetuned": after, "delta": metric_delta(after, before)})

    baseline_classes = {item["class_id"]: item for item in baseline["per_class"]}
    finetuned_classes = {item["class_id"]: item for item in finetuned["per_class"]}
    class_rows = []
    for class_id, before_item in baseline_classes.items():
        after_item = finetuned_classes.get(class_id, {})
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": before_item["class_name"],
                "support_pixels": before_item["support_pixels"],
                "baseline_iou": before_item.get("iou"),
                "finetuned_iou": after_item.get("iou"),
                "delta_iou": metric_delta(after_item.get("iou"), before_item.get("iou")),
                "baseline_dice_f1": before_item.get("dice_f1"),
                "finetuned_dice_f1": after_item.get("dice_f1"),
                "delta_dice_f1": metric_delta(after_item.get("dice_f1"), before_item.get("dice_f1")),
            }
        )

    comparison = {
        "baseline_weights": baseline.get("weights"),
        "finetuned_weights": finetuned.get("weights"),
        "images_evaluated": finetuned.get("images_evaluated"),
        "summary": summary_rows,
        "per_class": class_rows,
    }
    write_json(output_dir / "comparison_metrics.json", comparison)

    report = [
        "# U-Net Real Benchmark Comparison",
        "",
        f"- Images evaluated: `{finetuned.get('images_evaluated')}`",
        f"- Baseline weights: `{baseline.get('weights')}`",
        f"- Fine-tuned weights: `{finetuned.get('weights')}`",
        "",
        "## Summary",
        "",
        "| metric | baseline | fine-tuned | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(f"| {row['metric']} | {fmt(row['baseline'])} | {fmt(row['finetuned'])} | {fmt(row['delta'])} |")
    report.extend(
        [
            "",
            "## Per Class",
            "",
            "| class | support_pixels | baseline IoU | fine-tuned IoU | delta IoU | baseline Dice/F1 | fine-tuned Dice/F1 | delta Dice/F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        report.append(
            f"| {row['class_name']} | {row['support_pixels']} | {fmt(row['baseline_iou'])} | "
            f"{fmt(row['finetuned_iou'])} | {fmt(row['delta_iou'])} | {fmt(row['baseline_dice_f1'])} | "
            f"{fmt(row['finetuned_dice_f1'])} | {fmt(row['delta_dice_f1'])} |"
        )
    (output_dir / "comparison_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Comparison written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
