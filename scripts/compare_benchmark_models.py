from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_utils import ensure_dir, write_csv, write_json


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use name=path/to/benchmark_metrics.json")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")
    return name, Path(path.strip())


def class_lookup(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["class_name"]): item for item in metrics.get("per_class", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multiple U-Net benchmark metric files on the same real test split.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=parse_model_arg,
        help="Model metrics in name=path format. Pass at least two.",
    )
    parser.add_argument("--output", default="outputs/benchmark_real/public_plus_comparison")
    args = parser.parse_args()

    if len(args.model) < 2:
        print("Pass at least two --model name=metrics.json values.")
        return 1

    output_dir = Path(args.output)
    ensure_dir(output_dir)
    loaded: list[dict[str, Any]] = []
    for name, path in args.model:
        if not path.exists():
            print(f"Missing metrics file for {name}: {path}")
            return 1
        metrics = load_metrics(path)
        loaded.append({"name": name, "path": str(path), "metrics": metrics})

    image_counts = {item["metrics"].get("images_evaluated") for item in loaded}
    test_keys = {(item["metrics"].get("image_dir"), item["metrics"].get("mask_dir")) for item in loaded}
    warnings: list[str] = []
    if len(image_counts) != 1:
        warnings.append(f"Models were evaluated on different image counts: {sorted(image_counts)}")
    if len(test_keys) != 1:
        warnings.append("Models may not have been evaluated on the same image/mask directories.")

    summary_rows = []
    class_rows = []
    for item in loaded:
        metrics = item["metrics"]
        summary = metrics.get("summary", {})
        summary_rows.append(
            {
                "model": item["name"],
                "weights": metrics.get("weights"),
                "images_evaluated": metrics.get("images_evaluated"),
                "pixel_accuracy": summary.get("pixel_accuracy"),
                "mean_iou": summary.get("mean_iou"),
                "mean_dice_f1": summary.get("mean_dice_f1"),
            }
        )
        classes = class_lookup(metrics)
        for class_name in ["background", "dirty_area", "wet_surface"]:
            class_item = classes.get(class_name, {})
            class_rows.append(
                {
                    "model": item["name"],
                    "class_name": class_name,
                    "support_pixels": class_item.get("support_pixels"),
                    "iou": class_item.get("iou"),
                    "dice_f1": class_item.get("dice_f1"),
                }
            )

    comparison = {
        "warnings": warnings,
        "models": [{"name": item["name"], "path": item["path"], "weights": item["metrics"].get("weights")} for item in loaded],
        "summary": summary_rows,
        "per_class": class_rows,
    }
    write_json(output_dir / "comparison_metrics.json", comparison)
    write_csv(
        output_dir / "comparison_summary.csv",
        summary_rows,
        ["model", "weights", "images_evaluated", "pixel_accuracy", "mean_iou", "mean_dice_f1"],
    )
    write_csv(
        output_dir / "comparison_per_class.csv",
        class_rows,
        ["model", "class_name", "support_pixels", "iou", "dice_f1"],
    )

    report = [
        "# Public-Plus U-Net Benchmark Comparison",
        "",
        "## Summary",
        "",
        "| model | images | pixel accuracy | mean IoU | mean Dice/F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['model']} | {row['images_evaluated']} | {fmt(row['pixel_accuracy'])} | "
            f"{fmt(row['mean_iou'])} | {fmt(row['mean_dice_f1'])} |"
        )
    report.extend(["", "## Per Class", "", "| model | class | support pixels | IoU | Dice/F1 |", "|---|---|---:|---:|---:|"])
    for row in class_rows:
        report.append(
            f"| {row['model']} | {row['class_name']} | {fmt(row['support_pixels'])} | "
            f"{fmt(row['iou'])} | {fmt(row['dice_f1'])} |"
        )
    if warnings:
        report.extend(["", "## Warnings", ""])
        report.extend(f"- {warning}" for warning in warnings)
    (output_dir / "comparison_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Comparison written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
