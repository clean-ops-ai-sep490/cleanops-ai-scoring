from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> int:
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd)
    return int(completed.returncode)


def latest_best(pattern: str, root: Path = Path("outputs/unet/runs")) -> Path | None:
    candidates = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        best = candidate / "best.pt"
        if best.exists():
            return best
    return None


def load_public_new_count(report_path: Path) -> int:
    if not report_path.exists():
        return 0
    data = json.loads(report_path.read_text(encoding="utf-8"))
    written = data.get("written", {})
    count = 0
    for key, value in written.items():
        if key.endswith("_images_written") and not key.startswith(("existing_wet_", "real_val_")):
            count += int(value)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the selected public-data U-Net improvement pipeline.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-train", action="store_true", help="Only prepare/evaluate/compare existing checkpoints.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    py = args.python
    train_extra = ["--num-workers", str(args.num_workers), "--log-interval", str(args.log_interval)]
    if args.device:
        train_extra.extend(["--device", args.device])
    if args.amp:
        train_extra.append("--amp")

    steps = [
        [py, "scripts/prepare_unet_public_plus.py"],
        [
            py,
            "scripts/eval_benchmark.py",
            "--unet-root",
            "data/processed/unet_real",
            "--split",
            "test",
            "--output",
            "outputs/benchmark_real/public_plus_baseline",
            "--weights",
            "model-cache/active/unet/model.pth",
        ],
    ]
    for step in steps:
        code = run(step, repo)
        if code != 0:
            return code

    real_ft = latest_best("unet_resnet34_real_finetune_*/")
    comparison_models = [
        "baseline=outputs/benchmark_real/public_plus_baseline/benchmark_metrics.json",
    ]
    if real_ft is not None:
        code = run(
            [
                py,
                "scripts/eval_benchmark.py",
                "--unet-root",
                "data/processed/unet_real",
                "--split",
                "test",
                "--output",
                "outputs/benchmark_real/public_plus_existing_real_ft",
                "--weights",
                str(real_ft),
            ],
            repo,
        )
        if code != 0:
            return code
        comparison_models.append("existing_real_ft=outputs/benchmark_real/public_plus_existing_real_ft/benchmark_metrics.json")
    else:
        print("No existing real fine-tune checkpoint found; skipping existing_real_ft evaluation.", flush=True)

    public_new_count = load_public_new_count(repo / "data/processed/unet_public_plus/prepare_unet_public_plus_report.json")
    public_best: Path | None = None
    public_real_best: Path | None = None
    if args.skip_train:
        print("--skip-train was set; skipping stage 1 and stage 2 training.", flush=True)
    elif public_new_count <= 0:
        print("No new public DWSD/ZeroWaste/DMS/MINC samples found. Download them into data/raw before public-data training.", flush=True)
    else:
        code = run([py, "scripts/train_unet.py", "--config", "configs/unet_public_dirty.yaml", *train_extra], repo)
        if code != 0:
            return code
        public_best = latest_best("unet_resnet34_public_dirty_*/")
        if public_best is None:
            print("Stage 1 finished but no public best.pt was found.", flush=True)
            return 1
        code = run(
            [
                py,
                "scripts/eval_benchmark.py",
                "--unet-root",
                "data/processed/unet_real",
                "--split",
                "test",
                "--output",
                "outputs/benchmark_real/public_plus_public_dirty",
                "--weights",
                str(public_best),
            ],
            repo,
        )
        if code != 0:
            return code
        comparison_models.append("public_dirty=outputs/benchmark_real/public_plus_public_dirty/benchmark_metrics.json")

        code = run(
            [
                py,
                "scripts/train_unet.py",
                "--config",
                "configs/unet_public_dirty_real_finetune.yaml",
                "--checkpoint-init",
                str(public_best),
                *train_extra,
            ],
            repo,
        )
        if code != 0:
            return code
        public_real_best = latest_best("unet_resnet34_public_dirty_real_ft_*/")
        if public_real_best is None:
            print("Stage 2 finished but no public+real best.pt was found.", flush=True)
            return 1
        code = run(
            [
                py,
                "scripts/eval_benchmark.py",
                "--unet-root",
                "data/processed/unet_real",
                "--split",
                "test",
                "--output",
                "outputs/benchmark_real/public_plus_finetuned",
                "--weights",
                str(public_real_best),
            ],
            repo,
        )
        if code != 0:
            return code
        comparison_models.append("public_dirty_real_ft=outputs/benchmark_real/public_plus_finetuned/benchmark_metrics.json")

    compare_cmd = [py, "scripts/compare_benchmark_models.py", "--output", "outputs/benchmark_real/public_plus_comparison"]
    for model in comparison_models:
        compare_cmd.extend(["--model", model])
    if len(comparison_models) >= 2:
        code = run(compare_cmd, repo)
        if code != 0:
            return code
    else:
        print("Only one metrics file is available; skipping comparison.", flush=True)

    print("Public-plus pipeline finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
