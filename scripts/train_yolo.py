from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline YOLOv8n training entrypoint (standalone).")
    parser.add_argument("--data", default="configs/yolo_data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="outputs/yolo")
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Missing YOLO data config: {args.data}")
        return 1
    try:
        from ultralytics import YOLO
    except Exception:
        print("ultralytics is not installed. Install with: pip install ultralytics")
        return 1

    model = YOLO(args.model)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, project=args.project, name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
