from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from src.api.scoring_utils import score_image, summarize_penalty_detections
from src.models.unet_segmenter import UNetSegmenter


def load_unet_model(
    model_path: str,
    unet_device: torch.device,
    default_unet_img_size: int,
) -> Tuple[Optional[UNetSegmenter], int]:
    if not model_path or not Path(model_path).exists():
        return None, default_unet_img_size

    ckpt = torch.load(model_path, map_location=unet_device)
    encoder = "resnet50"
    unet_img_size = default_unet_img_size
    if isinstance(ckpt, dict):
        encoder = ckpt.get("encoder", "resnet50")
        unet_img_size = int(ckpt.get("img_size", default_unet_img_size))

    unet_model = UNetSegmenter(encoder_name=encoder, classes=3).to(unet_device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        unet_model.load_state_dict(ckpt["model_state"])
    else:
        unet_model.load_state_dict(ckpt)
    unet_model.eval()
    return unet_model, unet_img_size


def unet_predict_from_pil(
    img: Image.Image,
    unet_model: UNetSegmenter,
    unet_img_size: int,
    unet_device: torch.device,
    class_map: Dict[int, str],
) -> Dict[str, Any]:
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    resized = cv2.resize(rgb, (unet_img_size, unet_img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(unet_device)

    with torch.no_grad():
        logits = unet_model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    stain_pixels = int((pred == 1).sum())
    wet_pixels = int((pred == 2).sum())
    total_pixels = int(pred.size)

    stain_pct = (stain_pixels / total_pixels) * 100.0
    wet_pct = (wet_pixels / total_pixels) * 100.0
    dirty_pct = ((stain_pixels + wet_pixels) / total_pixels) * 100.0

    pred_original_size = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    return {
        "mask": pred,
        "mask_original_size": pred_original_size,
        "summary": {
            "input_size": [w, h],
            "model_input_size": unet_img_size,
            "class_mapping": class_map,
            "stain_or_water_pixels": stain_pixels,
            "wet_surface_pixels": wet_pixels,
            "stain_or_water_coverage_pct": round(stain_pct, 3),
            "wet_surface_coverage_pct": round(wet_pct, 3),
            "total_dirty_coverage_pct": round(dirty_pct, 3),
        },
        "rgb": rgb,
    }


def yolo_predict_from_pil(img: Image.Image, model, yolo_conf: float) -> Dict[str, Any]:
    results = model.predict(source=img, conf=yolo_conf, save=False, verbose=False)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            detections.append(
                {
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": float(f"{conf:.3f}"),
                    "bbox": [x1, y1, x2, y2],
                }
            )

    return {
        "detections_count": len(detections),
        "results": detections,
    }


def evaluate_image_with_artifacts(
    img: Image.Image,
    env_key: str,
    model,
    unet_model: UNetSegmenter,
    yolo_conf: float,
    unet_img_size: int,
    unet_device: torch.device,
    class_map: Dict[int, str],
    env_rules: Dict[str, Dict[str, object]],
    pending_lower_bound: float,
    scoring_penalty_labels: tuple[str, ...],
    scoring_object_penalty_per_detection: float,
):
    yolo_result = yolo_predict_from_pil(img, model=model, yolo_conf=yolo_conf)
    unet_result = unet_predict_from_pil(
        img,
        unet_model=unet_model,
        unet_img_size=unet_img_size,
        unet_device=unet_device,
        class_map=class_map,
    )

    penalty_summary = summarize_penalty_detections(
        yolo_result.get("results", []),
        scoring_penalty_labels,
    )
    score = score_image(
        total_dirty_coverage_pct=unet_result["summary"]["total_dirty_coverage_pct"],
        detections_count=yolo_result["detections_count"],
        env_key=env_key,
        env_rules=env_rules,
        pending_lower_bound=pending_lower_bound,
        object_penalty_per_detection=scoring_object_penalty_per_detection,
        **penalty_summary,
    )
    return yolo_result, unet_result, score


def evaluate_image(
    img: Image.Image,
    env_key: str,
    model,
    unet_model: UNetSegmenter,
    yolo_conf: float,
    unet_img_size: int,
    unet_device: torch.device,
    class_map: Dict[int, str],
    env_rules: Dict[str, Dict[str, object]],
    pending_lower_bound: float,
    scoring_penalty_labels: tuple[str, ...],
    scoring_object_penalty_per_detection: float,
) -> Dict[str, Any]:
    yolo_result, unet_result, score = evaluate_image_with_artifacts(
        img,
        env_key,
        model=model,
        unet_model=unet_model,
        yolo_conf=yolo_conf,
        unet_img_size=unet_img_size,
        unet_device=unet_device,
        class_map=class_map,
        env_rules=env_rules,
        pending_lower_bound=pending_lower_bound,
        scoring_penalty_labels=scoring_penalty_labels,
        scoring_object_penalty_per_detection=scoring_object_penalty_per_detection,
    )

    return {
        "yolo": yolo_result,
        "unet": unet_result["summary"],
        "scoring": score,
    }
