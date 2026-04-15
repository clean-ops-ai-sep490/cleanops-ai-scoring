from __future__ import annotations

import base64
import io
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from fastapi import Request

from src.api.temp_store import TempVisualizationStore


def render_unet_overlay(rgb: np.ndarray, pred_original_size: np.ndarray) -> bytes:
    overlay = rgb.copy()

    # Blend stain/water region in red.
    stain_region = pred_original_size == 1
    overlay[stain_region] = (overlay[stain_region] * 0.6 + np.array([255, 80, 80]) * 0.4).astype(np.uint8)

    # Blend wet surface region in cyan.
    wet_region = pred_original_size == 2
    overlay[wet_region] = (overlay[wet_region] * 0.6 + np.array([80, 255, 255]) * 0.4).astype(np.uint8)

    out = Image.fromarray(overlay)
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


def render_hybrid_overlay(
    rgb: np.ndarray,
    pred_original_size: np.ndarray,
    yolo_result: Dict[str, Any],
    scoring: Dict[str, Any],
    env_key: str,
    visualize_jpeg_quality: int,
) -> bytes:
    overlay = rgb.copy()

    stain_region = pred_original_size == 1
    overlay[stain_region] = (overlay[stain_region] * 0.5 + np.array([255, 80, 80]) * 0.5).astype(np.uint8)

    wet_region = pred_original_size == 2
    overlay[wet_region] = (overlay[wet_region] * 0.5 + np.array([80, 255, 255]) * 0.5).astype(np.uint8)

    composed = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    stain_mask = (pred_original_size == 1).astype(np.uint8) * 255
    wet_mask = (pred_original_size == 2).astype(np.uint8) * 255

    stain_contours = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wet_contours = cv2.findContours(wet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stain_contours = stain_contours[0] if len(stain_contours) == 2 else stain_contours[1]
    wet_contours = wet_contours[0] if len(wet_contours) == 2 else wet_contours[1]

    cv2.drawContours(composed, stain_contours, -1, (30, 30, 220), 2)
    cv2.drawContours(composed, wet_contours, -1, (220, 200, 30), 2)

    for item in yolo_result.get("results", []):
        x1, y1, x2, y2 = [int(v) for v in item.get("bbox", [0, 0, 0, 0])]
        class_name = str(item.get("class_name", "obj"))
        confidence = float(item.get("confidence", 0.0))
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(composed, (x1, y1), (x2, y2), (60, 200, 20), 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_top = max(0, y1 - text_h - 10)
        cv2.rectangle(composed, (x1, text_top), (x1 + text_w + 8, text_top + text_h + 8), (60, 200, 20), -1)
        cv2.putText(
            composed,
            label,
            (x1 + 4, text_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    verdict = str(scoring.get("verdict", "UNKNOWN")).upper()
    verdict_colors = {
        "PASS": (60, 175, 60),
        "PENDING": (0, 165, 255),
        "FAIL": (40, 40, 220),
    }
    verdict_color = verdict_colors.get(verdict, (180, 180, 180))

    panel_x1, panel_y1 = 12, 12
    panel_x2, panel_y2 = 560, 192
    cv2.rectangle(composed, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
    cv2.rectangle(composed, (panel_x1, panel_y1), (panel_x2, panel_y2), verdict_color, 2)

    info_lines = [
        f"ENV: {env_key}",
        f"VERDICT: {verdict}",
        f"QUALITY SCORE: {float(scoring.get('quality_score', 0.0)):.2f}",
        f"DIRTY COVERAGE: {float(100.0 - float(scoring.get('base_clean_score', 0.0))):.2f}%",
        f"YOLO DETECTIONS: {int(yolo_result.get('detections_count', 0))}",
    ]

    y_text = panel_y1 + 28
    for idx, line in enumerate(info_lines):
        color = (255, 255, 255) if idx != 1 else verdict_color
        cv2.putText(
            composed,
            line,
            (panel_x1 + 12, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        y_text += 30

    legend_y = panel_y2 + 28
    cv2.rectangle(composed, (panel_x1, legend_y), (panel_x1 + 18, legend_y + 18), (30, 30, 220), -1)
    cv2.putText(
        composed,
        "U-Net stain/water",
        (panel_x1 + 26, legend_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    legend_y += 28
    cv2.rectangle(composed, (panel_x1, legend_y), (panel_x1 + 18, legend_y + 18), (220, 200, 30), -1)
    cv2.putText(
        composed,
        "U-Net wet surface",
        (panel_x1 + 26, legend_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    legend_y += 28
    cv2.rectangle(composed, (panel_x1, legend_y), (panel_x1 + 18, legend_y + 18), (60, 200, 20), -1)
    cv2.putText(
        composed,
        "YOLO objects",
        (panel_x1 + 26, legend_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    out_rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(out_rgb)
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=visualize_jpeg_quality)
    buffer.seek(0)
    return buffer.getvalue()


def build_visualize_json_payload(
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    rendered: bytes,
) -> Dict[str, Any]:
    image_base64 = base64.b64encode(rendered).decode("ascii")
    return {
        "source_type": source_type,
        "source": source,
        "env": env_key,
        "mime_type": "image/jpeg",
        "encoding": "base64",
        "image_base64": image_base64,
        "scoring": scoring,
        "yolo": yolo_result,
        "unet": unet_result["summary"],
    }


def build_temp_visualization_url(request: Request, token: str, app_public_base_url: str) -> str:
    if app_public_base_url:
        return f"{app_public_base_url.rstrip('/')}/visualizations/{token}"
    return str(request.url_for("get_visualization_image", token=token))


def build_visualize_temp_url_payload(
    request: Request,
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    rendered: bytes,
    temp_visual_store: TempVisualizationStore,
    app_public_base_url: str,
) -> Dict[str, Any]:
    ticket = temp_visual_store.save(rendered, mime_type="image/jpeg")
    visualization_url = build_temp_visualization_url(request, ticket["token"], app_public_base_url)

    return {
        "source_type": source_type,
        "source": source,
        "env": env_key,
        "visualization": {
            "token": ticket["token"],
            "url": visualization_url,
            "mime_type": ticket["mime_type"],
            "byte_size": ticket["byte_size"],
            "ttl_seconds": ticket["ttl_seconds"],
            "expires_at_utc": ticket["expires_at_utc"],
        },
        "scoring": scoring,
        "yolo": yolo_result,
        "unet": unet_result["summary"],
    }


def build_visualize_blob_url_payload(
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    visualization_url: str,
    mime_type: str,
    byte_size: int,
) -> Dict[str, Any]:
    return {
        "source_type": source_type,
        "source": source,
        "env": env_key,
        "visualization": {
            "url": visualization_url,
            "mime_type": mime_type,
            "byte_size": byte_size,
        },
        "scoring": scoring,
        "yolo": yolo_result,
        "unet": unet_result["summary"],
    }
