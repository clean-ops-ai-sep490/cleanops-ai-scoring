from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any, Dict

import cv2
import numpy as np
from PIL import Image
from src.api.temp_store import TempVisualizationStore

if TYPE_CHECKING:
    from fastapi import Request
else:
    Request = Any


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def _truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 3)].rstrip()}..."


def _fit_font_scale_to_width(
    text: str,
    *,
    max_width_px: int,
    base_scale: float,
    min_scale: float,
    thickness: int,
) -> float:
    fitted = max(min_scale, base_scale)
    while fitted > min_scale:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fitted, thickness)
        if text_w <= max_width_px:
            return fitted
        fitted = round(fitted - 0.03, 3)
    return max(min_scale, fitted)


def _fit_text_to_width(
    text: str,
    *,
    max_width_px: int,
    font_scale: float,
    thickness: int,
    hard_limit: int = 80,
) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""
    if len(candidate) > hard_limit:
        candidate = _truncate_text(candidate, hard_limit)

    while candidate:
        (text_w, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if text_w <= max_width_px:
            return candidate
        if len(candidate) <= 4:
            return candidate
        candidate = _truncate_text(candidate, len(candidate) - 2)
    return ""


def _draw_labeled_box(
    image: np.ndarray,
    bbox: list[int],
    label: str,
    color: tuple[int, int, int],
    *,
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int],
    padding: int,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_top = max(0, y1 - text_h - (padding * 2))
    cv2.rectangle(
        image,
        (x1, text_top),
        (x1 + text_w + (padding * 2), text_top + text_h + (padding * 2)),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + padding, text_top + text_h + max(1, padding - 1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


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


def extract_dirty_region_candidates(
    pred_original_size: np.ndarray,
    *,
    max_regions: int = 6,
    min_area_ratio: float = 0.0025,
) -> list[Dict[str, Any]]:
    height, width = pred_original_size.shape[:2]
    image_area = max(1, height * width)
    region_id = 1
    candidates: list[Dict[str, Any]] = []
    class_meta = {
        1: "stain_or_water",
        2: "wet_surface",
    }

    for class_id, kind_hint in class_meta.items():
        mask = (pred_original_size == class_id).astype(np.uint8) * 255
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue

            area_ratio = area / image_area
            if area_ratio < min_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            candidates.append(
                {
                    "region_id": region_id,
                    "class_id": class_id,
                    "kind_hint": kind_hint,
                    "bbox_px": [int(x), int(y), int(x + w), int(y + h)],
                    "bbox_norm": [
                        round(x / max(1, width), 4),
                        round(y / max(1, height), 4),
                        round((x + w) / max(1, width), 4),
                        round((y + h) / max(1, height), 4),
                    ],
                    "area_pct": round(area_ratio * 100.0, 3),
                    "centroid_norm": [
                        round(cx / max(1, width), 4),
                        round(cy / max(1, height), 4),
                    ],
                }
            )
            region_id += 1

    candidates.sort(key=lambda item: float(item.get("area_pct", 0.0)), reverse=True)
    return candidates[:max_regions]


def render_hybrid_overlay(
    rgb: np.ndarray,
    pred_original_size: np.ndarray,
    yolo_result: Dict[str, Any],
    scoring: Dict[str, Any],
    env_key: str,
    visualize_jpeg_quality: int,
    visual_review: Dict[str, Any] | None = None,
    dirty_region_candidates: list[Dict[str, Any]] | None = None,
) -> bytes:
    overlay = rgb.copy()
    height, width = overlay.shape[:2]
    short_side = min(width, height)
    compact_mode = short_side < 720 or width < 960 or height < 720
    panel_scale = max(0.72, min(1.08, short_side / 900.0))
    if compact_mode:
        panel_scale *= 0.94
    contour_thickness = max(1, _clamp_int(panel_scale * 1.3, 1, 2))
    box_thickness = contour_thickness
    panel_thickness = contour_thickness
    headline_thickness = max(panel_thickness, _clamp_int(panel_scale * 1.6, 1, 2))
    label_font_scale = max(0.34, min(0.58, 0.42 * panel_scale))
    label_padding = _clamp_int(4 * panel_scale, 3, 7)
    panel_margin = _clamp_int(short_side * 0.018 * panel_scale, 8, 18)
    panel_padding = _clamp_int(10 * panel_scale, 8, 18)
    panel_font_scale = max(0.40, min(0.66, 0.48 * panel_scale))
    headline_font_scale = min(0.78, panel_font_scale * (1.18 if compact_mode else 1.14))
    legend_font_scale = max(0.32, panel_font_scale * 0.84)
    legend_box_size = _clamp_int(short_side * 0.022 * panel_scale, 12, 18)
    legend_gap = _clamp_int(short_side * 0.014 * panel_scale, 10, 16)

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

    cv2.drawContours(composed, stain_contours, -1, (30, 30, 220), contour_thickness)
    cv2.drawContours(composed, wet_contours, -1, (220, 200, 30), contour_thickness)

    dirty_region_candidates = dirty_region_candidates or []
    visual_review = visual_review or {}
    penalty_detection_indexes = {
        idx
        for idx in scoring.get("penalty_detection_indexes", [])
        if isinstance(idx, int)
    }
    # Production visualization should only draw object boxes that affect scoring.
    # Object advisory boxes are ignored here unless they become YOLO results and penalty indexes.
    advisory_object_boxes: list[Dict[str, Any]] = []
    advisory_dirty_boxes = [
        item for item in visual_review.get("advisory_dirty_boxes", []) if isinstance(item, dict)
    ]
    dirty_label_map = {
        int(item.get("region_id")): str(item.get("label", "")).strip()
        for item in visual_review.get("dirty_region_labels", [])
        if isinstance(item, dict) and isinstance(item.get("region_id"), int)
    }
    highlight_region_ids = {
        idx
        for idx in visual_review.get("highlight_dirty_region_ids", [])
        if isinstance(idx, int)
    }

    for idx, item in enumerate(yolo_result.get("results", [])):
        if idx not in penalty_detection_indexes:
            continue
        x1, y1, x2, y2 = [int(v) for v in item.get("bbox", [0, 0, 0, 0])]
        class_name = str(item.get("class_name", "obj"))
        confidence = float(item.get("confidence", 0.0))
        label = _truncate_text(f"{class_name} {confidence:.2f}", 28)
        _draw_labeled_box(
            composed,
            [x1, y1, x2, y2],
            label,
            (60, 200, 20),
            font_scale=label_font_scale,
            thickness=box_thickness,
            text_color=(0, 0, 0),
            padding=label_padding,
        )

    for candidate in dirty_region_candidates:
        region_id = int(candidate.get("region_id", 0))
        if region_id not in highlight_region_ids:
            continue

        x1, y1, x2, y2 = [int(v) for v in candidate.get("bbox_px", [0, 0, 0, 0])]
        label = dirty_label_map.get(region_id) or str(candidate.get("kind_hint", "dirty_zone")).replace("_", " ")
        _draw_labeled_box(
            composed,
            [x1, y1, x2, y2],
            _truncate_text(label, 28),
            (30, 30, 220),
            font_scale=label_font_scale,
            thickness=box_thickness,
            text_color=(255, 255, 255),
            padding=label_padding,
        )

    for item in advisory_object_boxes:
        x1, y1, x2, y2 = [int(v) for v in item.get("bbox_px", [0, 0, 0, 0])]
        label = _truncate_text(
            f"{str(item.get('label', 'object')).replace('_', ' ')} {float(item.get('confidence', 0.0)):.2f}",
            28,
        )
        _draw_labeled_box(
            composed,
            [x1, y1, x2, y2],
            label,
            (60, 200, 20),
            font_scale=label_font_scale,
            thickness=box_thickness,
            text_color=(0, 0, 0),
            padding=label_padding,
        )

    for item in advisory_dirty_boxes:
        x1, y1, x2, y2 = [int(v) for v in item.get("bbox_px", [0, 0, 0, 0])]
        label = _truncate_text(
            f"{str(item.get('label', 'dirty_zone')).replace('_', ' ')} {float(item.get('confidence', 0.0)):.2f}",
            28,
        )
        _draw_labeled_box(
            composed,
            [x1, y1, x2, y2],
            label,
            (30, 30, 220),
            font_scale=label_font_scale,
            thickness=box_thickness,
            text_color=(255, 255, 255),
            padding=label_padding,
        )

    verdict = str(scoring.get("verdict", "UNKNOWN")).upper()
    verdict_colors = {
        "PASS": (60, 175, 60),
        "PENDING": (0, 165, 255),
        "FAIL": (40, 40, 220),
    }
    verdict_color = verdict_colors.get(verdict, (180, 180, 180))

    core_lines = [
        f"VERDICT: {verdict}",
        f"QUALITY SCORE: {float(scoring.get('quality_score', 0.0)):.2f}",
        f"DIRTY COVERAGE: {float(100.0 - float(scoring.get('base_clean_score', 0.0))):.2f}%",
        f"PENALTY OBJECTS: {int(scoring.get('penalty_detections_count', 0))}",
    ]
    optional_lines: list[str] = []
    if not compact_mode:
        optional_lines.append(f"OBJECT PENALTY: {float(scoring.get('object_penalty', 0.0)):.2f}")
        optional_lines.append(f"ENV: {env_key}")
    if (highlight_region_ids or advisory_object_boxes or advisory_dirty_boxes) and not compact_mode:
        optional_lines.append("AI REVIEWED OVERLAY")
    overlay_summary = str(visual_review.get("overlay_summary", "")).strip()
    if overlay_summary and not compact_mode:
        optional_lines.append(f"NOTE: {_truncate_text(overlay_summary, 56)}")

    max_panel_height = int(height * (0.30 if compact_mode else 0.42))
    max_panel_width = max(170, min(420, int(width * 0.42)))
    panel_width = _clamp_int(width * (0.38 if compact_mode else 0.34), 190 if compact_mode else 200, max_panel_width)
    panel_x1, panel_y1 = panel_margin, panel_margin
    panel_x2 = min(width - panel_margin, panel_x1 + panel_width)
    max_text_width_px = max(80, panel_x2 - panel_x1 - (panel_padding * 2))

    mandatory_rows = [
        {"text": line, "kind": "headline" if idx < 2 else "body", "required": True}
        for idx, line in enumerate(core_lines)
    ]
    optional_rows = [{"text": line, "kind": "body", "required": False} for line in optional_lines]
    rendered_rows: list[dict[str, Any]] = []
    current_panel_height = panel_padding * 2

    for row in mandatory_rows + optional_rows:
        is_headline = row["kind"] == "headline"
        base_scale = headline_font_scale if is_headline else panel_font_scale
        min_scale = max(0.34, base_scale * (0.86 if is_headline else 0.9))
        thickness = headline_thickness if is_headline else panel_thickness
        fitted_scale = _fit_font_scale_to_width(
            row["text"],
            max_width_px=max_text_width_px,
            base_scale=base_scale,
            min_scale=min_scale,
            thickness=thickness,
        )
        fitted_text = _fit_text_to_width(
            row["text"],
            max_width_px=max_text_width_px,
            font_scale=fitted_scale,
            thickness=thickness,
            hard_limit=64 if is_headline else 72,
        )
        (_, text_h), _ = cv2.getTextSize(fitted_text or " ", cv2.FONT_HERSHEY_SIMPLEX, fitted_scale, thickness)
        line_gap = _clamp_int((7 if is_headline else 6) * panel_scale, 6, 14)
        projected_height = current_panel_height + text_h + line_gap
        if not row["required"] and projected_height > max_panel_height:
            continue
        rendered_rows.append(
            {
                "text": fitted_text,
                "font_scale": fitted_scale,
                "thickness": thickness,
                "text_h": text_h,
                "line_gap": line_gap,
                "is_headline": is_headline,
            }
        )
        current_panel_height = projected_height

    panel_y2 = min(height - panel_margin, panel_y1 + current_panel_height)
    cv2.rectangle(composed, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
    cv2.rectangle(composed, (panel_x1, panel_y1), (panel_x2, panel_y2), verdict_color, panel_thickness)

    y_text = panel_y1 + panel_padding
    for row in rendered_rows:
        baseline_y = y_text + row["text_h"]
        color = verdict_color if row["text"].startswith("VERDICT:") else (255, 255, 255)
        cv2.putText(
            composed,
            row["text"],
            (panel_x1 + panel_padding, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            row["font_scale"],
            color,
            row["thickness"],
            cv2.LINE_AA,
        )
        y_text = baseline_y + row["line_gap"]

    legend_items = [
        ((30, 30, 220), "Dirty area"),
        ((220, 200, 30), "Wet surface"),
        ((60, 200, 20), "Trash-like objects"),
    ]
    legend_required_height = len(legend_items) * (legend_box_size + (legend_gap // 2))
    show_legend = not compact_mode and (height - panel_y2 - panel_margin) >= legend_required_height
    if show_legend:
        legend_y = panel_y2 + legend_gap
        for color, text in legend_items:
            cv2.rectangle(
                composed,
                (panel_x1, legend_y),
                (panel_x1 + legend_box_size, legend_y + legend_box_size),
                color,
                -1,
            )
            cv2.putText(
                composed,
                text,
                (panel_x1 + legend_box_size + 8, legend_y + legend_box_size - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                legend_font_scale,
                (30, 30, 30),
                panel_thickness,
                cv2.LINE_AA,
            )
            legend_y += legend_box_size + (legend_gap // 2)

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
    llm_filter: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    image_base64 = base64.b64encode(rendered).decode("ascii")
    payload = {
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
    if llm_filter:
        payload.update(llm_filter)
    return payload


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
    llm_filter: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
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
    if llm_filter:
        payload.update(llm_filter)
    return payload
