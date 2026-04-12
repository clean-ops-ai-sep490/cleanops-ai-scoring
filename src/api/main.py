from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.openapi.utils import get_openapi
import base64
import uvicorn
from ultralytics import YOLO
import io
from PIL import Image
import os
import sys
import requests
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import get_env_rules, settings
from src.models.unet_segmenter import UNetSegmenter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cleaning AI POC API",
    description="API for YOLOv8 object detection and U-Net segmentation",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "production",
            "description": "Stable endpoints intended for backend integration and production usage.",
        },
        {
            "name": "test",
            "description": "Utility endpoints for debugging, inspection, and visual validation.",
        },
    ],
)


def _patch_binary_content_media_type(node: Any):
    if isinstance(node, dict):
        if (
            node.get("type") == "string"
            and node.get("contentMediaType") == "application/octet-stream"
        ):
            node.pop("contentMediaType", None)
            node["format"] = "binary"

        for value in node.values():
            _patch_binary_content_media_type(value)
    elif isinstance(node, list):
        for item in node:
            _patch_binary_content_media_type(item)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        openapi_version="3.0.3",
        routes=app.routes,
        tags=app.openapi_tags,
    )

    _patch_binary_content_media_type(schema)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

PROJECT_ROOT = settings.project_root
BASE_OUTPUT_DIR = str(settings.base_output_dir)
MODEL_PATH = settings.model_path
UNET_MODEL_PATH = settings.unet_model_path
UNET_IMG_SIZE = settings.unet_img_size
unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAP = {
    0: "background",
    1: "stain_or_water",
    2: "wet_surface",
}

ENV_RULES = get_env_rules()
MAX_BATCH_IMAGES = settings.max_batch_images
PENDING_LOWER_BOUND = settings.pending_lower_bound
YOLO_CONF = settings.yolo_conf
REQUEST_TIMEOUT_SEC = settings.request_timeout_sec
VISUALIZE_JPEG_QUALITY = max(20, min(100, settings.visualize_jpeg_quality))
APP_PUBLIC_BASE_URL = settings.app_public_base_url
VISUALIZE_TEMP_URL_TTL_SEC = max(30, settings.visualize_temp_url_ttl_sec)
VISUALIZE_TEMP_MAX_ITEMS = max(50, settings.visualize_temp_max_items)


class TempVisualizationStore:
    def __init__(self, ttl_sec: int, max_items: int):
        self._ttl_sec = ttl_sec
        self._max_items = max_items
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _prune_locked(self, now: datetime):
        expired_keys = [k for k, v in self._items.items() if v["expires_at"] <= now]
        for key in expired_keys:
            self._items.pop(key, None)

        overflow = len(self._items) - self._max_items
        if overflow > 0:
            sorted_keys = sorted(self._items.items(), key=lambda kv: kv[1]["expires_at"])
            for key, _ in sorted_keys[:overflow]:
                self._items.pop(key, None)

    def save(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._ttl_sec)
        token = uuid.uuid4().hex

        with self._lock:
            self._prune_locked(now)
            self._items[token] = {
                "image": image_bytes,
                "mime_type": mime_type,
                "expires_at": expires_at,
            }

        return {
            "token": token,
            "mime_type": mime_type,
            "byte_size": len(image_bytes),
            "ttl_seconds": self._ttl_sec,
            "expires_at_utc": expires_at.isoformat().replace("+00:00", "Z"),
        }

    def get(self, token: str) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._prune_locked(now)
            item = self._items.get(token)
            if item is None:
                return None
            if item["expires_at"] <= now:
                self._items.pop(token, None)
                return None
            return item


temp_visual_store = TempVisualizationStore(
    ttl_sec=VISUALIZE_TEMP_URL_TTL_SEC,
    max_items=VISUALIZE_TEMP_MAX_ITEMS,
)

# Load model global để tái sử dụng
try:
    if MODEL_PATH:
        model = YOLO(MODEL_PATH)
        print("✅ Load model thành công!")
        print(f"Model path: {MODEL_PATH}")
    else:
        model = None
        print("⚠️ Không có cấu hình MODEL_PATH, YOLO model chưa sẵn sàng.")
except Exception as e:
    model = None
    print(f"Lỗi khởi tạo model: {e}")


def load_unet_model(model_path: str):
    global UNET_IMG_SIZE
    if not model_path or not Path(model_path).exists():
        return None

    ckpt = torch.load(model_path, map_location=unet_device)
    encoder = "resnet50"
    if isinstance(ckpt, dict):
        encoder = ckpt.get("encoder", "resnet50")
        UNET_IMG_SIZE = int(ckpt.get("img_size", 384))

    unet_model = UNetSegmenter(encoder_name=encoder, classes=3).to(unet_device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        unet_model.load_state_dict(ckpt["model_state"])
    else:
        unet_model.load_state_dict(ckpt)
    unet_model.eval()
    return unet_model


try:
    unet_model = load_unet_model(UNET_MODEL_PATH)
    if unet_model is not None:
        print("✅ Load U-Net model thành công!")
        print(f"U-Net path: {UNET_MODEL_PATH}")
        print(f"U-Net input size: {UNET_IMG_SIZE}")
    else:
        print(f"⚠️ Không tìm thấy U-Net checkpoint tại {UNET_MODEL_PATH}")
except Exception as e:
    unet_model = None
    print(f"Lỗi khởi tạo U-Net: {e}")


def unet_predict_from_pil(img: Image.Image):
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    resized = cv2.resize(rgb, (UNET_IMG_SIZE, UNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
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
            "model_input_size": UNET_IMG_SIZE,
            "class_mapping": CLASS_MAP,
            "stain_or_water_pixels": stain_pixels,
            "wet_surface_pixels": wet_pixels,
            "stain_or_water_coverage_pct": round(stain_pct, 3),
            "wet_surface_coverage_pct": round(wet_pct, 3),
            "total_dirty_coverage_pct": round(dirty_pct, 3),
        },
        "rgb": rgb,
    }


def render_unet_overlay(rgb: np.ndarray, pred_original_size: np.ndarray):
    overlay = rgb.copy()

    # Blend vùng stain/water bằng màu đỏ.
    stain_region = pred_original_size == 1
    overlay[stain_region] = (overlay[stain_region] * 0.6 + np.array([255, 80, 80]) * 0.4).astype(np.uint8)

    # Blend vùng wet surface bằng màu cyan.
    wet_region = pred_original_size == 2
    overlay[wet_region] = (overlay[wet_region] * 0.6 + np.array([80, 255, 255]) * 0.4).astype(np.uint8)

    out = Image.fromarray(overlay)
    b = io.BytesIO()
    out.save(b, format="JPEG")
    b.seek(0)
    return b.getvalue()


def yolo_predict_from_pil(img: Image.Image):
    results = model.predict(source=img, conf=YOLO_CONF, save=False, verbose=False)

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


def normalize_env(env: Optional[str]):
    env_key = (env or "LOBBY_CORRIDOR").strip().upper()
    if env_key not in ENV_RULES:
        raise ValueError(
            f"Unsupported env '{env_key}'. Allowed envs: {', '.join(sorted(ENV_RULES.keys()))}"
        )
    return env_key


def clamp(v: float, lo: float, hi: float):
    return max(lo, min(v, hi))


def score_image(total_dirty_coverage_pct: float, detections_count: int, env_key: str):
    base_clean_score = 100.0 - float(total_dirty_coverage_pct)
    object_penalty = min(30.0, float(detections_count) * 5.0)
    quality_score = clamp(base_clean_score - object_penalty, 0.0, 100.0)

    pass_threshold = float(ENV_RULES[env_key]["pass_threshold"])
    if quality_score >= pass_threshold:
        verdict = "PASS"
    elif quality_score >= PENDING_LOWER_BOUND:
        verdict = "PENDING"
    else:
        verdict = "FAIL"

    reasons = []
    if total_dirty_coverage_pct >= 20.0:
        reasons.append("coverage high")
    if detections_count > 0:
        reasons.append("objects remain")
    if not reasons:
        reasons.append("good cleanliness")

    return {
        "base_clean_score": round(base_clean_score, 3),
        "object_penalty": round(object_penalty, 3),
        "quality_score": round(quality_score, 3),
        "pass_threshold": pass_threshold,
        "verdict": verdict,
        "reasons": reasons,
    }


def evaluate_image(img: Image.Image, env_key: str):
    yolo_result, unet_result, score = evaluate_image_with_artifacts(img, env_key)

    return {
        "yolo": yolo_result,
        "unet": unet_result["summary"],
        "scoring": score,
    }


def evaluate_image_with_artifacts(img: Image.Image, env_key: str):
    yolo_result = yolo_predict_from_pil(img)
    unet_result = unet_predict_from_pil(img)

    score = score_image(
        total_dirty_coverage_pct=unet_result["summary"]["total_dirty_coverage_pct"],
        detections_count=yolo_result["detections_count"],
        env_key=env_key,
    )
    return yolo_result, unet_result, score


def render_hybrid_overlay(
    rgb: np.ndarray,
    pred_original_size: np.ndarray,
    yolo_result: Dict[str, Any],
    scoring: Dict[str, Any],
    env_key: str,
):
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
    cv2.putText(composed, "U-Net stain/water", (panel_x1 + 26, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2, cv2.LINE_AA)

    legend_y += 28
    cv2.rectangle(composed, (panel_x1, legend_y), (panel_x1 + 18, legend_y + 18), (220, 200, 30), -1)
    cv2.putText(composed, "U-Net wet surface", (panel_x1 + 26, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2, cv2.LINE_AA)

    legend_y += 28
    cv2.rectangle(composed, (panel_x1, legend_y), (panel_x1 + 18, legend_y + 18), (60, 200, 20), -1)
    cv2.putText(composed, "YOLO objects", (panel_x1 + 26, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(out_rgb)
    b = io.BytesIO()
    out.save(b, format="JPEG", quality=VISUALIZE_JPEG_QUALITY)
    b.seek(0)
    return b.getvalue()


def build_visualize_json_payload(
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    rendered: bytes,
):
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


def build_temp_visualization_url(request: Request, token: str) -> str:
    if APP_PUBLIC_BASE_URL:
        return f"{APP_PUBLIC_BASE_URL.rstrip('/')}/visualizations/{token}"
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
):
    ticket = temp_visual_store.save(rendered, mime_type="image/jpeg")
    visualization_url = build_temp_visualization_url(request, ticket["token"])

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


def parse_url_items(image_urls: List[str]) -> List[str]:
    parsed: List[str] = []
    for raw in image_urls:
        if not isinstance(raw, str):
            continue

        candidate = raw.strip()
        if not candidate:
            continue

        # Swagger/UI integrations may send many URLs in one comma-separated string.
        parts = [p.strip() for p in candidate.split(",")]
        for part in parts:
            if part:
                parsed.append(part)

    return parsed

@app.get("/", tags=["production"])
def health_check():
    return {
        "status": "online",
        "yolo_loaded": model is not None,
        "unet_loaded": unet_model is not None,
        "yolo_model_path": MODEL_PATH,
        "unet_model_path": UNET_MODEL_PATH,
        "max_batch_images": MAX_BATCH_IMAGES,
        "pending_lower_bound": PENDING_LOWER_BOUND,
        "visualize_jpeg_quality": VISUALIZE_JPEG_QUALITY,
        "visualize_temp_url_ttl_sec": VISUALIZE_TEMP_URL_TTL_SEC,
        "visualize_temp_max_items": VISUALIZE_TEMP_MAX_ITEMS,
        "env_rules": ENV_RULES,
        "message": "Welcome to Cleaning AI Hybrid API (YOLO + U-Net)"
    }


@app.get("/visualizations/{token}", tags=["production"])
def get_visualization_image(token: str):
    item = temp_visual_store.get(token)
    if item is None:
        return JSONResponse(status_code=404, content={"error": "Visualization not found or expired."})

    return Response(
        content=item["image"],
        media_type=item["mime_type"],
        headers={"Cache-Control": "no-store, max-age=0"},
    )

@app.post("/predict", tags=["production"])
async def predict_image(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"}
        )
    
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result = yolo_predict_from_pil(img)

        return {
            "filename": file.filename,
            **yolo_result,
        }
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

class ImageURL(BaseModel):
    url: str


class EvaluateVisualizeRequest(BaseModel):
    url: str
    env: Optional[str] = "LOBBY_CORRIDOR"


@app.post("/predict-unet", tags=["production"])
async def predict_unet(file: UploadFile = File(...)):
    if not unet_model:
        return JSONResponse(
            status_code=500,
            content={"error": "U-Net model chưa được tải. Kiểm tra checkpoint unet_multiclass_best.pth"},
        )

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        result = unet_predict_from_pil(img)
        return {
            "filename": file.filename,
            **result["summary"],
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/predict-unet-url", tags=["production"])
async def predict_unet_url(payload: ImageURL):
    if not unet_model:
        return JSONResponse(
            status_code=500,
            content={"error": "U-Net model chưa được tải. Kiểm tra checkpoint unet_multiclass_best.pth"},
        )

    try:
        response = requests.get(payload.url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        result = unet_predict_from_pil(img)
        return {
            "url": payload.url,
            **result["summary"],
        }
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/predict-unet-url-visualize", tags=["test"])
async def predict_unet_url_visualize(payload: ImageURL):
    if not unet_model:
        return JSONResponse(
            status_code=500,
            content={"error": "U-Net model chưa được tải. Kiểm tra checkpoint unet_multiclass_best.pth"},
        )

    try:
        response = requests.get(payload.url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        result = unet_predict_from_pil(img)
        rendered = render_unet_overlay(result["rgb"], result["mask_original_size"])
        return Response(content=rendered, media_type="image/jpeg")
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict-url", tags=["production"])
async def predict_image_url(payload: ImageURL):
    if not model:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"}
        )
    
    try:
        response = requests.get(payload.url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        yolo_result = yolo_predict_from_pil(img)

        return {
            "url": payload.url,
            **yolo_result,
        }
        
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/evaluate-batch", tags=["production"])
async def evaluate_batch(
    files: List[UploadFile | str] = File(default=[]),
    image_urls: List[str] = Form(default=[]),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    upload_items = [u for u in files if isinstance(u, UploadFile) and (u.filename or "").strip()]
    url_items = parse_url_items(image_urls)

    total_images = len(upload_items) + len(url_items)
    if total_images == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Request must include at least 1 image via files or image_urls."},
        )
    if total_images > MAX_BATCH_IMAGES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Maximum {MAX_BATCH_IMAGES} images per request."},
        )

    try:
        env_key = normalize_env(env)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    results = []
    processed_count = 0
    skipped_count = 0
    pass_count = 0
    pending_count = 0
    fail_count = 0

    seq = 1
    for upload in upload_items:
        try:
            content = await upload.read()
            if not content:
                logger.warning("Skip empty upload in /evaluate-batch: filename=%s", upload.filename)
                skipped_count += 1
                seq += 1
                continue
            img = Image.open(io.BytesIO(content)).convert("RGB")
            eval_result = evaluate_image(img, env_key)

            verdict = eval_result["scoring"]["verdict"]
            pass_count += int(verdict == "PASS")
            pending_count += int(verdict == "PENDING")
            fail_count += int(verdict == "FAIL")

            results.append(
                {
                    "id": seq,
                    "source_type": "upload",
                    "source": upload.filename,
                    **eval_result,
                }
            )
            processed_count += 1
        except Exception as e:
            skipped_count += 1
            logger.warning(
                "Skip failed upload in /evaluate-batch: filename=%s, error=%s",
                upload.filename,
                str(e),
            )
        seq += 1

    for url in url_items:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            eval_result = evaluate_image(img, env_key)

            verdict = eval_result["scoring"]["verdict"]
            pass_count += int(verdict == "PASS")
            pending_count += int(verdict == "PENDING")
            fail_count += int(verdict == "FAIL")

            results.append(
                {
                    "id": seq,
                    "source_type": "url",
                    "source": url,
                    **eval_result,
                }
            )
            processed_count += 1
        except Exception as e:
            skipped_count += 1
            logger.warning(
                "Skip failed URL in /evaluate-batch: url=%s, error=%s",
                url,
                str(e),
            )
        seq += 1

    return {
        "env": env_key,
        "env_label": ENV_RULES[env_key]["label"],
        "max_batch_images": MAX_BATCH_IMAGES,
        "pending_lower_bound": PENDING_LOWER_BOUND,
        "summary": {
            "total_requested": total_images,
            "processed": processed_count,
            "skipped": skipped_count,
            "pass": pass_count,
            "pending": pending_count,
            "fail": fail_count,
        },
        "results": results,
    }

@app.post("/predict-url-visualize", tags=["test"])
async def predict_image_url_visualize(payload: ImageURL):
    """
    Nhận diện qua URL và trả về trực tiếp ảnh đã vẽ khung (ảnh kết quả)
    giúp dễ dàng xem bằng mắt thường.
    """
    if not model:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"}
        )
    
    try:
        # Tải ảnh gốc từ URL
        response = requests.get(payload.url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Dự đoán
        results = model.predict(source=img, conf=YOLO_CONF, save=False)
        
        # Lấy mảng dữ liệu ảnh đã được YOLO vẽ sẵn khung (numpy array)
        im_array = results[0].plot()  # raw numpy [H, W, C] (BGR)
        
        # Đảo màu BGR sang RGB cho Pillow (vì OpenCV vẽ theo chuẩn BGR)
        im_array_rgb = im_array[..., ::-1]
        
        # Chuyển numpy array thành dạng Image
        res_img = Image.fromarray(im_array_rgb)
        
        # Lưu vào Byte Buffer (bộ nhớ ảo) thay vì ghi ổ cứng
        img_byte_arr = io.BytesIO()
        res_img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Trả trực tiếp file ảnh .jpeg ra trình duyệt / màn hình UI
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
        
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/evaluate-visualize", tags=["test"])
async def evaluate_visualize(
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    """
    Tra ve anh da khoanh vung tong hop:
    - Bounding boxes tu YOLO
    - Mask/contour tu U-Net
    - Verdict + score tren anh
    """
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = normalize_env(env)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result, unet_result, scoring = evaluate_image_with_artifacts(img, env_key)
        rendered = render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )
        return Response(content=rendered, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/evaluate-url-visualize", tags=["test"])
async def evaluate_url_visualize(payload: EvaluateVisualizeRequest):
    """
    Giong /evaluate-visualize nhung nhan URL anh thay vi file upload.
    """
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = normalize_env(payload.env)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        response = requests.get(payload.url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        yolo_result, unet_result, scoring = evaluate_image_with_artifacts(img, env_key)
        rendered = render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )
        return Response(content=rendered, media_type="image/jpeg")
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/evaluate-visualize-json", tags=["production"])
async def evaluate_visualize_json(
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    """
    Tra ve JSON gom ket qua danh gia va anh overlay dang base64,
    phu hop de frontend render truc tiep khong can luu file tam.
    """
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = normalize_env(env)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result, unet_result, scoring = evaluate_image_with_artifacts(img, env_key)
        rendered = render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return build_visualize_json_payload(
            source_type="upload",
            source=file.filename or "upload",
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/evaluate-url-visualize-json", tags=["production"])
async def evaluate_url_visualize_json(payload: EvaluateVisualizeRequest):
    """
    Tuong tu /evaluate-visualize-json nhung nhan URL anh.
    """
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = normalize_env(payload.env)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        response = requests.get(payload.url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        yolo_result, unet_result, scoring = evaluate_image_with_artifacts(img, env_key)
        rendered = render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return build_visualize_json_payload(
            source_type="url",
            source=payload.url,
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/evaluate-visualize-link", tags=["production"])
async def evaluate_visualize_link(
    request: Request,
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    """
    Tra ve metadata + temporary URL cua anh overlay,
    phu hop mobile app de giam payload thay vi base64.
    """
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = normalize_env(env)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result, unet_result, scoring = evaluate_image_with_artifacts(img, env_key)
        rendered = render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return build_visualize_temp_url_payload(
            request=request,
            source_type="upload",
            source=file.filename or "upload",
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/evaluate-url-visualize-link", tags=["production"])
async def evaluate_url_visualize_link(request: Request, payload: EvaluateVisualizeRequest):
    """
    Tuong tu /evaluate-visualize-link nhung nhan URL anh.
    """
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = normalize_env(payload.env)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        response = requests.get(payload.url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        yolo_result, unet_result, scoring = evaluate_image_with_artifacts(img, env_key)
        rendered = render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return build_visualize_temp_url_payload(
            request=request,
            source_type="url",
            source=payload.url,
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )
