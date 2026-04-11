from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
from fastapi.openapi.utils import get_openapi
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
    yolo_result = yolo_predict_from_pil(img)
    unet_result = unet_predict_from_pil(img)

    score = score_image(
        total_dirty_coverage_pct=unet_result["summary"]["total_dirty_coverage_pct"],
        detections_count=yolo_result["detections_count"],
        env_key=env_key,
    )

    return {
        "yolo": yolo_result,
        "unet": unet_result["summary"],
        "scoring": score,
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
        "env_rules": ENV_RULES,
        "message": "Welcome to Cleaning AI Hybrid API (YOLO + U-Net)"
    }

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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )
