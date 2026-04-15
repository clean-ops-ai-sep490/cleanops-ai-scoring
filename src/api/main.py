from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
import uvicorn
from ultralytics import YOLO
import io
from PIL import Image
import os
import sys
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import get_env_rules, settings
from src.api.inference_utils import (
    evaluate_image as evaluate_image_impl,
    evaluate_image_with_artifacts as evaluate_image_with_artifacts_impl,
    load_unet_model as load_unet_model_impl,
    unet_predict_from_pil as unet_predict_from_pil_impl,
    yolo_predict_from_pil as yolo_predict_from_pil_impl,
)
from src.api.openapi_utils import apply_custom_openapi
from src.api.retrain_api import retrain_router
from src.api.schemas import EvaluateVisualizeRequest, ImageURL
from src.api.scoring_utils import normalize_env as normalize_env_impl
from src.api.scoring_utils import parse_url_items as parse_url_items_impl
from src.api.visualization_utils import (
    build_visualize_blob_url_payload as build_visualize_blob_url_payload_impl,
    build_visualize_json_payload as build_visualize_json_payload_impl,
    render_hybrid_overlay as render_hybrid_overlay_impl,
    render_unet_overlay as render_unet_overlay_impl,
)
from src.storage.model_loader import ObjectStorageConfig, ObjectStorageModelLoader
from src.storage.visualization_blob_store import VisualizationBlobConfig, VisualizationBlobStore

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
apply_custom_openapi(app)
app.include_router(retrain_router)

PROJECT_ROOT = settings.project_root
BASE_OUTPUT_DIR = str(settings.base_output_dir)
MODEL_PATH = settings.model_path
UNET_MODEL_PATH = settings.unet_model_path
UNET_IMG_SIZE = settings.unet_img_size
MODEL_CACHE_DIR = settings.model_cache_dir
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

MODEL_STORAGE = ObjectStorageConfig(
    enabled=settings.model_storage_enabled,
    connection_string=settings.model_storage_connection_string,
    container=settings.model_storage_container,
    force_refresh=settings.model_force_refresh,
)
MODEL_LOADER = ObjectStorageModelLoader(MODEL_STORAGE, logger)
MODEL_REQUIRE_BLOB = settings.model_require_blob

VISUALIZATION_BLOB_STORE = VisualizationBlobStore(
    VisualizationBlobConfig(
        enabled=settings.visualization_blob_enabled,
        connection_string=settings.visualization_blob_connection_string,
        container=settings.visualization_blob_container,
        prefix=settings.visualization_blob_prefix,
    ),
    logger,
)

YOLO_MODEL_SOURCE = "unknown"
UNET_MODEL_SOURCE = "unknown"


# Load model global để tái sử dụng
try:
    yolo_cache_path = Path(MODEL_CACHE_DIR) / "active" / "yolo" / "model.pt"
    yolo_fallback_paths = [] if MODEL_REQUIRE_BLOB else ([Path(MODEL_PATH)] if MODEL_PATH else [])
    yolo_resolved_path, YOLO_MODEL_SOURCE = MODEL_LOADER.resolve_model_path(
        settings.model_storage_active_yolo_key,
        yolo_cache_path,
        yolo_fallback_paths,
    )

    if not yolo_resolved_path and settings.yolo_weights_path and not MODEL_REQUIRE_BLOB:
        yolo_resolved_path = settings.yolo_weights_path
        YOLO_MODEL_SOURCE = "ultralytics-default"

    if yolo_resolved_path:
        model = YOLO(yolo_resolved_path)
        MODEL_PATH = yolo_resolved_path
        print("✅ Load model thành công!")
        print(f"Model path: {MODEL_PATH}")
        print(f"Model source: {YOLO_MODEL_SOURCE}")
    else:
        model = None
        print("⚠️ Không tìm thấy YOLO model từ object storage hoặc local fallback.")
        print(f"YOLO source: {YOLO_MODEL_SOURCE}")
except Exception as e:
    model = None
    YOLO_MODEL_SOURCE = f"load-error:{type(e).__name__}"
    print(f"Lỗi khởi tạo model: {e}")


try:
    unet_cache_path = Path(MODEL_CACHE_DIR) / "active" / "unet" / "model.pth"
    unet_fallback_paths = [] if MODEL_REQUIRE_BLOB else [Path(UNET_MODEL_PATH)]
    unet_resolved_path, UNET_MODEL_SOURCE = MODEL_LOADER.resolve_model_path(
        settings.model_storage_active_unet_key,
        unet_cache_path,
        unet_fallback_paths,
    )

    if unet_resolved_path:
        UNET_MODEL_PATH = unet_resolved_path

    unet_model, UNET_IMG_SIZE = load_unet_model_impl(
        UNET_MODEL_PATH,
        unet_device,
        UNET_IMG_SIZE,
    )
    if unet_model is not None:
        print("✅ Load U-Net model thành công!")
        print(f"U-Net path: {UNET_MODEL_PATH}")
        print(f"U-Net source: {UNET_MODEL_SOURCE}")
        print(f"U-Net input size: {UNET_IMG_SIZE}")
    else:
        print(f"⚠️ Không tìm thấy U-Net checkpoint tại {UNET_MODEL_PATH}")
        print(f"U-Net source: {UNET_MODEL_SOURCE}")
except Exception as e:
    unet_model = None
    UNET_MODEL_SOURCE = f"load-error:{type(e).__name__}"
    print(f"Lỗi khởi tạo U-Net: {e}")




def normalize_env(env: Optional[str]) -> str:
    return normalize_env_impl(env, ENV_RULES)


def parse_url_items(image_urls: List[str]) -> List[str]:
    return parse_url_items_impl(image_urls)


def yolo_predict_from_pil(img: Image.Image) -> Dict[str, Any]:
    return yolo_predict_from_pil_impl(
        img,
        model=model,
        yolo_conf=YOLO_CONF,
    )


def unet_predict_from_pil(img: Image.Image) -> Dict[str, Any]:
    return unet_predict_from_pil_impl(
        img,
        unet_model=unet_model,
        unet_img_size=UNET_IMG_SIZE,
        unet_device=unet_device,
        class_map=CLASS_MAP,
    )


def evaluate_image_with_artifacts(img: Image.Image, env_key: str):
    return evaluate_image_with_artifacts_impl(
        img,
        env_key,
        model=model,
        unet_model=unet_model,
        yolo_conf=YOLO_CONF,
        unet_img_size=UNET_IMG_SIZE,
        unet_device=unet_device,
        class_map=CLASS_MAP,
        env_rules=ENV_RULES,
        pending_lower_bound=PENDING_LOWER_BOUND,
    )


def evaluate_image(img: Image.Image, env_key: str) -> Dict[str, Any]:
    return evaluate_image_impl(
        img,
        env_key,
        model=model,
        unet_model=unet_model,
        yolo_conf=YOLO_CONF,
        unet_img_size=UNET_IMG_SIZE,
        unet_device=unet_device,
        class_map=CLASS_MAP,
        env_rules=ENV_RULES,
        pending_lower_bound=PENDING_LOWER_BOUND,
    )


def render_unet_overlay(rgb, pred_original_size):
    return render_unet_overlay_impl(rgb, pred_original_size)


def render_hybrid_overlay(
    rgb,
    pred_original_size,
    yolo_result: Dict[str, Any],
    scoring: Dict[str, Any],
    env_key: str,
):
    return render_hybrid_overlay_impl(
        rgb,
        pred_original_size,
        yolo_result=yolo_result,
        scoring=scoring,
        env_key=env_key,
        visualize_jpeg_quality=VISUALIZE_JPEG_QUALITY,
    )


def build_visualize_json_payload(
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    rendered: bytes,
):
    return build_visualize_json_payload_impl(
        source_type=source_type,
        source=source,
        env_key=env_key,
        yolo_result=yolo_result,
        unet_result=unet_result,
        scoring=scoring,
        rendered=rendered,
    )


def build_visualize_blob_payload(
    source_type: str,
    source: str,
    env_key: str,
    yolo_result: Dict[str, Any],
    unet_result: Dict[str, Any],
    scoring: Dict[str, Any],
    rendered: bytes,
):
    upload_info = VISUALIZATION_BLOB_STORE.upload_visualization(
        image_bytes=rendered,
        source_type=source_type,
        source=source,
        env_key=env_key,
    )

    return build_visualize_blob_url_payload_impl(
        source_type=source_type,
        source=source,
        env_key=env_key,
        yolo_result=yolo_result,
        unet_result=unet_result,
        scoring=scoring,
        visualization_url=upload_info["url"],
        mime_type=upload_info["mime_type"],
        byte_size=upload_info["byte_size"],
    )

@app.get("/", tags=["production"])
def health_check():
    return {
        "status": "online",
        "yolo_loaded": model is not None,
        "unet_loaded": unet_model is not None,
        "yolo_model_path": MODEL_PATH,
        "yolo_model_source": YOLO_MODEL_SOURCE,
        "unet_model_path": UNET_MODEL_PATH,
        "unet_model_source": UNET_MODEL_SOURCE,
        "model_storage_enabled": MODEL_STORAGE.enabled,
        "model_storage_container": MODEL_STORAGE.container,
        "model_require_blob": MODEL_REQUIRE_BLOB,
        "max_batch_images": MAX_BATCH_IMAGES,
        "pending_lower_bound": PENDING_LOWER_BOUND,
        "visualize_jpeg_quality": VISUALIZE_JPEG_QUALITY,
        "visualize_temp_url_ttl_sec": VISUALIZE_TEMP_URL_TTL_SEC,
        "visualize_temp_max_items": VISUALIZE_TEMP_MAX_ITEMS,
        "visualization_blob_enabled": settings.visualization_blob_enabled,
        "visualization_blob_container": settings.visualization_blob_container,
        "visualization_blob_prefix": settings.visualization_blob_prefix,
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
    files: List[UploadFile] = File(default=[]),
    image_urls: List[str] = Form(default=[]),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    if not model:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not unet_model:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    upload_items = [u for u in files if (u.filename or "").strip()]
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
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    """
    Tra ve metadata + public blob URL cua anh overlay,
    phu hop frontend/mobile de xem anh detect truc tiep.
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

        return build_visualize_blob_payload(
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
async def evaluate_url_visualize_link(payload: EvaluateVisualizeRequest):
    """
    Tuong tu /evaluate-visualize-link nhung nhan URL anh,
    ket qua visualization la blob URL public.
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

        return build_visualize_blob_payload(
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
