from __future__ import annotations

import io
from typing import List

import requests
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from src.api import app_state
from src.api.schemas import EvaluateVisualizeRequest, ImageURL

router = APIRouter(tags=["scoring"])


@router.post("/predict", tags=["production"])
async def predict_image(file: UploadFile = File(...)):
    if not app_state.MODEL:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"},
        )

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result = app_state.yolo_predict_from_pil(img)

        return {
            "filename": file.filename,
            **yolo_result,
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/predict-unet", tags=["production"])
async def predict_unet(file: UploadFile = File(...)):
    if not app_state.UNET_MODEL:
        return JSONResponse(
            status_code=500,
            content={"error": "U-Net model chưa được tải. Kiểm tra checkpoint unet_multiclass_best.pth"},
        )

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        result = app_state.unet_predict_from_pil(img)
        return {
            "filename": file.filename,
            **result["summary"],
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/predict-unet-url", tags=["production"])
async def predict_unet_url(payload: ImageURL):
    if not app_state.UNET_MODEL:
        return JSONResponse(
            status_code=500,
            content={"error": "U-Net model chưa được tải. Kiểm tra checkpoint unet_multiclass_best.pth"},
        )

    try:
        response = requests.get(payload.url, timeout=app_state.REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        result = app_state.unet_predict_from_pil(img)
        return {
            "url": payload.url,
            **result["summary"],
        }
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/predict-unet-url-visualize", tags=["test"])
async def predict_unet_url_visualize(payload: ImageURL):
    if not app_state.UNET_MODEL:
        return JSONResponse(
            status_code=500,
            content={"error": "U-Net model chưa được tải. Kiểm tra checkpoint unet_multiclass_best.pth"},
        )

    try:
        response = requests.get(payload.url, timeout=app_state.REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        result = app_state.unet_predict_from_pil(img)
        rendered = app_state.render_unet_overlay(result["rgb"], result["mask_original_size"])
        return Response(content=rendered, media_type="image/jpeg")
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/predict-url", tags=["production"])
async def predict_image_url(payload: ImageURL):
    if not app_state.MODEL:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"},
        )

    try:
        response = requests.get(payload.url, timeout=app_state.REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        yolo_result = app_state.yolo_predict_from_pil(img)

        return {
            "url": payload.url,
            **yolo_result,
        }
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-batch", tags=["production"])
async def evaluate_batch(
    files: List[UploadFile] = File(default=[]),
    image_urls: List[str] = Form(default=[]),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    upload_items = [u for u in files if (u.filename or "").strip()]
    url_items = app_state.parse_url_items(image_urls)

    total_images = len(upload_items) + len(url_items)
    if total_images == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Request must include at least 1 image via files or image_urls."},
        )
    if total_images > app_state.MAX_BATCH_IMAGES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Maximum {app_state.MAX_BATCH_IMAGES} images per request."},
        )

    try:
        env_key = app_state.normalize_env(env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

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
                app_state.logger.warning("Skip empty upload in /evaluate-batch: filename=%s", upload.filename)
                skipped_count += 1
                seq += 1
                continue
            img = Image.open(io.BytesIO(content)).convert("RGB")
            eval_result = app_state.evaluate_image(img, env_key)

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
        except Exception as exc:  # noqa: BLE001
            skipped_count += 1
            app_state.logger.warning(
                "Skip failed upload in /evaluate-batch: filename=%s, error=%s",
                upload.filename,
                str(exc),
            )
        seq += 1

    for url in url_items:
        try:
            response = requests.get(url, timeout=app_state.REQUEST_TIMEOUT_SEC)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            eval_result = app_state.evaluate_image(img, env_key)

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
        except Exception as exc:  # noqa: BLE001
            skipped_count += 1
            app_state.logger.warning(
                "Skip failed URL in /evaluate-batch: url=%s, error=%s",
                url,
                str(exc),
            )
        seq += 1

    return {
        "env": env_key,
        "env_label": app_state.ENV_RULES[env_key]["label"],
        "max_batch_images": app_state.MAX_BATCH_IMAGES,
        "pending_lower_bound": app_state.PENDING_LOWER_BOUND,
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


@router.post("/predict-url-visualize", tags=["test"])
async def predict_image_url_visualize(payload: ImageURL):
    if not app_state.MODEL:
        return JSONResponse(
            status_code=500,
            content={"error": "Model chưa được tải lên hệ thống. Vui lòng kiểm tra lại training!"},
        )

    try:
        response = requests.get(payload.url, timeout=app_state.REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        results = app_state.MODEL.predict(source=img, conf=app_state.YOLO_CONF, save=False)
        im_array = results[0].plot()
        im_array_rgb = im_array[..., ::-1]
        res_img = Image.fromarray(im_array_rgb)
        img_byte_arr = io.BytesIO()
        res_img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-visualize", tags=["test"])
async def evaluate_visualize(
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = app_state.normalize_env(env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result, unet_result, scoring = app_state.evaluate_image_with_artifacts(img, env_key)
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )
        return Response(content=rendered, media_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-url-visualize", tags=["test"])
async def evaluate_url_visualize(payload: EvaluateVisualizeRequest):
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = app_state.normalize_env(payload.env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        response = requests.get(payload.url, timeout=app_state.REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        yolo_result, unet_result, scoring = app_state.evaluate_image_with_artifacts(img, env_key)
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )
        return Response(content=rendered, media_type="image/jpeg")
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-visualize-json", tags=["production"])
async def evaluate_visualize_json(
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = app_state.normalize_env(env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result, unet_result, scoring = app_state.evaluate_image_with_artifacts(img, env_key)
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return app_state.build_visualize_json_payload(
            source_type="upload",
            source=file.filename or "upload",
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-url-visualize-json", tags=["production"])
async def evaluate_url_visualize_json(payload: EvaluateVisualizeRequest):
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = app_state.normalize_env(payload.env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        response = requests.get(payload.url, timeout=app_state.REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        yolo_result, unet_result, scoring = app_state.evaluate_image_with_artifacts(img, env_key)
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return app_state.build_visualize_json_payload(
            source_type="url",
            source=payload.url,
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-visualize-link", tags=["production"])
async def evaluate_visualize_link(
    file: UploadFile = File(...),
    env: str = Form(default="LOBBY_CORRIDOR"),
):
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = app_state.normalize_env(env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "File ảnh rỗng."})

        img = Image.open(io.BytesIO(content)).convert("RGB")
        yolo_result, unet_result, scoring = app_state.evaluate_image_with_artifacts(img, env_key)
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return app_state.build_visualize_blob_payload(
            source_type="upload",
            source=file.filename or "upload",
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/evaluate-url-visualize-link", tags=["production"])
async def evaluate_url_visualize_link(payload: EvaluateVisualizeRequest):
    if not app_state.MODEL:
        return JSONResponse(status_code=500, content={"error": "YOLO model chưa được tải."})
    if not app_state.UNET_MODEL:
        return JSONResponse(status_code=500, content={"error": "U-Net model chưa được tải."})

    try:
        env_key = app_state.normalize_env(payload.env)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        response = requests.get(payload.url, timeout=app_state.REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        yolo_result, unet_result, scoring = app_state.evaluate_image_with_artifacts(img, env_key)
        rendered = app_state.render_hybrid_overlay(
            rgb=unet_result["rgb"],
            pred_original_size=unet_result["mask_original_size"],
            yolo_result=yolo_result,
            scoring=scoring,
            env_key=env_key,
        )

        return app_state.build_visualize_blob_payload(
            source_type="url",
            source=payload.url,
            env_key=env_key,
            yolo_result=yolo_result,
            unet_result=unet_result,
            scoring=scoring,
            rendered=rendered,
        )
    except requests.exceptions.RequestException as exc:
        return JSONResponse(status_code=400, content={"error": f"Không thể tải ảnh từ URL: {str(exc)}"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})
