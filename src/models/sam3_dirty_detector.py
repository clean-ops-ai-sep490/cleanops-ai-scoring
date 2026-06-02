from __future__ import annotations

import logging
import time
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
import torch


def _normalize_label_key(raw: object) -> str:
    return "_".join(str(raw or "").strip().lower().replace("-", "_").split())


@dataclass(frozen=True)
class Sam3DetectorConfig:
    enabled: bool
    required: bool
    checkpoint_path: str
    resolution: int
    confidence_threshold: float
    default_prompts: tuple[str, ...]
    max_prompts: int
    min_mask_area_px: int
    device: str
    use_bfloat16: bool
    provider: str = "local"
    roboflow_api_url: str = "https://serverless.roboflow.com"
    roboflow_api_key: str | None = None
    roboflow_workspace: str = ""
    roboflow_workflow_id: str = ""
    roboflow_classes: tuple[str, ...] = ()
    roboflow_use_cache: bool = True


def parse_prompt_list(raw: str | Iterable[str] | None, default_prompts: tuple[str, ...], max_prompts: int) -> list[str]:
    if raw is None:
        candidates = list(default_prompts)
    elif isinstance(raw, str):
        candidates = raw.split(",")
    else:
        candidates = []
        for item in raw:
            if isinstance(item, str):
                candidates.extend(item.split(","))

    prompts: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        prompt = str(item or "").strip()
        if not prompt:
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
        if len(prompts) >= max(1, max_prompts):
            break

    return prompts or list(default_prompts[: max(1, max_prompts)])


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _normalize_masks(raw_masks: Any, *, width: int, height: int) -> list[np.ndarray]:
    masks = _as_numpy(raw_masks)
    if masks.size == 0:
        return []

    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 4:
        masks = np.squeeze(masks, axis=1) if masks.shape[1] == 1 else masks[:, 0, :, :]

    normalized: list[np.ndarray] = []
    for mask in masks:
        binary = np.asarray(mask > 0.5, dtype=np.uint8)
        if binary.shape[:2] != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        normalized.append(binary.astype(np.uint8))
    return normalized


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _coerce_bbox(raw_box: Any, fallback_mask: np.ndarray, *, width: int, height: int) -> list[int]:
    box_arr = _as_numpy(raw_box).reshape(-1)
    if box_arr.size < 4:
        return _bbox_from_mask(fallback_mask)

    x0, y0, x1, y1 = [float(v) for v in box_arr[:4]]
    return [
        max(0, min(width, int(round(x0)))),
        max(0, min(height, int(round(y0)))),
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
    ]


def _as_point_list(raw_points: Any) -> list[list[int]]:
    if not isinstance(raw_points, list):
        return []

    points: list[list[int]] = []
    for item in raw_points:
        if isinstance(item, dict):
            x = item.get("x")
            y = item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            continue
        try:
            points.append([int(round(float(x))), int(round(float(y)))])
        except (TypeError, ValueError):
            continue
    return points


def _iter_prediction_dicts(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_prediction_dicts(item)
        return

    if not isinstance(payload, dict):
        return

    looks_like_prediction = any(
        key in payload
        for key in ("x", "y", "width", "height", "bbox", "bbox_xyxy", "points", "polygon", "confidence", "class")
    )
    if looks_like_prediction and ("class" in payload or "label" in payload or "name" in payload):
        yield payload

    for key in ("predictions", "detections", "objects", "results", "output"):
        if key in payload:
            yield from _iter_prediction_dicts(payload[key])


def _bbox_from_prediction(prediction: dict[str, Any], *, width: int, height: int) -> list[int]:
    raw_bbox = prediction.get("bbox_xyxy") or prediction.get("bbox")
    if isinstance(raw_bbox, dict):
        raw_bbox = [
            raw_bbox.get("x1", raw_bbox.get("left", 0)),
            raw_bbox.get("y1", raw_bbox.get("top", 0)),
            raw_bbox.get("x2", raw_bbox.get("right", 0)),
            raw_bbox.get("y2", raw_bbox.get("bottom", 0)),
        ]
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        try:
            x0, y0, x1, y1 = [float(value) for value in raw_bbox[:4]]
            return [
                max(0, min(width, int(round(x0)))),
                max(0, min(height, int(round(y0)))),
                max(0, min(width, int(round(x1)))),
                max(0, min(height, int(round(y1)))),
            ]
        except (TypeError, ValueError):
            pass

    try:
        box_width = float(prediction.get("width", 0))
        box_height = float(prediction.get("height", 0))
        cx = float(prediction.get("x", 0))
        cy = float(prediction.get("y", 0))
    except (TypeError, ValueError):
        return [0, 0, 0, 0]

    x0 = cx - (box_width / 2.0)
    y0 = cy - (box_height / 2.0)
    x1 = cx + (box_width / 2.0)
    y1 = cy + (box_height / 2.0)
    return [
        max(0, min(width, int(round(x0)))),
        max(0, min(height, int(round(y0)))),
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
    ]


def _mask_from_roboflow_prediction(prediction: dict[str, Any], *, width: int, height: int) -> tuple[np.ndarray, str]:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = (
        _as_point_list(prediction.get("points"))
        or _as_point_list(prediction.get("polygon"))
        or _as_point_list(prediction.get("segmentation"))
    )
    if len(points) >= 3:
        clipped = np.array(
            [[max(0, min(width - 1, x)), max(0, min(height - 1, y))] for x, y in points],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [clipped], 1)
        return mask, "polygon"

    x0, y0, x1, y1 = _bbox_from_prediction(prediction, width=width, height=height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask, "bbox_fallback"


class Sam3DirtyDetector:
    def __init__(self, config: Sam3DetectorConfig, logger: logging.Logger | None = None):
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._roboflow_client: Any | None = None
        self._loaded = False
        self._last_error: str | None = None
        self._model_source = "disabled" if not config.enabled else "not-loaded"

        if config.enabled:
            self.load()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def load(self) -> None:
        if not self.config.enabled:
            return

        provider = self.config.provider.strip().lower()
        if provider == "roboflow":
            self._load_roboflow()
            return
        if provider != "local":
            self._loaded = False
            self._last_error = f"Unsupported SAM3_PROVIDER '{self.config.provider}'"
            self._model_source = "load-error:unsupported-provider"
            return

        try:
            if self.config.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("SAM3 requires CUDA, but torch.cuda.is_available() is false.")

            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            checkpoint_path = self.config.checkpoint_path.strip()
            build_kwargs: dict[str, Any] = {"device": self.config.device}
            if checkpoint_path:
                checkpoint = Path(checkpoint_path)
                if checkpoint.is_file():
                    build_kwargs["checkpoint_path"] = str(checkpoint)
                    self._model_source = str(checkpoint)
                else:
                    self._model_source = f"checkpoint-missing:{checkpoint_path}"

            try:
                self._model = build_sam3_image_model(**build_kwargs)
            except TypeError:
                self._model = build_sam3_image_model()
                if hasattr(self._model, "to"):
                    self._model.to(self.config.device)

            if hasattr(self._model, "eval"):
                self._model.eval()

            self._processor = Sam3Processor(
                self._model,
                resolution=self.config.resolution,
                confidence_threshold=self.config.confidence_threshold,
            )
            self._loaded = True
            if self._model_source == "not-loaded":
                self._model_source = "sam3-default"
            self._last_error = None
        except Exception as exc:  # noqa: BLE001
            self._loaded = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self._model_source == "not-loaded":
                self._model_source = f"load-error:{type(exc).__name__}"
            self._logger.exception("Failed to initialize SAM3 dirty detector.")

    def _load_roboflow(self) -> None:
        if not self.config.roboflow_api_key:
            self._loaded = False
            self._last_error = "Roboflow API key is not configured."
            self._model_source = "roboflow:not-configured"
            return
        if not self.config.roboflow_workspace or not self.config.roboflow_workflow_id:
            self._loaded = False
            self._last_error = "Roboflow workspace or workflow id is not configured."
            self._model_source = "roboflow:not-configured"
            return

        try:
            from inference_sdk import InferenceHTTPClient

            self._roboflow_client = InferenceHTTPClient(
                api_url=self.config.roboflow_api_url,
                api_key=self.config.roboflow_api_key,
            )
            self._loaded = True
            self._last_error = None
            self._model_source = f"roboflow:{self.config.roboflow_workspace}/{self.config.roboflow_workflow_id}"
        except Exception as exc:  # noqa: BLE001
            self._loaded = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._model_source = f"load-error:{type(exc).__name__}"
            self._logger.exception("Failed to initialize Roboflow SAM3-compatible detector.")

    def status_payload(self) -> dict[str, Any]:
        return {
            "sam3_enabled": self.config.enabled,
            "sam3_required": self.config.required,
            "sam3_loaded": self.loaded,
            "sam3_provider": self.config.provider,
            "sam3_device": self.config.device,
            "sam3_resolution": self.config.resolution,
            "sam3_confidence_threshold": self.config.confidence_threshold,
            "sam3_prompts": list(self.config.default_prompts),
            "sam3_checkpoint_path": self.config.checkpoint_path,
            "sam3_model_source": self._model_source,
            "sam3_last_error": self._last_error,
            "roboflow_configured": bool(
                self.config.roboflow_api_key
                and self.config.roboflow_workspace
                and self.config.roboflow_workflow_id
            ),
            "roboflow_workspace": self.config.roboflow_workspace,
            "roboflow_workflow_id": self.config.roboflow_workflow_id,
            "roboflow_classes": list(self.config.roboflow_classes),
            "roboflow_requested_classes": list(self.config.roboflow_classes),
        }

    def detect(
        self,
        image: Image.Image,
        *,
        prompts: str | Iterable[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        parsed_prompts = parse_prompt_list(prompts, self.config.default_prompts, self.config.max_prompts)
        width, height = image.size
        started_at = time.perf_counter()

        if not self.config.enabled:
            return self._empty_result(parsed_prompts, width, height, "disabled", started_at)
        if not self.loaded or self._processor is None:
            if self.config.provider.strip().lower() == "roboflow" and self._roboflow_client is not None:
                return self._detect_with_roboflow(image, parsed_prompts, started_at, threshold=threshold)
            return self._empty_result(parsed_prompts, width, height, "not_loaded", started_at)

        threshold_value = self.config.confidence_threshold if threshold is None else float(threshold)
        predictions: list[dict[str, Any]] = []
        regions: list[dict[str, Any]] = []
        union_mask = np.zeros((height, width), dtype=np.uint8)
        label_masks: dict[str, np.ndarray] = {}
        prediction_masks: list[dict[str, Any]] = []
        region_id = 1

        autocast_enabled = self.config.device == "cuda" and self.config.use_bfloat16
        autocast_dtype = torch.bfloat16 if self.config.use_bfloat16 else torch.float16

        try:
            with torch.no_grad():
                with torch.autocast(self.config.device, dtype=autocast_dtype, enabled=autocast_enabled):
                    state = self._processor.set_image(image.convert("RGB"))

                for prompt in parsed_prompts:
                    with torch.autocast(self.config.device, dtype=autocast_dtype, enabled=autocast_enabled):
                        output = self._processor.set_text_prompt(prompt=prompt, state=state)

                    masks = _normalize_masks(output.get("masks"), width=width, height=height)
                    boxes = _as_numpy(output.get("boxes"))
                    scores = _as_numpy(output.get("scores")).reshape(-1)

                    for idx, mask in enumerate(masks):
                        score = float(scores[idx]) if idx < len(scores) else 0.0
                        area_px = int(mask.sum())
                        if score < threshold_value or area_px < self.config.min_mask_area_px:
                            continue

                        raw_box = boxes[idx] if boxes.ndim >= 2 and idx < len(boxes) else []
                        bbox = _coerce_bbox(raw_box, mask, width=width, height=height)
                        x0, y0, x1, y1 = bbox
                        area_pct = round((area_px / max(1, width * height)) * 100.0, 3)
                        label_key = _normalize_label_key(prompt)
                        prediction = {
                            "class": prompt,
                            "prompt": prompt,
                            "label_normalized": label_key,
                            "confidence": round(score, 4),
                            "bbox_xyxy": bbox,
                            "x": x0,
                            "y": y0,
                            "width": max(0, x1 - x0),
                            "height": max(0, y1 - y0),
                            "mask_area_px": area_px,
                            "area_pct": area_pct,
                            "mask_source": "model_mask",
                        }
                        predictions.append(prediction)
                        regions.append(
                            {
                                "region_id": region_id,
                                "class_id": 3,
                                "kind_hint": prompt,
                                "label": prompt,
                                "label_normalized": label_key,
                                "confidence": round(score, 4),
                                "bbox_px": bbox,
                                "bbox_norm": [
                                    round(x0 / max(1, width), 4),
                                    round(y0 / max(1, height), 4),
                                    round(x1 / max(1, width), 4),
                                    round(y1 / max(1, height), 4),
                                ],
                                "area_pct": area_pct,
                                "mask_source": "model_mask",
                            }
                        )
                        prediction_masks.append(
                            {
                                "label": prompt,
                                "label_normalized": label_key,
                                "confidence": round(score, 4),
                                "area_pct": area_pct,
                                "mask_source": "model_mask",
                                "mask": mask,
                            }
                        )
                        union_mask = np.maximum(union_mask, mask)
                        if label_key:
                            label_masks[label_key] = np.maximum(
                                label_masks.get(label_key, np.zeros((height, width), dtype=np.uint8)),
                                mask,
                            )
                        region_id += 1

            union_area_px = int(union_mask.sum())
            return {
                "enabled": True,
                "loaded": True,
                "status": "ok",
                "prompts": parsed_prompts,
                "predictions": predictions,
                "regions": regions,
                "summary": {
                    "input_size": [width, height],
                    "resolution": self.config.resolution,
                    "confidence_threshold": threshold_value,
                    "prompts_count": len(parsed_prompts),
                    "predictions_count": len(predictions),
                    "union_mask_area_px": union_area_px,
                    "dirty_coverage_pct": round((union_area_px / max(1, width * height)) * 100.0, 3),
                    "max_confidence": round(max((float(p["confidence"]) for p in predictions), default=0.0), 4),
                    "elapsed_ms": int(round((time.perf_counter() - started_at) * 1000)),
                },
                "_mask_union": union_mask,
                "_label_masks": label_masks,
                "_prediction_masks": prediction_masks,
            }
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("SAM3 inference failed.")
            return self._empty_result(parsed_prompts, width, height, "error", started_at, error=self._last_error)

    def _detect_with_roboflow(
        self,
        image: Image.Image,
        prompts: list[str],
        started_at: float,
        *,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        width, height = image.size
        threshold_value = self.config.confidence_threshold if threshold is None else float(threshold)
        workflow_prompts = list(self.config.roboflow_classes or prompts or self.config.default_prompts)
        classes = ", ".join(workflow_prompts)

        try:
            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
                    temp_path = temp_image.name
                image.convert("RGB").save(temp_path, format="JPEG", quality=92)
                raw_result = self._roboflow_client.run_workflow(
                    workspace_name=self.config.roboflow_workspace,
                    workflow_id=self.config.roboflow_workflow_id,
                    images={"image": temp_path},
                    parameters={"classes": classes},
                    use_cache=self.config.roboflow_use_cache,
                )
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

            predictions: list[dict[str, Any]] = []
            regions: list[dict[str, Any]] = []
            union_mask = np.zeros((height, width), dtype=np.uint8)
            label_masks: dict[str, np.ndarray] = {}
            prediction_masks: list[dict[str, Any]] = []
            region_id = 1

            for raw_prediction in _iter_prediction_dicts(raw_result):
                label = str(
                    raw_prediction.get("class")
                    or raw_prediction.get("label")
                    or raw_prediction.get("name")
                    or "roboflow"
                ).strip()
                try:
                    score = float(
                        raw_prediction.get("confidence")
                        or raw_prediction.get("score")
                        or raw_prediction.get("confidence_percent", 0.0)
                    )
                except (TypeError, ValueError):
                    score = 0.0
                if score > 1.0:
                    score = score / 100.0

                mask, mask_source = _mask_from_roboflow_prediction(raw_prediction, width=width, height=height)
                area_px = int(mask.sum())
                if score < threshold_value or area_px < self.config.min_mask_area_px:
                    continue

                bbox = _bbox_from_mask(mask)
                x0, y0, x1, y1 = bbox
                area_pct = round((area_px / max(1, width * height)) * 100.0, 3)
                label_key = _normalize_label_key(label)
                predictions.append(
                    {
                        "class": label,
                        "prompt": label,
                        "label_normalized": label_key,
                        "confidence": round(score, 4),
                        "bbox_xyxy": bbox,
                        "x": x0,
                        "y": y0,
                        "width": max(0, x1 - x0),
                        "height": max(0, y1 - y0),
                        "mask_area_px": area_px,
                        "area_pct": area_pct,
                        "mask_source": mask_source,
                        "provider": "roboflow",
                    }
                )
                regions.append(
                    {
                        "region_id": region_id,
                        "class_id": 3,
                        "kind_hint": label,
                        "label": label,
                        "label_normalized": label_key,
                        "confidence": round(score, 4),
                        "bbox_px": bbox,
                        "bbox_norm": [
                            round(x0 / max(1, width), 4),
                            round(y0 / max(1, height), 4),
                            round(x1 / max(1, width), 4),
                            round(y1 / max(1, height), 4),
                        ],
                        "area_pct": area_pct,
                        "mask_source": mask_source,
                        "provider": "roboflow",
                    }
                )
                prediction_masks.append(
                    {
                        "label": label,
                        "label_normalized": label_key,
                        "confidence": round(score, 4),
                        "area_pct": area_pct,
                        "mask_source": mask_source,
                        "provider": "roboflow",
                        "mask": mask,
                    }
                )
                union_mask = np.maximum(union_mask, mask)
                if label_key:
                    label_masks[label_key] = np.maximum(
                        label_masks.get(label_key, np.zeros((height, width), dtype=np.uint8)),
                        mask,
                    )
                region_id += 1

            union_area_px = int(union_mask.sum())
            return {
                "enabled": True,
                "loaded": True,
                "status": "ok",
                "provider": "roboflow",
                "prompts": workflow_prompts,
                "requested_classes": workflow_prompts,
                "predictions": predictions,
                "regions": regions,
                "summary": {
                    "input_size": [width, height],
                    "resolution": self.config.resolution,
                    "confidence_threshold": threshold_value,
                    "prompts_count": len(workflow_prompts),
                    "requested_classes": workflow_prompts,
                    "predictions_count": len(predictions),
                    "union_mask_area_px": union_area_px,
                    "dirty_coverage_pct": round((union_area_px / max(1, width * height)) * 100.0, 3),
                    "max_confidence": round(max((float(p["confidence"]) for p in predictions), default=0.0), 4),
                    "elapsed_ms": int(round((time.perf_counter() - started_at) * 1000)),
                },
                "_mask_union": union_mask,
                "_label_masks": label_masks,
                "_prediction_masks": prediction_masks,
            }
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("Roboflow SAM3-compatible inference failed.")
            return self._empty_result(prompts, width, height, "error", started_at, error=self._last_error)

    def empty_result(
        self,
        image: Image.Image,
        *,
        prompts: str | Iterable[str] | None = None,
        status: str = "skipped",
        started_at: float | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        parsed_prompts = parse_prompt_list(prompts, self.config.default_prompts, self.config.max_prompts)
        width, height = image.size
        result = self._empty_result(
            parsed_prompts,
            width,
            height,
            status,
            time.perf_counter() if started_at is None else started_at,
            error=error,
        )
        result["skipped"] = True
        return result

    def _empty_result(
        self,
        prompts: list[str],
        width: int,
        height: int,
        status: str,
        started_at: float,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "loaded": self.loaded,
            "status": status,
            "provider": self.config.provider,
            "error": error or self._last_error,
            "prompts": prompts,
            "predictions": [],
            "regions": [],
            "summary": {
                "input_size": [width, height],
                "resolution": self.config.resolution,
                "confidence_threshold": self.config.confidence_threshold,
                "prompts_count": len(prompts),
                "predictions_count": 0,
                "union_mask_area_px": 0,
                "dirty_coverage_pct": 0.0,
                "max_confidence": 0.0,
                "elapsed_ms": int(round((time.perf_counter() - started_at) * 1000)),
            },
            "_mask_union": np.zeros((height, width), dtype=np.uint8),
            "_label_masks": {},
            "_prediction_masks": [],
        }


def public_sam3_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}
