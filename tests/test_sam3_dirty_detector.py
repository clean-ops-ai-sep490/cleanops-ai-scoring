from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sam3_dirty_detector import (
    Sam3DetectorConfig,
    Sam3DirtyDetector,
    parse_prompt_list,
    public_sam3_payload,
)


class _FakeProcessor:
    def set_image(self, image: Image.Image):
        return {"size": image.size}

    def set_text_prompt(self, *, prompt: str, state: dict[str, object]):
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:6, 3:8] = 1.0
        return {
            "masks": np.array([mask]),
            "boxes": np.array([[3, 2, 8, 6]], dtype=np.float32),
            "scores": np.array([0.73], dtype=np.float32),
        }


def _config() -> Sam3DetectorConfig:
    return Sam3DetectorConfig(
        enabled=True,
        required=False,
        checkpoint_path="",
        resolution=1008,
        confidence_threshold=0.3,
        default_prompts=("dirty area",),
        max_prompts=7,
        min_mask_area_px=1,
        device="cpu",
        use_bfloat16=False,
        provider="local",
    )


class Sam3DirtyDetectorTests(unittest.TestCase):
    def test_parse_prompt_list_dedupes_and_caps(self):
        prompts = parse_prompt_list(
            "Garbage, Stain, garbage, spill",
            default_prompts=("dirty area",),
            max_prompts=3,
        )

        self.assertEqual(prompts, ["Garbage", "Stain", "spill"])

    def test_detect_normalizes_masks_boxes_and_scores(self):
        detector = Sam3DirtyDetector.__new__(Sam3DirtyDetector)
        detector.config = _config()
        detector._logger = logging.getLogger("test-sam3")
        detector._model = object()
        detector._processor = _FakeProcessor()
        detector._loaded = True
        detector._last_error = None
        detector._model_source = "unit"

        result = detector.detect(Image.new("RGB", (10, 10), color=(200, 200, 200)), prompts="Stain")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["dirty_coverage_pct"], 20.0)
        self.assertEqual(result["predictions"][0]["prompt"], "Stain")
        self.assertEqual(result["predictions"][0]["confidence"], 0.73)
        self.assertEqual(result["predictions"][0]["bbox_xyxy"], [3, 2, 8, 6])
        self.assertEqual(result["predictions"][0]["mask_area_px"], 20)
        self.assertEqual(result["predictions"][0]["area_pct"], 20.0)
        self.assertEqual(result["predictions"][0]["mask_source"], "model_mask")
        self.assertEqual(result["predictions"][0]["label_normalized"], "stain")
        self.assertEqual(result["_prediction_masks"][0]["mask_source"], "model_mask")

    def test_public_payload_removes_internal_masks(self):
        public = public_sam3_payload({"predictions": [], "_mask_union": np.zeros((2, 2), dtype=np.uint8)})

        self.assertEqual(public, {"predictions": []})

    def test_empty_result_supports_timeout_fallback_payload(self):
        detector = Sam3DirtyDetector.__new__(Sam3DirtyDetector)
        detector.config = _config()
        detector._loaded = True
        detector._last_error = None

        result = detector.empty_result(
            Image.new("RGB", (4, 3), color=(200, 200, 200)),
            prompts="Trash",
            status="timeout",
            started_at=0.0,
            error="sam3_timeout_after_30s",
        )
        public = public_sam3_payload(result)

        self.assertEqual(public["status"], "timeout")
        self.assertTrue(public["skipped"])
        self.assertEqual(public["error"], "sam3_timeout_after_30s")
        self.assertEqual(public["summary"]["input_size"], [4, 3])
        self.assertEqual(public["summary"]["predictions_count"], 0)
        self.assertNotIn("_mask_union", public)

    def test_roboflow_provider_maps_predictions_to_sam3_payload(self):
        class FakeRoboflowClient:
            def run_workflow(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "predictions": [
                        {
                            "class": "Stain",
                            "confidence": 0.8,
                            "x": 5,
                            "y": 5,
                            "width": 4,
                            "height": 4,
                        }
                    ]
                }

        config = Sam3DetectorConfig(
            enabled=True,
            required=False,
            checkpoint_path="",
            resolution=512,
            confidence_threshold=0.3,
            default_prompts=("Garbage", "Stain"),
            max_prompts=7,
            min_mask_area_px=1,
            device="cpu",
            use_bfloat16=False,
            provider="roboflow",
            roboflow_api_key="test-key",
            roboflow_workspace="workspace",
            roboflow_workflow_id="workflow",
            roboflow_classes=("Garbage", "Trash", "Debris", "Stain", "Wet_Floor"),
        )
        detector = Sam3DirtyDetector.__new__(Sam3DirtyDetector)
        detector.config = config
        detector._logger = logging.getLogger("test-sam3-roboflow")
        detector._model = None
        detector._processor = None
        fake_client = FakeRoboflowClient()
        detector._roboflow_client = fake_client
        detector._loaded = True
        detector._last_error = None
        detector._model_source = "roboflow:workspace/workflow"

        result = detector.detect(Image.new("RGB", (10, 10), color=(200, 200, 200)), prompts="Stain")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "roboflow")
        self.assertEqual(result["summary"]["predictions_count"], 1)
        self.assertEqual(result["summary"]["dirty_coverage_pct"], 16.0)
        self.assertEqual(result["predictions"][0]["class"], "Stain")
        self.assertEqual(result["predictions"][0]["area_pct"], 16.0)
        self.assertEqual(result["predictions"][0]["mask_source"], "bbox_fallback")
        self.assertEqual(result["predictions"][0]["label_normalized"], "stain")
        self.assertEqual(
            fake_client.kwargs["parameters"]["classes"],
            "Garbage, Trash, Debris, Stain, Wet_Floor",
        )
        self.assertEqual(result["requested_classes"], ["Garbage", "Trash", "Debris", "Stain", "Wet_Floor"])
        self.assertEqual(
            result["summary"]["requested_classes"],
            ["Garbage", "Trash", "Debris", "Stain", "Wet_Floor"],
        )
        self.assertIn("stain", result["_label_masks"])
        self.assertEqual(result["_prediction_masks"][0]["mask_source"], "bbox_fallback")

    def test_roboflow_polygon_marks_mask_source_polygon(self):
        class FakeRoboflowClient:
            def run_workflow(self, **kwargs):
                return {
                    "predictions": [
                        {
                            "class": "Garbage",
                            "confidence": 0.9,
                            "points": [
                                {"x": 1, "y": 1},
                                {"x": 5, "y": 1},
                                {"x": 5, "y": 5},
                                {"x": 1, "y": 5},
                            ],
                        }
                    ]
                }

        config = Sam3DetectorConfig(
            enabled=True,
            required=False,
            checkpoint_path="",
            resolution=512,
            confidence_threshold=0.3,
            default_prompts=("Garbage",),
            max_prompts=7,
            min_mask_area_px=1,
            device="cpu",
            use_bfloat16=False,
            provider="roboflow",
            roboflow_api_key="test-key",
            roboflow_workspace="workspace",
            roboflow_workflow_id="workflow",
            roboflow_classes=("Garbage",),
        )
        detector = Sam3DirtyDetector.__new__(Sam3DirtyDetector)
        detector.config = config
        detector._logger = logging.getLogger("test-sam3-roboflow-polygon")
        detector._model = None
        detector._processor = None
        detector._roboflow_client = FakeRoboflowClient()
        detector._loaded = True
        detector._last_error = None
        detector._model_source = "roboflow:workspace/workflow"

        result = detector.detect(Image.new("RGB", (10, 10), color=(200, 200, 200)), prompts="Garbage")

        self.assertEqual(result["predictions"][0]["mask_source"], "polygon")
        self.assertEqual(result["predictions"][0]["label_normalized"], "garbage")


if __name__ == "__main__":
    unittest.main()
