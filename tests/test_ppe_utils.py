from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

from src.api.ppe_utils import evaluate_ppe_payload


class _AddingFilter:
    def should_verify_ppe(self, **_: object) -> tuple[bool, None]:
        return True, None

    def refine_ppe_detected_items(
        self,
        image: Image.Image,
        *,
        required_objects: list[str],
        detected_items: list[dict[str, object]],
        allowed_labels: list[str],
        min_confidence: float,
        source: str,
    ) -> list[dict[str, object]]:
        return [
            *detected_items,
            {
                "name": "gloves",
                "confidence": 85.0,
                "image_index": 0,
                "source": "filter",
            },
        ]


class PpeUtilsTests(unittest.IsolatedAsyncioTestCase):
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_uses_filter_items_for_final_pass(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
    ) -> None:
        load_image_from_url_mock.return_value = Image.new("RGB", (64, 64), color=(200, 200, 200))
        collect_filtered_detections_mock.return_value = [
            {
                "name": "helmet",
                "confidence": 85.0,
                "image_index": 0,
                "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
            }
        ]

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["helmet", "gloves"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            llm_filter=_AddingFilter(),
            allowed_labels=["helmet", "gloves"],
        )

        self.assertEqual(response["status"], "PASS")
        self.assertEqual(response["message"], "All required PPE items detected")
        self.assertEqual(response["missing_items"], [])
        self.assertEqual(response["detected_items"][0]["source"], "detector")
        self.assertEqual(response["detected_items"][1]["source"], "filter")
        self.assertNotIn("llm_filter", response)


if __name__ == "__main__":
    unittest.main()
