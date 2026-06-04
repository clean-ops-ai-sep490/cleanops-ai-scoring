from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

from src.api.ppe_utils import (
    GEMINI_MAX_VERIFICATION_ATTEMPTS,
    GeminiPpeConfig,
    evaluate_ppe_payload,
    verify_missing_items_with_gemini,
)


class GeminiResponseStub:
    def __init__(self, text: str):
        self._text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": self._text,
                            }
                        ]
                    }
                }
            ]
        }


class PpeUtilsTests(unittest.IsolatedAsyncioTestCase):
    @patch("src.api.ppe_utils.requests.post")
    def test_verify_missing_items_with_gemini_uses_structured_deterministic_json(
        self,
        requests_post_mock,
    ) -> None:
        requests_post_mock.return_value = GeminiResponseStub(
            """
            {
              "items": [
                {"name": "gloves", "present": true, "confidence": 0.91, "evidence": "visible gloves"},
                {"name": "surgical_mask", "present": false, "confidence": 0.2, "evidence": ""}
              ],
              "notes": "clear enough"
            }
            """
        )

        response = verify_missing_items_with_gemini(
            images=[Image.new("RGB", (64, 64), color=(200, 200, 200))],
            missing_items=["gloves", "surgical_mask"],
            config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        payload = requests_post_mock.call_args.kwargs["json"]
        generation_config = payload["generationConfig"]
        self.assertEqual(generation_config["temperature"], 0)
        self.assertEqual(generation_config["candidateCount"], 1)
        self.assertEqual(generation_config["responseMimeType"], "application/json")
        self.assertIn("responseSchema", generation_config)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["confirmed_items"], ["gloves"])
        self.assertEqual(response["remaining_missing_items"], ["surgical_mask"])
        self.assertEqual(len(response["item_reviews"]), 2)

    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_uses_detector_items_only(
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
        )

        self.assertEqual(response["status"], "FAIL")
        self.assertEqual(response["message"], "Missing items: gloves")
        self.assertEqual(response["missing_items"], ["gloves"])
        self.assertEqual(response["detected_items"][0]["source"], "detector")
        self.assertEqual(response["gemini_review"]["status"], "skipped")

    @patch("src.api.ppe_utils.verify_missing_items_with_gemini")
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_uses_gemini_missing_only(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
        verify_missing_items_mock,
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
        verify_missing_items_mock.return_value = {
            "status": "ok",
            "confirmed_items": ["gloves"],
            "remaining_missing_items": [],
        }

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["helmet", "gloves"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            gemini_config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        self.assertEqual(response["status"], "PASS")
        self.assertEqual(response["missing_items"], [])
        self.assertEqual(response["detected_items"][-1]["source"], "gemini")
        self.assertEqual(response["gemini_review"]["status"], "ok")
        self.assertEqual(response["gemini_review"]["attempts"], 1)

    @patch("src.api.ppe_utils.verify_missing_items_with_gemini")
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_retries_malformed_gemini_response(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
        verify_missing_items_mock,
    ) -> None:
        load_image_from_url_mock.return_value = Image.new("RGB", (64, 64), color=(200, 200, 200))
        collect_filtered_detections_mock.return_value = []
        verify_missing_items_mock.side_effect = [
            {
                "status": "malformed",
                "reason": "incomplete_item_reviews",
                "confirmed_items": [],
                "remaining_missing_items": ["gloves"],
            },
            {
                "status": "ok",
                "confirmed_items": ["gloves"],
                "remaining_missing_items": [],
            },
        ]

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["gloves"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            gemini_config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        self.assertEqual(response["status"], "PASS")
        self.assertEqual(response["missing_items"], [])
        self.assertEqual(verify_missing_items_mock.call_count, 2)
        self.assertEqual(response["gemini_review"]["attempts"], 2)
        self.assertEqual(response["gemini_review"]["raw_statuses"], ["malformed", "ok"])

    @patch("src.api.ppe_utils.verify_missing_items_with_gemini")
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_retries_partial_gemini_confirmation(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
        verify_missing_items_mock,
    ) -> None:
        load_image_from_url_mock.return_value = Image.new("RGB", (64, 64), color=(200, 200, 200))
        collect_filtered_detections_mock.return_value = []
        verify_missing_items_mock.side_effect = [
            {
                "status": "ok",
                "confirmed_items": ["surgical_mask"],
                "remaining_missing_items": ["gloves"],
            },
            {
                "status": "ok",
                "confirmed_items": ["gloves", "surgical_mask"],
                "remaining_missing_items": [],
            },
        ]

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["gloves", "surgical_mask"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            gemini_config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        self.assertEqual(response["status"], "PASS")
        self.assertEqual(response["missing_items"], [])
        self.assertEqual(verify_missing_items_mock.call_count, 2)
        self.assertEqual(response["gemini_review"]["attempts"], 2)
        self.assertEqual(response["gemini_review"]["raw_statuses"], ["ok", "ok"])

    @patch("src.api.ppe_utils.verify_missing_items_with_gemini")
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_stops_retrying_negative_review_after_attempt_budget(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
        verify_missing_items_mock,
    ) -> None:
        load_image_from_url_mock.return_value = Image.new("RGB", (64, 64), color=(200, 200, 200))
        collect_filtered_detections_mock.return_value = []
        verify_missing_items_mock.return_value = {
            "status": "ok",
            "confirmed_items": [],
            "remaining_missing_items": ["gloves"],
            "item_reviews": [
                {
                    "name": "gloves",
                    "present": False,
                    "confidence": 0.1,
                    "evidence": "",
                }
            ],
        }

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["gloves"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            gemini_config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        self.assertEqual(response["status"], "FAIL")
        self.assertEqual(response["missing_items"], ["gloves"])
        self.assertEqual(verify_missing_items_mock.call_count, GEMINI_MAX_VERIFICATION_ATTEMPTS)
        self.assertEqual(response["gemini_review"]["attempts"], GEMINI_MAX_VERIFICATION_ATTEMPTS)

    @patch("src.api.ppe_utils.verify_missing_items_with_gemini")
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_falls_back_when_gemini_errors(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
        verify_missing_items_mock,
    ) -> None:
        load_image_from_url_mock.return_value = Image.new("RGB", (64, 64), color=(200, 200, 200))
        collect_filtered_detections_mock.return_value = []
        verify_missing_items_mock.return_value = {
            "status": "error",
            "confirmed_items": [],
            "remaining_missing_items": ["helmet"],
            "error": "timeout",
        }

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["helmet"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            gemini_config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        self.assertEqual(response["status"], "FAIL")
        self.assertEqual(response["missing_items"], ["helmet"])
        self.assertEqual(response["gemini_review"]["status"], "error")

    @patch("src.api.ppe_utils.verify_missing_items_with_gemini")
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_skips_gemini_when_deadline_exceeded_before_call(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
        verify_missing_items_mock,
    ) -> None:
        load_image_from_url_mock.return_value = Image.new("RGB", (64, 64), color=(200, 200, 200))
        collect_filtered_detections_mock.return_value = []

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["helmet"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            gemini_deadline_sec=0.0,
            gemini_config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        verify_missing_items_mock.assert_not_called()
        self.assertEqual(response["status"], "FAIL")
        self.assertEqual(response["missing_items"], ["helmet"])
        self.assertEqual(response["gemini_review"]["status"], "skipped")
        self.assertEqual(response["gemini_review"]["reason"], "deadline_exceeded")

    @patch("src.api.ppe_utils.verify_missing_items_with_gemini")
    @patch("src.api.ppe_utils.load_image_from_url")
    @patch("src.api.ppe_utils.collect_filtered_detections")
    async def test_evaluate_ppe_payload_skips_gemini_when_call_exceeds_deadline(
        self,
        collect_filtered_detections_mock,
        load_image_from_url_mock,
        verify_missing_items_mock,
    ) -> None:
        load_image_from_url_mock.return_value = Image.new("RGB", (64, 64), color=(200, 200, 200))
        collect_filtered_detections_mock.return_value = []

        def slow_gemini(*_args, **_kwargs):
            import time

            time.sleep(0.2)
            return {
                "status": "ok",
                "confirmed_items": ["helmet"],
                "remaining_missing_items": [],
            }

        verify_missing_items_mock.side_effect = slow_gemini

        response = await evaluate_ppe_payload(
            image_urls=["https://example.test/ppe.jpg"],
            required_objects=["helmet"],
            model=object(),
            timeout_sec=1,
            min_confidence=25.0,
            gemini_deadline_sec=0.01,
            gemini_config=GeminiPpeConfig(
                enabled=True,
                mode="missing_only",
                api_key="test-key",
                model="gemini-test",
                base_url="https://example.test/v1beta",
                timeout_sec=1,
            ),
        )

        self.assertEqual(response["status"], "FAIL")
        self.assertEqual(response["missing_items"], ["helmet"])
        self.assertEqual(response["gemini_review"]["status"], "skipped")
        self.assertEqual(response["gemini_review"]["reason"], "deadline_exceeded")


if __name__ == "__main__":
    unittest.main()
