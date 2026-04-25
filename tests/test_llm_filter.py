from __future__ import annotations

import logging
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.llm_filter import GeminiFilterConfig, GeminiLLMFilter
from src.api.visualization_utils import extract_dirty_region_candidates


def _json_candidate(text: str) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": text,
                        }
                    ]
                }
            }
        ]
    }


def _make_filter(*, queue_enabled: bool = False, deadline_sec: float = 1.0) -> GeminiLLMFilter:
    return GeminiLLMFilter(
        GeminiFilterConfig(
            enabled=True,
            mode="quota_saver",
            model="gemini-2.5-pro",
            timeout_sec=0.2,
            batch_concurrency=2,
            queue_enabled=queue_enabled,
            queue_mode="global_fifo",
            deadline_sec=deadline_sec,
            retry_429_max_retries=0,
            retry_5xx_max_retries=1,
            cooldown_sec=90,
            enable_borderline_only=True,
            scoring_pass_window=10.0,
            ppe_verify_on_missing_only=True,
            retry_initial_delay_ms=10,
            retry_max_delay_ms=10,
            retryable_status_codes=(429, 500, 502, 503, 504),
            max_image_dimension=768,
            jpeg_quality=65,
            api_key="test-key",
            base_url="https://example.test/v1beta",
        ),
        logging.getLogger("test-llm-filter"),
    )


def _make_http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    response._content = b"{}"
    return requests.HTTPError(f"http {status_code}", response=response)


class GeminiLLMFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = _make_filter()
        self.image = Image.new("RGB", (64, 64), color=(200, 200, 200))

    @patch("src.api.llm_filter.requests.post")
    def test_invoke_json_parses_json_only_response(self, post_mock: Mock):
        post_mock.return_value.raise_for_status.return_value = None
        post_mock.return_value.json.return_value = _json_candidate(
            '```json\n{"keep_indexes":[0],"reason":"ok"}\n```'
        )

        result = self.filter._invoke_json("prompt", self.image, kind="yolo", source="unit")  # noqa: SLF001

        self.assertEqual(result, {"keep_indexes": [0], "reason": "ok"})

    @patch("src.api.llm_filter.requests.post")
    def test_invoke_json_returns_none_on_malformed_json(self, post_mock: Mock):
        post_mock.return_value.raise_for_status.return_value = None
        post_mock.return_value.json.return_value = _json_candidate("this is not valid json")

        result = self.filter._invoke_json("prompt", self.image, kind="yolo", source="unit")  # noqa: SLF001

        self.assertIsNone(result)
        self.assertIn("parse_error", self.filter.status_payload()["llm_filter_last_error"])
        self.assertEqual(self.filter.status_payload()["llm_filter_last_result"], "non_retryable_error")

    def test_refine_yolo_result_keeps_only_existing_indexes(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={"verified_detection_indexes": [1]}
        )
        raw = {
            "detections_count": 2,
            "results": [
                {"class_name": "trash", "confidence": 0.91, "bbox": [0, 0, 10, 10]},
                {"class_name": "bottle", "confidence": 0.74, "bbox": [10, 10, 20, 20]},
            ],
        }

        refined = self.filter.refine_yolo_result(self.image, raw, source="unit")

        self.assertEqual(refined["detections_count"], 1)
        self.assertEqual(refined["results"][0]["class_name"], "bottle")

    def test_refine_yolo_result_can_add_guarded_advisory_detection(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "verified_detection_indexes": [],
                "advisory_object_boxes": [
                    {
                        "label": "trash",
                        "confidence": 0.88,
                        "bbox_norm": [0.1, 0.1, 0.35, 0.35],
                        "reason": "clear trash",
                    }
                ],
            }
        )
        raw = {
            "detections_count": 0,
            "results": [],
        }

        refined = self.filter.refine_yolo_result(
            self.image,
            raw,
            allowed_labels=["trash"],
            label_to_id={"trash": 0},
            source="unit",
        )

        self.assertEqual(refined["detections_count"], 1)
        self.assertEqual(refined["results"][0]["class_name"], "trash")
        self.assertEqual(refined["results"][0]["class_id"], 0)

    def test_verify_dirty_evidence_clamps_large_adjustment(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "highlight_dirty_region_ids": [1],
                "dirty_region_labels": [{"region_id": 1, "label": "wet_area"}],
                "stain_delta_pct": 60.0,
                "wet_delta_pct": 40.0,
                "reasons": ["visible wet patch"],
                "confidence_note": "high confidence",
            }
        )
        raw = {
            "input_size": [100, 100],
            "model_input_size": 100,
            "class_mapping": {0: "background", 1: "stain_or_water", 2: "wet_surface"},
            "stain_or_water_pixels": 200,
            "wet_surface_pixels": 100,
            "stain_or_water_coverage_pct": 2.0,
            "wet_surface_coverage_pct": 1.0,
            "total_dirty_coverage_pct": 3.0,
        }

        verified = self.filter.verify_dirty_evidence(
            self.image,
            raw,
            dirty_region_candidates=[
                {"region_id": 1, "kind_hint": "wet_surface", "bbox_norm": [0.1, 0.1, 0.4, 0.5], "area_pct": 5.0}
            ],
            source="unit",
        )
        refined = verified["summary"]

        self.assertLessEqual(refined["total_dirty_coverage_pct"], 18.0)
        self.assertGreaterEqual(refined["stain_or_water_coverage_pct"], 0.0)
        self.assertGreaterEqual(refined["wet_surface_coverage_pct"], 0.0)
        self.assertEqual(verified["review"]["highlight_dirty_region_ids"], [1])
        self.assertEqual(verified["review"]["dirty_region_labels"], [{"region_id": 1, "label": "wet_area"}])

    def test_refine_scoring_normalizes_quality_to_match_verdict(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "verdict": "PASS",
                "quality_score": 40.0,
                "reasons": ["image looks clean"],
            }
        )
        raw_scoring = {
            "base_clean_score": 88.0,
            "object_penalty": 10.0,
            "quality_score": 78.0,
            "pass_threshold": 90.0,
            "verdict": "PENDING",
            "reasons": ["objects remain"],
        }

        refined = self.filter.refine_scoring(
            self.image,
            env_key="LOBBY_CORRIDOR",
            yolo_result={"detections_count": 1, "results": []},
            unet_summary={"total_dirty_coverage_pct": 12.0},
            scoring=raw_scoring,
            pending_lower_bound=50.0,
            source="unit",
        )

        self.assertEqual(refined["verdict"], "PASS")
        self.assertGreaterEqual(refined["quality_score"], 90.0)
        self.assertEqual(refined["reasons"], ["image looks clean"])

    def test_verify_scoring_evidence_combines_object_and_dirty_review(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "verified_detection_indexes": [],
                "advisory_object_boxes": [
                    {
                        "label": "trash",
                        "confidence": 0.8,
                        "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                        "reason": "visible trash",
                    }
                ],
                "highlight_dirty_region_ids": [2],
                "dirty_region_labels": [{"region_id": 2, "label": "wet_area"}],
                "stain_delta_pct": 2.0,
                "wet_delta_pct": 3.0,
                "advisory_dirty_boxes": [],
                "reasons": ["verified visible mess"],
                "confidence_note": "good visibility",
            }
        )
        result = self.filter.verify_scoring_evidence(
            self.image,
            env_key="LOBBY_CORRIDOR",
            yolo_result={"detections_count": 1, "results": [{"class_name": "chair", "confidence": 0.9, "bbox": [1, 1, 8, 8]}]},
            unet_summary={
                "input_size": [100, 100],
                "model_input_size": 100,
                "stain_or_water_coverage_pct": 2.0,
                "wet_surface_coverage_pct": 3.0,
                "total_dirty_coverage_pct": 5.0,
            },
            dirty_region_candidates=[{"region_id": 2, "kind_hint": "wet_surface", "bbox_norm": [0.2, 0.2, 0.5, 0.6], "area_pct": 7.0}],
            scoring={"quality_score": 86.0, "pass_threshold": 90.0},
            pending_lower_bound=50.0,
            allowed_labels=["trash"],
            label_to_id={"trash": 0},
            source="unit",
        )

        self.assertEqual(result["yolo"]["detections_count"], 1)
        self.assertEqual(result["yolo"]["results"][0]["class_name"], "trash")
        self.assertEqual(result["review"]["highlight_dirty_region_ids"], [2])
        self.assertGreater(result["summary"]["total_dirty_coverage_pct"], 5.0)

    def test_verify_scoring_evidence_visualize_enhanced_keeps_overlay_geometry(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "verified_detection_indexes": [0],
                "highlight_dirty_region_ids": [2],
                "stain_delta_pct": 1.0,
                "wet_delta_pct": 2.0,
                "advisory_dirty_boxes": [
                    {
                        "label": "dirty_zone",
                        "confidence": 0.81,
                        "bbox_norm": [0.45, 0.4, 0.75, 0.78],
                        "reason": "visible debris cluster",
                    }
                ],
                "overlay_summary": "highlight debris cluster on the right",
                "reasons": ["verified visible debris"],
                "confidence_note": "good visibility",
            }
        )
        result = self.filter.verify_scoring_evidence(
            self.image,
            env_key="LOBBY_CORRIDOR",
            yolo_result={
                "detections_count": 1,
                "results": [{"class_name": "trash", "confidence": 0.9, "bbox": [1, 1, 8, 8]}],
            },
            unet_summary={
                "input_size": [100, 100],
                "model_input_size": 100,
                "stain_or_water_coverage_pct": 2.0,
                "wet_surface_coverage_pct": 3.0,
                "total_dirty_coverage_pct": 5.0,
            },
            dirty_region_candidates=[{"region_id": 2, "kind_hint": "wet_surface", "bbox_norm": [0.2, 0.2, 0.5, 0.6], "area_pct": 7.0}],
            scoring={"quality_score": 86.0, "pass_threshold": 90.0},
            pending_lower_bound=50.0,
            allowed_labels=["trash"],
            label_to_id={"trash": 0},
            source="unit-visual",
            visualize_enhanced=True,
        )

        self.assertEqual(result["review"]["keep_detection_indexes"], [0])
        self.assertEqual(result["review"]["highlight_dirty_region_ids"], [2])
        self.assertEqual(result["review"]["dirty_region_labels"], [])
        self.assertEqual(result["review"]["advisory_object_boxes"], [])
        self.assertEqual(len(result["review"]["advisory_dirty_boxes"]), 1)
        self.assertEqual(result["review"]["advisory_dirty_boxes"][0]["label"], "dirty_zone")
        self.assertIn("debris", result["review"]["overlay_summary"])

    @patch("src.api.llm_filter.requests.post")
    def test_invoke_json_captures_raw_preview_on_parse_failure(self, post_mock: Mock):
        post_mock.return_value.raise_for_status.return_value = None
        post_mock.return_value.json.return_value = _json_candidate(
            "```json\n{'verified_detection_indexes':[0], 'highlight_dirty_region_ids':[1],\n```"
        )

        result = self.filter._invoke_json("prompt", self.image, kind="scoring_verification", source="unit-raw")  # noqa: SLF001

        self.assertIsNone(result)
        metadata = self.filter.response_metadata("unit-raw", ["scoring_verification"])
        preview = metadata["stages"]["scoring_verification"]["response_preview"]
        self.assertIn("raw_response_preview", preview)
        self.assertIn("verified_detection_indexes", preview["raw_response_preview"])

    def test_verify_scoring_evidence_skips_confident_clean_cases(self):
        result = self.filter.verify_scoring_evidence(
            self.image,
            env_key="LOBBY_CORRIDOR",
            yolo_result={"detections_count": 0, "results": []},
            unet_summary={
                "model_input_size": 100,
                "stain_or_water_coverage_pct": 0.0,
                "wet_surface_coverage_pct": 4.0,
                "total_dirty_coverage_pct": 4.0,
            },
            dirty_region_candidates=[],
            scoring={"quality_score": 95.0, "pass_threshold": 90.0},
            pending_lower_bound=50.0,
            allowed_labels=["trash"],
            label_to_id={"trash": 0},
            source="unit-clean",
        )

        self.assertEqual(result["yolo"]["detections_count"], 0)
        metadata = self.filter.response_metadata("unit-clean", ["scoring_verification"])
        self.assertEqual(metadata["stages"]["scoring_verification"]["status"], "skipped")
        self.assertEqual(metadata["stages"]["scoring_verification"]["error"], "cv_confident")

    def test_refine_ppe_detected_items_returns_subset_only(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={"present_objects": ["helmet"]}
        )
        raw_items = [
            {"name": "helmet", "confidence": 96.2, "image_index": 0},
            {"name": "gloves", "confidence": 83.5, "image_index": 0},
        ]

        refined = self.filter.refine_ppe_detected_items(
            self.image,
            required_objects=["helmet", "gloves"],
            detected_items=raw_items,
            source="unit",
        )

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0]["name"], "helmet")

    def test_refine_ppe_detected_items_can_add_required_visible_item(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "present_objects": ["helmet"],
                "visible_missed_objects": ["gloves"],
            }
        )
        raw_items = [
            {"name": "helmet", "confidence": 96.2, "image_index": 0},
        ]

        refined = self.filter.refine_ppe_detected_items(
            self.image,
            required_objects=["helmet", "gloves"],
            detected_items=raw_items,
            allowed_labels=["helmet", "gloves"],
            min_confidence=25.0,
            source="unit",
        )

        self.assertEqual(len(refined), 2)
        self.assertEqual(refined[0]["name"], "helmet")
        self.assertEqual(refined[1]["name"], "gloves")
        self.assertGreaterEqual(refined[1]["confidence"], 85.0)

    def test_refine_ppe_detected_items_does_not_add_legacy_visible_missing_field(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "present_objects": [],
                "visible_missing_objects": ["helmet"],
            }
        )

        refined = self.filter.refine_ppe_detected_items(
            self.image,
            required_objects=["helmet"],
            detected_items=[],
            allowed_labels=["helmet"],
            min_confidence=25.0,
            source="unit",
        )

        self.assertEqual(refined, [])

    def test_review_visual_overlay_sanitizes_advisory_boxes(self):
        self.filter._invoke_json = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value={
                "keep_detection_indexes": [0, 9],
                "highlight_dirty_region_ids": [2, 99],
                "dirty_region_labels": [{"region_id": 2, "label": "wet_area"}],
                "advisory_object_boxes": [
                    {
                        "label": "trash",
                        "confidence": 0.91,
                        "bbox_norm": [0.1, 0.2, 0.4, 0.5],
                        "reason": "visible object",
                    },
                    {
                        "label": "banana",
                        "confidence": 0.99,
                        "bbox_norm": [0.1, 0.2, 0.9, 0.95],
                        "reason": "invalid label",
                    },
                ],
                "advisory_dirty_boxes": [
                    {
                        "label": "stain",
                        "confidence": 0.82,
                        "bbox_norm": [0.25, 0.25, 0.55, 0.65],
                        "reason": "visible stain",
                    },
                    {
                        "label": "stain",
                        "confidence": 0.5,
                        "bbox_norm": [0.1, 0.1, 0.2, 0.2],
                        "reason": "too low confidence",
                    },
                ],
                "overlay_summary": "visible wet patch",
            }
        )

        review = self.filter.review_visual_overlay(
            self.image,
            yolo_result={
                "detections_count": 1,
                "results": [{"class_name": "trash", "confidence": 0.9, "bbox": [1, 1, 5, 5]}],
            },
            unet_summary={"total_dirty_coverage_pct": 8.2},
            dirty_region_candidates=[
                {"region_id": 2, "kind_hint": "wet_surface", "bbox_norm": [0.2, 0.2, 0.5, 0.6], "area_pct": 8.0},
            ],
            scoring={"verdict": "PENDING", "quality_score": 80.0},
            source="unit",
        )

        self.assertEqual(review["keep_detection_indexes"], [0])
        self.assertEqual(review["highlight_dirty_region_ids"], [2])
        self.assertEqual(review["dirty_region_labels"], [{"region_id": 2, "label": "wet_area"}])
        self.assertEqual(len(review["advisory_object_boxes"]), 1)
        self.assertEqual(review["advisory_object_boxes"][0]["label"], "trash")
        self.assertEqual(len(review["advisory_dirty_boxes"]), 1)
        self.assertEqual(review["advisory_dirty_boxes"][0]["label"], "stain")

    def test_extract_dirty_region_candidates_returns_sorted_regions(self):
        mask = [[0] * 20 for _ in range(20)]
        for y in range(2, 10):
            for x in range(3, 12):
                mask[y][x] = 1
        for y in range(12, 18):
            for x in range(12, 18):
                mask[y][x] = 2

        import numpy as np

        candidates = extract_dirty_region_candidates(np.array(mask, dtype=np.uint8), max_regions=6, min_area_ratio=0.01)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["region_id"], 1)
        self.assertIn(candidates[0]["kind_hint"], {"stain_or_water", "wet_surface"})
        self.assertEqual(len(candidates[0]["bbox_px"]), 4)

    def test_queue_processes_jobs_in_fifo_order(self):
        queue_filter = _make_filter(queue_enabled=True, deadline_sec=0.5)
        start_order: list[str] = []
        finish_order: list[str] = []
        first_started = threading.Event()
        release_first = threading.Event()

        def fake_send(url: str, body: dict[str, object]):
            marker = str(body["contents"][0]["parts"][0]["text"])
            start_order.append(marker)
            if marker == "first":
                first_started.set()
                release_first.wait(timeout=1)
            finish_order.append(marker)
            return _json_candidate('{"keep_indexes":[0]}')

        queue_filter._send_request = fake_send  # type: ignore[method-assign]  # noqa: SLF001

        results: list[tuple[str, dict[str, object] | None]] = []

        def worker(marker: str):
            result = queue_filter._invoke_json(marker, self.image, kind="yolo", source=marker)  # noqa: SLF001
            results.append((marker, result))

        thread_one = threading.Thread(target=worker, args=("first",))
        thread_two = threading.Thread(target=worker, args=("second",))
        thread_one.start()
        first_started.wait(timeout=1)
        thread_two.start()
        time.sleep(0.05)
        self.assertEqual(queue_filter.status_payload()["llm_filter_queue_depth"], 2)
        release_first.set()
        thread_one.join(timeout=1)
        thread_two.join(timeout=1)

        self.assertEqual(start_order, ["first", "second"])
        self.assertEqual(finish_order, ["first", "second"])
        self.assertEqual(len(results), 2)
        self.assertEqual(queue_filter.status_payload()["llm_filter_queue_depth"], 0)

    def test_retryable_429_opens_cooldown_and_expires_without_looping(self):
        queue_filter = _make_filter(queue_enabled=True, deadline_sec=0.05)
        queue_filter._send_request = Mock(side_effect=_make_http_error(429))  # type: ignore[method-assign]  # noqa: SLF001

        result = queue_filter._invoke_json("prompt", self.image, kind="scoring", source="unit")  # noqa: SLF001

        self.assertIsNone(result)
        self.assertEqual(queue_filter.status_payload()["llm_filter_last_result"], "expired")
        self.assertIn("expired", queue_filter.status_payload()["llm_filter_last_error"])
        self.assertIsNotNone(queue_filter.status_payload()["llm_filter_cooldown_until"])

    def test_cooldown_skips_followup_requests(self):
        queue_filter = _make_filter(queue_enabled=False, deadline_sec=0.2)
        queue_filter._open_cooldown(30)  # type: ignore[attr-defined]  # noqa: SLF001

        result = queue_filter._invoke_json("prompt", self.image, kind="scoring", source="unit")  # noqa: SLF001

        self.assertIsNone(result)
        metadata = queue_filter.response_metadata("unit", ["scoring"])
        self.assertTrue(metadata["cooldown_active"])
        self.assertEqual(metadata["stages"]["scoring"]["status"], "skipped")
        self.assertEqual(metadata["stages"]["scoring"]["error"], "429_circuit_open")

    def test_non_retryable_400_falls_back_immediately(self):
        queue_filter = _make_filter(queue_enabled=True, deadline_sec=0.2)
        send_mock = Mock(side_effect=_make_http_error(400))
        queue_filter._send_request = send_mock  # type: ignore[method-assign]  # noqa: SLF001

        result = queue_filter._invoke_json("prompt", self.image, kind="scoring", source="unit")  # noqa: SLF001

        self.assertIsNone(result)
        self.assertEqual(send_mock.call_count, 1)
        self.assertEqual(queue_filter.status_payload()["llm_filter_last_result"], "non_retryable_error")
        self.assertIn("http_error:400", queue_filter.status_payload()["llm_filter_last_error"])

    def test_response_metadata_exposes_stage_trace(self):
        self.filter._set_trace(  # type: ignore[attr-defined]  # noqa: SLF001
            "visual",
            "unit-source",
            status="expired",
            error="http_429",
            attempts=4,
            response_preview=None,
        )

        metadata = self.filter.response_metadata("unit-source", ["scoring", "visual"])

        self.assertEqual(metadata["model"], "gemini-2.5-pro")
        self.assertIn("visual", metadata["stages"])
        self.assertEqual(metadata["stages"]["visual"]["status"], "expired")
        self.assertEqual(metadata["stages"]["visual"]["error"], "http_429")


if __name__ == "__main__":
    unittest.main()
