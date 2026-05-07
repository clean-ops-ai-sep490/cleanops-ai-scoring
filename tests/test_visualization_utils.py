from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.visualization_utils import build_visualize_blob_url_payload, normalize_visual_review, render_hybrid_overlay


class VisualizationUtilsTests(unittest.TestCase):
    def _render_with_capture(
        self,
        *,
        yolo_result: dict[str, object],
        scoring: dict[str, object],
        visual_review: dict[str, object] | None = None,
    ):
        drawn_boxes: list[tuple[list[int], str]] = []
        panel_texts: list[str] = []
        rectangles: list[tuple[tuple[int, int], tuple[int, int], int]] = []

        def fake_draw_box(_image, bbox, label, _color, **_kwargs):
            drawn_boxes.append((bbox, label))

        def fake_put_text(image, text, *args, **kwargs):
            panel_texts.append(str(text))
            return image

        original_rectangle = render_hybrid_overlay.__globals__["cv2"].rectangle

        def fake_rectangle(image, pt1, pt2, color, thickness, *args, **kwargs):
            rectangles.append((pt1, pt2, int(thickness)))
            return original_rectangle(image, pt1, pt2, color, thickness, *args, **kwargs)

        rgb = np.full((1000, 1000, 3), 240, dtype=np.uint8)
        mask = np.zeros((1000, 1000), dtype=np.uint8)

        with patch("src.api.visualization_utils._draw_labeled_box", side_effect=fake_draw_box):
            with patch("src.api.visualization_utils.cv2.putText", side_effect=fake_put_text):
                with patch("src.api.visualization_utils.cv2.rectangle", side_effect=fake_rectangle):
                    rendered = render_hybrid_overlay(
                        rgb=rgb,
                        pred_original_size=mask,
                        yolo_result=yolo_result,
                        scoring=scoring,
                        env_key="LOBBY_CORRIDOR",
                        visualize_jpeg_quality=92,
                        visual_review=visual_review or {"keep_detection_indexes": [0, 1]},
                        dirty_region_candidates=[],
                    )

        self.assertGreater(len(rendered), 0)
        return drawn_boxes, panel_texts, rectangles

    def test_renderer_draws_only_penalty_object_boxes(self):
        drawn_boxes, panel_texts, _ = self._render_with_capture(
            yolo_result={
                "detections_count": 2,
                "results": [
                    {"class_name": "trash", "confidence": 0.95, "bbox": [10, 20, 110, 120]},
                    {"class_name": "toilet", "confidence": 0.91, "bbox": [210, 220, 310, 320]},
                ],
            },
            scoring={
                "verdict": "PENDING",
                "quality_score": 75.0,
                "base_clean_score": 85.0,
                "object_penalty": 10.0,
                "penalty_detections_count": 1,
                "penalty_detection_indexes": [0],
            },
        )

        self.assertEqual(len(drawn_boxes), 1)
        self.assertIn("trash", drawn_boxes[0][1])
        self.assertTrue(any(text == "PENALTY OBJECTS: 1" for text in panel_texts))

    def test_renderer_hides_ignored_objects_when_no_penalty_indexes(self):
        drawn_boxes, panel_texts, _ = self._render_with_capture(
            yolo_result={
                "detections_count": 1,
                "results": [
                    {"class_name": "toilet", "confidence": 0.91, "bbox": [210, 220, 310, 320]},
                ],
            },
            scoring={
                "verdict": "PASS",
                "quality_score": 95.0,
                "base_clean_score": 95.0,
                "object_penalty": 0.0,
                "penalty_detections_count": 0,
                "penalty_detection_indexes": [],
            },
        )

        self.assertEqual(drawn_boxes, [])
        self.assertTrue(any(text == "PENALTY OBJECTS: 0" for text in panel_texts))

    def test_renderer_fills_and_labels_advisory_dirty_regions_consistently(self):
        drawn_boxes, panel_texts, rectangles = self._render_with_capture(
            yolo_result={"detections_count": 0, "results": []},
            scoring={
                "verdict": "FAIL",
                "quality_score": 45.0,
                "base_clean_score": 89.5,
                "object_penalty": 0.0,
                "penalty_detections_count": 0,
                "penalty_detection_indexes": [],
            },
            visual_review={
                "floor_condition": "dirty",
                "needs_cleaning": True,
                "estimated_dirty_coverage_pct": 10.5,
                "advisory_dirty_boxes": [
                    {
                        "label": "dirty_zone",
                        "bbox_px": [100, 100, 300, 300],
                        "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                        "reason": "dirty zone alias",
                    }
                ],
                "reasons": ["dirty zone alias"],
                "overlay_summary": "dirty zone alias",
            },
        )

        self.assertTrue(any(label == "Dirty area" for _, label in drawn_boxes))
        self.assertFalse(any("dirty zone" in label.lower() for _, label in drawn_boxes))
        self.assertTrue(any(text == "FLOOR CONDITION: Dirty" for text in panel_texts))
        self.assertTrue(any(text == "CLEANING REQUIRED: YES" for text in panel_texts))
        self.assertTrue(any(thickness == -1 for _, _, thickness in rectangles))

    def test_normalize_visual_review_converts_dirty_zone_alias(self):
        review = normalize_visual_review(
            {
                "floor_condition": "dirty",
                "needs_cleaning": True,
                "advisory_dirty_boxes": [
                    {
                        "label": "dirty_zone",
                        "bbox_px": [1, 2, 3, 4],
                        "bbox_norm": [0.1, 0.2, 0.3, 0.4],
                        "reason": "dirty zone",
                    }
                ],
                "reasons": ["dirty zone"],
                "overlay_summary": "dirty zone",
            }
        )

        self.assertEqual(review["floor_condition"], "dirty")
        self.assertEqual(review["advisory_dirty_boxes"][0]["label"], "dirty_area")
        self.assertEqual(review["advisory_dirty_boxes"][0]["label_text"], "Dirty area")
        self.assertEqual(review["evidence_regions"][0]["type"], "dirty_area")
        self.assertIn("Dirty area", review["overlay_summary"])

    def test_visualize_payload_includes_normalized_visual_review(self):
        payload = build_visualize_blob_url_payload(
            source_type="url",
            source="https://example.test/a.jpg",
            env_key="LOBBY_CORRIDOR",
            yolo_result={"detections_count": 0, "results": []},
            unet_result={"summary": {"total_dirty_coverage_pct": 10.5}},
            scoring={"verdict": "FAIL", "quality_score": 45.0},
            visualization_url="https://example.test/v.jpg",
            mime_type="image/jpeg",
            byte_size=123,
            visual_review={
                "floor_condition": "dirty",
                "needs_cleaning": True,
                "advisory_dirty_boxes": [{"label": "dirty_zone", "bbox_px": [1, 2, 3, 4]}],
            },
        )

        self.assertEqual(payload["visual_review"]["floor_condition"], "dirty")
        self.assertTrue(payload["visual_review"]["needs_cleaning"])
        self.assertEqual(payload["visual_review"]["advisory_dirty_boxes"][0]["label"], "dirty_area")


if __name__ == "__main__":
    unittest.main()
