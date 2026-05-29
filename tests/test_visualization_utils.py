from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.visualization_utils import render_hybrid_overlay


class VisualizationUtilsTests(unittest.TestCase):
    def _render_with_capture(
        self,
        *,
        yolo_result: dict[str, object],
        scoring: dict[str, object],
        sam3_result: dict[str, object] | None = None,
        visual_review: dict[str, object] | None = None,
        dirty_region_candidates: list[dict[str, object]] | None = None,
    ):
        drawn_boxes: list[tuple[list[int], str]] = []
        panel_texts: list[str] = []

        def fake_draw_box(_image, bbox, label, _color, **_kwargs):
            drawn_boxes.append((bbox, label))

        def fake_put_text(image, text, *args, **kwargs):
            panel_texts.append(str(text))
            return image

        rgb = np.full((1000, 1000, 3), 240, dtype=np.uint8)
        mask = np.zeros((1000, 1000), dtype=np.uint8)

        with patch("src.api.visualization_utils._draw_labeled_box", side_effect=fake_draw_box):
            with patch("src.api.visualization_utils.cv2.putText", side_effect=fake_put_text):
                rendered = render_hybrid_overlay(
                    rgb=rgb,
                    pred_original_size=mask,
                    yolo_result=yolo_result,
                    scoring=scoring,
                    env_key="LOBBY_CORRIDOR",
                    visualize_jpeg_quality=92,
                    visual_review=visual_review or {"keep_detection_indexes": [0, 1]},
                    dirty_region_candidates=dirty_region_candidates or [],
                    sam3_result=sam3_result,
                )

        self.assertGreater(len(rendered), 0)
        return drawn_boxes, panel_texts

    def test_renderer_draws_only_penalty_object_boxes(self):
        drawn_boxes, panel_texts = self._render_with_capture(
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
        drawn_boxes, panel_texts = self._render_with_capture(
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

    def test_renderer_does_not_draw_sam3_prompt_boxes(self):
        sam3_mask = np.zeros((1000, 1000), dtype=np.uint8)
        sam3_mask[100:180, 120:220] = 1
        drawn_boxes, _ = self._render_with_capture(
            yolo_result={"detections_count": 0, "results": []},
            scoring={
                "verdict": "PENDING",
                "quality_score": 82.0,
                "base_clean_score": 82.0,
                "object_penalty": 0.0,
                "penalty_detections_count": 0,
                "penalty_detection_indexes": [],
            },
            sam3_result={
                "_mask_union": sam3_mask,
                "predictions": [
                    {
                        "prompt": "dirty area",
                        "confidence": 0.81,
                        "bbox_xyxy": [120, 100, 220, 180],
                    }
                ],
            },
        )

        self.assertEqual(drawn_boxes, [])

    def test_renderer_does_not_draw_dirty_region_boxes(self):
        drawn_boxes, _ = self._render_with_capture(
            yolo_result={"detections_count": 0, "results": []},
            scoring={
                "verdict": "PENDING",
                "quality_score": 60.0,
                "base_clean_score": 60.0,
                "object_penalty": 0.0,
                "penalty_detections_count": 0,
                "penalty_detection_indexes": [],
            },
            visual_review={
                "highlight_dirty_region_ids": [1],
                "dirty_region_labels": [{"region_id": 1, "label": "stain"}],
                "advisory_dirty_boxes": [{"bbox_px": [20, 20, 120, 120], "label": "stain"}],
            },
            dirty_region_candidates=[
                {"region_id": 1, "bbox_px": [20, 20, 120, 120], "kind_hint": "stain"}
            ],
        )

        self.assertEqual(drawn_boxes, [])

    def test_renderer_shows_calibration_context(self):
        _, panel_texts = self._render_with_capture(
            yolo_result={"detections_count": 0, "results": []},
            scoring={
                "verdict": "PENDING",
                "raw_verdict": "PASS",
                "quality_score": 84.999,
                "base_clean_score": 100.0,
                "object_penalty": 0.0,
                "penalty_detections_count": 0,
                "penalty_detection_indexes": [],
                "calibrated": True,
                "calibration_rules": ["high_risk_weak_evidence_review"],
            },
        )

        self.assertTrue(any(text == "RAW: PASS -> PENDING" for text in panel_texts))
        self.assertTrue(any(text == "CALIBRATED: review required" for text in panel_texts))


if __name__ == "__main__":
    unittest.main()
