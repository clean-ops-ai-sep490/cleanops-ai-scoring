from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.scoring_utils import (
    calibrate_score,
    combine_dirty_coverage,
    merge_unet_and_sam3_masks,
    score_image,
    summarize_penalty_detections,
)


ENV_RULES = {
    "LOBBY_CORRIDOR": {
        "pass_threshold": 90.0,
        "label": "Lobby",
    }
}
PENALTY_LABELS = (
    "metal",
    "paper",
    "plastic",
    "trash",
    "marks",
    "garbage",
    "rubbish",
    "litter",
    "waste",
    "debris",
    "bottle",
    "plastic_bottle",
    "can",
    "cup",
    "cardboard",
    "bag",
    "trash_bag",
)


def _score(detections: list[dict[str, object]]) -> dict[str, object]:
    penalty_summary = summarize_penalty_detections(detections, PENALTY_LABELS)
    return score_image(
        total_dirty_coverage_pct=0.0,
        detections_count=len(detections),
        env_key="LOBBY_CORRIDOR",
        env_rules=ENV_RULES,
        pending_lower_bound=50.0,
        object_penalty_per_detection=10.0,
        **penalty_summary,
    )


class ScoringUtilsTests(unittest.TestCase):
    def test_one_trash_detection_penalty_is_ten(self):
        scoring = _score([{"class_name": "trash"}])

        self.assertEqual(scoring["penalty_detections_count"], 1)
        self.assertEqual(scoring["object_penalty"], 10.0)
        self.assertEqual(scoring["quality_score"], 90.0)

    def test_three_penalty_detections_penalize_thirty(self):
        scoring = _score(
            [
                {"class_name": "trash"},
                {"class_name": "paper"},
                {"class_name": "plastic"},
            ]
        )

        self.assertEqual(scoring["penalty_detections_count"], 3)
        self.assertEqual(scoring["object_penalty"], 30.0)
        self.assertEqual(scoring["quality_score"], 70.0)

    def test_four_penalty_detections_cap_at_forty(self):
        scoring = _score(
            [
                {"class_name": "trash"},
                {"class_name": "paper"},
                {"class_name": "plastic"},
                {"class_name": "bottle"},
            ]
        )

        self.assertEqual(scoring["penalty_detections_count"], 4)
        self.assertEqual(scoring["object_penalty"], 40.0)
        self.assertEqual(scoring["quality_score"], 60.0)

    def test_five_penalty_detections_stay_capped_at_forty(self):
        scoring = _score(
            [
                {"class_name": "trash"},
                {"class_name": "paper"},
                {"class_name": "plastic"},
                {"class_name": "bottle"},
                {"class_name": "can"},
            ]
        )

        self.assertEqual(scoring["penalty_detections_count"], 5)
        self.assertEqual(scoring["object_penalty"], 40.0)
        self.assertEqual(scoring["quality_score"], 60.0)

    def test_non_trash_like_objects_are_not_penalized(self):
        scoring = _score(
            [
                {"class_name": "trash"},
                {"class_name": "toilet"},
                {"class_name": "chair"},
            ]
        )

        self.assertEqual(scoring["penalty_detections_count"], 1)
        self.assertEqual(scoring["ignored_detections_count"], 2)
        self.assertEqual(scoring["object_penalty"], 10.0)
        self.assertEqual(scoring["penalty_detection_labels"], ["trash"])
        self.assertEqual(scoring["ignored_detection_labels"], ["chair", "toilet"])
        self.assertEqual(scoring["penalty_detection_indexes"], [0])
        self.assertEqual(scoring["ignored_detection_indexes"], [1, 2])

    def test_label_variants_are_normalized(self):
        scoring = _score(
            [
                {"class_name": "trash-bag"},
                {"class_name": "plastic bottle"},
                {"class_name": "Garbage"},
            ]
        )

        self.assertEqual(scoring["penalty_detections_count"], 3)
        self.assertEqual(scoring["object_penalty"], 30.0)
        self.assertEqual(
            scoring["penalty_detection_labels"],
            ["garbage", "plastic_bottle", "trash_bag"],
        )

    def test_verified_false_positive_marks_detection_can_be_removed(self):
        scoring = _score([])

        self.assertEqual(scoring["penalty_detections_count"], 0)
        self.assertEqual(scoring["object_penalty"], 0.0)
        self.assertEqual(scoring["quality_score"], 100.0)

    def test_label_filter_uses_penalty_labels_not_total_detections(self):
        scoring = _score(
            [
                {"class_name": "toilet"},
                {"class_name": "chair"},
                {"class_name": "paper"},
            ]
        )

        self.assertEqual(scoring["penalty_detections_count"], 1)
        self.assertEqual(scoring["ignored_detections_count"], 2)
        self.assertEqual(scoring["object_penalty"], 10.0)
        self.assertEqual(scoring["penalty_detection_indexes"], [2])
        self.assertEqual(scoring["ignored_detection_indexes"], [0, 1])

    def test_combined_dirty_coverage_uses_max_of_unet_and_sam3(self):
        combined = combine_dirty_coverage(3.25, 8.5)

        self.assertEqual(combined["unet_dirty_coverage_pct"], 3.25)
        self.assertEqual(combined["sam3_dirty_coverage_pct"], 8.5)
        self.assertEqual(combined["combined_dirty_coverage_pct"], 8.5)
        self.assertEqual(combined["dirty_coverage_source"], "sam3")

    def test_merged_mask_unions_unet_and_roboflow_without_double_count(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[0:4, 0:4] = 1
        stain_mask = np.zeros((10, 10), dtype=np.uint8)
        stain_mask[2:6, 2:6] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {"_label_masks": {"Stain": stain_mask}},
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
        )

        self.assertEqual(merged["dirty_coverage_source"], "merged")
        self.assertEqual(merged["combined_dirty_coverage_pct"], 28.0)
        self.assertEqual(merged["merged_stain_or_water_coverage_pct"], 28.0)
        self.assertEqual(merged["roboflow_label_class_counts"]["stain_or_water"], 1)

    def test_merged_mask_maps_wet_floor_to_wet_surface(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        wet_mask = np.zeros((10, 10), dtype=np.uint8)
        wet_mask[1:4, 1:4] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {"_label_masks": {"Wet_Floor": wet_mask}},
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
        )

        merged_mask = merged["merged_mask"]
        self.assertEqual(merged["dirty_coverage_source"], "sam3")
        self.assertEqual(merged["combined_dirty_coverage_pct"], 9.0)
        self.assertEqual(merged["merged_wet_surface_coverage_pct"], 9.0)
        self.assertTrue(np.all(merged_mask[1:4, 1:4] == 2))

    def test_calibration_downgrades_high_risk_weak_pass_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "PASS",
                "quality_score": 100.0,
                "pass_threshold": 85.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 0.0,
                "sam3_dirty_coverage_pct": 0.0,
                "combined_dirty_coverage_pct": 0.0,
                "dirty_coverage_source": "equal",
                "sam3_predictions_count": 0,
                "reasons": ["good cleanliness"],
            },
            env_key="RESTROOM",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["raw_verdict"], "PASS")
        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertTrue(calibrated["calibrated"])
        self.assertIn("high_risk_weak_evidence_review", calibrated["calibration_rules"])

    def test_calibration_downgrades_pass_with_ignored_objects_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "PASS",
                "quality_score": 94.374,
                "pass_threshold": 90.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 4,
                "ignored_detection_labels": ["person", "skis", "snowboard"],
                "unet_dirty_coverage_pct": 5.626,
                "sam3_dirty_coverage_pct": 0.0,
                "combined_dirty_coverage_pct": 5.626,
                "dirty_coverage_source": "unet",
                "sam3_predictions_count": 0,
                "reasons": ["good cleanliness"],
            },
            env_key="LOBBY_CORRIDOR",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertTrue(calibrated["calibrated"])
        self.assertIn("ignored_objects_review", calibrated["calibration_rules"])

    def test_calibration_ignores_non_review_ignored_object_labels(self):
        calibrated = calibrate_score(
            {
                "verdict": "PASS",
                "quality_score": 99.715,
                "pass_threshold": 90.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 4,
                "ignored_detection_labels": ["book", "couch", "suitcase"],
                "unet_dirty_coverage_pct": 0.285,
                "sam3_dirty_coverage_pct": 0.0,
                "combined_dirty_coverage_pct": 0.285,
                "dirty_coverage_source": "unet",
                "sam3_predictions_count": 0,
                "reasons": ["good cleanliness"],
            },
            env_key="LOBBY_CORRIDOR",
            pending_lower_bound=50.0,
            ignored_object_review_labels=("person", "skis", "snowboard"),
        )

        self.assertEqual(calibrated["verdict"], "PASS")
        self.assertFalse(calibrated["calibrated"])

    def test_calibration_moves_unet_only_fail_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "FAIL",
                "quality_score": 43.907,
                "pass_threshold": 90.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 56.093,
                "sam3_dirty_coverage_pct": 0.0,
                "combined_dirty_coverage_pct": 56.093,
                "dirty_coverage_source": "unet",
                "sam3_predictions_count": 0,
                "reasons": ["coverage high"],
            },
            env_key="LOBBY_CORRIDOR",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertGreaterEqual(calibrated["quality_score"], 50.0)
        self.assertIn("unet_only_high_coverage_review", calibrated["calibration_rules"])

    def test_calibration_moves_single_sam3_large_mask_fail_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "FAIL",
                "quality_score": 39.502,
                "pass_threshold": 80.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 33.523,
                "sam3_dirty_coverage_pct": 60.498,
                "combined_dirty_coverage_pct": 60.498,
                "dirty_coverage_source": "sam3",
                "sam3_predictions_count": 1,
                "reasons": ["coverage high"],
            },
            env_key="OUTDOOR_LANDSCAPE",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertIn("single_sam3_large_mask_review", calibrated["calibration_rules"])

    def test_calibration_moves_auxiliary_merged_fail_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "FAIL",
                "quality_score": 35.0,
                "pass_threshold": 90.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 12.0,
                "sam3_dirty_coverage_pct": 55.0,
                "combined_dirty_coverage_pct": 61.0,
                "dirty_coverage_source": "merged",
                "sam3_predictions_count": 5,
                "reasons": ["coverage high"],
            },
            env_key="LOBBY_CORRIDOR",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertIn("auxiliary_segmentation_review", calibrated["calibration_rules"])

    def test_calibration_moves_weak_merged_union_fail_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "FAIL",
                "quality_score": 42.0,
                "pass_threshold": 90.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 31.0,
                "sam3_dirty_coverage_pct": 41.0,
                "combined_dirty_coverage_pct": 57.0,
                "dirty_coverage_source": "merged",
                "sam3_predictions_count": 1,
                "reasons": ["coverage high"],
            },
            env_key="LOBBY_CORRIDOR",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertIn("auxiliary_segmentation_review", calibrated["calibration_rules"])

    def test_calibration_keeps_strong_multi_source_dirty_fail(self):
        calibrated = calibrate_score(
            {
                "verdict": "PENDING",
                "quality_score": 55.0,
                "pass_threshold": 85.0,
                "penalty_detections_count": 2,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 30.0,
                "sam3_dirty_coverage_pct": 48.0,
                "combined_dirty_coverage_pct": 48.0,
                "dirty_coverage_source": "sam3",
                "sam3_predictions_count": 3,
                "reasons": ["coverage high", "trash-like objects remain"],
            },
            env_key="RESTROOM",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "FAIL")
        self.assertTrue(calibrated["calibrated"])
        self.assertIn("strong_multi_source_dirty", calibrated["calibration_rules"])

    def test_calibration_moves_coverage_only_fail_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "FAIL",
                "quality_score": 28.0,
                "pass_threshold": 90.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 58.0,
                "sam3_dirty_coverage_pct": 62.0,
                "combined_dirty_coverage_pct": 82.0,
                "dirty_coverage_source": "merged",
                "sam3_predictions_count": 2,
                "reasons": ["coverage high"],
            },
            env_key="LOBBY_CORRIDOR",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertTrue(calibrated["calibrated"])
        self.assertIn("coverage_only_fail_review", calibrated["calibration_rules"])


if __name__ == "__main__":
    unittest.main()
