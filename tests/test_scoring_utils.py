from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.scoring_utils import (
    DEFAULT_ROBOFLOW_DIRTY_LABELS,
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
    },
    "RESTROOM": {
        "pass_threshold": 85.0,
        "label": "Restroom",
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
    def test_default_roboflow_dirty_labels_exclude_stained_floor(self):
        self.assertNotIn("stained_floor", DEFAULT_ROBOFLOW_DIRTY_LABELS)
        self.assertIn("stain", DEFAULT_ROBOFLOW_DIRTY_LABELS)
        self.assertIn("trash", DEFAULT_ROBOFLOW_DIRTY_LABELS)

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
        self.assertEqual(merged["dirty_coverage_source"], "equal")
        self.assertEqual(merged["combined_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_scored_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_advisory_coverage_pct"], 9.0)
        self.assertEqual(merged["merged_wet_surface_coverage_pct"], 0.0)
        self.assertTrue(np.all(merged_mask[1:4, 1:4] == 0))
        self.assertIn("wet_surface_advisory", merged["sam3_filter_rules"])

    def test_large_single_bbox_fallback_is_advisory_not_scored(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        aux_mask = np.ones((10, 10), dtype=np.uint8)

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": aux_mask,
                        "mask_source": "bbox_fallback",
                    }
                ]
            },
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertEqual(merged["combined_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_raw_coverage_pct"], 100.0)
        self.assertEqual(merged["sam3_scored_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_advisory_coverage_pct"], 100.0)
        self.assertEqual(merged["sam3_scored_predictions_count"], 0)
        self.assertEqual(merged["sam3_advisory_predictions_count"], 1)
        self.assertIn("large_bbox_fallback_advisory", merged["sam3_filter_rules"])

    def test_large_single_polygon_normal_env_is_advisory_when_unet_clean(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        aux_mask = np.zeros((10, 10), dtype=np.uint8)
        aux_mask[:, :6] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": aux_mask,
                        "mask_source": "polygon",
                    }
                ]
            },
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertEqual(merged["combined_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_advisory_coverage_pct"], 60.0)
        self.assertIn("single_giant_aux_ignored", merged["sam3_filter_rules"])

    def test_large_single_polygon_high_risk_is_advisory_for_review(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        aux_mask = np.zeros((10, 10), dtype=np.uint8)
        aux_mask[:, :6] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": aux_mask,
                        "mask_source": "polygon",
                    }
                ]
            },
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="RESTROOM",
        )

        self.assertEqual(merged["combined_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_advisory_coverage_pct"], 60.0)
        self.assertIn("single_giant_aux_high_risk_review", merged["sam3_filter_rules"])

    def test_multiple_dirty_regions_are_scored(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        masks = []
        for col in (0, 3, 6):
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[0:2, col : col + 2] = 1
            masks.append(mask)

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {"label": "Stain", "label_normalized": "stain", "mask": mask, "mask_source": "polygon"}
                    for mask in masks
                ]
            },
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertEqual(merged["sam3_scored_predictions_count"], 3)
        self.assertEqual(merged["sam3_scored_coverage_pct"], 12.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 12.0)
        self.assertEqual(merged["sam3_penalty_detections_count"], 0)
        self.assertIn("aux_prediction_scored", merged["sam3_filter_rules"])

    def test_sam3_trash_predictions_count_as_object_penalty_without_dirty_coverage(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        trash_a = np.zeros((10, 10), dtype=np.uint8)
        trash_a[0:2, 0:2] = 1
        trash_b = np.zeros((10, 10), dtype=np.uint8)
        trash_b[3:5, 7:9] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {"label": "Trash", "label_normalized": "trash", "mask": trash_a, "mask_source": "polygon"},
                    {"label": "Trash", "label_normalized": "trash", "mask": trash_b, "mask_source": "polygon"},
                ]
            },
            dirty_labels=("Trash", "Stain"),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )
        yolo_penalty = summarize_penalty_detections([], PENALTY_LABELS)
        penalty_count = int(yolo_penalty["penalty_detections_count"]) + int(
            merged["sam3_penalty_detections_count"]
        )
        scoring = score_image(
            total_dirty_coverage_pct=merged["combined_dirty_coverage_pct"],
            detections_count=0,
            env_key="LOBBY_CORRIDOR",
            env_rules=ENV_RULES,
            pending_lower_bound=50.0,
            penalty_detections_count=penalty_count,
            object_penalty_per_detection=10.0,
            penalty_detection_labels=sorted(
                set(yolo_penalty["penalty_detection_labels"] + merged["sam3_penalty_detection_labels"])
            ),
        )

        self.assertEqual(merged["sam3_scored_predictions_count"], 0)
        self.assertEqual(merged["sam3_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_object_penalty_detections_count"], 2)
        self.assertEqual(merged["sam3_penalty_detections_count"], 2)
        self.assertEqual(merged["sam3_penalty_detection_labels"], ["trash"])
        self.assertEqual(merged["sam3_penalty_detection_indexes"], [0, 1])
        self.assertEqual(scoring["penalty_detections_count"], 2)
        self.assertEqual(scoring["object_penalty"], 20.0)
        self.assertEqual(scoring["quality_score"], 80.0)
        self.assertEqual(scoring["verdict"], "PENDING")
        self.assertIn("trash-like objects remain", scoring["reasons"])

    def test_overlapping_sam3_trash_labels_count_as_one_object_penalty(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        trash_mask = np.zeros((10, 10), dtype=np.uint8)
        trash_mask[1:8, 1:8] = 1
        garbage_mask = np.zeros((10, 10), dtype=np.uint8)
        garbage_mask[2:8, 2:8] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Trash",
                        "label_normalized": "trash",
                        "mask": trash_mask,
                        "mask_source": "polygon",
                        "confidence": 0.52,
                    },
                    {
                        "label": "Garbage",
                        "label_normalized": "garbage",
                        "mask": garbage_mask,
                        "mask_source": "polygon",
                        "confidence": 0.74,
                    },
                ]
            },
            dirty_labels=("Trash", "Garbage", "Stain"),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertEqual(merged["sam3_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_object_penalty_detections_count"], 1)
        self.assertEqual(merged["sam3_penalty_detections_count"], 1)
        self.assertEqual(merged["sam3_penalty_detection_labels"], ["garbage"])
        self.assertEqual(merged["sam3_penalty_detection_indexes"], [1])
        self.assertIn("object_trash_penalty", merged["sam3_filter_rules"])

    def test_sam3_stain_still_increases_dirty_coverage(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        stain_mask = np.zeros((10, 10), dtype=np.uint8)
        stain_mask[0:2, 0:5] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": stain_mask,
                        "mask_source": "polygon",
                    }
                ]
            },
            dirty_labels=("Trash", "Stain"),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertEqual(merged["sam3_scored_predictions_count"], 1)
        self.assertEqual(merged["sam3_dirty_coverage_pct"], 10.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 10.0)
        self.assertEqual(merged["sam3_penalty_detections_count"], 0)

    def test_yolo_and_sam3_penalty_counts_are_additive_and_capped(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        masks = []
        for col in (0, 3, 6):
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[0:2, col : col + 2] = 1
            masks.append(mask)

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {"label": "Garbage", "label_normalized": "garbage", "mask": mask, "mask_source": "polygon"}
                    for mask in masks
                ]
            },
            dirty_labels=("Garbage",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )
        yolo_penalty = summarize_penalty_detections(
            [{"class_name": "trash"}, {"class_name": "bottle"}],
            PENALTY_LABELS,
        )
        penalty_count = int(yolo_penalty["penalty_detections_count"]) + int(
            merged["sam3_penalty_detections_count"]
        )

        scoring = score_image(
            total_dirty_coverage_pct=merged["combined_dirty_coverage_pct"],
            detections_count=2,
            env_key="LOBBY_CORRIDOR",
            env_rules=ENV_RULES,
            pending_lower_bound=50.0,
            penalty_detections_count=penalty_count,
            object_penalty_per_detection=10.0,
            penalty_detection_labels=sorted(
                set(yolo_penalty["penalty_detection_labels"] + merged["sam3_penalty_detection_labels"])
            ),
        )

        self.assertEqual(merged["sam3_penalty_detections_count"], 3)
        self.assertEqual(scoring["penalty_detections_count"], 5)
        self.assertEqual(scoring["object_penalty"], 40.0)

    def test_floor_like_unet_overmask_is_discounted_in_normal_env(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[4:10, :] = 1
        advisory_mask = np.zeros((10, 10), dtype=np.uint8)
        advisory_mask[4:10, :] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": advisory_mask,
                        "mask_source": "polygon",
                    }
                ]
            },
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertTrue(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "floor_like_overmask_discount")
        self.assertEqual(merged["raw_combined_dirty_coverage_pct"], 60.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 6.0)
        self.assertEqual(merged["effective_dirty_coverage_pct"], 6.0)
        self.assertEqual(merged["coverage_adjustment_factor"], 0.1)
        self.assertEqual(merged["unet_component_count"], 1)
        self.assertEqual(merged["unet_largest_component_area_pct"], 60.0)

        scoring = score_image(
            total_dirty_coverage_pct=merged["combined_dirty_coverage_pct"],
            detections_count=0,
            env_key="LOBBY_CORRIDOR",
            env_rules=ENV_RULES,
            pending_lower_bound=50.0,
        )
        scoring.update(merged)
        calibrated = calibrate_score(scoring, env_key="LOBBY_CORRIDOR", pending_lower_bound=50.0)

        self.assertEqual(calibrated["verdict"], "PASS")
        self.assertEqual(calibrated["quality_score"], 94.0)

    def test_large_unet_only_floor_like_mask_stays_review_in_normal_env(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[4:10, :] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {"_prediction_masks": []},
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )
        scoring = score_image(
            total_dirty_coverage_pct=merged["combined_dirty_coverage_pct"],
            detections_count=0,
            env_key="LOBBY_CORRIDOR",
            env_rules=ENV_RULES,
            pending_lower_bound=50.0,
        )
        scoring.update(merged)
        calibrated = calibrate_score(scoring, env_key="LOBBY_CORRIDOR", pending_lower_bound=50.0)

        self.assertTrue(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "floor_like_overmask_review_unet_only")
        self.assertEqual(merged["raw_combined_dirty_coverage_pct"], 60.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 60.0)
        self.assertEqual(merged["effective_dirty_coverage_pct"], 60.0)
        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertTrue(calibrated["review_required"])

    def test_wet_only_unet_floor_like_mask_is_not_discounted_to_pass(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[4:10, :] = 2
        advisory_mask = np.zeros((10, 10), dtype=np.uint8)
        advisory_mask[4:10, :] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": advisory_mask,
                        "mask_source": "polygon",
                    }
                ]
            },
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertFalse(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "")
        self.assertEqual(merged["raw_combined_dirty_coverage_pct"], 60.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 60.0)

    def test_aux_advisory_allows_reflective_floor_with_some_dirty_signal_discount(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[7:8, :] = 1
        unet_mask[8:10, :] = 2
        advisory_mask = np.zeros((10, 10), dtype=np.uint8)
        advisory_mask[4:10, :] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": advisory_mask,
                        "mask_source": "polygon",
                    }
                ]
            },
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertTrue(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "floor_like_overmask_discount")
        self.assertEqual(merged["raw_combined_dirty_coverage_pct"], 30.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 3.0)

    def test_reflective_unet_floor_signal_can_discount_without_aux_advisory(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[7:8, :] = 1
        unet_mask[8:10, :] = 2

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {"_prediction_masks": []},
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertTrue(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "floor_like_overmask_discount")
        self.assertEqual(merged["raw_combined_dirty_coverage_pct"], 30.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 3.0)

    def test_floor_like_unet_overmask_in_restroom_stays_pending_review(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[4:10, :] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {"_prediction_masks": []},
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="RESTROOM",
        )
        scoring = score_image(
            total_dirty_coverage_pct=merged["combined_dirty_coverage_pct"],
            detections_count=0,
            env_key="RESTROOM",
            env_rules=ENV_RULES,
            pending_lower_bound=50.0,
        )
        scoring.update(merged)
        calibrated = calibrate_score(scoring, env_key="RESTROOM", pending_lower_bound=50.0)

        self.assertTrue(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "floor_like_overmask_high_risk_review")
        self.assertEqual(merged["raw_combined_dirty_coverage_pct"], 60.0)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 24.0)
        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertTrue(calibrated["review_required"])
        self.assertIn("high_risk_floor_like_review", calibrated["calibration_rules"])

    def test_fragmented_unet_mask_is_not_discounted(self):
        unet_mask = np.zeros((20, 20), dtype=np.uint8)
        for row in (2, 6, 10, 14):
            unet_mask[row : row + 2, 0:10] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {"_prediction_masks": []},
            dirty_labels=("Stain",),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertFalse(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "")
        self.assertEqual(merged["raw_combined_dirty_coverage_pct"], merged["combined_dirty_coverage_pct"])
        self.assertGreater(merged["unet_component_count"], 3)

    def test_strong_sam3_scored_evidence_prevents_floor_discount(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[4:10, :] = 1
        masks = []
        for col in (0, 3, 6):
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[0:2, col : col + 2] = 1
            masks.append(mask)

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {"label": "Stain", "label_normalized": "stain", "mask": mask, "mask_source": "polygon"}
                    for mask in masks
                ]
            },
            dirty_labels=("Garbage", "Stain"),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertFalse(merged["floor_like_overmask_detected"])
        self.assertEqual(merged["coverage_adjustment_reason"], "")
        self.assertEqual(merged["sam3_scored_predictions_count"], 3)
        self.assertEqual(merged["combined_dirty_coverage_pct"], 72.0)

    def test_single_giant_stain_on_floor_like_unet_is_advisory(self):
        unet_mask = np.zeros((10, 10), dtype=np.uint8)
        unet_mask[4:10, :] = 1
        stain_mask = np.zeros((10, 10), dtype=np.uint8)
        stain_mask[4:10, :] = 1

        merged = merge_unet_and_sam3_masks(
            unet_mask,
            {
                "_prediction_masks": [
                    {
                        "label": "Stain",
                        "label_normalized": "stain",
                        "mask": stain_mask,
                        "mask_source": "polygon",
                    }
                ]
            },
            dirty_labels=("Garbage", "Stain"),
            wet_labels=("Wet_Floor",),
            env_key="LOBBY_CORRIDOR",
        )

        self.assertEqual(merged["sam3_scored_coverage_pct"], 0.0)
        self.assertEqual(merged["sam3_advisory_coverage_pct"], 60.0)
        self.assertEqual(merged["sam3_penalty_detections_count"], 0)
        self.assertIn("giant_stain_floor_like_advisory", merged["sam3_filter_rules"])
        self.assertEqual(merged["coverage_adjustment_reason"], "floor_like_overmask_discount")
        self.assertEqual(merged["combined_dirty_coverage_pct"], 6.0)

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
        self.assertEqual(calibrated["raw_quality_score"], 100.0)
        self.assertEqual(calibrated["quality_score"], 100.0)
        self.assertEqual(calibrated["decision_score"], 84.999)
        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertTrue(calibrated["review_required"])
        self.assertEqual(calibrated["verdict_source"], "calibration")
        self.assertEqual(calibrated["verdict_reason_code"], "high_risk_weak_evidence_review")
        self.assertIn("high_risk_weak_evidence_review", calibrated["risk_flags"])
        self.assertTrue(calibrated["calibrated"])
        self.assertIn("high_risk_weak_evidence_review", calibrated["calibration_rules"])

    def test_calibration_downgrades_high_risk_advisory_aux_pass_to_pending(self):
        calibrated = calibrate_score(
            {
                "verdict": "PASS",
                "quality_score": 100.0,
                "pass_threshold": 85.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 0.0,
                "sam3_dirty_coverage_pct": 0.0,
                "sam3_advisory_coverage_pct": 60.0,
                "combined_dirty_coverage_pct": 0.0,
                "dirty_coverage_source": "equal",
                "sam3_predictions_count": 1,
                "reasons": ["good cleanliness"],
            },
            env_key="RESTROOM",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertTrue(calibrated["review_required"])
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
        self.assertEqual(calibrated["quality_score"], 94.374)
        self.assertEqual(calibrated["decision_score"], 89.999)
        self.assertTrue(calibrated["review_required"])
        self.assertTrue(calibrated["calibrated"])
        self.assertIn("ignored_objects_review", calibrated["calibration_rules"])

    def test_calibration_downgrades_near_threshold_dirty_pass_without_changing_quality(self):
        calibrated = calibrate_score(
            {
                "verdict": "PASS",
                "quality_score": 93.804,
                "pass_threshold": 90.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 6.196,
                "sam3_dirty_coverage_pct": 0.0,
                "combined_dirty_coverage_pct": 6.196,
                "dirty_coverage_source": "unet",
                "sam3_predictions_count": 0,
                "reasons": ["good cleanliness"],
            },
            env_key="LOBBY_CORRIDOR",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "PENDING")
        self.assertEqual(calibrated["quality_score"], 93.804)
        self.assertEqual(calibrated["decision_score"], 89.999)
        self.assertTrue(calibrated["review_required"])
        self.assertIn("near_threshold_dirty_review", calibrated["calibration_rules"])

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
        self.assertEqual(calibrated["quality_score"], 43.907)
        self.assertGreaterEqual(calibrated["decision_score"], 50.0)
        self.assertTrue(calibrated["review_required"])
        self.assertIn("unet_only_high_coverage_review", calibrated["calibration_rules"])

    def test_calibration_moves_dense_dirty_pending_to_fail(self):
        calibrated = calibrate_score(
            {
                "verdict": "PENDING",
                "quality_score": 52.704,
                "pass_threshold": 85.0,
                "penalty_detections_count": 0,
                "ignored_detections_count": 0,
                "unet_dirty_coverage_pct": 20.575,
                "sam3_dirty_coverage_pct": 46.743,
                "combined_dirty_coverage_pct": 47.296,
                "dirty_coverage_source": "merged",
                "sam3_predictions_count": 43,
                "reasons": ["coverage high"],
            },
            env_key="RESTROOM",
            pending_lower_bound=50.0,
        )

        self.assertEqual(calibrated["verdict"], "FAIL")
        self.assertEqual(calibrated["quality_score"], 52.704)
        self.assertLess(calibrated["decision_score"], 50.0)
        self.assertFalse(calibrated["review_required"])
        self.assertIn("dense_dirty_regions_fail", calibrated["calibration_rules"])

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
        self.assertEqual(calibrated["quality_score"], 39.502)
        self.assertGreaterEqual(calibrated["decision_score"], 50.0)
        self.assertTrue(calibrated["review_required"])
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
        self.assertEqual(calibrated["quality_score"], 35.0)
        self.assertGreaterEqual(calibrated["decision_score"], 50.0)
        self.assertTrue(calibrated["review_required"])
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
        self.assertEqual(calibrated["quality_score"], 42.0)
        self.assertGreaterEqual(calibrated["decision_score"], 50.0)
        self.assertTrue(calibrated["review_required"])
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
        self.assertEqual(calibrated["quality_score"], 55.0)
        self.assertLess(calibrated["decision_score"], 50.0)
        self.assertFalse(calibrated["review_required"])
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
        self.assertEqual(calibrated["quality_score"], 28.0)
        self.assertGreaterEqual(calibrated["decision_score"], 50.0)
        self.assertTrue(calibrated["review_required"])
        self.assertTrue(calibrated["calibrated"])
        self.assertIn("coverage_only_fail_review", calibrated["calibration_rules"])


if __name__ == "__main__":
    unittest.main()
