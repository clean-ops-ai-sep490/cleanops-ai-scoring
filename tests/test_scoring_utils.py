from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.scoring_utils import score_image, summarize_penalty_detections


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

    def test_llm_failure_fallback_uses_label_filter_not_total_detections(self):
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


if __name__ == "__main__":
    unittest.main()
