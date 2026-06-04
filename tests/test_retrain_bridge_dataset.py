(from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_retrain_bridge_dataset import (
    annotation_to_mask,
    annotation_to_yolo_lines,
    normalize_label,
)


class RetrainBridgeDatasetTests(unittest.TestCase):
    def test_dirty_area_is_normalized_for_bridge_export(self) -> None:
        self.assertEqual(normalize_label("dirty_area"), "dirty_area")

    def test_dirty_area_exports_to_yolo_dirty_class(self) -> None:
        labels = [
            {
                "label": "dirty_area",
                "shapeType": "polygon",
                "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
            }
        ]

        lines = annotation_to_yolo_lines(labels, width=100, height=100)

        self.assertEqual(lines, ["0 0.200000 0.300000 0.200000 0.200000"])

    def test_dirty_area_polygon_rasterizes_to_mask_class_one(self) -> None:
        labels = [
            {
                "label": "dirty_area",
                "shapeType": "polygon",
                "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
            }
        ]

        mask = annotation_to_mask(labels, width=50, height=50)

        self.assertGreater(int(np.count_nonzero(mask == 1)), 0)
        self.assertEqual(int(np.count_nonzero(mask == 2)), 0)


if __name__ == "__main__":
    unittest.main()
