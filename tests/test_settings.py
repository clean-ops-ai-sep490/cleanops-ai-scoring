from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings


class SettingsTests(unittest.TestCase):
    def test_roboflow_defaults_exclude_stained_floor(self):
        self.assertEqual(settings.roboflow_classes, ("Garbage", "Trash", "Debris", "Stain", "Wet_Floor"))
        self.assertNotIn("Stained_Floor", settings.roboflow_classes)
        self.assertNotIn("Stained_Floor", settings.roboflow_dirty_labels)


if __name__ == "__main__":
    unittest.main()
