"""Unit tests for configuration and settings."""

import os
import unittest
from pathlib import Path

from API.config import Settings


class ConfigTests(unittest.TestCase):
    def test_default_settings(self):
        settings = Settings()
        self.assertEqual(settings.threshold, 0.5)
        self.assertEqual(settings.app_title, "Predictive Maintenance API")
        self.assertTrue(settings.model_path.name.endswith(".txt"))

    def test_env_override_settings(self):
        os.environ["PREDICT_THRESHOLD"] = "0.75"
        os.environ["API_PORT"] = "9000"
        try:
            settings = Settings()
            self.assertEqual(settings.threshold, 0.75)
            self.assertEqual(settings.port, 9000)
        finally:
            del os.environ["PREDICT_THRESHOLD"]
            del os.environ["API_PORT"]


if __name__ == "__main__":
    unittest.main()
