"""Unit tests for ModelService."""

import unittest
from pathlib import Path

from API.config import DEFAULT_MODEL_PATH
from API.schemas import Prediction, SensorInput
from API.services.model_service import ModelService


class ModelServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ModelService(model_path=DEFAULT_MODEL_PATH, threshold=0.5)

    def test_service_initialization(self):
        self.assertTrue(self.service.is_loaded)
        self.assertGreater(len(self.service.feature_names), 0)
        self.assertIn("type_encoded", self.service.feature_names)
        self.assertIn("power_w", self.service.feature_names)

    def test_predict_sensor(self):
        sensor = SensorInput(
            type="L",
            air_temperature_k=300.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=50.0,
        )
        prediction = self.service.predict_sensor(sensor)

        self.assertIsInstance(prediction, Prediction)
        self.assertIn(prediction.machine_failure, (0, 1))
        self.assertGreaterEqual(prediction.failure_probability, 0.0)
        self.assertLessEqual(prediction.failure_probability, 1.0)

    def test_nonexistent_model_raises_filenotfound(self):
        invalid_path = Path("/nonexistent/path/model.txt")
        with self.assertRaises(FileNotFoundError):
            ModelService(model_path=invalid_path)

    def test_custom_threshold(self):
        sensor = SensorInput(
            type="L",
            air_temperature_k=300.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=50.0,
        )
        # Threshold 0.0: any non-negative probability triggers failure
        service_zero = ModelService(model_path=DEFAULT_MODEL_PATH, threshold=0.0)
        pred_zero = service_zero.predict_sensor(sensor)
        self.assertEqual(pred_zero.machine_failure, 1)

        # Threshold 1.0: machine failure requires 100% probability
        service_one = ModelService(model_path=DEFAULT_MODEL_PATH, threshold=1.0)
        pred_one = service_one.predict_sensor(sensor)
        self.assertEqual(pred_one.machine_failure, 0)


if __name__ == "__main__":
    unittest.main()
