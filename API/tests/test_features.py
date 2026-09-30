"""Unit tests for feature engineering service."""

import math
import unittest

from API.schemas import SensorInput
from API.services.feature_service import make_features


class FeatureServiceTests(unittest.TestCase):
    def test_make_features_calculations(self):
        sensor = SensorInput(
            type="M",
            air_temperature_k=300.0,
            process_temperature_k=310.5,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=210.0,
        )
        df = make_features(sensor)

        self.assertEqual(len(df), 1)
        row = df.iloc[0]

        # Verify type encoding
        self.assertEqual(row["type_encoded"], 1)

        # Verify temperature difference
        self.assertAlmostEqual(row["temp_diff_k"], 10.5)

        # Verify power: Torque * Speed * 2 * pi / 60
        expected_power = round(40.0 * 1500.0 * 2 * math.pi / 60, 2)
        self.assertAlmostEqual(row["power_w"], expected_power)

        # Verify strain: Tool wear * Torque
        self.assertAlmostEqual(row["strain_min_nm"], 210.0 * 40.0)

        # Verify tool_wear_critical: >= 200 should be 1
        self.assertEqual(row["tool_wear_critical"], 1)

    def test_tool_wear_critical_threshold(self):
        # Under 200 min
        sensor_normal = SensorInput(
            type="L",
            air_temperature_k=298.0,
            process_temperature_k=308.0,
            rotational_speed_rpm=1400.0,
            torque_nm=35.0,
            tool_wear_min=199.0,
        )
        df_normal = make_features(sensor_normal)
        self.assertEqual(df_normal.iloc[0]["tool_wear_critical"], 0)

        # Exact 200 min
        sensor_critical = SensorInput(
            type="H",
            air_temperature_k=298.0,
            process_temperature_k=308.0,
            rotational_speed_rpm=1400.0,
            torque_nm=35.0,
            tool_wear_min=200.0,
        )
        df_critical = make_features(sensor_critical)
        self.assertEqual(df_critical.iloc[0]["tool_wear_critical"], 1)
        self.assertEqual(df_critical.iloc[0]["type_encoded"], 2)


if __name__ == "__main__":
    unittest.main()
