import unittest
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi.testclient import TestClient

from api import MODEL_PATH, app


class PredictionTests(unittest.TestCase):
    def test_raw_inputs_match_preprocessed_model_predictions(self):
        root = Path(__file__).resolve().parents[1]
        raw = pd.read_csv(root / "dataset" / "ai4i2020.csv")
        prepared = pd.read_csv(root / "dataset" / "ai4i2020_preprocessed.csv")
        model = lgb.Booster(model_str=MODEL_PATH.read_text(encoding="utf-8"))
        indices = [0, int(raw.index[raw["Machine failure"] == 1][0]), 9999]
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            schema = client.get("/openapi.json").json()
            self.assertEqual(set(schema["components"]["schemas"]["Prediction"]["properties"]), {"machine_failure", "failure_probability"})
            for index in indices:
                row = raw.iloc[index]
                payload = {
                    "type": row["Type"],
                    "air_temperature_k": float(row["Air temperature [K]"]),
                    "process_temperature_k": float(row["Process temperature [K]"]),
                    "rotational_speed_rpm": float(row["Rotational speed [rpm]"]),
                    "torque_nm": float(row["Torque [Nm]"]),
                    "tool_wear_min": float(row["Tool wear [min]"]),
                }
                response = client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()), {"machine_failure", "failure_probability"})
                expected = float(model.predict(prepared.iloc[[index]][model.feature_name()], num_threads=1)[0])
                self.assertAlmostEqual(response.json()["failure_probability"], expected)
                self.assertEqual(response.json()["machine_failure"], int(expected >= 0.5))
                for invalid in [{**payload, "type": "X"}, {**payload, "rotational_speed_rpm": 0}, {**payload, "tool_wear_min": -1}, {**payload, "machine_failure": 1}, {}]:
                    self.assertEqual(client.post("/predict", json=invalid).status_code, 422)


if __name__ == "__main__":
    unittest.main()
