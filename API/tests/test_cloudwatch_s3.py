"""Unit and integration tests for CloudWatch Retraining/RL logging and S3 export."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from API.api import create_app
from API.config import Settings
from API.schemas import FeedbackInput, S3ExportRequest, SensorInput
from API.services.cloudwatch_service import CloudWatchLogService


class CloudWatchAndS3ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            cw_log_group="/mlops/test/inference",
            cw_log_stream="test-stream-v2",
            cw_logging_enabled=True,
            s3_bucket_name="test-mlops-s3-bucket",
            s3_export_prefix="exports/rl",
            local_log_dir=Path(self.temp_dir.name),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cloudwatch_put_log_events_and_s3_export_with_mock_aws(self):
        mock_logs = MagicMock()
        mock_logs.put_log_events.return_value = {"nextSequenceToken": "seq-001"}
        mock_logs.get_log_events.return_value = {"events": []}
        mock_logs.create_export_task.return_value = {"taskId": "cw-export-task-999"}
        mock_s3 = MagicMock()

        cw_service = CloudWatchLogService(
            settings=self.settings,
            logs_client=mock_logs,
            s3_client=mock_s3,
        )

        app = create_app(settings=self.settings)
        with TestClient(app) as client:
            app.state.cloudwatch_service = cw_service

            # 1. Inference request logs state s_t and action a_t to CloudWatch
            payload = {
                "type": "L",
                "air_temperature_k": 302.5,
                "process_temperature_k": 310.5,
                "rotational_speed_rpm": 1320.0,
                "torque_nm": 52.0,
                "tool_wear_min": 110.0,
            }
            pred_res = client.post("/predict", json=payload, headers={"X-Equipment-ID": "CNC-MILL-07"})
            self.assertEqual(pred_res.status_code, 200)
            self.assertIn("X-Prediction-ID", pred_res.headers)
            pred_id = pred_res.headers["X-Prediction-ID"]
            self.assertEqual(pred_res.json()["machine_failure"], 1)

            # Verify CloudWatch put_log_events was called
            mock_logs.create_log_group.assert_called_once_with(logGroupName="/mlops/test/inference")
            mock_logs.create_log_stream.assert_called_once_with(
                logGroupName="/mlops/test/inference",
                logStreamName="test-stream-v2",
            )
            self.assertEqual(mock_logs.put_log_events.call_count, 1)

            # 2. Feedback request logs RL reward R_t, ground-truth label, and next state S_{t+1}
            fb_payload = {
                "prediction_id": pred_id,
                "equipment_id": "CNC-MILL-07",
                "operator_action": "COOLING_REPAIR",
                "actual_failure": 1,
                "actual_failure_mode": "HDF",
                "downtime_minutes": 10.0,
                "maintenance_cost_usd": 100.0,
                "next_state_sensors": {
                    "type": "L",
                    "air_temperature_k": 299.8,
                    "process_temperature_k": 310.0,
                    "rotational_speed_rpm": 1500.0,
                    "torque_nm": 40.0,
                    "tool_wear_min": 110.0,
                },
                "notes": "HDF 방열 불량 즉시 조치 완료",
            }
            fb_res = client.post("/feedback", json=fb_payload)
            self.assertEqual(fb_res.status_code, 200)
            fb_data = fb_res.json()
            self.assertEqual(fb_data["outcome_type"], "TP")
            # Base TP (+100.0) - (0.5 * 10 + 0.05 * 100 = 10.0) = +90.0
            self.assertAlmostEqual(fb_data["computed_reward"], 90.0)
            self.assertTrue(fb_data["logged_to_cloudwatch"])
            self.assertEqual(mock_logs.put_log_events.call_count, 2)

            # 3. Inspect recent logs
            recent_res = client.get("/logs/recent?limit=10")
            self.assertEqual(recent_res.status_code, 200)
            recent_data = recent_res.json()
            self.assertEqual(recent_data["count"], 2)
            self.assertEqual(recent_data["records"][0]["event_type"], "INFERENCE_PREDICTION")
            self.assertEqual(recent_data["records"][1]["event_type"], "FEEDBACK_RL_TRANSITION")
            self.assertIsNotNone(recent_data["records"][1]["rl_transition"]["next_state_s_t1"])

            # 4. Export Log Stream to S3 via CloudWatch Export Task
            export_res = client.post(
                "/logs/export-s3",
                json={"export_mode": "cloudwatch_export_task"},
            )
            self.assertEqual(export_res.status_code, 200)
            export_data = export_res.json()
            self.assertEqual(export_data["status"], "completed")
            self.assertEqual(export_data["export_mode"], "cloudwatch_export_task")
            self.assertEqual(export_data["task_id"], "cw-export-task-999")
            self.assertEqual(export_data["exported_records"], 2)
            self.assertTrue(export_data["s3_uri"].startswith("s3://test-mlops-s3-bucket/exports/rl/"))

            # 5. Export Log Stream to S3 via Direct S3 Object Upload
            direct_res = client.post(
                "/logs/export-s3",
                json={"export_mode": "s3_direct_upload"},
            )
            self.assertEqual(direct_res.status_code, 200)
            self.assertEqual(direct_res.json()["export_mode"], "s3_direct_upload")
            mock_s3.put_object.assert_called_once()

    def test_rl_reward_outcomes_fp_fn_tn(self):
        cw_service = CloudWatchLogService(
            settings=Settings(
                cw_logging_enabled=False,
                local_log_dir=Path(self.temp_dir.name),
            )
        )
        # False Negative (predicted 0, actual 1): catastrophic penalty (-200)
        fn_res = cw_service.log_feedback_event(
            FeedbackInput(
                prediction_id="pred-fn-test",
                predicted_failure=0,
                operator_action="EMERGENCY_STOP",
                actual_failure=1,
                actual_failure_mode="PWF",
                downtime_minutes=60.0,
                maintenance_cost_usd=500.0,
            )
        )
        self.assertEqual(fn_res.outcome_type, "FN")
        self.assertLess(fn_res.computed_reward, -200.0)

        # False Positive (predicted 1, actual 0): false alarm penalty (-30)
        fp_res = cw_service.log_feedback_event(
            FeedbackInput(
                prediction_id="pred-fp-test",
                predicted_failure=1,
                operator_action="INSPECT",
                actual_failure=0,
                actual_failure_mode="NONE",
                downtime_minutes=10.0,
                maintenance_cost_usd=20.0,
            )
        )
        self.assertEqual(fp_res.outcome_type, "FP")
        self.assertAlmostEqual(fp_res.computed_reward, -36.0)


if __name__ == "__main__":
    unittest.main()
