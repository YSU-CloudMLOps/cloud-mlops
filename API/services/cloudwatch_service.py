"""CloudWatch logging and S3 export service for ML retraining and reinforcement learning."""

import json
import logging
import math
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
import pandas as pd

from API.config import Settings, get_settings
from API.schemas import (
    FeedbackInput,
    FeedbackResponse,
    Prediction,
    S3ExportRequest,
    S3ExportResponse,
    SensorInput,
)
from API.services.feature_service import make_features

logger = logging.getLogger("mlops.cloudwatch")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Base RL reward matrix for industrial predictive maintenance
RL_BASE_REWARDS: dict[str, float] = {
    "TP": 100.0,   # True Positive: Failure accurately predicted & catastrophic breakdown prevented
    "TN": 10.0,    # True Negative: Normal operation accurately maintained without interruption
    "FP": -30.0,   # False Positive: False alarm causing unnecessary line stop / inspection cost
    "FN": -200.0,  # False Negative: Missed failure causing unplanned machine breakdown & scrap
}


def compute_binary_entropy(prob: float) -> float:
    """Compute binary Shannon entropy H(p) in bits [0.0, 1.0] for Active Learning uncertainty."""
    p = max(min(float(prob), 1.0 - 1e-12), 1e-12)
    return round(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)), 6)


def recommend_maintenance_action(probability: float, threshold: float = 0.5) -> tuple[str, str]:
    """Map failure probability to discrete RL action space and risk tier."""
    if probability >= threshold:
        return "IMMEDIATE_SHUTDOWN_MAINTENANCE", "CRITICAL"
    if probability >= 0.25:
        return "PREVENTIVE_INSPECTION", "WARNING"
    return "CONTINUE_NORMAL_OPERATION", "SAFE"


class CloudWatchLogService:
    """Manages structured Retraining & Reinforcement Learning logs in CloudWatch and S3 exports."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        logs_client: Optional[Any] = None,
        s3_client: Optional[Any] = None,
    ):
        self.settings = settings or get_settings()
        self.aws_region = self.settings.aws_region
        self.log_group = self.settings.cw_log_group
        self.log_stream = self.settings.cw_log_stream
        self.cw_enabled = self.settings.cw_logging_enabled
        self.s3_bucket = self.settings.s3_bucket_name
        self.s3_prefix = self.settings.s3_export_prefix
        self.local_log_dir = Path(self.settings.local_log_dir)

        self._logs_client = logs_client
        self._s3_client = s3_client
        self._sequence_token: Optional[str] = None
        self._stream_initialized = False

        # In-memory cache of recent inference events for fast feedback join & inspection
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=500)
        self._predictions_by_id: dict[str, dict[str, Any]] = {}

        # Ensure local log directories exist
        self.local_log_dir.mkdir(parents=True, exist_ok=True)
        (self.local_log_dir / "s3_exports").mkdir(parents=True, exist_ok=True)

    @property
    def local_stream_path(self) -> Path:
        """Path to local JSONL mirror of the CloudWatch Log Stream."""
        safe_stream = self.log_stream.replace("/", "_")
        return self.local_log_dir / f"{safe_stream}.jsonl"

    def _get_logs_client(self) -> Any:
        if self._logs_client is None:
            self._logs_client = boto3.client("logs", region_name=self.aws_region)
        return self._logs_client

    def _get_s3_client(self) -> Any:
        if self._s3_client is None:
            self._s3_client = boto3.client("s3", region_name=self.aws_region)
        return self._s3_client

    def _ensure_cloudwatch_stream(self, client: Any) -> None:
        """Create CloudWatch Log Group and Log Stream if they do not already exist."""
        if self._stream_initialized:
            return
        try:
            client.create_log_group(logGroupName=self.log_group)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "ResourceAlreadyExistsException":
                raise

        try:
            client.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "ResourceAlreadyExistsException":
                raise

        self._stream_initialized = True

    def _put_cloudwatch_event(self, payload: dict[str, Any]) -> bool:
        """Send a single structured JSON event to AWS CloudWatch Logs if enabled."""
        if not self.cw_enabled and self._logs_client is None:
            return False

        try:
            client = self._get_logs_client()
            self._ensure_cloudwatch_stream(client)
            event_body = {
                "timestamp": int(payload.get("epoch_ms", int(time.time() * 1000))),
                "message": json.dumps(payload, ensure_ascii=False),
            }
            kwargs: dict[str, Any] = {
                "logGroupName": self.log_group,
                "logStreamName": self.log_stream,
                "logEvents": [event_body],
            }
            if self._sequence_token:
                kwargs["sequenceToken"] = self._sequence_token

            response = client.put_log_events(**kwargs)
            self._sequence_token = response.get("nextSequenceToken")
            return True
        except (NoCredentialsError, BotoCoreError, ClientError) as exc:
            logger.debug(f"CloudWatch put_log_events skipped/fallback: {exc}")
            return False

    def _persist_event(self, payload: dict[str, Any]) -> bool:
        """Emit structured log to stdout, local JSONL mirror, and AWS CloudWatch Logs."""
        sent_to_cw = self._put_cloudwatch_event(payload)
        payload["cloudwatch_target"]["delivered_to_aws"] = sent_to_cw

        serialized = json.dumps(payload, ensure_ascii=False)
        # 1. Emit structured JSON line to stdout (collected by Docker awslogs / CloudWatch agent)
        logger.info(serialized)

        # 2. Append to local JSONL log stream mirror
        with self.local_stream_path.open("a", encoding="utf-8") as fh:
            fh.write(serialized + "\n")

        # 3. Save in memory ring buffer
        self._recent_events.append(payload)
        return sent_to_cw

    def log_inference_event(
        self,
        sensor: SensorInput,
        features_df: pd.DataFrame,
        prediction: Prediction,
        threshold: float,
        latency_ms: float,
        prediction_id: Optional[str] = None,
        equipment_id: str = "CNC-MILL-01",
    ) -> dict[str, Any]:
        """Build and record a structured Retraining + RL Inference event."""
        now_utc = datetime.now(timezone.utc)
        epoch_ms = int(now_utc.timestamp() * 1000)
        pred_id = prediction_id or f"pred-{now_utc.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        event_id = f"evt-{uuid.uuid4().hex[:12]}"

        feature_dict = {
            k: (int(v) if isinstance(v, (bool, int)) or k in (
                "type_encoded",
                "tool_wear_critical",
                "hdf_risk",
                "pwf_risk",
                "osf_risk",
                "twf_zone",
                "physical_risk_sum",
            ) else round(float(v), 6))
            for k, v in features_df.iloc[0].to_dict().items()
        }

        prob = float(prediction.failure_probability)
        pred_label = int(prediction.machine_failure)
        recommended_action, risk_tier = recommend_maintenance_action(prob, threshold)
        propensity = round(prob if pred_label == 1 else (1.0 - prob), 6)
        entropy = compute_binary_entropy(prob)

        log_entry: dict[str, Any] = {
            "schema_version": "1.0",
            "event_type": "INFERENCE_PREDICTION",
            "event_id": event_id,
            "prediction_id": pred_id,
            "equipment_id": equipment_id,
            "timestamp": now_utc.isoformat(),
            "epoch_ms": epoch_ms,
            "cloudwatch_target": {
                "log_group": self.log_group,
                "log_stream": self.log_stream,
                "delivered_to_aws": False,
            },
            "model_metadata": {
                "model_name": "LightGBM-v2-23feat",
                "model_version": "2.0.0",
                "feature_count": len(feature_dict),
                "decision_threshold": threshold,
            },
            "rl_trajectory": {
                "state_s_t": {
                    "raw_sensors": sensor.model_dump(),
                    "engineered_features": feature_dict,
                    "physical_risk_flags": {
                        "hdf_risk": feature_dict.get("hdf_risk", 0),
                        "pwf_risk": feature_dict.get("pwf_risk", 0),
                        "osf_risk": feature_dict.get("osf_risk", 0),
                        "twf_zone": feature_dict.get("twf_zone", 0),
                        "physical_risk_sum": feature_dict.get("physical_risk_sum", 0),
                    },
                },
                "action_a_t": {
                    "predicted_failure": pred_label,
                    "failure_probability": round(prob, 6),
                    "recommended_action": recommended_action,
                    "risk_tier": risk_tier,
                    "policy_propensity": propensity,
                    "uncertainty_entropy": entropy,
                },
            },
            "retraining_metadata": {
                "active_learning_priority": "HIGH" if 0.20 <= prob <= 0.80 else "NORMAL",
                "is_boundary_sample": bool(
                    feature_dict.get("physical_risk_sum", 0) > 0 or (0.15 <= prob <= 0.85)
                ),
                "ground_truth_status": "PENDING_FEEDBACK",
            },
            "telemetry": {
                "inference_latency_ms": round(latency_ms, 3),
            },
        }

        self._predictions_by_id[pred_id] = log_entry
        self._persist_event(log_entry)
        return log_entry

    def log_feedback_event(self, feedback: FeedbackInput) -> FeedbackResponse:
        """Record ground-truth outcome and RL reward transition (S_t, A_t, R_t, S_{t+1})."""
        now_utc = datetime.now(timezone.utc)
        epoch_ms = int(now_utc.timestamp() * 1000)
        event_id = f"fb-{uuid.uuid4().hex[:12]}"

        prior = self._predictions_by_id.get(feedback.prediction_id)
        if prior is not None:
            pred_label = int(prior["rl_trajectory"]["action_a_t"]["predicted_failure"])
            pred_prob = float(prior["rl_trajectory"]["action_a_t"]["failure_probability"])
            prior_state = prior["rl_trajectory"]["state_s_t"]
            recommended_action = prior["rl_trajectory"]["action_a_t"]["recommended_action"]
        else:
            pred_label = int(feedback.predicted_failure) if feedback.predicted_failure is not None else 0
            pred_prob = float(pred_label)
            prior_state = None
            recommended_action = "UNKNOWN"

        actual = int(feedback.actual_failure)
        if pred_label == 1 and actual == 1:
            outcome_type = "TP"
        elif pred_label == 0 and actual == 0:
            outcome_type = "TN"
        elif pred_label == 1 and actual == 0:
            outcome_type = "FP"
        else:
            outcome_type = "FN"

        base_reward = RL_BASE_REWARDS[outcome_type]
        operational_penalty = round(
            0.5 * feedback.downtime_minutes + 0.05 * feedback.maintenance_cost_usd, 2
        )
        computed_reward = round(base_reward - operational_penalty, 2)

        next_state_dict: Optional[dict[str, Any]] = None
        if feedback.next_state_sensors is not None:
            next_df = make_features(feedback.next_state_sensors)
            next_state_dict = {
                "raw_sensors": feedback.next_state_sensors.model_dump(),
                "engineered_features": {
                    k: round(float(v), 6) for k, v in next_df.iloc[0].to_dict().items()
                },
            }

        feedback_entry: dict[str, Any] = {
            "schema_version": "1.0",
            "event_type": "FEEDBACK_RL_TRANSITION",
            "event_id": event_id,
            "prediction_id": feedback.prediction_id,
            "equipment_id": feedback.equipment_id,
            "timestamp": now_utc.isoformat(),
            "epoch_ms": epoch_ms,
            "cloudwatch_target": {
                "log_group": self.log_group,
                "log_stream": self.log_stream,
                "delivered_to_aws": False,
            },
            "rl_transition": {
                "state_s_t": prior_state,
                "model_action_a_t": {
                    "predicted_failure": pred_label,
                    "failure_probability": pred_prob,
                    "recommended_action": recommended_action,
                },
                "operator_action": feedback.operator_action,
                "reward_r_t": {
                    "computed_reward": computed_reward,
                    "base_reward": base_reward,
                    "operational_penalty": operational_penalty,
                    "downtime_minutes": feedback.downtime_minutes,
                    "maintenance_cost_usd": feedback.maintenance_cost_usd,
                },
                "next_state_s_t1": next_state_dict,
                "terminal_done": bool(actual == 1 or feedback.operator_action in ("REPLACE_TOOL", "EMERGENCY_STOP")),
            },
            "retraining_label": {
                "actual_failure": actual,
                "actual_failure_mode": feedback.actual_failure_mode,
                "outcome_type": outcome_type,
                "is_misclassified": outcome_type in ("FP", "FN"),
                "sample_weight_multiplier": 5.0 if outcome_type in ("FP", "FN") else 1.0,
                "notes": feedback.notes,
            },
        }

        delivered = self._persist_event(feedback_entry)
        return FeedbackResponse(
            status="logged",
            event_id=event_id,
            prediction_id=feedback.prediction_id,
            outcome_type=outcome_type,
            computed_reward=computed_reward,
            logged_to_cloudwatch=delivered,
            log_group=self.log_group,
            log_stream=self.log_stream,
        )

    def get_recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent structured log records from memory or local stream file."""
        if self._recent_events:
            return list(self._recent_events)[-limit:]
        if not self.local_stream_path.exists():
            return []
        lines = self.local_stream_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        return records[-limit:]

    def export_log_stream_to_s3(
        self,
        request: Optional[S3ExportRequest] = None,
    ) -> S3ExportResponse:
        """Export a CloudWatch Log Stream to Amazon S3 (and local S3 export snapshot)."""
        req = request or S3ExportRequest()
        log_group = req.log_group_name or self.log_group
        log_stream = req.log_stream_name or self.log_stream
        s3_bucket = req.s3_bucket or self.s3_bucket
        s3_prefix = (req.s3_prefix or self.s3_prefix).strip("/")

        now_utc = datetime.now(timezone.utc)
        to_time_ms = req.to_time_ms or int(now_utc.timestamp() * 1000)
        from_time_ms = req.from_time_ms or (to_time_ms - 24 * 60 * 60 * 1000)
        date_partition = now_utc.strftime("year=%Y/month=%m/day=%d")
        batch_stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
        safe_stream = log_stream.replace("/", "_")
        s3_object_key = f"{s3_prefix}/{date_partition}/{safe_stream}-{batch_stamp}.jsonl"
        s3_uri = f"s3://{s3_bucket}/{s3_object_key}"

        records = self.get_recent_logs(limit=1000)

        # Try fetching events from CloudWatch Logs if client is configured
        if (self.cw_enabled or self._logs_client is not None) and req.export_mode != "local_archive":
            try:
                logs_client = self._get_logs_client()
                cw_resp = logs_client.get_log_events(
                    logGroupName=log_group,
                    logStreamName=log_stream,
                    startTime=from_time_ms,
                    endTime=to_time_ms,
                    startFromHead=True,
                )
                cw_events = cw_resp.get("events", [])
                if cw_events:
                    records = [json.loads(ev["message"]) for ev in cw_events if ev.get("message")]
            except (NoCredentialsError, BotoCoreError, ClientError, json.JSONDecodeError) as exc:
                logger.debug(f"CloudWatch get_log_events fallback to local stream buffer: {exc}")

        # Always save a concrete local S3 export bundle in logs/s3_exports/
        export_dir = self.local_log_dir / "s3_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        local_export_file = export_dir / f"{safe_stream}-{batch_stamp}.jsonl"
        jsonl_body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else "")
        local_export_file.write_text(jsonl_body, encoding="utf-8")

        task_id = f"export-{uuid.uuid4().hex[:12]}"
        executed_mode = "local_s3_mirror"

        # Mode 1: CloudWatch native create_export_task
        if req.export_mode in ("cloudwatch_export_task", "auto") and (
            self.cw_enabled or self._logs_client is not None
        ):
            try:
                logs_client = self._get_logs_client()
                cw_task = logs_client.create_export_task(
                    taskName=f"mlops-rl-export-{batch_stamp}",
                    logGroupName=log_group,
                    logStreamNamePrefix=log_stream,
                    fromTime=from_time_ms,
                    to=to_time_ms,
                    destination=s3_bucket,
                    destinationPrefix=f"{s3_prefix}/{date_partition}/{safe_stream}",
                )
                task_id = cw_task.get("taskId", task_id)
                executed_mode = "cloudwatch_export_task"
            except (NoCredentialsError, BotoCoreError, ClientError) as exc:
                logger.debug(f"create_export_task fallback: {exc}")

        # Mode 2: Direct S3 put_object upload
        if executed_mode != "cloudwatch_export_task" and req.export_mode in ("s3_direct_upload", "auto") and (
            self.cw_enabled or self._s3_client is not None
        ):
            try:
                s3_client = self._get_s3_client()
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=s3_object_key,
                    Body=jsonl_body.encode("utf-8"),
                    ContentType="application/x-ndjson",
                    Metadata={
                        "log-group": log_group,
                        "log-stream": log_stream,
                        "record-count": str(len(records)),
                    },
                )
                executed_mode = "s3_direct_upload"
            except (NoCredentialsError, BotoCoreError, ClientError) as exc:
                logger.debug(f"S3 put_object fallback: {exc}")

        return S3ExportResponse(
            status="completed",
            export_mode=executed_mode,
            task_id=task_id,
            log_group=log_group,
            log_stream=log_stream,
            s3_uri=s3_uri,
            local_export_path=str(local_export_file),
            exported_records=len(records),
            exported_at=now_utc.isoformat(),
            sample_record=records[-1] if records else None,
        )
