"""Request and response schemas for the predictive maintenance API."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class SensorInput(BaseModel):
    """Raw sensor inputs collected from milling machine."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "type": "L",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 100.0,
            }
        },
    )

    type: Literal["L", "M", "H"] = Field(
        description="Product quality variant: L (Low/Standard), M (Medium), H (High)"
    )
    air_temperature_k: float = Field(
        gt=0,
        description="Air/ambient temperature in Kelvin [K]",
    )
    process_temperature_k: float = Field(
        gt=0,
        description="Process operating temperature in Kelvin [K]",
    )
    rotational_speed_rpm: float = Field(
        gt=0,
        description="Spindle rotational speed in revolutions per minute [rpm]",
    )
    torque_nm: float = Field(
        ge=0,
        description="Spindle motor torque in Newton-meters [Nm]",
    )
    tool_wear_min: float = Field(
        ge=0,
        description="Cumulative tool wear duration in minutes [min]",
    )


class Prediction(BaseModel):
    """Prediction result indicating machine failure probability and binary status."""

    machine_failure: Literal[0, 1] = Field(
        description="Predicted failure state (0: Normal, 1: Machine Failure)"
    )
    failure_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Model estimated probability of machine failure [0.0 ~ 1.0]",
    )


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = Field(default="ok", description="Service health status")
    model_loaded: bool = Field(default=True, description="Whether ML model is loaded and ready")
    version: str = Field(default="1.0.0", description="API version")


class FeedbackInput(BaseModel):
    """Field feedback & ground-truth outcome for Retraining and Reinforcement Learning (MDP Reward/Transition)."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "prediction_id": "pred-20261028-0001",
                "equipment_id": "CNC-MILL-01",
                "predicted_failure": 1,
                "operator_action": "REPLACE_TOOL",
                "actual_failure": 1,
                "actual_failure_mode": "OSF",
                "downtime_minutes": 15.0,
                "maintenance_cost_usd": 120.0,
                "next_state_sensors": {
                    "type": "L",
                    "air_temperature_k": 299.5,
                    "process_temperature_k": 309.2,
                    "rotational_speed_rpm": 1520.0,
                    "torque_nm": 38.5,
                    "tool_wear_min": 0.0,
                },
                "notes": "과부하 경보(OSF) 확인 후 공구 즉시 교체 완료, 실제 절삭날 미세 균열 확인됨.",
            }
        },
    )

    prediction_id: str = Field(
        min_length=1,
        description="Unique prediction ID returned in X-Prediction-ID header or inference log",
    )
    equipment_id: str = Field(
        default="CNC-MILL-01",
        description="Target CNC milling machine identifier",
    )
    predicted_failure: Optional[Literal[0, 1]] = Field(
        default=None,
        description="Predicted label at inference time (optional if linked via prediction_id)",
    )
    operator_action: Literal[
        "CONTINUE",
        "INSPECT",
        "REPLACE_TOOL",
        "COOLING_REPAIR",
        "POWER_CHECK",
        "EMERGENCY_STOP",
    ] = Field(
        description="Actual maintenance action executed by field engineer / shift supervisor"
    )
    actual_failure: Literal[0, 1] = Field(
        description="Ground-truth machine failure label observed in the field (0: Normal, 1: Failure)"
    )
    actual_failure_mode: Literal["NONE", "TWF", "HDF", "PWF", "OSF", "RNF"] = Field(
        default="NONE",
        description="Specific physical failure root cause verified during inspection",
    )
    downtime_minutes: float = Field(
        default=0.0,
        ge=0.0,
        description="Unplanned or maintenance line downtime in minutes",
    )
    maintenance_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Direct maintenance and replacement cost in USD",
    )
    next_state_sensors: Optional[SensorInput] = Field(
        default=None,
        description="Post-action sensor state (S_{t+1}) for Markov Decision Process transition logging",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Operator inspection notes",
    )


class FeedbackResponse(BaseModel):
    """Response returned after logging retraining/RL feedback event."""

    status: str = Field(default="logged", description="Feedback logging status")
    event_id: str = Field(description="Unique feedback log event ID")
    prediction_id: str = Field(description="Associated inference prediction ID")
    outcome_type: Literal["TP", "TN", "FP", "FN"] = Field(
        description="Confusion matrix outcome category"
    )
    computed_reward: float = Field(
        description="Calculated RL scalar reward R_t based on operational cost-benefit matrix"
    )
    logged_to_cloudwatch: bool = Field(
        description="Whether the event was pushed to AWS CloudWatch Logs"
    )
    log_group: str = Field(description="Target CloudWatch Log Group name")
    log_stream: str = Field(description="Target CloudWatch Log Stream name")


class S3ExportRequest(BaseModel):
    """Request payload for exporting a CloudWatch Log Stream to Amazon S3."""

    log_group_name: Optional[str] = Field(
        default=None,
        description="CloudWatch Log Group name (defaults to configured CW_LOG_GROUP)",
    )
    log_stream_name: Optional[str] = Field(
        default=None,
        description="CloudWatch Log Stream name to export (defaults to configured CW_LOG_STREAM)",
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="Target Amazon S3 bucket name (defaults to configured S3_LOG_BUCKET)",
    )
    s3_prefix: Optional[str] = Field(
        default=None,
        description="Target S3 object key prefix (defaults to configured S3_EXPORT_PREFIX)",
    )
    from_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start timestamp in epoch milliseconds (defaults to 24 hours ago)",
    )
    to_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="End timestamp in epoch milliseconds (defaults to current time)",
    )
    export_mode: Literal[
        "auto",
        "cloudwatch_export_task",
        "s3_direct_upload",
        "local_archive",
    ] = Field(
        default="auto",
        description="Export strategy: 'auto' tries AWS CloudWatch/S3 first and archives locally",
    )


class S3ExportResponse(BaseModel):
    """Result metadata after exporting a Log Stream to S3."""

    status: str = Field(description="Export execution status")
    export_mode: str = Field(description="Executed export mechanism")
    task_id: str = Field(description="CloudWatch Export Task ID or S3 export batch ID")
    log_group: str = Field(description="Source CloudWatch Log Group")
    log_stream: str = Field(description="Source CloudWatch Log Stream")
    s3_uri: str = Field(description="Destination Amazon S3 URI (s3://bucket/key)")
    local_export_path: Optional[str] = Field(
        default=None,
        description="Local exported snapshot path for offline verification",
    )
    exported_records: int = Field(
        ge=0,
        description="Number of structured log events exported",
    )
    exported_at: str = Field(description="ISO-8601 UTC export timestamp")
    sample_record: Optional[dict[str, Any]] = Field(
        default=None,
        description="Sample exported log record for schema verification",
    )
