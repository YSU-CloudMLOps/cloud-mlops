"""Application configuration and environment settings."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lightgbm_model.txt"
DEFAULT_THRESHOLD = 0.5
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


class Settings(BaseModel):
    """Configuration settings for the predictive maintenance inference API."""

    model_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("PREDICT_MODEL_PATH", os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        )
    )
    threshold: float = Field(
        default_factory=lambda: float(
            os.getenv("PREDICT_THRESHOLD", os.getenv("THRESHOLD", str(DEFAULT_THRESHOLD)))
        )
    )
    app_title: str = "Predictive Maintenance API"
    app_version: str = "1.1.0"
    app_description: str = (
        "Production-ready FastAPI service for AI4I 2020 Predictive Maintenance "
        "using LightGBM inference model with CloudWatch Retraining/RL logging and S3 export."
    )
    api_prefix: str = ""
    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # CloudWatch & S3 MLOps Logging Configuration
    aws_region: str = Field(
        default_factory=lambda: os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))
    )
    cw_log_group: str = Field(
        default_factory=lambda: os.getenv("CW_LOG_GROUP", "/mlops/predictive-maintenance/inference")
    )
    cw_log_stream: str = Field(
        default_factory=lambda: os.getenv("CW_LOG_STREAM", "api-inference-stream-v2")
    )
    cw_logging_enabled: bool = Field(
        default_factory=lambda: os.getenv("CW_LOGGING_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    s3_bucket_name: str = Field(
        default_factory=lambda: os.getenv("S3_LOG_BUCKET", "cloud-mlops-pdm-logs")
    )
    s3_export_prefix: str = Field(
        default_factory=lambda: os.getenv("S3_EXPORT_PREFIX", "cloudwatch-exports/retraining-rl")
    )
    local_log_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("LOCAL_LOG_DIR", str(DEFAULT_LOG_DIR)))
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
