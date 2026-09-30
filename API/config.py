"""Application configuration and environment settings."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lightgbm_model.txt"
DEFAULT_THRESHOLD = 0.5


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
    app_version: str = "1.0.0"
    app_description: str = (
        "Production-ready FastAPI service for AI4I 2020 Predictive Maintenance "
        "using LightGBM inference model."
    )
    api_prefix: str = ""
    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
