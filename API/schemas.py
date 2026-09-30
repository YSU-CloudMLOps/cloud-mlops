"""Request and response schemas for the predictive maintenance API."""

from typing import Literal
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
