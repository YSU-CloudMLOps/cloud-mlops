"""Serve the baseline LightGBM model from six raw input features."""

from contextlib import asynccontextmanager
from math import pi
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parent / "models" / "lightgbm_model.txt"
THRESHOLD = 0.5


class SensorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)


class Prediction(BaseModel):
    machine_failure: Literal[0, 1]
    failure_probability: float


def make_features(sensor: SensorInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "type_encoded": {"L": 0, "M": 1, "H": 2}[sensor.type],
        "air_temperature_k": sensor.air_temperature_k,
        "process_temperature_k": sensor.process_temperature_k,
        "rotational_speed_rpm": sensor.rotational_speed_rpm,
        "torque_nm": sensor.torque_nm,
        "tool_wear_min": sensor.tool_wear_min,
        "temp_diff_k": round(sensor.process_temperature_k - sensor.air_temperature_k, 2),
        "power_w": round(sensor.torque_nm * sensor.rotational_speed_rpm * 2 * pi / 60, 2),
        "strain_min_nm": round(sensor.tool_wear_min * sensor.torque_nm, 2),
        "tool_wear_critical": int(sensor.tool_wear_min >= 200),
    }])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Normalize Git's Windows CRLF checkout back to the native model's LF offsets.
    app.state.model = lgb.Booster(model_str=MODEL_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Predictive Maintenance API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(sensor: SensorInput, request: Request):
    features = make_features(sensor)
    model = request.app.state.model
    probability = float(model.predict(features[model.feature_name()], num_threads=1)[0])
    return Prediction(
        machine_failure=int(probability >= THRESHOLD),
        failure_probability=probability,
    )
