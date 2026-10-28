"""Prediction endpoints for machine failure inference with CloudWatch RL/Retraining logging."""

import time
from fastapi import APIRouter, Request, Response

from API.schemas import Prediction, SensorInput
from API.services.cloudwatch_service import CloudWatchLogService
from API.services.feature_service import make_features

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=Prediction,
    summary="Predict Machine Failure",
    description=(
        "Calculate machine failure probability and predicted failure state from raw sensor inputs, "
        "and emit structured Retraining & Reinforcement Learning trajectory logs to CloudWatch."
    ),
)
def predict(sensor: SensorInput, request: Request, response: Response) -> Prediction:
    """Predict equipment failure probability, classification label, and record MLOps log."""
    start_time = time.perf_counter()
    features = make_features(sensor)

    model_service = getattr(request.app.state, "model_service", None)
    if model_service is not None:
        probability = model_service.predict_probability(features)
        threshold = model_service.threshold
        prediction = Prediction(
            machine_failure=1 if probability >= threshold else 0,
            failure_probability=probability,
        )
    else:
        # Fallback to direct model reference on app.state
        model = request.app.state.model
        threshold = getattr(request.app.state, "threshold", 0.5)
        probability = float(model.predict(features[model.feature_name()], num_threads=1)[0])
        prediction = Prediction(
            machine_failure=int(probability >= threshold),
            failure_probability=probability,
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    cw_service = getattr(request.app.state, "cloudwatch_service", None)
    if cw_service is None:
        cw_service = CloudWatchLogService(settings=getattr(request.app.state, "settings", None))
        request.app.state.cloudwatch_service = cw_service

    equipment_id = request.headers.get("X-Equipment-ID", "CNC-MILL-01")
    log_entry = cw_service.log_inference_event(
        sensor=sensor,
        features_df=features,
        prediction=prediction,
        threshold=threshold,
        latency_ms=latency_ms,
        equipment_id=equipment_id,
    )

    response.headers["X-Prediction-ID"] = log_entry["prediction_id"]
    response.headers["X-Log-Group"] = cw_service.log_group
    response.headers["X-Log-Stream"] = cw_service.log_stream

    return prediction
