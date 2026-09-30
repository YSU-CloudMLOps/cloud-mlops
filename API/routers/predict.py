"""Prediction endpoints for machine failure inference."""

from fastapi import APIRouter, Request

from API.schemas import Prediction, SensorInput

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=Prediction,
    summary="Predict Machine Failure",
    description="Calculate machine failure probability and predicted failure state from raw sensor inputs.",
)
def predict(sensor: SensorInput, request: Request) -> Prediction:
    """Predict equipment failure probability and classification label."""
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is not None:
        return model_service.predict_sensor(sensor)

    # Fallback to direct model reference on app.state
    from API.services.feature_service import make_features
    features = make_features(sensor)
    model = request.app.state.model
    threshold = getattr(request.app.state, "threshold", 0.5)
    probability = float(model.predict(features[model.feature_name()], num_threads=1)[0])
    return Prediction(
        machine_failure=int(probability >= threshold),
        failure_probability=probability,
    )
