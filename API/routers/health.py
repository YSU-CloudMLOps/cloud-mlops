"""Health check and service status endpoints."""

from fastapi import APIRouter, Request

from API.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Check the operational status of the inference API and model readiness.",
)
def health(request: Request) -> dict:
    """Return health status of the API service and ML model."""
    model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "version": request.app.version,
    }
