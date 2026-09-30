"""Routers package containing API endpoints."""

from API.routers.health import router as health_router
from API.routers.predict import router as predict_router

__all__ = ["health_router", "predict_router"]
