"""Main application module and entrypoint for Predictive Maintenance API."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from API.config import Settings, get_settings
from API.routers.health import router as health_router
from API.routers.predict import router as predict_router
from API.schemas import Prediction, SensorInput
from API.services.feature_service import make_features
from API.services.model_service import ModelService

# Backward-compatible exports
_settings = get_settings()
MODEL_PATH = _settings.model_path
THRESHOLD = _settings.threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup initialization and shutdown cleanup."""
    settings: Settings = getattr(app.state, "settings", get_settings())
    # Initialize ModelService which loads the model and normalizes CRLF line endings
    model_service = ModelService(model_path=settings.model_path, threshold=settings.threshold)
    app.state.model_service = model_service
    app.state.model = model_service.model
    app.state.threshold = settings.threshold
    yield


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure FastAPI application instance."""
    app_settings = settings or get_settings()

    application = FastAPI(
        title=app_settings.app_title,
        version=app_settings.app_version,
        description=app_settings.app_description,
        lifespan=lifespan,
    )
    application.state.settings = app_settings

    # Enable Cross-Origin Resource Sharing (CORS) for web frontend
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include modular routers
    application.include_router(health_router, prefix=app_settings.api_prefix)
    application.include_router(predict_router, prefix=app_settings.api_prefix)

    return application


# Global application instance for ASGI servers (e.g. uvicorn API.api:app)
app = create_app()

__all__ = [
    "app",
    "create_app",
    "lifespan",
    "MODEL_PATH",
    "THRESHOLD",
    "SensorInput",
    "Prediction",
    "make_features",
]
