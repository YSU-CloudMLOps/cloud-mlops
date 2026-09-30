"""Machine learning model loader and inference service."""

import logging
from pathlib import Path
from typing import Optional, Union

import lightgbm as lgb
import pandas as pd

from API.config import Settings, get_settings
from API.schemas import Prediction, SensorInput
from API.services.feature_service import make_features

logger = logging.getLogger(__name__)


class ModelService:
    """Service wrapping LightGBM model loading and inference logic."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
    ):
        settings: Settings = get_settings()
        self.model_path = Path(model_path) if model_path else settings.model_path
        self.threshold = threshold if threshold is not None else settings.threshold
        self._model: Optional[lgb.Booster] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load LightGBM model from text file, normalizing line endings."""
        if not self.model_path.exists():
            error_msg = f"Model file not found at: {self.model_path.resolve()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading model from: {self.model_path.resolve()}")
        # Normalize Git's Windows CRLF checkout back to native LF offsets.
        raw_text = self.model_path.read_text(encoding="utf-8")
        self._model = lgb.Booster(model_str=raw_text)
        logger.info(f"Model successfully loaded with {len(self.feature_names)} features.")

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self._model is not None

    @property
    def model(self) -> lgb.Booster:
        """Return the underlying LightGBM Booster instance."""
        if self._model is None:
            raise RuntimeError("Model is not loaded.")
        return self._model

    @property
    def feature_names(self) -> list[str]:
        """Return expected feature names in order."""
        return list(self.model.feature_name())

    def predict_probability(self, features: pd.DataFrame) -> float:
        """Calculate failure probability for engineered features."""
        # Align features with model expectations
        aligned_features = features[self.feature_names]
        prob = float(self.model.predict(aligned_features, num_threads=1)[0])
        return prob

    def predict_sensor(self, sensor: SensorInput) -> Prediction:
        """Transform raw sensor input and return structured Prediction."""
        features = make_features(sensor)
        probability = self.predict_probability(features)
        machine_failure = 1 if probability >= self.threshold else 0

        return Prediction(
            machine_failure=machine_failure,
            failure_probability=probability,
        )
