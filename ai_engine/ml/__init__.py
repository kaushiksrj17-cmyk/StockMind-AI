"""Machine Learning Engine for StockMind-AI.

Exports:
- TimeSeriesFeaturePipeline, chronological_train_test_split, TimeSeriesScaler
- TimeSeriesFeatureSelector
- calculate_classification_metrics, calculate_regression_metrics, evaluate_model_performance
- train_direction_model, train_return_model, train_full_ml_pipeline
- DirectionalEnsemble, MovementRangeEnsemble
- ModelRegistry, model_registry
- MLPredictor, ml_predictor, PREDICTION_DISCLAIMER
"""

from ai_engine.ml.preprocessing import (
    TimeSeriesFeaturePipeline,
    chronological_train_test_split,
    TimeSeriesScaler,
)
from ai_engine.ml.feature_selection import TimeSeriesFeatureSelector
from ai_engine.ml.evaluate import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    evaluate_model_performance,
)
from ai_engine.ml.train import (
    train_direction_model,
    train_return_model,
    train_full_ml_pipeline,
    get_classifier,
    get_regressor,
)
from ai_engine.ml.ensemble import (
    DirectionalEnsemble,
    MovementRangeEnsemble,
)
from ai_engine.ml.model_registry import (
    ModelRegistry,
    model_registry,
)
from ai_engine.ml.predict import (
    MLPredictor,
    ml_predictor,
    PREDICTION_DISCLAIMER,
)

__all__ = [
    "TimeSeriesFeaturePipeline",
    "chronological_train_test_split",
    "TimeSeriesScaler",
    "TimeSeriesFeatureSelector",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "evaluate_model_performance",
    "train_direction_model",
    "train_return_model",
    "train_full_ml_pipeline",
    "get_classifier",
    "get_regressor",
    "DirectionalEnsemble",
    "MovementRangeEnsemble",
    "ModelRegistry",
    "model_registry",
    "MLPredictor",
    "ml_predictor",
    "PREDICTION_DISCLAIMER",
]
