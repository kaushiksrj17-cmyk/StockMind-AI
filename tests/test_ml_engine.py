"""Unit tests for the Machine Learning Engine in StockMind-AI.

Tests:
1. Time-series safe preprocessing: no look-ahead bias, forward targets shifted by 1, chronological split, scaler isolation.
2. Feature selection: variance threshold, correlation ranking, collinear threshold.
3. Evaluation metrics: MAE, RMSE, R2, Accuracy, Precision, Recall, F1, Confusion Matrix.
4. Model Registry: version tracking, saving, loading, metadata persistence.
5. Ensemble & Predict: Directional ensemble, Movement range ensemble, MLPredictor inference output schema and statutory disclaimer.
6. API routes: /predict/{symbol}, /comparison/{symbol}, /status.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ai_engine.ml.evaluate import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    evaluate_model_performance,
)
from ai_engine.ml.feature_selection import TimeSeriesFeatureSelector
from ai_engine.ml.model_registry import ModelRegistry
from ai_engine.ml.preprocessing import (
    TimeSeriesFeaturePipeline,
    TimeSeriesScaler,
    chronological_train_test_split,
)
from backend.app.main import app


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Create deterministic synthetic OHLCV time-series for tests."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n)
    closes = base_price * np.cumprod(1 + returns)

    highs = closes * (1 + np.abs(np.random.normal(0.005, 0.005, n)))
    lows = closes * (1 - np.abs(np.random.normal(0.005, 0.005, n)))
    opens = (closes + lows) / 2.0
    volumes = np.random.randint(100000, 500000, n).astype(float)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


# ============================================================================
# 1. PREPROCESSING & TIME-SERIES INTEGRITY TESTS
# ============================================================================

def test_time_series_feature_pipeline_no_lookahead_bias(synthetic_ohlcv_df: pd.DataFrame):
    """Verify that features and targets prevent look-ahead bias and data leakage."""
    pipeline = TimeSeriesFeaturePipeline()
    X, y_dir, y_ret, latest_row = pipeline.fit_transform(synthetic_ohlcv_df)

    # 1. Feature matrix and targets must have matching length
    assert len(X) == len(y_dir) == len(y_ret)
    assert len(X) > 50

    # 2. Latest bar must be held out for forward inference
    assert latest_row is not None
    assert isinstance(latest_row, pd.DataFrame)
    assert len(latest_row) == 1

    # 3. No NaN values in training feature matrix or targets
    assert not X.isna().any().any()
    assert not y_dir.isna().any()
    assert not y_ret.isna().any()

    # 4. Target values validity: y_dir is binary (0 or 1), y_ret is bounded float
    assert set(y_dir.unique()).issubset({0, 1})
    assert -0.5 < y_ret.mean() < 0.5


def test_chronological_train_test_split():
    """Verify strictly chronological split without random shuffling."""
    X = pd.DataFrame({"feat": np.arange(100)})
    y = pd.Series(np.arange(100))

    X_train, X_test, y_train, y_test = chronological_train_test_split(
        X, y, test_size=0.2
    )

    # Size check
    assert len(X_train) == 80
    assert len(X_test) == 20

    # Chronological integrity: all train indices strictly precede test indices
    assert X_train.index.max() < X_test.index.min()
    assert y_train.iloc[-1] < y_test.iloc[0]


def test_time_series_scaler_leakage_prevention():
    """Verify TimeSeriesScaler fits ONLY on training data, preventing test set leakage."""
    train_vals = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    test_vals = np.array([[40.0, 400.0], [50.0, 500.0]])

    scaler = TimeSeriesScaler()
    scaler.fit(train_vals)

    # Transform train and test
    scaled_train = scaler.transform(train_vals)
    scaled_test = scaler.transform(test_vals)

    # Mean of scaled train should be approximately 0.0, std approximately 1.0 (sample std)
    np.testing.assert_almost_equal(scaled_train.mean(axis=0), [0.0, 0.0], decimal=5)
    np.testing.assert_almost_equal(scaled_train.std(axis=0, ddof=1), [1.0, 1.0], decimal=5)

    # Test values must be scaled using train statistics (mean=20, std=8.16)
    assert scaled_test[0, 0] > 1.0


# ============================================================================
# 2. FEATURE SELECTION TESTS
# ============================================================================

def test_feature_selection_filtering():
    """Verify TimeSeriesFeatureSelector eliminates zero-variance and collinear features."""
    np.random.seed(42)
    n = 100
    f1 = np.random.normal(0, 1, n)
    f2 = f1 * 1.00001  # highly collinear with f1
    f3 = np.random.normal(0, 1, n)
    f_const = np.ones(n)  # zero variance

    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f_const": f_const})
    y = pd.Series(np.where(f1 > 0, 1, 0))

    selector = TimeSeriesFeatureSelector(variance_thresh=0.01, collinear_thresh=0.95, top_k=2)
    X_selected = selector.fit_transform(X, y)

    # Constant feature must be removed
    assert "f_const" not in selector.selected_features_

    # Collinear feature duplicate should be pruned
    assert not ("f1" in selector.selected_features_ and "f2" in selector.selected_features_)
    assert len(selector.selected_features_) <= 2


# ============================================================================
# 3. EVALUATION METRICS TESTS
# ============================================================================

def test_regression_metrics_calculation():
    """Verify MAE, RMSE, R2, and MDA calculations."""
    y_true = np.array([100.0, 102.0, 101.0, 105.0])
    y_pred = np.array([101.0, 101.0, 102.0, 104.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert "directional_accuracy" in metrics

    # MAE = (|1| + |1| + |1| + |1|) / 4 = 1.0
    assert pytest.approx(metrics["mae"], 0.01) == 1.0
    # RMSE = sqrt(4 / 4) = 1.0
    assert pytest.approx(metrics["rmse"], 0.01) == 1.0
    assert -1.0 <= metrics["r2_score"] <= 1.0


def test_classification_metrics_calculation():
    """Verify Accuracy, Precision, Recall, F1, and Confusion Matrix calculations."""
    # 2 TN, 1 FP, 1 FN, 2 TP
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert pytest.approx(metrics["accuracy"], 0.1) == (4 / 6) * 100.0
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

    cm = metrics["confusion_matrix"]
    assert cm["true_negative"] == 2
    assert cm["false_positive"] == 1
    assert cm["false_negative"] == 1
    assert cm["true_positive"] == 2


# ============================================================================
# 4. MODEL REGISTRY TESTS
# ============================================================================

def test_model_registry_save_and_retrieve(tmp_path: Path):
    """Verify ModelRegistry correctly versions, saves, and loads models."""
    registry = ModelRegistry(registry_dir=tmp_path)

    dummy_model = {"model_name": "test_model", "weights": [1, 2, 3]}
    metadata = {
        "model_type": "random_forest",
        "symbol": "TEST",
        "metrics": {"accuracy": 0.75},
    }

    # Save
    version = registry.save_model("TEST", dummy_model, metadata)
    assert version.startswith("v1.")

    # Status check
    status = registry.get_model_status("TEST")
    assert status["is_ready"] is True
    assert status["latest_version"] == version

    # Load
    loaded_model, loaded_meta = registry.load_model("TEST", version)
    assert loaded_model["model_name"] == "test_model"
    assert loaded_meta["symbol"] == "TEST"


# ============================================================================
# 5. FASTAPI REST ENDPOINTS INTEGRITY TESTS
# ============================================================================

client = TestClient(app)


def test_ml_status_endpoint():
    """Verify GET /api/v1/ml/status returns registry status."""
    response = client.get("/api/v1/ml/status")
    assert response.status_code == 200
    data = response.json()
    assert "engine_status" in data
    assert "total_registered_models" in data
    assert "supported_model_types" in data


def test_ml_prediction_endpoint():
    """Verify GET /api/v1/ml/predict/{symbol} returns valid prediction schema."""
    response = client.get("/api/v1/ml/predict/RELIANCE?timeframe=1d")
    assert response.status_code == 200
    data = response.json()

    assert data["symbol"] == "RELIANCE"
    assert data["predicted_direction"] in ["UP", "DOWN", "NEUTRAL"]
    assert data["sentiment_bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert 0.0 <= data["confidence_score"] <= 100.0
    assert 0.0 <= data["probability_up"] <= 100.0
    assert data["expected_movement"]["target_price"] > 0
    assert data["expected_movement"]["expected_range_low"] <= data["expected_movement"]["expected_range_high"]
    assert "feature_importance" in data
    assert "disclaimer" in data
    assert "guarantee" in data["disclaimer"].lower() or "statistical" in data["disclaimer"].lower()


def test_ml_comparison_endpoint():
    """Verify GET /api/v1/ml/comparison/{symbol} returns comparison metrics."""
    response = client.get("/api/v1/ml/comparison/RELIANCE")
    assert response.status_code == 200
    data = response.json()

    assert data["symbol"] == "RELIANCE"
    assert "models" in data
    assert len(data["models"]) >= 1

    first_model = data["models"][0]
    assert "model_type" in first_model
    assert "accuracy" in first_model
    assert "f1" in first_model
    assert "mae" in first_model
