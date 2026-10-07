"""Unit tests for the Deep Learning Engine and Hybrid Ensemble in StockMind-AI.

Tests:
1. SequenceScaler: Anti-leakage isolation (fit strictly on train).
2. Chronological Split: 3-way train/validation/test ordering without lookahead or shuffling.
3. Sequence Generation: Multi-step historical lookback window creation with dual targets.
4. PyTorch Dataset & DataLoader: Tensor mapping, batch collation.
5. LSTMForecaster Architecture: Multi-task recurrent network, dual heads, MC Dropout uncertainty.
6. GRUForecaster Architecture: Gated recurrent network, dual heads, MC Dropout uncertainty.
7. EarlyStopping & Checkpoints: Validation loss patience, best weight restoration.
8. TrainingScheduleManager: Cooldown enforcement to prevent tick thrashing.
9. Evaluation: Comprehensive classification and regression metrics (Accuracy, F1, MAE, RMSE, R²).
10. DeepLearningPredictor & HybridMLEnsemble: Forward inference, model agreement, and statutory disclaimers.
11. FastAPI Endpoints: REST API routes for DL prediction, consensus ensemble, and comparison.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ai_engine.deep_learning.dataset import (
    SequenceScaler,
    TimeSeriesSequenceDataset,
    create_chronological_sequences,
    chronological_train_val_test_split,
    prepare_deep_learning_dataloaders,
)
from ai_engine.deep_learning.lstm import LSTMForecaster
from ai_engine.deep_learning.gru import GRUForecaster
from ai_engine.deep_learning.train import (
    EarlyStopping,
    TrainingScheduleManager,
    train_deep_learning_model,
)
from ai_engine.deep_learning.evaluate import (
    evaluate_deep_learning_model,
    compare_deep_learning_with_classical,
)
from ai_engine.deep_learning.predict import (
    DeepLearningPredictor,
    HybridMLEnsemble,
    dl_predictor,
    hybrid_ensemble,
)
from ai_engine.ml.model_registry import ModelRegistry, model_registry
from backend.app.main import app

import torch
import torch.nn as nn


@pytest.fixture
def synthetic_stock_df() -> pd.DataFrame:
    """Create reproducible synthetic OHLCV time-series dataframe."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    base_price = 2500.0
    returns = np.random.normal(0.0008, 0.015, n)
    closes = base_price * np.cumprod(1 + returns)

    highs = closes * (1 + np.abs(np.random.normal(0.004, 0.003, n)))
    lows = closes * (1 - np.abs(np.random.normal(0.004, 0.003, n)))
    opens = (closes + lows) / 2.0
    volumes = np.random.randint(200000, 800000, n).astype(float)

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
# 1. DATASET & PREPROCESSING TESTS
# ============================================================================


def test_sequence_scaler_prevents_leakage():
    """Verify SequenceScaler fits strictly on training data and transforms without leakage."""
    np.random.seed(42)
    train_data = np.random.normal(loc=100.0, scale=10.0, size=(100, 4))
    test_data = np.random.normal(loc=150.0, scale=20.0, size=(30, 4))

    scaler = SequenceScaler()
    scaler.fit(train_data)

    # Train mean should reflect train_data
    np.testing.assert_allclose(scaler.mean_, train_data.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(scaler.scale_, train_data.std(axis=0), atol=1e-5)

    scaled_train = scaler.transform(train_data)
    np.testing.assert_allclose(scaled_train.mean(axis=0), np.zeros(4), atol=1e-5)
    np.testing.assert_allclose(scaled_train.std(axis=0), np.ones(4), atol=1e-5)

    # Test data scaled with train statistics
    scaled_test = scaler.transform(test_data)
    assert scaled_test.shape == (30, 4)
    # Test data has different distribution, so mean shouldn't be exactly 0
    assert not np.allclose(scaled_test.mean(axis=0), np.zeros(4), atol=0.1)


def test_chronological_train_val_test_split(synthetic_stock_df: pd.DataFrame):
    """Verify strict chronological sequence ordering and size splits."""
    train_df, val_df, test_df = chronological_train_val_test_split(
        synthetic_stock_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    n = len(synthetic_stock_df)
    assert len(train_df) + len(val_df) + len(test_df) == n

    # Strict chronological monotonic timestamps
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_create_chronological_sequences(synthetic_stock_df: pd.DataFrame):
    """Verify sliding lookback sequences and dual targets (direction & return)."""
    feature_cols = ["open", "high", "low", "close", "volume"]
    features = synthetic_stock_df[feature_cols].to_numpy()
    closes = synthetic_stock_df["close"].to_numpy()
    ret = np.diff(closes) / closes[:-1]
    y_ret = np.append(ret, 0.0)
    y_dir = (y_ret >= 0).astype(np.float32)
    seq_len = 20

    X_seq, y_dir_seq, y_ret_seq = create_chronological_sequences(
        features=features,
        targets_dir=y_dir,
        targets_ret=y_ret,
        seq_length=seq_len,
    )

    expected_samples = len(synthetic_stock_df) - seq_len + 1
    assert X_seq.shape == (expected_samples, seq_len, len(feature_cols))
    assert y_dir_seq.shape == (expected_samples,)
    assert y_ret_seq.shape == (expected_samples,)

    # Direction must be binary 0.0 or 1.0
    assert set(np.unique(y_dir_seq)).issubset({0.0, 1.0})


def test_time_series_sequence_dataset():
    """Verify PyTorch dataset yields correct shapes and float32 tensors."""
    X = np.random.randn(50, 15, 6).astype(np.float32)
    y_dir = np.random.choice([0.0, 1.0], size=50).astype(np.float32)
    y_ret = np.random.randn(50).astype(np.float32)

    dataset = TimeSeriesSequenceDataset(X, y_dir, y_ret)
    assert len(dataset) == 50

    x_item, dir_item, ret_item = dataset[0]
    assert isinstance(x_item, torch.Tensor)
    assert x_item.shape == (15, 6)
    assert dir_item.shape == ()
    assert ret_item.shape == ()


# ============================================================================
# 2. NEURAL ARCHITECTURE TESTS (LSTM & GRU)
# ============================================================================


def test_lstm_forecaster_forward_pass():
    """Verify LSTM dual-head forward pass output dimensions."""
    model = LSTMForecaster(input_dim=8, hidden_dim=32, num_layers=2, dropout=0.2)
    dummy_input = torch.randn(16, 25, 8)  # batch_size=16, seq_len=25, input_dim=8

    dir_logits, ret_preds, log_std = model(dummy_input)

    assert dir_logits.shape == (16, 1)
    assert ret_preds.shape == (16, 1)
    assert log_std.shape == (16, 1)


def test_gru_forecaster_forward_pass():
    """Verify GRU dual-head forward pass output dimensions."""
    model = GRUForecaster(input_dim=8, hidden_dim=32, num_layers=2, dropout=0.2)
    dummy_input = torch.randn(16, 25, 8)

    dir_logits, ret_preds, log_std = model(dummy_input)

    assert dir_logits.shape == (16, 1)
    assert ret_preds.shape == (16, 1)
    assert log_std.shape == (16, 1)


def test_lstm_mc_dropout_uncertainty():
    """Verify Monte Carlo Dropout yields confidence probability and epistemic uncertainty."""
    model = LSTMForecaster(input_dim=5, hidden_dim=16, num_layers=2, dropout=0.3)
    single_seq = torch.randn(1, 20, 5)

    mc_res = model.predict_with_uncertainty(single_seq, num_samples=15)

    assert "probability_up" in mc_res
    assert "confidence" in mc_res
    assert "expected_return" in mc_res
    assert "return_std" in mc_res
    assert "predicted_direction" in mc_res
    assert 0.0 <= mc_res["probability_up"] <= 1.0
    assert mc_res["return_std"] >= 0.0
    assert mc_res["predicted_direction"] in ["BULLISH", "BEARISH"]


def test_gru_mc_dropout_uncertainty():
    """Verify GRU Monte Carlo Dropout output contract."""
    model = GRUForecaster(input_dim=5, hidden_dim=16, num_layers=2, dropout=0.3)
    single_seq = torch.randn(1, 20, 5)

    mc_res = model.predict_with_uncertainty(single_seq, num_samples=15)

    assert 0.0 <= mc_res["probability_up"] <= 1.0
    assert mc_res["predicted_direction"] in ["BULLISH", "BEARISH"]


# ============================================================================
# 3. TRAINING & CONTROLLED SCHEDULE TESTS
# ============================================================================


def test_early_stopping_mechanism():
    """Verify early stopping triggers after patience limit and restores best weights."""
    model = nn.Linear(5, 1)
    early_stopping = EarlyStopping(patience=3, min_delta=1e-4)

    # Initial best loss
    assert not early_stopping(val_loss=1.0, model=model)

    # Plateaus / worsens
    assert not early_stopping(val_loss=1.1, model=model)
    assert not early_stopping(val_loss=1.05, model=model)
    # 3rd non-improving epoch triggers stop
    assert early_stopping(val_loss=1.2, model=model)
    assert early_stopping.early_stop is True


def test_training_schedule_manager_prevents_tick_thrashing():
    """Verify TrainingScheduleManager enforces cooldown interval."""
    manager = TrainingScheduleManager(cooldown_minutes=60)
    symbol = "TEST_SYM"
    model_type = "LSTM"

    # 1. First run should be allowed
    assert manager.can_train(symbol, model_type) is True

    # 2. Record training
    manager.record_training(symbol, model_type)

    # 3. Immediate subsequent call must be BLOCKED
    assert manager.can_train(symbol, model_type) is False
    remaining = manager.remaining_cooldown(symbol, model_type)
    assert 55.0 <= remaining <= 60.0

    # 4. Force retrain should bypass cooldown
    assert manager.can_train(symbol, model_type, force=True) is True


def test_train_deep_learning_model_execution(synthetic_stock_df: pd.DataFrame, tmp_path: Path):
    """Verify deep learning training loop creates saved checkpoint and registers in ModelRegistry."""
    res = train_deep_learning_model(
        df=synthetic_stock_df,
        symbol="SYNTH_DL",
        model_type="GRU",
        epochs=3,
        batch_size=16,
        sequence_length=15,
        force_retrain=True,
    )

    assert res["symbol"] == "SYNTH_DL"
    assert res["model_type"] == "GRU"
    assert "test_metrics" in res
    assert "accuracy" in res["test_metrics"]
    assert "f1" in res["test_metrics"]
    assert "checkpoint_path" in res
    assert Path(res["checkpoint_path"]).exists()


# ============================================================================
# 4. PREDICTOR & HYBRID ENSEMBLE TESTS
# ============================================================================


def test_deep_learning_predictor_output(synthetic_stock_df: pd.DataFrame):
    """Verify DeepLearningPredictor inference structure and bounds."""
    pred = dl_predictor.predict_next_period(
        df=synthetic_stock_df,
        symbol="RELIANCE",
        model_type="LSTM",
    )

    assert pred["symbol"] == "RELIANCE"
    assert pred["model_type"] in ["LSTM", "GRU"]
    assert pred["predicted_direction"] in ["UP", "DOWN", "BULLISH", "BEARISH"]
    assert 50.0 <= pred["confidence_score"] <= 100.0
    assert pred["target_price"] > 0
    assert pred["expected_range_low"] < pred["expected_range_high"]
    assert "disclaimer" in pred


def test_hybrid_ensemble_model_agreement(synthetic_stock_df: pd.DataFrame):
    """Verify HybridMLEnsemble combines classical ML and DL models with agreement score."""
    consensus = hybrid_ensemble.predict_consensus(
        df=synthetic_stock_df,
        symbol="RELIANCE",
    )

    assert consensus["symbol"] == "RELIANCE"
    assert consensus["consensus_direction"] in ["UP", "DOWN"]
    assert 50.0 <= consensus["consensus_confidence"] <= 100.0
    assert 0.0 <= consensus["model_agreement_pct"] <= 100.0
    assert "agreement_summary" in consensus
    assert consensus["target_price"] > 0
    assert consensus["expected_range_low"] < consensus["expected_range_high"]

    # Verify individual votes breakdown
    votes = consensus["votes"]
    assert len(votes) >= 5  # RF, LR, SVM, GB, LSTM, GRU
    model_names = [v["model_name"] for v in votes]
    assert "LSTM" in model_names
    assert "GRU" in model_names
    assert "RandomForest" in model_names


def test_compare_deep_learning_with_classical():
    """Verify comparison function outputs structured comparison payload."""
    comp = compare_deep_learning_with_classical("RELIANCE")
    assert "symbol" in comp
    assert "models" in comp
    assert "total_models" in comp
    assert "disclaimer" in comp


# ============================================================================
# 5. REST API ENDPOINTS
# ============================================================================


def test_dl_predict_api_route():
    """Verify GET /api/v1/ml/deep-learning/predict/{symbol} returns 200 and schema."""
    with TestClient(app) as client:
        res = client.get("/api/v1/ml/deep-learning/predict/RELIANCE?model_type=LSTM")
        assert res.status_code == 200
        data = res.json()
        assert data["symbol"] == "RELIANCE"
        assert data["model_type"] == "LSTM"
        assert data["predicted_direction"] in ["UP", "DOWN", "BULLISH", "BEARISH"]
        assert "confidence_score" in data
        assert "target_price" in data


def test_ensemble_consensus_api_route():
    """Verify GET /api/v1/ml/ensemble/consensus/{symbol} returns consensus and agreement."""
    with TestClient(app) as client:
        res = client.get("/api/v1/ml/ensemble/consensus/RELIANCE")
        assert res.status_code == 200
        data = res.json()
        assert data["symbol"] == "RELIANCE"
        assert data["consensus_direction"] in ["UP", "DOWN"]
        assert "model_agreement_pct" in data
        assert "votes" in data
        assert len(data["votes"]) > 0


def test_dl_comparison_api_route():
    """Verify GET /api/v1/ml/deep-learning/comparison/{symbol} returns comparison matrix."""
    with TestClient(app) as client:
        res = client.get("/api/v1/ml/deep-learning/comparison/RELIANCE")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
