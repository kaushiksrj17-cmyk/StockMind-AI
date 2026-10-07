"""Unit and Integration Tests for Explainable AI (SHAP) and Real-Time AI Signal Engine."""

import datetime
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_engine.explainability.shap_explainer import (
    FeatureImportance,
    SHAPExplainer,
    SHAPExplanationResult,
    shap_explainer,
)
from ai_engine.signals.realtime_pipeline import (
    CandleBar,
    RealTimeSignalPipeline,
    realtime_signal_pipeline,
)
from ai_engine.signals.signal_engine import (
    AISignalEngine,
    AISignalResult,
    ai_signal_engine,
)
from backend.app.main import app


# ============================================================================
# 1. SHAP EXPLAINABILITY TESTS
# ============================================================================

def test_shap_explainer_basic_attribution():
    """Verify SHAP explanation generation and impact direction separation."""
    features = {
        "Technical AI Score": 78.0,
        "ML Direction Prob": 82.0,
        "News Sentiment": 45.0,
        "VWAP Momentum": 15.0,
        "Annual Volatility": 12.0,
        "Sharpe Ratio": 40.0,
    }

    res = shap_explainer.explain_instance(
        features=features,
        symbol="RELIANCE",
        model_confidence=0.84,
        model_agreement_pct=88.0,
    )

    assert res.symbol == "RELIANCE"
    assert res.predicted_label in ("STRONG BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG BEARISH")
    assert 0.0 < res.predicted_value < 1.0
    assert res.model_confidence_pct == pytest.approx(84.0, rel=1e-2)
    assert res.model_agreement_pct == pytest.approx(88.0, rel=1e-2)
    assert len(res.top_contributing_features) > 0

    # Verify positive and negative groupings
    for p in res.positive_impacts:
        assert p.impact_direction == "POSITIVE"
        assert p.shap_value >= 0.0

    for n in res.negative_impacts:
        assert n.impact_direction == "NEGATIVE"
        assert n.shap_value <= 0.0

    # Verify narrative and non-guarantee disclaimer
    assert len(res.explanation_narrative) > 0
    assert "EXPLAINABILITY SIMULATION NOTICE" in res.compliance_disclaimer


def test_shap_exact_shapley_efficiency():
    """Verify that exact Shapley value computation satisfies the efficiency axiom."""
    # Define a simple non-linear model function
    def toy_model(X):
        # f(x0, x1, x2) = 2*x0 + 0.5*x1^2 - x2
        return 2.0 * X[:, 0] + 0.5 * (X[:, 1] ** 2) - X[:, 2]

    x_inst = np.array([1.5, 2.0, 0.5])
    baseline = np.array([0.0, 0.0, 0.0])
    feature_names = ["feat0", "feat1", "feat2"]

    shap_vals, base_val = SHAPExplainer._compute_exact_shapley_values(
        predict_fn=toy_model,
        x_instance=x_inst,
        baseline=baseline,
        feature_names=feature_names,
        num_permutations=50,
    )

    expected_diff = float(toy_model(x_inst.reshape(1, -1))[0] - toy_model(baseline.reshape(1, -1))[0])
    # Efficiency axiom: sum(phi_i) == f(x) - f(baseline)
    assert sum(shap_vals) == pytest.approx(expected_diff, rel=1e-3)


# ============================================================================
# 2. AI SIGNAL ENGINE TESTS
# ============================================================================

def test_ai_signal_engine_strong_bullish_synthesis():
    """Verify strong bullish multi-factor consensus generation."""
    signal = ai_signal_engine.generate_signal(
        symbol="TCS",
        live_quote={"last_price": 4200.0, "vwap": 4150.0, "change_percent": 2.4},
        technical_data={"technical_score": 88.0},
        ml_prediction={"direction": "UP", "probability": 0.85},
        deep_learning_forecast={"direction": "UP", "confidence": 0.82},
        sentiment_data={"sentiment_score": 0.70},
        risk_metrics={"annualized_volatility_pct": 14.0, "sharpe_ratio": 2.4, "max_drawdown_pct": 6.0},
        regime_data={"primary_regime": "BULL", "confidence_pct": 85.0},
        data_mode="LIVE DATA",
    )

    assert signal.symbol == "TCS"
    assert signal.signal in ("STRONG BULLISH", "BULLISH")
    assert signal.composite_score > 35.0
    assert signal.confidence_pct >= 65.0
    assert signal.data_mode == "LIVE DATA"
    assert len(signal.supporting_factors) > 0
    assert len(signal.explanation) > 0
    assert "PROBABILISTIC MARKET ANALYSIS NOTICE" in signal.compliance_disclaimer


def test_ai_signal_engine_strong_bearish_synthesis():
    """Verify strong bearish multi-factor consensus generation."""
    signal = ai_signal_engine.generate_signal(
        symbol="INTRADAY_BREAKDOWN",
        live_quote={"last_price": 950.0, "vwap": 985.0, "change_percent": -4.2},
        technical_data={"technical_score": 15.0},
        ml_prediction={"direction": "DOWN", "probability": 0.88},
        deep_learning_forecast={"direction": "DOWN", "confidence": 0.84},
        sentiment_data={"sentiment_score": -0.65},
        risk_metrics={"annualized_volatility_pct": 32.0, "sharpe_ratio": -0.5, "max_drawdown_pct": 22.0},
        regime_data={"primary_regime": "BEAR", "confidence_pct": 90.0},
        data_mode="REPLAY DATA",
    )

    assert signal.signal in ("STRONG BEARISH", "BEARISH")
    assert signal.composite_score < -35.0
    assert len(signal.opposing_factors) > 0


def test_ai_signal_engine_neutral_synthesis():
    """Verify neutral equilibrium state."""
    signal = ai_signal_engine.generate_signal(
        symbol="SIDEWAYS_ASSET",
        live_quote={"last_price": 1000.0, "vwap": 1000.0, "change_percent": 0.05},
        technical_data={"technical_score": 50.0},
        ml_prediction={"direction": "NEUTRAL", "probability": 0.50},
        regime_data={"primary_regime": "SIDEWAYS", "confidence_pct": 60.0},
    )

    assert signal.signal == "NEUTRAL"
    assert abs(signal.composite_score) < 20.0


# ============================================================================
# 3. REAL-TIME STREAMING PIPELINE TESTS
# ============================================================================

def test_realtime_pipeline_lifecycle():
    """Verify real-time execution flow:
    LIVE TICK -> candle update -> indicator update -> feature update -> fast AI inference -> signal update.
    """
    pipeline = RealTimeSignalPipeline()

    # Pre-seed slow model cache (ensuring expensive models are not run on ticks)
    pipeline.update_slow_model_cache(
        symbol="RELIANCE",
        ml_prediction={"direction": "UP", "probability": 0.75},
        deep_learning_forecast={"direction": "UP", "confidence": 0.72},
        sentiment_data={"sentiment_score": 0.35},
        regime_data={"primary_regime": "BULL", "confidence_pct": 80.0},
    )

    # Simulate live tick stream
    tick_1 = pipeline.process_live_tick(
        symbol="RELIANCE",
        price=2900.0,
        volume=500.0,
        day_open=2880.0,
        data_mode="LIVE DATA",
    )
    assert tick_1.symbol == "RELIANCE"
    assert tick_1.data_mode == "LIVE DATA"
    assert tick_1.factor_telemetry["last_price"] == 2900.0

    # Check candle state
    bar = pipeline._candle_bars["RELIANCE"]
    assert bar.high == 2900.0
    assert bar.low == 2900.0
    assert bar.volume == 500.0

    # Simulate higher tick
    tick_2 = pipeline.process_live_tick(
        symbol="RELIANCE",
        price=2925.0,
        volume=800.0,
        data_mode="LIVE DATA",
    )
    assert tick_2.factor_telemetry["last_price"] == 2925.0
    assert bar.high == 2925.0
    assert bar.volume == 1300.0

    # Verify cached retrieval
    cached_sig = pipeline.get_latest_signal("RELIANCE")
    assert cached_sig is not None
    assert cached_sig.symbol == "RELIANCE"


# ============================================================================
# 4. FASTAPI SIGNALS & EXPLAINABILITY API TESTS
# ============================================================================

def test_api_signals_endpoint():
    """Verify GET /api/v1/signals/{symbol}."""
    client = TestClient(app)

    resp = client.get("/api/v1/signals/RELIANCE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert data["signal"] in ("STRONG BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG BEARISH")
    assert "confidence_pct" in data
    assert "composite_score" in data
    assert "supporting_factors" in data
    assert "opposing_factors" in data
    assert "explanation" in data
    assert "compliance_disclaimer" in data


def test_api_explainability_endpoint():
    """Verify GET /api/v1/signals/{symbol}/explain."""
    client = TestClient(app)

    resp = client.get("/api/v1/signals/RELIANCE/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert "top_contributing_features" in data
    assert "positive_impacts" in data
    assert "negative_impacts" in data
    assert "explanation_narrative" in data
    assert "model_confidence_pct" in data
    assert "model_agreement_pct" in data
