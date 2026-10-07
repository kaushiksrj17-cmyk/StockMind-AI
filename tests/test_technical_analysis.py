"""Unit tests for Technical Analysis Engine, Indicators, Patterns, and REST APIs."""

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient

from ai_engine.feature_engineering.moving_averages import (
    compute_sma,
    compute_ema,
    compute_all_moving_averages,
    detect_ma_crossovers,
    analyze_moving_averages,
)
from ai_engine.feature_engineering.oscillators import (
    compute_rsi,
    compute_macd,
    compute_stochastic_oscillator,
    compute_roc,
    compute_momentum,
    analyze_oscillators,
)
from ai_engine.feature_engineering.volatility import (
    compute_bollinger_bands,
    compute_atr,
    compute_historical_volatility,
    analyze_volatility,
)
from ai_engine.feature_engineering.trend import (
    compute_adx,
    analyze_trend,
)
from ai_engine.feature_engineering.volume import (
    compute_obv,
    compute_vwap,
    compute_relative_volume,
    detect_price_volume_divergence,
    analyze_volume,
)
from ai_engine.feature_engineering.price_action import (
    compute_support_resistance,
    compute_fibonacci_levels,
    detect_candlestick_patterns,
)
from ai_engine.feature_engineering.technical_score import (
    compute_ai_technical_score,
    STATISTICAL_DISCLAIMER,
)
from ai_engine.feature_engineering.engine import TechnicalAnalysisEngine


@pytest.fixture
def sample_ohlcv_df():
    """Generate reproducible 120-bar synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    
    # Generate an upward-trending price series
    trend = np.linspace(100.0, 150.0, n)
    noise = np.random.normal(0, 1.5, n)
    close = trend + noise
    open_p = close - np.random.uniform(-1.0, 1.0, n)
    high = np.maximum(open_p, close) + np.random.uniform(0.5, 2.0, n)
    low = np.minimum(open_p, close) - np.random.uniform(0.5, 2.0, n)
    volume = np.random.uniform(500000, 2000000, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# =====================================================================
# 1. Moving Averages Tests
# =====================================================================

def test_sma_calculation():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma3 = compute_sma(series, 3)
    assert len(sma3) == 5
    # SMA of last 3: (30 + 40 + 50) / 3 = 40.0
    assert pytest.approx(sma3.iloc[-1], 0.01) == 40.0


def test_ema_calculation():
    series = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    ema = compute_ema(series, 3)
    assert len(ema) == 6
    # EMA should respond faster than SMA in an upward move
    sma = compute_sma(series, 3)
    assert ema.iloc[-1] >= sma.iloc[-1] - 0.5


def test_golden_cross_and_death_cross_detection():
    # Golden cross scenario: fast crosses from below to above slow
    fast = pd.Series([10.0, 12.0, 15.0, 18.0, 22.0])
    slow = pd.Series([15.0, 15.0, 16.0, 16.5, 17.0])
    cross = detect_ma_crossovers(fast, slow, lookback=4)
    assert cross["golden_cross"] is True
    assert cross["death_cross"] is False
    assert cross["fast_above_slow"] is True

    # Death cross scenario: fast crosses from above to below slow
    fast_bear = pd.Series([25.0, 22.0, 18.0, 14.0, 10.0])
    slow_bear = pd.Series([16.0, 16.0, 16.0, 16.0, 16.0])
    cross_bear = detect_ma_crossovers(fast_bear, slow_bear, lookback=4)
    assert cross_bear["death_cross"] is True
    assert cross_bear["golden_cross"] is False
    assert cross_bear["fast_above_slow"] is False


def test_analyze_moving_averages(sample_ohlcv_df):
    result = analyze_moving_averages(sample_ohlcv_df)
    assert "values" in result
    for key in ["sma_20", "sma_50", "sma_100", "sma_200", "ema_9", "ema_21", "ema_50", "ema_200"]:
        assert key in result["values"]
        assert result["values"][key] > 0
    assert "is_bullish_stack" in result
    assert "bias_score" in result


# =====================================================================
# 2. Oscillators & Momentum Tests
# =====================================================================

def test_rsi_bounds_and_dynamics():
    # Monotonically increasing close series -> RSI should be high (> 70)
    up_series = pd.Series(np.linspace(100, 200, 50))
    rsi_up = compute_rsi(up_series, period=14)
    assert len(rsi_up) == 50
    assert 0.0 <= rsi_up.iloc[-1] <= 100.0
    assert rsi_up.iloc[-1] > 70.0

    # Monotonically decreasing series -> RSI should be low (< 30)
    down_series = pd.Series(np.linspace(200, 100, 50))
    rsi_down = compute_rsi(down_series, period=14)
    assert rsi_down.iloc[-1] < 30.0


def test_macd_computation():
    close = pd.Series(np.linspace(100, 150, 40))
    macd_df = compute_macd(close, fast_period=12, slow_period=26, signal_period=9)
    assert "macd_line" in macd_df.columns
    assert "macd_signal" in macd_df.columns
    assert "macd_histogram" in macd_df.columns
    # Histogram should equal macd_line - macd_signal
    diff = macd_df["macd_line"] - macd_df["macd_signal"]
    assert np.allclose(macd_df["macd_histogram"], diff, atol=1e-5)


def test_stochastic_oscillator():
    high = pd.Series([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    low = pd.Series([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    close = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26])
    stoch = compute_stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    assert "stoch_k" in stoch.columns
    assert "stoch_d" in stoch.columns
    assert 0.0 <= stoch["stoch_k"].iloc[-1] <= 100.0
    assert 0.0 <= stoch["stoch_d"].iloc[-1] <= 100.0


def test_roc_and_momentum():
    series = pd.Series([100.0, 102.0, 105.0, 110.0, 115.0])
    roc = compute_roc(series, period=2)
    # (115 - 105) / 105 * 100 = ~9.52%
    assert pytest.approx(roc.iloc[-1], 0.1) == 9.52

    mom = compute_momentum(series, period=2)
    # 115 - 105 = 10.0
    assert pytest.approx(mom.iloc[-1], 0.1) == 10.0


def test_analyze_oscillators(sample_ohlcv_df):
    osc = analyze_oscillators(sample_ohlcv_df)
    assert "rsi" in osc
    assert "macd" in osc
    assert "stochastic" in osc
    assert "roc_12" in osc
    assert "momentum_10" in osc
    assert "oscillator_score" in osc


# =====================================================================
# 3. Volatility Tests
# =====================================================================

def test_bollinger_bands():
    series = pd.Series(np.linspace(100, 120, 30))
    bb = compute_bollinger_bands(series, period=20, num_std=2.0)
    assert "bb_upper" in bb.columns
    assert "bb_middle" in bb.columns
    assert "bb_lower" in bb.columns
    assert "bb_pct_b" in bb.columns
    assert "bb_bandwidth" in bb.columns

    last = bb.iloc[-1]
    assert last["bb_upper"] >= last["bb_middle"]
    assert last["bb_middle"] >= last["bb_lower"]


def test_atr():
    high = pd.Series([105.0, 108.0, 112.0, 115.0])
    low = pd.Series([98.0, 101.0, 104.0, 108.0])
    close = pd.Series([102.0, 106.0, 110.0, 114.0])
    atr = compute_atr(high, low, close, period=3)
    assert len(atr) == 4
    assert atr.iloc[-1] > 0.0


def test_historical_volatility():
    close = pd.Series([100.0, 102.0, 99.0, 103.0, 101.0, 104.0, 98.0, 105.0])
    vol = compute_historical_volatility(close, window=5)
    assert len(vol) == 8
    assert vol.iloc[-1] >= 0.0


def test_analyze_volatility(sample_ohlcv_df):
    res = analyze_volatility(sample_ohlcv_df)
    assert "bollinger_bands" in res
    assert "atr" in res
    assert "annualized_volatility_pct" in res


# =====================================================================
# 4. Trend Tests
# =====================================================================

def test_adx_and_trend(sample_ohlcv_df):
    high = sample_ohlcv_df["high"]
    low = sample_ohlcv_df["low"]
    close = sample_ohlcv_df["close"]
    adx_df = compute_adx(high, low, close, period=14)
    assert "plus_di" in adx_df.columns
    assert "minus_di" in adx_df.columns
    assert "adx" in adx_df.columns
    assert 0.0 <= adx_df["adx"].iloc[-1] <= 100.0

    trend = analyze_trend(sample_ohlcv_df)
    assert trend["trend_direction"] in ["UPTREND", "DOWNTREND", "WEAK_UPTREND", "WEAK_DOWNTREND", "SIDEWAYS"]
    assert trend["trend_strength"] in ["VERY_STRONG", "STRONG", "MODERATE", "WEAK_OR_CONSOLIDATING"]


# =====================================================================
# 5. Volume Tests
# =====================================================================

def test_obv_and_vwap(sample_ohlcv_df):
    close = sample_ohlcv_df["close"]
    vol = sample_ohlcv_df["volume"]
    high = sample_ohlcv_df["high"]
    low = sample_ohlcv_df["low"]

    obv = compute_obv(close, vol)
    assert len(obv) == len(close)

    vwap = compute_vwap(high, low, close, vol)
    assert len(vwap) == len(close)
    # VWAP should stay within the general price range
    assert vwap.iloc[-1] > low.min()
    assert vwap.iloc[-1] < high.max()


def test_price_volume_divergence():
    # Bearish divergence: price rising, OBV falling
    close = pd.Series(np.linspace(100, 150, 20))
    obv = pd.Series(np.linspace(1000000, 500000, 20))
    vol = pd.Series([10000] * 20)
    div = detect_price_volume_divergence(close, obv, vol, window=14)
    assert div["has_divergence"] is True
    assert div["divergence_type"] == "BEARISH_DIVERGENCE"


def test_analyze_volume(sample_ohlcv_df):
    vol_res = analyze_volume(sample_ohlcv_df)
    assert "volume" in vol_res
    assert "rvol_20" in vol_res
    assert "vwap" in vol_res
    assert "obv" in vol_res
    assert "divergence" in vol_res


# =====================================================================
# 6. Price Action, Fibonacci & Candlestick Patterns Tests
# =====================================================================

def test_support_and_resistance(sample_ohlcv_df):
    sr = compute_support_resistance(sample_ohlcv_df, window=5)
    assert "current_price" in sr
    assert "support_levels" in sr
    assert "resistance_levels" in sr
    assert "nearest_support" in sr
    assert "nearest_resistance" in sr
    assert sr["nearest_support"] <= sr["current_price"]
    assert sr["nearest_resistance"] >= sr["current_price"]
    assert "pivot_points" in sr


def test_fibonacci_levels(sample_ohlcv_df):
    fib = compute_fibonacci_levels(sample_ohlcv_df, lookback=60)
    assert "swing_high" in fib
    assert "swing_low" in fib
    assert "levels" in fib
    assert fib["swing_high"] >= fib["swing_low"]
    assert "61.8% (Golden)" in fib["levels"]
    assert "nearest_level" in fib


def test_candlestick_pattern_recognition():
    # Construct a clean Bullish Engulfing pattern:
    # Bar 0: Neutral green
    # Bar 1: Red candle (open 100, high 101, low 94, close 95)
    # Bar 2: Strong green candle engulfing Bar 1 (open 93, high 104, low 92, close 103)
    engulf_df = pd.DataFrame([
        {"open": 98.0, "high": 100.0, "low": 97.0, "close": 99.0},
        {"open": 100.0, "high": 101.0, "low": 94.0, "close": 95.0},
        {"open": 93.0, "high": 104.0, "low": 92.0, "close": 103.0},
    ])
    patterns = detect_candlestick_patterns(engulf_df, lookback_bars=2)
    names = [p["name"] for p in patterns]
    assert "Bullish Engulfing" in names

    # Construct a Doji pattern (open ~ close)
    doji_df = pd.DataFrame([
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0},
        {"open": 101.0, "high": 103.0, "low": 99.0, "close": 101.5},
        {"open": 101.5, "high": 110.0, "low": 90.0, "close": 101.6},  # body = 0.1, range = 20.0
    ])
    doji_patterns = detect_candlestick_patterns(doji_df, lookback_bars=2)
    doji_names = [p["name"] for p in doji_patterns]
    assert "Doji" in doji_names


# =====================================================================
# 7. AI Technical Score & Engine Tests
# =====================================================================

def test_ai_technical_score_bounds_and_structure(sample_ohlcv_df):
    engine = TechnicalAnalysisEngine()
    result = engine.analyze(sample_ohlcv_df, symbol="RELIANCE")

    assert "ai_technical_score" in result
    score_data = result["ai_technical_score"]
    score = score_data["score"]
    assert 0.0 <= score <= 100.0

    assert score_data["classification"] in [
        "STRONG_BULLISH",
        "BULLISH",
        "NEUTRAL",
        "BEARISH",
        "STRONG_BEARISH",
    ]
    assert "component_scores" in score_data
    for pillar in ["trend", "momentum", "volume", "volatility", "price_action"]:
        assert pillar in score_data["component_scores"]
        assert 0.0 <= score_data["component_scores"][pillar] <= 100.0

    # Ensure weights sum to 1.0
    weights = score_data["component_weights"]
    assert pytest.approx(sum(weights.values()), 0.001) == 1.0

    # Verify plain-English explanations
    assert len(score_data["indicator_explanations"]) >= 5
    assert len(score_data["score_explanation"]) > 20

    # Verify mandatory statistical disclaimer
    assert "do not guarantee future price movements" in score_data["disclaimer"].lower()
    assert result["disclaimer"] == STATISTICAL_DISCLAIMER


def test_engine_chart_data_series(sample_ohlcv_df):
    engine = TechnicalAnalysisEngine()
    res = engine.analyze(sample_ohlcv_df, symbol="TCS")
    chart_data = res["chart_data"]
    assert "dates" in chart_data
    assert "close" in chart_data
    assert "sma_20" in chart_data
    assert "sma_50" in chart_data
    assert "rsi" in chart_data
    assert "macd_line" in chart_data
    assert len(chart_data["close"]) == len(chart_data["dates"])


# =====================================================================
# 8. FastAPI Technical Analysis Endpoints Tests
# =====================================================================

@pytest.mark.asyncio
async def test_get_technical_analysis_api(client: AsyncClient):
    response = await client.get("/api/v1/technical/analysis/RELIANCE?timeframe=1d&limit=100")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "RELIANCE"
    assert "current_price" in data
    assert "ai_technical_score" in data
    assert 0.0 <= data["ai_technical_score"]["score"] <= 100.0
    assert "moving_averages" in data
    assert "oscillators" in data
    assert "volatility" in data
    assert "support_resistance" in data
    assert "fibonacci_levels" in data
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_get_technical_score_api(client: AsyncClient):
    response = await client.get("/api/v1/technical/score/TCS")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TCS"
    assert "ai_technical_score" in data
    assert "score" in data["ai_technical_score"]
    assert "classification" in data["ai_technical_score"]
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_get_technical_indicators_api(client: AsyncClient):
    response = await client.get("/api/v1/technical/indicators/INFY")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "INFY"
    assert "moving_averages" in data
    assert "oscillators" in data
    assert "volatility" in data
    assert "trend" in data
    assert "volume_profile" in data
    assert "disclaimer" in data
