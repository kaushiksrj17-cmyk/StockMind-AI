"""Oscillators & Momentum Feature Engineering Module.

Computes RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence),
Stochastic Oscillator (%K, %D), Rate of Change (ROC), and Momentum indicators.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing technique.

    Args:
        series: Closing price series.
        period: RSI period (standard: 14).

    Returns:
        Series of RSI values bounded between 0 and 100.
    """
    if len(series) < 2:
        return pd.Series(50.0, index=series.index)

    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    # Wilder's smoothing alpha = 1 / period
    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fill edge cases (zero loss -> 100, zero gain -> 0)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    # Warmup fill with neutral 50
    return rsi.bfill().fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Compute MACD Line, Signal Line, and MACD Histogram."""
    if len(series) == 0:
        return pd.DataFrame(columns=["macd_line", "macd_signal", "macd_histogram"])

    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
        },
        index=series.index,
    )


def compute_stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
) -> pd.DataFrame:
    """Compute Slow Stochastic Oscillator %K and %D."""
    if len(close) < k_period:
        default_val = pd.Series(50.0, index=close.index)
        return pd.DataFrame({"stoch_k": default_val, "stoch_d": default_val})

    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()

    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    fast_k = 100.0 * (close - lowest_low) / denom
    fast_k = fast_k.fillna(50.0).clip(0.0, 100.0)

    # Smooth into slow %K and %D
    slow_k = fast_k.rolling(window=slowing, min_periods=1).mean()
    slow_d = slow_k.rolling(window=d_period, min_periods=1).mean()

    return pd.DataFrame({"stoch_k": slow_k, "stoch_d": slow_d}, index=close.index)


def compute_roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Compute Price Rate of Change (ROC) in percentage."""
    if len(series) <= period:
        return pd.Series(0.0, index=series.index)
    shifted = series.shift(period).replace(0.0, np.nan)
    roc = ((series - shifted) / shifted) * 100.0
    return roc.fillna(0.0)


def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Compute Absolute Momentum (Price - Price n bars ago)."""
    if len(series) <= period:
        return pd.Series(0.0, index=series.index)
    mom = series - series.shift(period)
    return mom.fillna(0.0)


def analyze_oscillators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate and interpret all oscillator metrics on the provided OHLCV DataFrame."""
    if df.empty or "close" not in df.columns:
        return {}

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    rsi_series = compute_rsi(close, period=14)
    macd_df = compute_macd(close, fast_period=12, slow_period=26, signal_period=9)
    stoch_df = compute_stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    roc_series = compute_roc(close, period=12)
    mom_series = compute_momentum(close, period=10)

    latest_rsi = float(rsi_series.iloc[-1])
    latest_macd = float(macd_df["macd_line"].iloc[-1])
    latest_signal = float(macd_df["macd_signal"].iloc[-1])
    latest_hist = float(macd_df["macd_histogram"].iloc[-1])
    prev_hist = float(macd_df["macd_histogram"].iloc[-2]) if len(macd_df) > 1 else latest_hist

    latest_stoch_k = float(stoch_df["stoch_k"].iloc[-1])
    latest_stoch_d = float(stoch_df["stoch_d"].iloc[-1])
    latest_roc = float(roc_series.iloc[-1])
    latest_mom = float(mom_series.iloc[-1])

    # Interpretations
    # RSI Condition
    if latest_rsi >= 70.0:
        rsi_condition = "OVERBOUGHT"
    elif latest_rsi <= 30.0:
        rsi_condition = "OVERSOLD"
    elif latest_rsi >= 55.0:
        rsi_condition = "BULLISH_EXPANSION"
    elif latest_rsi <= 45.0:
        rsi_condition = "BEARISH_CONTRACTION"
    else:
        rsi_condition = "NEUTRAL"

    # MACD Condition
    macd_bullish = latest_macd > latest_signal
    hist_accelerating = latest_hist > prev_hist
    if macd_bullish and hist_accelerating:
        macd_condition = "STRONG_BULLISH"
    elif macd_bullish:
        macd_condition = "BULLISH_DECELERATING"
    elif not macd_bullish and not hist_accelerating:
        macd_condition = "STRONG_BEARISH"
    else:
        macd_condition = "BEARISH_DECELERATING"

    # Stochastic Condition
    if latest_stoch_k >= 80.0 and latest_stoch_d >= 80.0:
        stoch_condition = "OVERBOUGHT"
    elif latest_stoch_k <= 20.0 and latest_stoch_d <= 20.0:
        stoch_condition = "OVERSOLD"
    elif latest_stoch_k > latest_stoch_d:
        stoch_condition = "BULLISH_CROSS"
    else:
        stoch_condition = "BEARISH_CROSS"

    # Oscillator composite score (-5 to +5)
    osc_score = 0
    if 45.0 <= latest_rsi <= 65.0:
        osc_score += 1
    elif latest_rsi > 65.0 and latest_rsi < 75.0:
        osc_score += 1
    elif latest_rsi <= 30.0:
        osc_score += 1  # Mean reversion opportunity
    elif latest_rsi >= 80.0:
        osc_score -= 1  # Overextended

    if macd_bullish:
        osc_score += 1
    else:
        osc_score -= 1
    if hist_accelerating:
        osc_score += 1
    else:
        osc_score -= 1

    if latest_stoch_k > latest_stoch_d:
        osc_score += 1
    else:
        osc_score -= 1

    if latest_roc > 0:
        osc_score += 1
    else:
        osc_score -= 1

    return {
        "rsi": round(latest_rsi, 2),
        "rsi_condition": rsi_condition,
        "macd": {
            "macd_line": round(latest_macd, 2),
            "signal_line": round(latest_signal, 2),
            "histogram": round(latest_hist, 2),
            "condition": macd_condition,
            "is_bullish": macd_bullish,
            "is_accelerating": hist_accelerating,
        },
        "stochastic": {
            "stoch_k": round(latest_stoch_k, 2),
            "stoch_d": round(latest_stoch_d, 2),
            "condition": stoch_condition,
        },
        "roc_12": round(latest_roc, 2),
        "momentum_10": round(latest_mom, 2),
        "oscillator_score": osc_score,
    }
