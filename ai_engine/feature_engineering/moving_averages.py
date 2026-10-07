"""Moving Averages Feature Engineering Module.

Computes Simple Moving Averages (SMA 20, 50, 100, 200), Exponential Moving Averages (EMA),
and detects Golden Cross / Death Cross crossover signals.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Compute Simple Moving Average for a specified lookback period."""
    if len(series) == 0:
        return pd.Series(dtype=float)
    return series.rolling(window=period, min_periods=1).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average using standard smoothing alpha = 2 / (period + 1)."""
    if len(series) == 0:
        return pd.Series(dtype=float)
    return series.ewm(span=period, adjust=False).mean()


def compute_all_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard moving averages (SMA 20, 50, 100, 200, EMA 9, 21, 50, 200).

    Args:
        df: DataFrame containing at least a 'close' column.

    Returns:
        DataFrame with added moving average columns.
    """
    out = df.copy()
    close = out["close"]

    # Simple Moving Averages
    out["sma_20"] = compute_sma(close, 20)
    out["sma_50"] = compute_sma(close, 50)
    out["sma_100"] = compute_sma(close, 100)
    out["sma_200"] = compute_sma(close, 200)

    # Exponential Moving Averages
    out["ema_9"] = compute_ema(close, 9)
    out["ema_21"] = compute_ema(close, 21)
    out["ema_50"] = compute_ema(close, 50)
    out["ema_200"] = compute_ema(close, 200)

    return out


def detect_ma_crossovers(
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    lookback: int = 5,
) -> Dict[str, Any]:
    """Detect whether a bullish cross (Golden Cross) or bearish cross (Death Cross) occurred.

    Args:
        fast_ma: Fast moving average series (e.g. SMA 50 or SMA 20).
        slow_ma: Slow moving average series (e.g. SMA 200 or SMA 50).
        lookback: Number of recent bars to inspect for a crossover event.

    Returns:
        Dictionary with crossover flags, current status, and bar index.
    """
    if len(fast_ma) < 2 or len(slow_ma) < 2:
        return {
            "golden_cross": False,
            "death_cross": False,
            "fast_above_slow": False,
            "crossover_bar_ago": None,
        }

    # Inspect current alignment
    fast_curr = float(fast_ma.iloc[-1])
    slow_curr = float(slow_ma.iloc[-1])
    fast_above_slow = fast_curr > slow_curr

    # Check for crossover within lookback window
    window = min(lookback, len(fast_ma) - 1)
    golden_cross = False
    death_cross = False
    crossover_bar_ago = None

    for i in range(1, window + 1):
        prev_f = fast_ma.iloc[-i - 1]
        prev_s = slow_ma.iloc[-i - 1]
        curr_f = fast_ma.iloc[-i]
        curr_s = slow_ma.iloc[-i]

        if prev_f <= prev_s and curr_f > curr_s:
            golden_cross = True
            crossover_bar_ago = i - 1
            break
        elif prev_f >= prev_s and curr_f < curr_s:
            death_cross = True
            crossover_bar_ago = i - 1
            break

    return {
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "fast_above_slow": fast_above_slow,
        "crossover_bar_ago": crossover_bar_ago,
    }


def analyze_moving_averages(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive moving average breakdown for the latest bar."""
    if df.empty or "close" not in df.columns:
        return {}

    ma_df = compute_all_moving_averages(df)
    latest_close = float(df["close"].iloc[-1])

    sma20 = float(ma_df["sma_20"].iloc[-1])
    sma50 = float(ma_df["sma_50"].iloc[-1])
    sma100 = float(ma_df["sma_100"].iloc[-1])
    sma200 = float(ma_df["sma_200"].iloc[-1])

    ema9 = float(ma_df["ema_9"].iloc[-1])
    ema21 = float(ma_df["ema_21"].iloc[-1])
    ema50 = float(ma_df["ema_50"].iloc[-1])
    ema200 = float(ma_df["ema_200"].iloc[-1])

    # 50 / 200 Classical Golden / Death Cross
    classical_cross = detect_ma_crossovers(ma_df["sma_50"], ma_df["sma_200"])
    # 20 / 50 Tactical Short-Term Cross
    tactical_cross = detect_ma_crossovers(ma_df["sma_20"], ma_df["sma_50"])

    # Stack alignment: Bullish stack is Price > SMA20 > SMA50 > SMA200
    is_bullish_stack = latest_close > sma20 > sma50 > sma200
    is_bearish_stack = latest_close < sma20 < sma50 < sma200

    bias_score = 0
    if latest_close > sma20:
        bias_score += 1
    if latest_close > sma50:
        bias_score += 1
    if latest_close > sma100:
        bias_score += 1
    if latest_close > sma200:
        bias_score += 1
    if classical_cross["fast_above_slow"]:
        bias_score += 1
    if classical_cross["golden_cross"]:
        bias_score += 2
    if classical_cross["death_cross"]:
        bias_score -= 2

    return {
        "values": {
            "sma_20": round(sma20, 2),
            "sma_50": round(sma50, 2),
            "sma_100": round(sma100, 2),
            "sma_200": round(sma200, 2),
            "ema_9": round(ema9, 2),
            "ema_21": round(ema21, 2),
            "ema_50": round(ema50, 2),
            "ema_200": round(ema200, 2),
        },
        "price_vs_sma20_pct": round(((latest_close - sma20) / sma20) * 100, 2) if sma20 else 0.0,
        "price_vs_sma50_pct": round(((latest_close - sma50) / sma50) * 100, 2) if sma50 else 0.0,
        "price_vs_sma200_pct": round(((latest_close - sma200) / sma200) * 100, 2) if sma200 else 0.0,
        "golden_cross": classical_cross["golden_cross"],
        "death_cross": classical_cross["death_cross"],
        "classical_50_200_status": "BULLISH_ABOVE" if classical_cross["fast_above_slow"] else "BEARISH_BELOW",
        "tactical_20_50_status": "BULLISH_ABOVE" if tactical_cross["fast_above_slow"] else "BEARISH_BELOW",
        "is_bullish_stack": is_bullish_stack,
        "is_bearish_stack": is_bearish_stack,
        "bias_score": bias_score,  # -2 to +7
    }
