"""Trend Analysis Feature Engineering Module.

Computes ADX (Average Directional Index), +DI, -DI, Trend Direction, and Trend Strength.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Compute Average Directional Index (ADX), +DI, and -DI."""
    if len(close) < 2:
        zero_series = pd.Series(0.0, index=close.index)
        return pd.DataFrame({"plus_di": zero_series, "minus_di": zero_series, "adx": zero_series})

    # 1. Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=close.index)
    minus_dm_series = pd.Series(minus_dm, index=close.index)

    # 2. True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 3. Smoothed Series (Wilder's alpha = 1 / period)
    smoothed_tr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm_series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm_series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # 4. Directional Indicators (+DI, -DI)
    tr_clean = smoothed_tr.replace(0.0, np.nan)
    plus_di = (100.0 * smoothed_plus_dm / tr_clean).fillna(0.0)
    minus_di = (100.0 * smoothed_minus_dm / tr_clean).fillna(0.0)

    # 5. Directional Index (DX) & ADX
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().fillna(0.0)

    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": adx,
        },
        index=close.index,
    )


def analyze_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive directional and trend strength evaluation."""
    if df.empty or "close" not in df.columns:
        return {}

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    adx_df = compute_adx(high, low, close, period=14)
    latest_adx = float(adx_df["adx"].iloc[-1])
    latest_plus_di = float(adx_df["plus_di"].iloc[-1])
    latest_minus_di = float(adx_df["minus_di"].iloc[-1])

    # Trend Direction based on DI and price moving averages
    sma50 = close.rolling(50, min_periods=1).mean().iloc[-1]
    latest_close = close.iloc[-1]

    if latest_plus_di > latest_minus_di and latest_close >= sma50:
        trend_direction = "UPTREND"
    elif latest_minus_di > latest_plus_di and latest_close <= sma50:
        trend_direction = "DOWNTREND"
    elif latest_plus_di > latest_minus_di:
        trend_direction = "WEAK_UPTREND"
    elif latest_minus_di > latest_plus_di:
        trend_direction = "WEAK_DOWNTREND"
    else:
        trend_direction = "SIDEWAYS"

    # Trend Strength based on ADX levels
    if latest_adx >= 35.0:
        trend_strength = "VERY_STRONG"
    elif latest_adx >= 25.0:
        trend_strength = "STRONG"
    elif latest_adx >= 20.0:
        trend_strength = "MODERATE"
    else:
        trend_strength = "WEAK_OR_CONSOLIDATING"

    return {
        "adx": round(latest_adx, 2),
        "plus_di": round(latest_plus_di, 2),
        "minus_di": round(latest_minus_di, 2),
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
        "is_trending": bool(latest_adx >= 25.0),
    }
