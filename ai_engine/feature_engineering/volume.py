"""Volume Analysis & Flow Feature Engineering Module.

Computes On-Balance Volume (OBV), Volume-Weighted Average Price (VWAP),
Volume Trends (RVOL, Surge), and Price-Volume Divergence detection.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV)."""
    if len(close) < 2:
        return pd.Series(0.0, index=close.index)

    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    return obv


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute Volume-Weighted Average Price (VWAP)."""
    if len(close) == 0:
        return pd.Series(dtype=float)

    typical_price = (high + low + close) / 3.0
    cum_vol_price = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0.0, np.nan)

    vwap = cum_vol_price / cum_vol
    return vwap.bfill().fillna(close)


def compute_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute Relative Volume (RVOL) compared to 20-period average."""
    vol_sma = volume.rolling(window=period, min_periods=1).mean().replace(0.0, np.nan)
    rvol = volume / vol_sma
    return rvol.fillna(1.0)


def detect_price_volume_divergence(
    close: pd.Series,
    obv: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> Dict[str, Any]:
    """Detect whether price and volume/OBV are diverging."""
    if len(close) < window:
        return {
            "has_divergence": False,
            "divergence_type": "NONE",
            "description": "Insufficient history for divergence evaluation.",
        }

    recent_close = close.tail(window)
    recent_obv = obv.tail(window)

    # Compute slopes over the window
    x = np.arange(len(recent_close))
    price_slope = np.polyfit(x, recent_close, 1)[0]
    obv_slope = np.polyfit(x, recent_obv, 1)[0]

    # Normalize slopes by standard deviations
    price_std = recent_close.std() or 1.0
    obv_std = recent_obv.std() or 1.0

    norm_price_slope = price_slope / price_std
    norm_obv_slope = obv_slope / obv_std

    # Bearish Divergence: Price trending upwards while OBV is trending downwards
    if norm_price_slope > 0.05 and norm_obv_slope < -0.05:
        return {
            "has_divergence": True,
            "divergence_type": "BEARISH_DIVERGENCE",
            "description": "Price rising with declining volume/OBV, signaling weak buying conviction (potential exhaustion).",
        }
    # Bullish Divergence: Price trending downwards while OBV is rising (accumulation)
    elif norm_price_slope < -0.05 and norm_obv_slope > 0.05:
        return {
            "has_divergence": True,
            "divergence_type": "BULLISH_DIVERGENCE",
            "description": "Price declining while volume/OBV is rising, indicating institutional accumulation into weakness.",
        }

    return {
        "has_divergence": False,
        "divergence_type": "CONFIRMING",
        "description": "Volume flow confirms prevailing price trajectory.",
    }


def analyze_volume(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive volume profile and order flow evaluation."""
    if df.empty or "close" not in df.columns or "volume" not in df.columns:
        return {}

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    volume = df["volume"]

    obv_series = compute_obv(close, volume)
    vwap_series = compute_vwap(high, low, close, volume)
    rvol_series = compute_relative_volume(volume, period=20)
    divergence = detect_price_volume_divergence(close, obv_series, volume, window=14)

    latest_close = float(close.iloc[-1])
    latest_vol = float(volume.iloc[-1])
    latest_obv = float(obv_series.iloc[-1])
    latest_vwap = float(vwap_series.iloc[-1])
    latest_rvol = float(rvol_series.iloc[-1])

    is_above_vwap = latest_close >= latest_vwap
    is_volume_surge = latest_rvol >= 1.5

    return {
        "volume": int(latest_vol),
        "rvol_20": round(latest_rvol, 2),
        "is_volume_surge": is_volume_surge,
        "vwap": round(latest_vwap, 2),
        "price_vs_vwap_pct": round(((latest_close - latest_vwap) / latest_vwap) * 100, 2) if latest_vwap else 0.0,
        "is_above_vwap": is_above_vwap,
        "obv": int(latest_obv),
        "divergence": divergence,
    }
