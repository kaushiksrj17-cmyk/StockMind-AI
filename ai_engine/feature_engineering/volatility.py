"""Volatility Feature Engineering Module.

Computes Bollinger Bands, Average True Range (ATR), and Realized/Historical Volatility metrics.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands (Upper, Middle, Lower, %B, and Bandwidth)."""
    if len(series) == 0:
        return pd.DataFrame(columns=["bb_middle", "bb_upper", "bb_lower", "bb_pct_b", "bb_bandwidth"])

    middle = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std().fillna(0.0)

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    band_range = (upper - lower).replace(0.0, np.nan)
    pct_b = (series - lower) / band_range
    pct_b = pct_b.fillna(0.5)

    bandwidth = ((upper - lower) / middle.replace(0.0, np.nan)) * 100.0
    bandwidth = bandwidth.fillna(0.0)

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        },
        index=series.index,
    )


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR) using Wilder's smoothing."""
    if len(close) < 2:
        return pd.Series(0.0, index=close.index)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing alpha = 1 / period
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.bfill().fillna(0.0)


def compute_historical_volatility(
    close: pd.Series,
    window: int = 20,
    trading_days: int = 252,
) -> pd.Series:
    """Compute Annualized Realized / Historical Volatility in percentage."""
    if len(close) < 2:
        return pd.Series(0.0, index=close.index)

    log_returns = np.log(close / close.shift(1)).fillna(0.0)
    rolling_std = log_returns.rolling(window=window, min_periods=2).std().fillna(0.0)
    annualized_vol = rolling_std * np.sqrt(trading_days) * 100.0
    return annualized_vol.bfill().fillna(0.0)


def analyze_volatility(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive volatility profiling on OHLCV data."""
    if df.empty or "close" not in df.columns:
        return {}

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    bb_df = compute_bollinger_bands(close, period=20, num_std=2.0)
    atr_series = compute_atr(high, low, close, period=14)
    vol_series = compute_historical_volatility(close, window=20)

    latest_close = float(close.iloc[-1])
    bb_upper = float(bb_df["bb_upper"].iloc[-1])
    bb_middle = float(bb_df["bb_middle"].iloc[-1])
    bb_lower = float(bb_df["bb_lower"].iloc[-1])
    bb_pct_b = float(bb_df["bb_pct_b"].iloc[-1])
    bb_bandwidth = float(bb_df["bb_bandwidth"].iloc[-1])
    latest_atr = float(atr_series.iloc[-1])
    latest_vol = float(vol_series.iloc[-1])

    # Squeeze detection: bandwidth in lower 20th percentile of its 50-day range
    recent_bw = bb_df["bb_bandwidth"].tail(50)
    is_squeeze = bool(bb_bandwidth <= recent_bw.quantile(0.20)) if len(recent_bw) >= 10 else False

    # Position evaluation
    if latest_close >= bb_upper:
        bb_position = "ABOVE_UPPER_BAND"
    elif latest_close <= bb_lower:
        bb_position = "BELOW_LOWER_BAND"
    elif latest_close > bb_middle:
        bb_position = "UPPER_CHANNEL"
    else:
        bb_position = "LOWER_CHANNEL"

    atr_pct = (latest_atr / latest_close) * 100.0 if latest_close else 0.0

    return {
        "bollinger_bands": {
            "upper": round(bb_upper, 2),
            "middle": round(bb_middle, 2),
            "lower": round(bb_lower, 2),
            "pct_b": round(bb_pct_b, 4),
            "bandwidth_pct": round(bb_bandwidth, 2),
            "position": bb_position,
            "is_squeeze": is_squeeze,
        },
        "atr": round(latest_atr, 2),
        "atr_pct_of_price": round(atr_pct, 2),
        "annualized_volatility_pct": round(latest_vol, 2),
    }
