"""Time-Series Safe Machine Learning Preprocessing Pipeline.

Strictly prevents:
- Data leakage: All scalers and normalizers are fit ONLY on historical training splits.
- Look-ahead bias: At bar t, features use information strictly at or before t.
- Random shuffling: Only chronological train-test and walk-forward cross-validation splits.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ai_engine.feature_engineering.moving_averages import (
    compute_sma,
    compute_ema,
)
from ai_engine.feature_engineering.oscillators import (
    compute_rsi,
    compute_macd,
    compute_stochastic_oscillator,
    compute_roc,
    compute_momentum,
)
from ai_engine.feature_engineering.volatility import (
    compute_bollinger_bands,
    compute_atr,
    compute_historical_volatility,
)
from ai_engine.feature_engineering.trend import compute_adx
from ai_engine.feature_engineering.volume import (
    compute_obv,
    compute_vwap,
    compute_relative_volume,
)


class TimeSeriesFeaturePipeline:
    """Generates stationary technical and statistical features for predictive ML models."""

    def __init__(self, forward_horizon: int = 1) -> None:
        self.forward_horizon = forward_horizon
        self.feature_names: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """Convenience method returning (X, y_direction, y_return, X_latest)."""
        X, y_dir, y_ret, _, X_latest = self.extract_features_and_targets(df)
        return X, y_dir, y_ret, X_latest

    def extract_features_and_targets(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """Extract stationary time-series features and aligned forward targets.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            Tuple of:
            - X: Feature matrix for all labeled historical bars.
            - y_direction: Next-period direction (1 for Up, 0 for Down).
            - y_return: Next-period fractional return ((Close[t+1] - Close[t]) / Close[t]).
            - y_range: Next-period fractional high-low range ((High[t+1] - Low[t+1]) / Close[t]).
            - X_latest: Single-row feature DataFrame for bar T (used for out-of-sample forward inference).
        """
        data = df.copy()
        data.columns = [c.lower() for c in data.columns]

        # Ensure required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in data.columns:
                raise ValueError(f"Required OHLCV column '{col}' missing from DataFrame.")
            data[col] = pd.to_numeric(data[col], errors="coerce")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_p = data["open"]
        vol = data["volume"]

        features = pd.DataFrame(index=data.index)

        # -------------------------------------------------------------
        # 1. Stationary Returns & Price Action Lags (Past only)
        # -------------------------------------------------------------
        features["ret_1d"] = close.pct_change(1)
        features["ret_2d"] = close.pct_change(2)
        features["ret_5d"] = close.pct_change(5)
        features["ret_10d"] = close.pct_change(10)
        features["ret_20d"] = close.pct_change(20)

        # Intraday shape (High-Low spread, Close-Open spread)
        features["hl_spread_pct"] = (high - low) / close
        features["co_spread_pct"] = (close - open_p) / open_p.replace(0.0, np.nan)
        features["upper_shadow_pct"] = (high - np.maximum(open_p, close)) / close
        features["lower_shadow_pct"] = (np.minimum(open_p, close) - low) / close

        # -------------------------------------------------------------
        # 2. Moving Average Distances (% deviation from trend)
        # -------------------------------------------------------------
        sma20 = compute_sma(close, 20)
        sma50 = compute_sma(close, 50)
        sma200 = compute_sma(close, 200)
        ema21 = compute_ema(close, 21)

        features["dist_sma20"] = (close - sma20) / sma20.replace(0.0, np.nan)
        features["dist_sma50"] = (close - sma50) / sma50.replace(0.0, np.nan)
        features["dist_sma200"] = (close - sma200) / sma200.replace(0.0, np.nan)
        features["dist_ema21"] = (close - ema21) / ema21.replace(0.0, np.nan)
        features["sma50_sma200_spread"] = (sma50 - sma200) / sma200.replace(0.0, np.nan)

        # -------------------------------------------------------------
        # 3. Oscillators (Wilder's RSI, MACD, Stochastic, ROC, Momentum)
        # -------------------------------------------------------------
        features["rsi_14"] = compute_rsi(close, 14) / 100.0  # normalize [0, 1]
        features["rsi_diff_5"] = features["rsi_14"].diff(5)

        macd_df = compute_macd(close, 12, 26, 9)
        features["macd_line_pct"] = macd_df["macd_line"] / close
        features["macd_signal_pct"] = macd_df["macd_signal"] / close
        features["macd_hist_pct"] = macd_df["macd_histogram"] / close

        stoch_df = compute_stochastic_oscillator(high, low, close, 14, 3)
        features["stoch_k"] = stoch_df["stoch_k"] / 100.0
        features["stoch_d"] = stoch_df["stoch_d"] / 100.0
        features["stoch_diff"] = features["stoch_k"] - features["stoch_d"]

        features["roc_12"] = compute_roc(close, 12) / 100.0
        features["momentum_10_pct"] = compute_momentum(close, 10) / close

        # -------------------------------------------------------------
        # 4. Volatility (Bollinger Bands, ATR, Realized Volatility)
        # -------------------------------------------------------------
        bb_df = compute_bollinger_bands(close, 20, 2.0)
        features["bb_pct_b"] = bb_df["bb_pct_b"]
        features["bb_bandwidth"] = bb_df["bb_bandwidth"] / 100.0

        atr_series = compute_atr(high, low, close, 14)
        features["atr_pct"] = atr_series / close
        features["realized_vol_20"] = compute_historical_volatility(close, 20) / 100.0

        # -------------------------------------------------------------
        # 5. Trend & Direction (ADX, +DI, -DI)
        # -------------------------------------------------------------
        adx_df = compute_adx(high, low, close, 14)
        features["adx"] = adx_df["adx"] / 100.0
        features["plus_di"] = adx_df["plus_di"] / 100.0
        features["minus_di"] = adx_df["minus_di"] / 100.0
        features["di_diff"] = features["plus_di"] - features["minus_di"]

        # -------------------------------------------------------------
        # 6. Volume Dynamics & Order Flow
        # -------------------------------------------------------------
        features["rvol_20"] = compute_relative_volume(vol, 20)
        vwap_series = compute_vwap(high, low, close, vol)
        features["dist_vwap"] = (close - vwap_series) / vwap_series.replace(0.0, np.nan)

        obv_series = compute_obv(close, vol)
        obv_norm = obv_series.pct_change(10).fillna(0.0)
        features["obv_momentum_10"] = obv_norm

        self.feature_names = list(features.columns)

        # -------------------------------------------------------------
        # 7. Strictly Aligned Forward Targets (Shifted by -forward_horizon)
        # -------------------------------------------------------------
        future_close = close.shift(-self.forward_horizon)
        future_high = high.shift(-self.forward_horizon)
        future_low = low.shift(-self.forward_horizon)

        # Direction target: 1 for UP, 0 for DOWN
        target_direction = (future_close > close).astype(int)

        # Return target: fractional price return to next period close
        target_return = (future_close - close) / close

        # Range target: fractional volatility range of next period
        target_range = (future_high - future_low) / close

        # Separate the latest complete bar (for forward out-of-sample inference)
        latest_idx = features.index[-1]
        X_latest = features.loc[[latest_idx]].copy().fillna(0.0)

        # Drop NaN values caused by rolling windows and the last row (which has no target)
        valid_mask = ~(features.isna().any(axis=1) | target_direction.isna())
        # The last row must be excluded from training set because target is NaN
        valid_mask.iloc[-self.forward_horizon :] = False

        X_labeled = features[valid_mask].copy().fillna(0.0)
        y_dir = target_direction[valid_mask].copy()
        y_ret = target_return[valid_mask].copy()
        y_rng = target_range[valid_mask].copy()

        return X_labeled, y_dir, y_ret, y_rng, X_latest


def chronological_train_test_split(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform strictly chronological train-test split without random shuffling.

    Ensures the training set strictly precedes the testing set in time.
    """
    if len(X) < 10:
        raise ValueError("Insufficient data points for time-series train-test split.")

    split_idx = int(len(X) * (1.0 - test_size))
    # Ensure at least 5 test points
    split_idx = max(5, min(split_idx, len(X) - 5))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()

    if isinstance(y, pd.Series):
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()
    else:
        y_train = pd.Series(y[:split_idx], index=X_train.index)
        y_test = pd.Series(y[split_idx:], index=X_test.index)

    return X_train, X_test, y_train, y_test


class TimeSeriesScaler:
    """Robust feature scaler that prevents look-ahead bias and data leakage.

    Fitted strictly on training data; preserves DataFrame column names.
    """

    def __init__(self) -> None:
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.columns: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "TimeSeriesScaler":
        """Compute mean and std strictly on the training set."""
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        else:
            X_df = X
        self.columns = list(X_df.columns)
        for col in self.columns:
            m = float(X_df[col].mean())
            s = float(X_df[col].std())
            self.means[col] = m
            self.stds[col] = s if s > 1e-8 else 1.0
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Standardize using training set parameters without leakage."""
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        out = X_df.copy()
        for col in self.columns:
            if col in out.columns:
                m = self.means.get(col, 0.0)
                s = self.stds.get(col, 1.0)
                out[col] = (out[col] - m) / s
        out = out.fillna(0.0)
        return out.to_numpy() if is_numpy else out

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Fit strictly on X and return transformed matrix."""
        return self.fit(X).transform(X)
