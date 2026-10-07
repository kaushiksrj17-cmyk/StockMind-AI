"""Quantitative Trading Strategy Implementations for StockMind-AI Backtesting Engine.

Strictly avoids look-ahead bias and future data leakage by calculating signals
only using past-and-present bar values.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for quantitative backtesting strategies."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate chronological trade position signals: 1 for Long, 0 for Flat.
        
        Signal at row index t is generated using ONLY information up to bar t.
        """
        pass


class MovingAverageCrossStrategy(BaseStrategy):
    """Moving Average Crossover Strategy (e.g. Golden Cross / Death Cross)."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50, use_ema: bool = True) -> None:
        super().__init__(f"{'EMA' if use_ema else 'SMA'} Crossover ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        if self.use_ema:
            fast_ma = close.ewm(span=self.fast_period, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow_period, adjust=False).mean()
        else:
            fast_ma = close.rolling(window=self.fast_period, min_periods=self.fast_period).mean()
            slow_ma = close.rolling(window=self.slow_period, min_periods=self.slow_period).mean()

        signal = pd.Series(0, index=df.index, dtype=int)
        # Position is Long when Fast MA is above Slow MA
        condition = fast_ma > slow_ma
        signal[condition] = 1
        return signal


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy (Buy Oversold, Exit Overbought)."""

    def __init__(self, rsi_period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        super().__init__(f"RSI Mean Reversion ({oversold}/{overbought})")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def _calculate_rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period).mean()

        rs = avg_gain / np.maximum(avg_loss, 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._calculate_rsi(df["close"].astype(float))
        signal = pd.Series(0, index=df.index, dtype=int)

        in_position = False
        for i in range(len(df)):
            curr_rsi = rsi.iloc[i]
            if not in_position and curr_rsi <= self.oversold:
                in_position = True
            elif in_position and curr_rsi >= self.overbought:
                in_position = False
            signal.iloc[i] = 1 if in_position else 0

        return signal


class BollingerBreakoutStrategy(BaseStrategy):
    """Bollinger Band Volatility Breakout with Middle Band Exit."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        super().__init__(f"Bollinger Band Breakout ({period}, {num_std}σ)")
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        sma = close.rolling(window=self.period, min_periods=self.period).mean()
        std = close.rolling(window=self.period, min_periods=self.period).std().fillna(0.0)
        upper = sma + self.num_std * std

        signal = pd.Series(0, index=df.index, dtype=int)
        in_position = False

        for i in range(len(df)):
            c = close.iloc[i]
            u = upper.iloc[i]
            m = sma.iloc[i]

            if not in_position and c > u:
                in_position = True
            elif in_position and c < m:
                in_position = False

            signal.iloc[i] = 1 if in_position else 0

        return signal


class MACDStrategy(BaseStrategy):
    """MACD Trend Following Strategy."""

    def __init__(self, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9) -> None:
        super().__init__(f"MACD Trend ({fast_span}/{slow_span}/{signal_span})")
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.signal_span = signal_span

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        ema_fast = close.ewm(span=self.fast_span, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_span, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_span, adjust=False).mean()

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[macd_line > signal_line] = 1
        return signal


class MultiFactorConsensusStrategy(BaseStrategy):
    """Multi-Factor AI Strategy (Trend + Momentum + Volume Confirmation)."""

    def __init__(self) -> None:
        super().__init__("Multi-Factor Quantitative Consensus")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, min_periods=14).mean()
        loss = -delta.clip(upper=0.0).ewm(alpha=1.0 / 14, min_periods=14).mean()
        rsi = 100.0 - (100.0 / (1.0 + (gain / np.maximum(loss, 1e-6))))

        vol_sma20 = volume.rolling(20, min_periods=5).mean().fillna(volume.iloc[0])

        trend_factor = (close > ema20) & (ema20 > ema50)
        momentum_factor = (rsi > 48.0) & (rsi < 72.0)
        volume_factor = volume >= (vol_sma20 * 0.85)

        # Composite consensus requires at least 2 of 3 factors plus trend
        signal = pd.Series(0, index=df.index, dtype=int)
        consensus = trend_factor & (momentum_factor | volume_factor)
        signal[consensus] = 1
        return signal
