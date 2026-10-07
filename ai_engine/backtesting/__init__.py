"""Quantitative Backtesting Engine module exports for StockMind-AI."""

from ai_engine.backtesting.backtest_engine import (
    BACKTEST_DISCLAIMER,
    BacktestEngine,
    BacktestResult,
    TradeRecord,
    backtest_engine,
)
from ai_engine.backtesting.strategies import (
    BaseStrategy,
    BollingerBreakoutStrategy,
    MACDStrategy,
    MovingAverageCrossStrategy,
    MultiFactorConsensusStrategy,
    RSIMeanReversionStrategy,
)

__all__ = [
    "BACKTEST_DISCLAIMER",
    "BacktestEngine",
    "BacktestResult",
    "BaseStrategy",
    "BollingerBreakoutStrategy",
    "MACDStrategy",
    "MovingAverageCrossStrategy",
    "MultiFactorConsensusStrategy",
    "RSIMeanReversionStrategy",
    "TradeRecord",
    "backtest_engine",
]
