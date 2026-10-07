"""Feature Engineering and Technical Indicator modules for StockMind-AI.

Exports:
- TechnicalAnalysisEngine (unified orchestrator)
- Moving Average computations and Golden/Death Cross detection
- Oscillators (RSI, MACD, Stochastic, ROC, Momentum)
- Volatility (Bollinger Bands, ATR, Historical Volatility)
- Trend (ADX, +DI, -DI)
- Volume (OBV, VWAP, RVOL, Divergence)
- Price Action (Support/Resistance, Fibonacci levels, Candlestick patterns)
- AI Technical Score & Plain English Explanations
"""

from ai_engine.feature_engineering.engine import TechnicalAnalysisEngine
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

__all__ = [
    "TechnicalAnalysisEngine",
    "compute_sma",
    "compute_ema",
    "compute_all_moving_averages",
    "detect_ma_crossovers",
    "analyze_moving_averages",
    "compute_rsi",
    "compute_macd",
    "compute_stochastic_oscillator",
    "compute_roc",
    "compute_momentum",
    "analyze_oscillators",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_historical_volatility",
    "analyze_volatility",
    "compute_adx",
    "analyze_trend",
    "compute_obv",
    "compute_vwap",
    "compute_relative_volume",
    "detect_price_volume_divergence",
    "analyze_volume",
    "compute_support_resistance",
    "compute_fibonacci_levels",
    "detect_candlestick_patterns",
    "compute_ai_technical_score",
    "STATISTICAL_DISCLAIMER",
]
