"""Technical Analysis Engine Orchestrator.

Combines Moving Averages, Oscillators, Volatility, Trend, Volume,
Support/Resistance, Fibonacci levels, Candlestick Patterns, and the AI Technical Score
into a unified institutional market intelligence engine.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from ai_engine.feature_engineering.moving_averages import (
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


class TechnicalAnalysisEngine:
    """Institutional-grade Technical Analysis Engine for Indian Equities and Indices."""

    def __init__(self, default_lookback: int = 200) -> None:
        self.default_lookback = default_lookback

    def validate_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame, enforce required columns, and handle ordering."""
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        out = df.copy()

        # Handle column names (case-insensitive conversion)
        out.columns = [c.lower() for c in out.columns]

        if "close" not in out.columns:
            raise ValueError("Input DataFrame must contain a 'close' price column.")

        # If OHLC columns are missing, synthesize from close defensively
        if "open" not in out.columns:
            out["open"] = out["close"].shift(1).fillna(out["close"])
        if "high" not in out.columns:
            out["high"] = out[["open", "close"]].max(axis=1)
        if "low" not in out.columns:
            out["low"] = out[["open", "close"]].min(axis=1)
        if "volume" not in out.columns:
            out["volume"] = 1000000.0

        # Enforce float types
        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        # Sort chronologically if date/timestamp column exists
        for dt_col in ["timestamp", "date", "datetime", "time"]:
            if dt_col in out.columns:
                out[dt_col] = pd.to_datetime(out[dt_col])
                out = out.sort_values(by=dt_col)
                break

        return out.reset_index(drop=True)

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY50",
        timeframe: str = "1D",
    ) -> Dict[str, Any]:
        """Perform comprehensive technical analysis across all indicators.

        Args:
            df: Historical OHLCV DataFrame.
            symbol: Ticker symbol (e.g. 'RELIANCE.NS', 'NIFTY50').
            timeframe: Analysis timeframe ('1D', '1H', etc.).

        Returns:
            Dictionary containing indicators, signals, patterns, score, and explanations.
        """
        prepared_df = self.validate_and_prepare_df(df)
        n = len(prepared_df)

        current_close = float(prepared_df["close"].iloc[-1])
        prev_close = float(prepared_df["close"].iloc[-2]) if n > 1 else current_close
        price_change = current_close - prev_close
        price_change_pct = (price_change / prev_close) * 100.0 if prev_close else 0.0

        # 1. Moving Averages
        ma_df = compute_all_moving_averages(prepared_df)
        ma_analysis = analyze_moving_averages(prepared_df)
        crossover_50_200 = detect_ma_crossovers(ma_df["sma_50"], ma_df["sma_200"])
        crossover_20_50 = detect_ma_crossovers(ma_df["sma_20"], ma_df["sma_50"])

        # 2. Oscillators & Momentum
        osc_analysis = analyze_oscillators(prepared_df)
        macd_df = compute_macd(prepared_df["close"])
        rsi_series = compute_rsi(prepared_df["close"])
        stoch_df = compute_stochastic_oscillator(prepared_df["high"], prepared_df["low"], prepared_df["close"])

        # 3. Volatility
        vol_analysis = analyze_volatility(prepared_df)
        bb_df = compute_bollinger_bands(prepared_df["close"])
        atr_series = compute_atr(prepared_df["high"], prepared_df["low"], prepared_df["close"])
        hist_vol_series = compute_historical_volatility(prepared_df["close"])

        # 4. Trend
        trend_analysis = analyze_trend(prepared_df)
        adx_df = compute_adx(prepared_df["high"], prepared_df["low"], prepared_df["close"])

        # 5. Volume
        volume_analysis = analyze_volume(prepared_df)
        obv_series = compute_obv(prepared_df["close"], prepared_df["volume"])
        vwap_series = compute_vwap(prepared_df["high"], prepared_df["low"], prepared_df["close"], prepared_df["volume"])

        # 6. Price Action & Patterns
        sr_analysis = compute_support_resistance(prepared_df)
        fib_analysis = compute_fibonacci_levels(prepared_df)
        patterns = detect_candlestick_patterns(prepared_df)

        # 7. AI Technical Score (0-100)
        score_data = compute_ai_technical_score(
            df=prepared_df,
            ma_bias=ma_analysis,
            crossover_data=crossover_50_200,
            oscillator_summary=osc_analysis,
            volatility_summary=vol_analysis,
            trend_summary=trend_analysis,
            volume_summary=volume_analysis,
            patterns=patterns,
            support_resistance=sr_analysis,
            fibonacci_data=fib_analysis,
        )

        # 8. Date labels extraction
        date_series = None
        for dt_col in ["timestamp", "date", "datetime"]:
            if dt_col in prepared_df.columns:
                date_series = [str(x) for x in prepared_df[dt_col].dt.strftime("%Y-%m-%d").values]
                break
        if date_series is None:
            date_series = [f"Bar-{i+1}" for i in range(n)]

        # Historical series tail for Plotly visualization (last 100 bars)
        chart_window = min(100, n)
        chart_dates = date_series[-chart_window:]
        chart_close = [round(float(x), 2) for x in prepared_df["close"].tail(chart_window)]
        chart_open = [round(float(x), 2) for x in prepared_df["open"].tail(chart_window)]
        chart_high = [round(float(x), 2) for x in prepared_df["high"].tail(chart_window)]
        chart_low = [round(float(x), 2) for x in prepared_df["low"].tail(chart_window)]
        chart_vol = [int(x) for x in prepared_df["volume"].tail(chart_window)]

        chart_sma20 = [round(float(x), 2) for x in ma_df["sma_20"].tail(chart_window)]
        chart_sma50 = [round(float(x), 2) for x in ma_df["sma_50"].tail(chart_window)]
        chart_sma200 = [round(float(x), 2) for x in ma_df["sma_200"].tail(chart_window)]
        chart_ema21 = [round(float(x), 2) for x in ma_df["ema_21"].tail(chart_window)]

        chart_bb_upper = [round(float(x), 2) for x in bb_df["bb_upper"].tail(chart_window)]
        chart_bb_middle = [round(float(x), 2) for x in bb_df["bb_middle"].tail(chart_window)]
        chart_bb_lower = [round(float(x), 2) for x in bb_df["bb_lower"].tail(chart_window)]

        chart_rsi = [round(float(x), 2) for x in rsi_series.tail(chart_window)]
        chart_macd = [round(float(x), 2) for x in macd_df["macd_line"].tail(chart_window)]
        chart_macd_sig = [round(float(x), 2) for x in macd_df["macd_signal"].tail(chart_window)]
        chart_macd_hist = [round(float(x), 2) for x in macd_df["macd_histogram"].tail(chart_window)]

        chart_vwap = [round(float(x), 2) for x in vwap_series.tail(chart_window)]
        chart_obv = [int(x) for x in obv_series.tail(chart_window)]

        # Assemble unified result
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "as_of": datetime.now().isoformat(),
            "current_price": round(current_close, 2),
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "ai_technical_score": score_data,
            "moving_averages": {
                **ma_analysis,
                "golden_cross_50_200": crossover_50_200["golden_cross"],
                "death_cross_50_200": crossover_50_200["death_cross"],
                "tactical_cross_20_50": crossover_20_50["golden_cross"],
            },
            "oscillators": osc_analysis,
            "volatility": vol_analysis,
            "trend": trend_analysis,
            "volume_profile": volume_analysis,
            "support_resistance": sr_analysis,
            "fibonacci_levels": fib_analysis,
            "candlestick_patterns": patterns,
            "chart_data": {
                "dates": chart_dates,
                "open": chart_open,
                "high": chart_high,
                "low": chart_low,
                "close": chart_close,
                "volume": chart_vol,
                "sma_20": chart_sma20,
                "sma_50": chart_sma50,
                "sma_200": chart_sma200,
                "ema_21": chart_ema21,
                "bb_upper": chart_bb_upper,
                "bb_middle": chart_bb_middle,
                "bb_lower": chart_bb_lower,
                "rsi": chart_rsi,
                "macd_line": chart_macd,
                "macd_signal": chart_macd_sig,
                "macd_histogram": chart_macd_hist,
                "vwap": chart_vwap,
                "obv": chart_obv,
            },
            "disclaimer": STATISTICAL_DISCLAIMER,
        }
