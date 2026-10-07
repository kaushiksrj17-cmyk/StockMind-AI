"""Institutional Market Regime Detection Engine for StockMind-AI.

Classifies market state into 4 fundamental regimes:
1. BULL (Sustained upward trend, positive momentum, stable volatility)
2. BEAR (Sustained downward trend, negative momentum, risk-off pressure)
3. SIDEWAYS (Mean-reverting, range-bound, low directional conviction)
4. HIGH_VOLATILITY (Turbulent swings, wide ATR dispersion, elevated tail risk)

Calculates probabilistic regime distribution, regime stability, and strategic institutional guidance.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class RegimeClassification:
    """Quantitative regime assessment result."""
    symbol: str
    primary_regime: str  # BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY
    confidence_pct: float
    regime_probabilities: Dict[str, float]  # e.g. {"BULL": 0.65, "BEAR": 0.05, ...}
    trend_score: float   # -100 (Strong Bearish) to +100 (Strong Bullish)
    volatility_score: float  # 0 (Compressed) to 100 (Extreme Turbulance)
    trend_strength_adx: float  # 0 to 100
    stability_score: float  # 0 to 100 (Consistency of regime across rolling windows)
    bars_in_regime: int
    recommended_stance: str
    institutional_explanation: str
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "primary_regime": self.primary_regime,
            "confidence_pct": round(self.confidence_pct, 1),
            "regime_probabilities": {k: round(v, 3) for k, v in self.regime_probabilities.items()},
            "trend_score": round(self.trend_score, 1),
            "volatility_score": round(self.volatility_score, 1),
            "trend_strength_adx": round(self.trend_strength_adx, 1),
            "stability_score": round(self.stability_score, 1),
            "bars_in_regime": self.bars_in_regime,
            "recommended_stance": self.recommended_stance,
            "institutional_explanation": self.institutional_explanation,
            "supporting_metrics": self.supporting_metrics,
        }


class MarketRegimeEngine:
    """Quantitative engine classifying macro & micro market states."""

    def __init__(
        self,
        volatility_high_threshold_pct: float = 26.0,
        adx_trend_threshold: float = 22.0,
    ) -> None:
        self.volatility_high_threshold_pct = volatility_high_threshold_pct
        self.adx_trend_threshold = adx_trend_threshold

    @staticmethod
    def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        """Compute ADX, +DI, and -DI."""
        n = len(close)
        if n < period + 2:
            return 15.0, 20.0, 20.0

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = pd.Series(tr).ewm(alpha=1.0 / period, min_periods=period).mean().values
        plus_smooth = pd.Series(plus_dm).ewm(alpha=1.0 / period, min_periods=period).mean().values
        minus_smooth = pd.Series(minus_dm).ewm(alpha=1.0 / period, min_periods=period).mean().values

        tr_smooth = np.where(tr_smooth < 1e-4, 1e-4, tr_smooth)
        plus_di = 100.0 * (plus_smooth / tr_smooth)
        minus_di = 100.0 * (minus_smooth / tr_smooth)

        dx_denom = plus_di + minus_di
        dx_denom = np.where(dx_denom < 1e-4, 1e-4, dx_denom)
        dx = 100.0 * (np.abs(plus_di - minus_di) / dx_denom)
        adx_series = pd.Series(dx).ewm(alpha=1.0 / period, min_periods=period).mean().values

        adx_val = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 18.0
        p_di = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 20.0
        m_di = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 20.0
        return adx_val, p_di, m_di

    def detect_regime(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY50",
    ) -> RegimeClassification:
        """Classify market regime from price history."""
        if df is None or len(df) < 20:
            return RegimeClassification(
                symbol=symbol,
                primary_regime="SIDEWAYS",
                confidence_pct=50.0,
                regime_probabilities={"BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.40, "HIGH_VOLATILITY": 0.10},
                trend_score=0.0,
                volatility_score=20.0,
                trend_strength_adx=15.0,
                stability_score=50.0,
                bars_in_regime=5,
                recommended_stance="Neutral / Hold Cash / Await Clear Trend",
                institutional_explanation="Insufficient data points for robust statistical regime decomposition.",
            )

        data = df.copy()
        data.columns = [str(c).strip().lower() for c in data.columns]
        close = data["close"].astype(float).values
        high = data["high"].astype(float).values if "high" in data.columns else close
        low = data["low"].astype(float).values if "low" in data.columns else close
        n = len(close)

        # 1. Volatility Calculations
        returns = np.zeros(n)
        returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-6)
        vol_20d_ann = float(np.std(returns[-20:], ddof=1) * math.sqrt(252) * 100.0) if n >= 20 else 18.0

        # Bollinger Band Width
        sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
        std20 = pd.Series(close).rolling(20, min_periods=5).std().fillna(1.0).values
        upper_bb = sma20 + 2.0 * std20
        lower_bb = sma20 - 2.0 * std20
        bb_width_pct = float((upper_bb[-1] - lower_bb[-1]) / max(sma20[-1], 1.0) * 100.0)

        # Volatility Score (0 to 100)
        vol_score = min(100.0, max(0.0, (vol_20d_ann / 40.0) * 80.0 + (bb_width_pct / 15.0) * 20.0))

        # 2. Moving Averages & Trend Score
        ema20 = pd.Series(close).ewm(span=20).mean().values[-1]
        ema50 = pd.Series(close).ewm(span=50).mean().values[-1] if n >= 30 else ema20
        sma200 = pd.Series(close).rolling(200, min_periods=30).mean().values[-1]

        cur_price = close[-1]
        # Moving average alignment score (-50 to +50)
        ma_alignment = 0.0
        if cur_price > ema20:
            ma_alignment += 15.0
        else:
            ma_alignment -= 15.0
        if ema20 > ema50:
            ma_alignment += 20.0
        else:
            ma_alignment -= 20.0
        if cur_price > sma200:
            ma_alignment += 15.0
        else:
            ma_alignment -= 15.0

        # Linear regression slope over last 30 bars (-50 to +50)
        lookback_reg = min(30, n)
        x_axis = np.arange(lookback_reg)
        y_axis = close[-lookback_reg:]
        slope, _ = np.polyfit(x_axis, y_axis, 1)
        slope_pct = (slope * lookback_reg) / max(y_axis[0], 1.0) * 100.0
        slope_score = float(np.clip(slope_pct * 5.0, -50.0, 50.0))

        trend_score = float(np.clip(ma_alignment + slope_score, -100.0, 100.0))

        # 3. ADX & Directional Strength
        adx_val, plus_di, minus_di = self._compute_adx(high, low, close, period=14)

        # 4. Probabilistic Regime Scoring
        # We model unnormalized logits for the 4 states:
        logits = {
            "BULL": 0.0,
            "BEAR": 0.0,
            "SIDEWAYS": 0.0,
            "HIGH_VOLATILITY": 0.0,
        }

        # High Volatility logit
        if vol_20d_ann >= self.volatility_high_threshold_pct or vol_score >= 65.0:
            logits["HIGH_VOLATILITY"] += 2.8 + (vol_20d_ann - self.volatility_high_threshold_pct) * 0.15
        else:
            logits["HIGH_VOLATILITY"] += max(0.0, (vol_score - 40.0) * 0.05)

        # Bull logit
        if trend_score > 15.0:
            logits["BULL"] += (trend_score / 25.0) + (adx_val / 20.0 if plus_di > minus_di else 0.5)
        # Bear logit
        if trend_score < -15.0:
            logits["BEAR"] += (abs(trend_score) / 25.0) + (adx_val / 20.0 if minus_di > plus_di else 0.5)

        # Sideways logit
        if abs(trend_score) <= 25.0 and adx_val < self.adx_trend_threshold:
            logits["SIDEWAYS"] += 2.5 + (self.adx_trend_threshold - adx_val) * 0.15 + (40.0 - vol_score) * 0.04
        else:
            logits["SIDEWAYS"] += max(0.2, (30.0 - abs(trend_score)) * 0.04)

        # Softmax normalization
        exp_vals = {k: math.exp(v) for k, v in logits.items()}
        sum_exp = sum(exp_vals.values())
        probs = {k: exp_vals[k] / sum_exp for k in exp_vals}

        primary_regime = max(probs, key=probs.get)
        confidence_pct = probs[primary_regime] * 100.0

        # Estimate bars in regime (rolling backward scan)
        bars_in_regime = 1
        for i in range(2, min(40, n)):
            sub_c = close[:-i]
            if len(sub_c) < 10:
                break
            sub_ret = (sub_c[-1] - sub_c[0]) / max(sub_c[0], 1.0)
            if primary_regime == "BULL" and sub_ret > 0:
                bars_in_regime += 1
            elif primary_regime == "BEAR" and sub_ret < 0:
                bars_in_regime += 1
            elif primary_regime == "SIDEWAYS" and abs(sub_ret) < 0.04:
                bars_in_regime += 1
            elif primary_regime == "HIGH_VOLATILITY":
                bars_in_regime += 1
            else:
                break

        # Stance & Institutional Explanation
        if primary_regime == "BULL":
            recommended_stance = "Trend Following / Accumulate Quality / Expand Long Exposure"
            explanation = (
                f"Sustained bullish market structure with trend score of {trend_score:+.1f}/100. "
                f"Price trades above key moving averages with ADX at {adx_val:.1f} (+DI: {plus_di:.1f} > -DI: {minus_di:.1f}). "
                f"Controlled volatility ({vol_20d_ann:.1f}% ann.) supports orderly trend continuation."
            )
        elif primary_regime == "BEAR":
            recommended_stance = "Defensive / Tactical Short / Hedge Longs / Capital Preservation"
            explanation = (
                f"Bearish breakdown detected with negative trend momentum ({trend_score:+.1f}/100). "
                f"Distribution pressure evidenced by -DI ({minus_di:.1f}) exceeding +DI ({plus_di:.1f}) and ADX at {adx_val:.1f}. "
                f"Downside risk elevated; defensive asset rotation recommended."
            )
        elif primary_regime == "HIGH_VOLATILITY":
            recommended_stance = "Volatility Arbitrage / Downsize Position Sizing / Widen Stop Losses"
            explanation = (
                f"Regime dominated by heightened volatility shocks ({vol_20d_ann:.1f}% annualized). "
                f"Bollinger Band width stands at {bb_width_pct:.1f}%, indicating sharp price dispersion and fat tail events. "
                f"Directional persistence is low; risk controls take precedence."
            )
        else:  # SIDEWAYS
            recommended_stance = "Mean Reversion / Range Trading / Sell Resistance, Buy Support"
            explanation = (
                f"Market is oscillating inside a consolidation corridor. "
                f"ADX of {adx_val:.1f} reflects absent directional conviction. "
                f"Moving averages are compressing with flat trajectory, favoring range-bound execution."
            )

        stability = float(np.clip(100.0 - abs(vol_20d_ann - 15.0) * 2.0, 30.0, 95.0))

        return RegimeClassification(
            symbol=symbol.upper(),
            primary_regime=primary_regime,
            confidence_pct=confidence_pct,
            regime_probabilities=probs,
            trend_score=trend_score,
            volatility_score=vol_score,
            trend_strength_adx=adx_val,
            stability_score=stability,
            bars_in_regime=bars_in_regime,
            recommended_stance=recommended_stance,
            institutional_explanation=explanation,
            supporting_metrics={
                "annualized_volatility_pct": round(vol_20d_ann, 2),
                "bb_width_pct": round(bb_width_pct, 2),
                "adx_14": round(adx_val, 2),
                "plus_di": round(plus_di, 2),
                "minus_di": round(minus_di, 2),
                "slope_score": round(slope_score, 2),
                "ma_alignment": round(ma_alignment, 2),
            },
        )


# Global singleton instance
regime_engine = MarketRegimeEngine()
