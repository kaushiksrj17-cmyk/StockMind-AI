"""Consolidated Multi-Factor Real-Time AI Signal Engine for StockMind-AI.

Fuses 8 institutional analytical dimensions:
1. Live Market Data (Tick momentum, VWAP deviation, order-flow pressure)
2. Technical Analysis (AI Technical Score 0-100, RSI, MACD, Moving Averages)
3. Machine Learning (Ensemble Directional Classifier probability)
4. Deep Learning (LSTM / GRU sequential trend forecast)
5. Financial NLP Sentiment (FinBERT & Loughran-McDonald score)
6. Quantitative Risk Intelligence (VaR, Volatility, Sharpe, Drawdown)
7. Anomaly Radar (Isolation Forest outlier flag, volume spikes, divergence)
8. Market Regime (Macro Bull, Bear, Sideways, High Volatility classification)

Produces 5 discrete signal classifications:
- STRONG BULLISH
- BULLISH
- NEUTRAL
- BEARISH
- STRONG BEARISH

Guarantees:
- Every signal includes confidence, supporting factors, opposing factors, and explanation.
- Predictions explicitly disclaimed as non-guaranteed probabilistic modeling.
- Never runs expensive LSTM/Transformer models on every tick (uses fast cached inference).
"""

from dataclasses import dataclass, field
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ai_engine.explainability.shap_explainer import shap_explainer


AI_SIGNAL_DISCLAIMER = (
    "PROBABILISTIC MARKET ANALYSIS NOTICE: This signal represents quantitative mathematical "
    "modeling and multi-factor pattern classification. It does NOT constitute financial advice, "
    "an investment recommendation, or a guarantee of future returns. All trading involves capital risk."
)


@dataclass
class AISignalResult:
    """Comprehensive multi-factor institutional signal output."""
    symbol: str
    signal: str  # STRONG BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG BEARISH
    composite_score: float  # -100.0 (Extreme Bearish) to +100.0 (Extreme Bullish)
    confidence_pct: float   # 0.0 to 100.0
    timestamp: str          # ISO 8601
    model_version: str      # e.g. "v3.2-institutional-ensemble"
    data_mode: str          # "LIVE DATA", "HISTORICAL DATA", "REPLAY DATA"
    supporting_factors: List[str] = field(default_factory=list)
    opposing_factors: List[str] = field(default_factory=list)
    explanation: str = ""
    component_scores: Dict[str, float] = field(default_factory=dict)
    factor_telemetry: Dict[str, Any] = field(default_factory=dict)
    shap_top_features: List[Dict[str, Any]] = field(default_factory=list)
    compliance_disclaimer: str = AI_SIGNAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "composite_score": round(self.composite_score, 1),
            "confidence_pct": round(self.confidence_pct, 1),
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "data_mode": self.data_mode,
            "supporting_factors": self.supporting_factors,
            "opposing_factors": self.opposing_factors,
            "explanation": self.explanation,
            "component_scores": {k: round(v, 1) for k, v in self.component_scores.items()},
            "factor_telemetry": self.factor_telemetry,
            "shap_top_features": self.shap_top_features,
            "compliance_disclaimer": self.compliance_disclaimer,
        }


class AISignalEngine:
    """Multi-factor AI Signal Aggregator and Real-Time Decision Engine."""

    def __init__(self, model_version: str = "v3.2-multi-factor") -> None:
        self.model_version = model_version

    def generate_signal(
        self,
        symbol: str,
        live_quote: Optional[Dict[str, Any]] = None,
        technical_data: Optional[Dict[str, Any]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None,
        deep_learning_forecast: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        anomaly_report: Optional[Dict[str, Any]] = None,
        regime_data: Optional[Dict[str, Any]] = None,
        data_mode: str = "REPLAY DATA",
    ) -> AISignalResult:
        """Synthesize 8 analytical dimensions into a high-conviction quantitative signal."""
        clean_symbol = symbol.strip().upper()
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Parse Live Market Factor (-100 to +100)
        quote = live_quote or {}
        change_pct = float(quote.get("change_percent") or quote.get("day_change_pct") or 0.0)
        vwap = float(quote.get("vwap") or quote.get("last_price") or 0.0)
        last_price = float(quote.get("last_price") or quote.get("close") or vwap)
        vwap_diff_pct = ((last_price - vwap) / vwap * 100.0) if vwap > 0 else 0.0

        # Market flow score: combines intraday change and VWAP relation
        market_score = float(np.clip(change_pct * 15.0 + vwap_diff_pct * 25.0, -100.0, 100.0))

        # 2. Parse Technical Analysis Factor (-100 to +100)
        tech = technical_data or {}
        tech_ai_score = float(tech.get("technical_score", 50.0))  # 0 to 100
        # Map 0..100 to -100..+100
        tech_score = (tech_ai_score - 50.0) * 2.0

        # 3. Parse Machine Learning Factor (-100 to +100)
        ml = ml_prediction or {}
        ml_dir = str(ml.get("direction", "NEUTRAL")).upper()
        ml_prob = float(ml.get("probability") or ml.get("confidence") or 0.50)
        if "UP" in ml_dir or "BULL" in ml_dir:
            ml_score = (ml_prob - 0.50) * 200.0
        elif "DOWN" in ml_dir or "BEAR" in ml_dir:
            ml_score = -(ml_prob - 0.50) * 200.0
        else:
            ml_score = 0.0
        ml_score = float(np.clip(ml_score, -100.0, 100.0))

        # 4. Parse Deep Learning Factor (-100 to +100)
        dl = deep_learning_forecast or {}
        dl_dir = str(dl.get("direction", "NEUTRAL")).upper()
        dl_conf = float(dl.get("confidence") or dl.get("model_confidence") or 0.50)
        if "UP" in dl_dir or "BULL" in dl_dir:
            dl_score = (dl_conf - 0.50) * 200.0
        elif "DOWN" in dl_dir or "BEAR" in dl_dir:
            dl_score = -(dl_conf - 0.50) * 200.0
        else:
            dl_score = 0.0
        dl_score = float(np.clip(dl_score, -100.0, 100.0))

        # 5. Parse News Sentiment Factor (-100 to +100)
        sent = sentiment_data or {}
        net_sent = float(sent.get("sentiment_score") or sent.get("compound_score") or 0.0)  # -1.0 to +1.0
        sent_score = float(np.clip(net_sent * 100.0, -100.0, 100.0))

        # 6. Parse Risk Intelligence Factor (-100 to +100)
        risk = risk_metrics or {}
        vol_ann = float(risk.get("annualized_volatility_pct", 18.0))
        max_dd = float(risk.get("max_drawdown_pct", 10.0))
        sharpe = float(risk.get("sharpe_ratio", 1.2))
        # High sharpe and low volatility supports positive conviction; high volatility penalizes
        risk_score = float(np.clip((sharpe - 1.0) * 35.0 - (vol_ann - 18.0) * 1.5 - (max_dd - 10.0) * 1.0, -100.0, 100.0))

        # 7. Parse Anomaly Factor (-100 to +100)
        anom = anomaly_report or {}
        is_anom = bool(anom.get("is_current_bar_anomalous", False))
        anom_sev = str(anom.get("current_severity", "NORMAL")).upper()
        div_stat = str(anom.get("divergence_status", "")).upper()

        anomaly_score = 0.0
        if "ACCUMULATION" in div_stat or "ABSORPTION" in div_stat:
            anomaly_score = +45.0
        elif "DISTRIBUTION" in div_stat:
            anomaly_score = -55.0
        elif is_anom and anom_sev in ("CRITICAL", "HIGH"):
            # Severe unclassified anomaly introduces risk-off penalty
            anomaly_score = -40.0

        # 8. Parse Market Regime Factor (-100 to +100)
        regime = regime_data or {}
        prim_reg = str(regime.get("primary_regime", "SIDEWAYS")).upper()
        reg_conf = float(regime.get("confidence_pct", 50.0)) / 100.0
        if prim_reg == "BULL":
            regime_score = +60.0 * reg_conf
        elif prim_reg == "BEAR":
            regime_score = -60.0 * reg_conf
        elif prim_reg == "HIGH_VOLATILITY":
            regime_score = -30.0 * reg_conf
        else:  # SIDEWAYS
            regime_score = 0.0

        # ====================================================================
        # COMPOSITE MULTI-FACTOR WEIGHTED SYNTHESIS
        # ====================================================================
        weights = {
            "technical": 0.24,
            "machine_learning": 0.20,
            "deep_learning": 0.16,
            "sentiment": 0.12,
            "regime": 0.10,
            "market_flow": 0.08,
            "risk": 0.06,
            "anomaly": 0.04,
        }

        component_scores = {
            "technical": tech_score,
            "machine_learning": ml_score,
            "deep_learning": dl_score,
            "sentiment": sent_score,
            "regime": regime_score,
            "market_flow": market_score,
            "risk": risk_score,
            "anomaly": anomaly_score,
        }

        composite_score = sum(weights[k] * component_scores[k] for k in weights)
        composite_score = float(np.clip(composite_score, -100.0, 100.0))

        # Calibrate Confidence
        # Higher absolute score and model alignment increases confidence
        concordance = sum(1 for v in component_scores.values() if (v * composite_score) > 0)
        alignment_pct = (concordance / len(component_scores)) * 100.0
        base_conf = 50.0 + (abs(composite_score) / 100.0) * 35.0 + (alignment_pct / 100.0) * 15.0
        confidence_pct = float(np.clip(base_conf, 50.0, 96.0))

        # ====================================================================
        # SIGNAL CLASSIFICATION
        # ====================================================================
        if composite_score >= 50.0:
            signal_label = "STRONG BULLISH"
        elif composite_score >= 20.0:
            signal_label = "BULLISH"
        elif composite_score <= -50.0:
            signal_label = "STRONG BEARISH"
        elif composite_score <= -20.0:
            signal_label = "BEARISH"
        else:
            signal_label = "NEUTRAL"

        # ====================================================================
        # IDENTIFY SUPPORTING & OPPOSING FACTORS
        # ====================================================================
        supporting: List[str] = []
        opposing: List[str] = []

        is_bullish_bias = composite_score >= 0.0

        factor_descriptions = {
            "technical": (f"AI Technical Score ({tech_ai_score:.0f}/100) indicates bullish structure", f"AI Technical Score ({tech_ai_score:.0f}/100) reflects bearish weakness"),
            "machine_learning": (f"ML Ensemble projects upward direction ({ml_prob*100.0:.0f}% prob)", f"ML Ensemble projects downward pressure ({ml_prob*100.0:.0f}% prob)"),
            "deep_learning": (f"LSTM/GRU sequential model forecasts positive momentum", f"LSTM/GRU sequential model projects downward drift"),
            "sentiment": (f"News sentiment is net positive (score: {net_sent:+.2f})", f"News sentiment reflects negative headwinds (score: {net_sent:+.2f})"),
            "regime": (f"Macro market regime is {prim_reg}", f"Macro market regime ({prim_reg}) offers resistance"),
            "market_flow": (f"Live market trades above VWAP with {change_pct:+.2f}% day change", f"Live market trades below VWAP with {change_pct:+.2f}% day change"),
            "risk": (f"Risk-adjusted metrics support entry (Sharpe: {sharpe:.2f})", f"Elevated volatility ({vol_ann:.1f}% ann.) constraints sizing"),
            "anomaly": (f"Order flow indicates institutional accumulation", f"Anomaly radar flags distribution pressure"),
        }

        for comp, score_val in component_scores.items():
            pos_desc, neg_desc = factor_descriptions[comp]
            if is_bullish_bias:
                if score_val >= 15.0:
                    supporting.append(pos_desc)
                elif score_val <= -15.0:
                    opposing.append(neg_desc)
            else:
                if score_val <= -15.0:
                    supporting.append(neg_desc)
                elif score_val >= 15.0:
                    opposing.append(pos_desc)

        if not supporting:
            supporting.append("Equilibrium between opposing forces; indicators hovering near neutral thresholds.")
        if not opposing:
            opposing.append("No significant opposing technical or fundamental divergence identified.")

        # ====================================================================
        # DETAILED NATURAL LANGUAGE EXPLANATION
        # ====================================================================
        supp_summary = "; ".join(supporting[:2])
        opp_summary = "; ".join(opposing[:2]) if opposing else "None"

        explanation = (
            f"Consolidated AI Engine issues a {signal_label} signal (Composite Score: {composite_score:+.1f}/100) "
            f"with {confidence_pct:.1f}% model confidence. "
            f"Primary positive confirmation from: {supp_summary}. "
            f"Headwinds and countervailing factors: {opp_summary}."
        )

        # ====================================================================
        # FAST SHAP EXPLAINABILITY ATTRIBUTION
        # ====================================================================
        feat_dict = {
            "Technical Score": tech_ai_score,
            "ML Direction Prob": ml_prob * 100.0,
            "Deep Learning Conf": dl_conf * 100.0,
            "News Sentiment": net_sent * 100.0,
            "VWAP Deviation %": vwap_diff_pct,
            "Annual Volatility": vol_ann,
            "Sharpe Ratio": sharpe * 20.0,
        }
        shap_res = shap_explainer.explain_instance(
            features=feat_dict,
            symbol=clean_symbol,
            model_confidence=confidence_pct / 100.0,
            model_agreement_pct=alignment_pct,
        )

        return AISignalResult(
            symbol=clean_symbol,
            signal=signal_label,
            composite_score=composite_score,
            confidence_pct=confidence_pct,
            timestamp=now_ts,
            model_version=self.model_version,
            data_mode=data_mode,
            supporting_factors=supporting[:4],
            opposing_factors=opposing[:3],
            explanation=explanation,
            component_scores=component_scores,
            factor_telemetry={
                "last_price": last_price,
                "change_percent": change_pct,
                "vwap": vwap,
                "technical_score": tech_ai_score,
                "ml_probability": ml_prob,
                "news_sentiment": net_sent,
                "market_regime": prim_reg,
                "annual_volatility": vol_ann,
                "anomaly_status": anom_sev,
            },
            shap_top_features=[f.to_dict() for f in shap_res.top_contributing_features],
            compliance_disclaimer=AI_SIGNAL_DISCLAIMER,
        )


# Global singleton instance
ai_signal_engine = AISignalEngine()
