"""AI Technical Score & Narrative Engine.

Aggregates calculated technical indicators across Trend, Momentum, Volatility,
Volume, and Price Action into a normalized 0–100 AI Technical Score.
Generates classification, trend direction, trend strength, plain-English
indicator explanations, and a transparent score breakdown.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


STATISTICAL_DISCLAIMER = (
    "Technical indicators represent mathematical derivations of past price and volume data. "
    "They do not guarantee future price movements or performance and should not be construed "
    "as financial or investment advice."
)


def compute_ai_technical_score(
    df: pd.DataFrame,
    ma_bias: Dict[str, Any],
    crossover_data: Dict[str, Any],
    oscillator_summary: Dict[str, Any],
    volatility_summary: Dict[str, Any],
    trend_summary: Dict[str, Any],
    volume_summary: Dict[str, Any],
    patterns: List[Dict[str, Any]],
    support_resistance: Dict[str, Any],
    fibonacci_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the composite 0-100 AI Technical Score and accompanying insights.

    Weights:
    - Trend & Moving Averages: 30%
    - Momentum & Oscillators: 30%
    - Volume Dynamics: 15%
    - Volatility & Bollinger Bands: 15%
    - Candlestick Patterns & Price Action: 10%
    """
    current_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0

    # -------------------------------------------------------------
    # 1. Trend & Moving Averages Component (0 - 100)
    # -------------------------------------------------------------
    trend_pts = 50.0  # neutral base

    # Moving Average Stack Bias
    stack_bias = ma_bias.get("bias")
    if not stack_bias:
        if ma_bias.get("is_bullish_stack"):
            stack_bias = "BULLISH_STACK"
        elif ma_bias.get("is_bearish_stack"):
            stack_bias = "BEARISH_STACK"
        elif ma_bias.get("bias_score", 0) > 2:
            stack_bias = "MILD_BULLISH"
        elif ma_bias.get("bias_score", 0) < 0:
            stack_bias = "MILD_BEARISH"
        else:
            stack_bias = "NEUTRAL"

    if stack_bias == "BULLISH_STACK":
        trend_pts += 25.0
    elif stack_bias == "BEARISH_STACK":
        trend_pts -= 25.0
    elif stack_bias == "MILD_BULLISH":
        trend_pts += 12.0
    elif stack_bias == "MILD_BEARISH":
        trend_pts -= 12.0

    # Golden / Death Cross
    golden_cross = crossover_data.get("golden_cross", False)
    death_cross = crossover_data.get("death_cross", False)
    if golden_cross:
        trend_pts += 15.0
    if death_cross:
        trend_pts -= 15.0

    # ADX and Directional Bias
    adx_direction = trend_summary.get("trend_direction") or trend_summary.get("direction", "SIDEWAYS")
    adx_val = float(trend_summary.get("adx", 20.0))
    plus_di = float(trend_summary.get("plus_di", 20.0))
    minus_di = float(trend_summary.get("minus_di", 20.0))

    if plus_di > minus_di:
        boost = min(15.0, (adx_val / 40.0) * 15.0)
        trend_pts += boost
    else:
        penalty = min(15.0, (adx_val / 40.0) * 15.0)
        trend_pts -= penalty

    trend_score = max(0.0, min(100.0, trend_pts))

    # -------------------------------------------------------------
    # 2. Momentum & Oscillators Component (0 - 100)
    # -------------------------------------------------------------
    mom_pts = 50.0

    # RSI (14)
    rsi_val = float(oscillator_summary.get("rsi", 50.0))
    if rsi_val >= 70.0:
        mom_pts += 12.0
    elif rsi_val >= 55.0:
        mom_pts += 20.0
    elif rsi_val <= 30.0:
        mom_pts -= 12.0
    elif rsi_val <= 45.0:
        mom_pts -= 20.0

    # MACD
    macd_dict = oscillator_summary.get("macd", {})
    macd_hist = float(oscillator_summary.get("macd_histogram", macd_dict.get("histogram", 0.0)))
    macd_line = float(oscillator_summary.get("macd_line", macd_dict.get("macd_line", 0.0)))
    macd_signal = float(oscillator_summary.get("macd_signal", macd_dict.get("signal_line", 0.0)))
    if macd_line > macd_signal:
        mom_pts += 12.0
    else:
        mom_pts -= 12.0
    if macd_hist > 0:
        mom_pts += 6.0
    else:
        mom_pts -= 6.0

    # Stochastic Oscillator
    stoch_dict = oscillator_summary.get("stochastic", {})
    stoch_k = float(oscillator_summary.get("stoch_k", stoch_dict.get("stoch_k", 50.0)))
    stoch_d = float(oscillator_summary.get("stoch_d", stoch_dict.get("stoch_d", 50.0)))
    if stoch_k > stoch_d:
        mom_pts += 7.0
    else:
        mom_pts -= 7.0

    # ROC / Momentum
    roc_val = float(oscillator_summary.get("roc", oscillator_summary.get("roc_12", 0.0)))
    if roc_val > 2.0:
        mom_pts += 5.0
    elif roc_val < -2.0:
        mom_pts -= 5.0

    momentum_score = max(0.0, min(100.0, mom_pts))

    # -------------------------------------------------------------
    # 3. Volume Dynamics Component (0 - 100)
    # -------------------------------------------------------------
    vol_pts = 50.0

    # VWAP Benchmark
    vwap_val = volume_summary.get("vwap")
    if vwap_val and current_price > 0:
        if current_price >= vwap_val:
            vol_pts += 15.0
        else:
            vol_pts -= 15.0

    # Relative Volume (RVOL)
    rvol = float(volume_summary.get("rvol_20", 1.0))
    is_surge = volume_summary.get("is_volume_surge", volume_summary.get("volume_surge", False))
    if is_surge:
        open_val = float(df["open"].iloc[-1]) if "open" in df.columns else current_price
        if current_price >= open_val:
            vol_pts += 15.0
        else:
            vol_pts -= 15.0
    elif rvol > 1.2:
        vol_pts += 5.0

    # Price-Volume Divergence
    pv_div_data = volume_summary.get("divergence", "CONFIRMING")
    if isinstance(pv_div_data, dict):
        pv_div = pv_div_data.get("divergence_type", "CONFIRMING")
    else:
        pv_div = str(pv_div_data)

    if pv_div == "BULLISH_DIVERGENCE":
        vol_pts += 20.0
    elif pv_div == "BEARISH_DIVERGENCE":
        vol_pts -= 20.0

    volume_score = max(0.0, min(100.0, vol_pts))

    # -------------------------------------------------------------
    # 4. Volatility & Bollinger Bands Component (0 - 100)
    # -------------------------------------------------------------
    volat_pts = 50.0
    bb_dict = volatility_summary.get("bollinger_bands", {})
    pct_b = float(volatility_summary.get("bb_pct_b", bb_dict.get("pct_b", 0.5)))
    is_squeeze = volatility_summary.get("bb_squeeze", bb_dict.get("is_squeeze", False))

    # Position in bands
    if pct_b > 0.8:
        volat_pts += 15.0
    elif pct_b > 0.5:
        volat_pts += 10.0
    elif pct_b < 0.2:
        volat_pts -= 15.0
    else:
        volat_pts -= 10.0

    # Volatility trend
    ann_vol = float(volatility_summary.get("annualized_volatility", 0.25))
    if ann_vol < 0.35:
        # Controlled orderly volatility is positive for sustainable trends
        volat_pts += 5.0
    else:
        # Erratic wild volatility adds uncertainty
        volat_pts -= 5.0

    volatility_score = max(0.0, min(100.0, volat_pts))

    # -------------------------------------------------------------
    # 5. Price Action & Patterns Component (0 - 100)
    # -------------------------------------------------------------
    pa_pts = 50.0
    for p in patterns:
        bias = p.get("bias", "NEUTRAL")
        sig = p.get("significance", "MODERATE")
        mult = 12.0 if sig == "VERY_HIGH" else (8.0 if sig == "HIGH" else 4.0)
        # Recent patterns have higher weight
        decay = 1.0 / (1.0 + p.get("bars_ago", 0))

        if bias == "BULLISH":
            pa_pts += mult * decay
        elif bias == "BEARISH":
            pa_pts -= mult * decay

    # Distance to Support vs Resistance
    nearest_sup = support_resistance.get("nearest_support", current_price * 0.95)
    nearest_res = support_resistance.get("nearest_resistance", current_price * 1.05)
    sup_dist = (current_price - nearest_sup) / current_price if current_price > 0 else 0.05
    res_dist = (nearest_res - current_price) / current_price if current_price > 0 else 0.05

    # If comfortably above support with upside room to resistance
    if res_dist > sup_dist:
        pa_pts += 6.0
    else:
        pa_pts -= 6.0

    price_action_score = max(0.0, min(100.0, pa_pts))

    # -------------------------------------------------------------
    # Composite Weighted AI Technical Score (0 - 100)
    # -------------------------------------------------------------
    weights = {
        "trend": 0.30,
        "momentum": 0.30,
        "volume": 0.15,
        "volatility": 0.15,
        "price_action": 0.10,
    }

    final_score = (
        (trend_score * weights["trend"])
        + (momentum_score * weights["momentum"])
        + (volume_score * weights["volume"])
        + (volatility_score * weights["volatility"])
        + (price_action_score * weights["price_action"])
    )
    final_score = round(max(0.0, min(100.0, final_score)), 1)

    # -------------------------------------------------------------
    # Classification & Labeling
    # -------------------------------------------------------------
    if final_score >= 80.0:
        classification = "STRONG_BULLISH"
        summary_label = "Strong Bullish Momentum"
    elif final_score >= 60.0:
        classification = "BULLISH"
        summary_label = "Bullish Technical Bias"
    elif final_score >= 40.0:
        classification = "NEUTRAL"
        summary_label = "Neutral / Rangebound Consolidation"
    elif final_score >= 20.0:
        classification = "BEARISH"
        summary_label = "Bearish Technical Bias"
    else:
        classification = "STRONG_BEARISH"
        summary_label = "Strong Bearish Distribution"

    trend_dir = trend_summary.get("direction", "SIDEWAYS")
    trend_str = trend_summary.get("strength", "MODERATE")

    # -------------------------------------------------------------
    # Indicator Explanations (Plain English)
    # -------------------------------------------------------------
    indicator_explanations = {
        "moving_averages": (
            f"Price is trading relative to key averages with a '{stack_bias.replace('_', ' ').title()}' configuration. "
            + ("A Golden Cross (50-SMA crossing above 200-SMA) is active, reinforcing medium-term upside. " if golden_cross else "")
            + ("A Death Cross (50-SMA crossing below 200-SMA) is active, indicating extended downside risk. " if death_cross else "")
        ),
        "oscillators": (
            f"RSI-14 is currently {rsi_val:.1f} ({'Overbought' if rsi_val >= 70 else 'Oversold' if rsi_val <= 30 else 'Neutral'}). "
            f"MACD line is {round(macd_line, 2)} vs Signal {round(macd_signal, 2)} with a {'positive' if macd_hist >= 0 else 'negative'} "
            f"histogram spread of {round(macd_hist, 2)}. Stochastic %K is at {stoch_k:.1f} vs %D at {stoch_d:.1f}."
        ),
        "volatility": (
            f"Bollinger Bands show %B at {pct_b:.2f} ({'touching upper band' if pct_b >= 1.0 else 'touching lower band' if pct_b <= 0.0 else 'trading inside bands'}). "
            + ("Bollinger Band Squeeze detected, indicating impending volatility expansion. " if is_squeeze else "")
            + f"Annualized realized historical volatility is {ann_vol * 100.0:.1f}%."
        ),
        "volume": (
            f"Volume analysis indicates RVOL of {rvol:.2f}x average. "
            + (f"Current price is {'above' if current_price >= (vwap_val or 0) else 'below'} VWAP ({round(vwap_val or 0, 2)}). " if vwap_val else "")
            + (f"Price-volume relationship is showing '{pv_div.replace('_', ' ').title()}'. " if pv_div != "CONFIRMING" else "Price and volume are moving in statistical confirmation. ")
        ),
        "price_action": (
            f"Nearest dynamic support is at ₹{nearest_sup:,.2f} and resistance is at ₹{nearest_res:,.2f}. "
            f"Recent price action generated {len(patterns)} pattern signal(s): "
            + (", ".join([f"{p['name']} ({p['bias']})" for p in patterns[:3]]) if patterns else "No high-conviction pattern detected.")
        ),
    }

    # -------------------------------------------------------------
    # Technical Score Detailed Explanation
    # -------------------------------------------------------------
    pos_drivers = []
    neg_drivers = []

    if trend_score >= 60:
        pos_drivers.append(f"Trend & MAs (+{round(trend_score * weights['trend'], 1)} pts)")
    elif trend_score <= 40:
        neg_drivers.append(f"Trend weakness (-{round((50 - trend_score) * weights['trend'], 1)} pts)")

    if momentum_score >= 60:
        pos_drivers.append(f"Bullish momentum (+{round(momentum_score * weights['momentum'], 1)} pts)")
    elif momentum_score <= 40:
        neg_drivers.append(f"Negative oscillator drag (-{round((50 - momentum_score) * weights['momentum'], 1)} pts)")

    if volume_score >= 60:
        pos_drivers.append(f"Volume accumulation (+{round(volume_score * weights['volume'], 1)} pts)")
    elif volume_score <= 40:
        neg_drivers.append(f"Volume distribution / divergence (-{round((50 - volume_score) * weights['volume'], 1)} pts)")

    pos_str = ", ".join(pos_drivers) if pos_drivers else "No dominant positive factors"
    neg_str = ", ".join(neg_drivers) if neg_drivers else "No dominant negative drags"

    score_explanation = (
        f"The AI Technical Score of {final_score}/100 reflects a '{summary_label}'. "
        f"Key positive drivers include: {pos_str}. "
        f"Downside or drag factors include: {neg_str}. "
        f"Trend direction is evaluated as {trend_dir} with {trend_str.replace('_', ' ').lower()} conviction."
    )

    return {
        "score": final_score,
        "classification": classification,
        "summary_label": summary_label,
        "trend_direction": trend_dir,
        "trend_strength": trend_str,
        "component_scores": {
            "trend": round(trend_score, 1),
            "momentum": round(momentum_score, 1),
            "volume": round(volume_score, 1),
            "volatility": round(volatility_score, 1),
            "price_action": round(price_action_score, 1),
        },
        "component_weights": weights,
        "indicator_explanations": indicator_explanations,
        "score_explanation": score_explanation,
        "disclaimer": STATISTICAL_DISCLAIMER,
    }
