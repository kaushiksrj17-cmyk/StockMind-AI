"""Real-Time AI Signals and Explainability API Routes."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, status

from ai_engine.explainability.shap_explainer import shap_explainer
from ai_engine.signals.realtime_pipeline import realtime_signal_pipeline
from ai_engine.signals.signal_engine import ai_signal_engine
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/signals", tags=["Real-Time AI Signal & Explainability Engine"])


@router.get(
    "/{symbol}",
    summary="Get Consolidated Multi-Factor AI Signal",
)
async def get_ai_signal(
    symbol: str,
):
    """Retrieve multi-factor institutional AI signal combining:
    Live Market Data + Technicals + ML + Deep Learning + News Sentiment + Risk + Anomaly + Market Regime.
    
    Generates STRONG BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG BEARISH.
    Includes confidence, supporting/opposing factors, explanation, and SHAP top features.
    """
    clean_sym = symbol.strip().upper()

    # 1. Check if real-time pipeline already has an active cached signal
    cached = realtime_signal_pipeline.get_latest_signal(clean_sym)

    # 2. Fetch latest live market quote
    quote_data = None
    try:
        quote = await market_service.get_quote(clean_sym)
        if quote:
            quote_data = {
                "symbol": clean_sym,
                "last_price": float(quote.last_price),
                "open": float(quote.open_price),
                "high": float(quote.high_price),
                "low": float(quote.low_price),
                "close": float(quote.last_price),
                "volume": float(quote.volume),
                "change_percent": float(quote.change_percent),
            }
    except Exception:
        quote_data = None

    data_mode_label = f"{market_service.data_mode.value.upper()} DATA"

    # If quote is available, process tick or generate fresh signal
    if quote_data:
        signal = realtime_signal_pipeline.process_live_tick(
            symbol=clean_sym,
            price=quote_data["last_price"],
            volume=quote_data["volume"],
            day_open=quote_data["open"],
            data_mode=data_mode_label,
        )
        return signal.to_dict()

    if cached:
        return cached.to_dict()

    # Fallback generation
    signal = ai_signal_engine.generate_signal(
        symbol=clean_sym,
        data_mode=data_mode_label,
    )
    return signal.to_dict()


@router.get(
    "/{symbol}/explain",
    summary="Get SHAP-Based Feature Attribution & Interpretability",
)
async def get_signal_explanation(
    symbol: str,
):
    """Retrieve detailed SHAP explainability decomposition:
    - Top contributing features
    - Positive feature impacts (forces pushing bullish)
    - Negative feature impacts (forces pushing bearish)
    - Natural language prediction explanation
    - Model confidence and ensemble agreement
    """
    clean_sym = symbol.strip().upper()

    # Get latest signal for features
    signal_res = realtime_signal_pipeline.get_latest_signal(clean_sym)
    if not signal_res:
        signal_res = ai_signal_engine.generate_signal(symbol=clean_sym)

    # Build representative feature set
    features = {
        "Technical AI Score": signal_res.component_scores.get("technical", 50.0) / 2.0 + 50.0,
        "ML Direction Prob": (signal_res.component_scores.get("machine_learning", 0.0) / 200.0 + 0.5) * 100.0,
        "Deep Learning Conf": (signal_res.component_scores.get("deep_learning", 0.0) / 200.0 + 0.5) * 100.0,
        "News Sentiment": signal_res.component_scores.get("sentiment", 0.0),
        "Market Regime Conviction": signal_res.component_scores.get("regime", 0.0),
        "VWAP Momentum": signal_res.component_scores.get("market_flow", 0.0),
        "Risk Headroom": signal_res.component_scores.get("risk", 0.0),
        "Anomaly Bias": signal_res.component_scores.get("anomaly", 0.0),
    }

    explanation = shap_explainer.explain_instance(
        features=features,
        symbol=clean_sym,
        model_confidence=signal_res.confidence_pct / 100.0,
        model_agreement_pct=85.0,
    )
    return explanation.to_dict()
