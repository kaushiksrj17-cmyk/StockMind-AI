"""Technical Analysis REST API Routes.

Exposes endpoints for computing mathematical technical indicators,
moving average cross signals, candlestick patterns, support/resistance,
Fibonacci retracements, and the 0-100 AI Technical Score.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
import pandas as pd

from ai_engine.feature_engineering.engine import TechnicalAnalysisEngine
from backend.app.schemas.technical import (
    TechnicalAnalysisResponse,
    TechnicalScoreOnlyResponse,
    TechnicalIndicatorsOnlyResponse,
)
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/technical", tags=["Technical Analysis Engine"])
engine = TechnicalAnalysisEngine()


async def _get_df_for_symbol(symbol: str, timeframe: str = "1d", limit: int = 200) -> pd.DataFrame:
    """Helper to fetch historical OHLCV data from market_service and return a prepared DataFrame."""
    bars = await market_service.get_historical_ohlc(symbol=symbol, timeframe=timeframe, limit=limit)
    if not bars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical price data unavailable for symbol '{symbol}'. Ensure it is a valid instrument.",
        )

    records = [
        b.model_dump() if hasattr(b, "model_dump") else (b.dict() if hasattr(b, "dict") else b.__dict__)
        for b in bars
    ]
    df = pd.DataFrame(records)
    return df


@router.get(
    "/analysis/{symbol}",
    response_model=TechnicalAnalysisResponse,
    summary="Comprehensive Technical Analysis",
)
async def get_technical_analysis(
    symbol: str,
    timeframe: str = Query(default="1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
    limit: int = Query(default=200, ge=30, le=1000, description="Lookback bars to compute indicators"),
):
    """Run full technical analysis engine on a stock or index.

    Calculates:
    - Moving averages (SMA 20/50/100/200, EMA 9/21/50/200, Golden & Death Crosses)
    - Oscillators (RSI-14, MACD, Stochastic %K/%D, ROC, Momentum)
    - Volatility (Bollinger Bands, ATR, Historical Volatility)
    - Trend (ADX, +DI, -DI, Trend Strength & Direction)
    - Volume flow (OBV, VWAP, RVOL, Price-Volume Divergence)
    - Price Action (Support & Resistance pivots, Fibonacci levels, Candlestick patterns)
    - AI Technical Score (0-100) with Bullish/Bearish/Neutral classification and explanations
    """
    df = await _get_df_for_symbol(symbol=symbol, timeframe=timeframe, limit=limit)
    try:
        result = engine.analyze(df=df, symbol=symbol, timeframe=timeframe)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Technical analysis computation error: {str(exc)}",
        )


@router.get(
    "/score/{symbol}",
    response_model=TechnicalScoreOnlyResponse,
    summary="AI Technical Score & Classification",
)
async def get_technical_score(
    symbol: str,
    timeframe: str = Query(default="1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
    limit: int = Query(default=200, ge=30, le=1000, description="Lookback bars"),
):
    """Retrieve 0-100 AI Technical Score, classification, and plain-English explanations."""
    df = await _get_df_for_symbol(symbol=symbol, timeframe=timeframe, limit=limit)
    try:
        result = engine.analyze(df=df, symbol=symbol, timeframe=timeframe)
        return {
            "symbol": result["symbol"],
            "current_price": result["current_price"],
            "ai_technical_score": result["ai_technical_score"],
            "disclaimer": result["disclaimer"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Score computation error: {str(exc)}",
        )


@router.get(
    "/indicators/{symbol}",
    response_model=TechnicalIndicatorsOnlyResponse,
    summary="Raw Technical Indicators",
)
async def get_technical_indicators(
    symbol: str,
    timeframe: str = Query(default="1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
    limit: int = Query(default=200, ge=30, le=1000, description="Lookback bars"),
):
    """Retrieve pure mathematical indicator values for external algorithmic consumers."""
    df = await _get_df_for_symbol(symbol=symbol, timeframe=timeframe, limit=limit)
    try:
        result = engine.analyze(df=df, symbol=symbol, timeframe=timeframe)
        return {
            "symbol": result["symbol"],
            "current_price": result["current_price"],
            "moving_averages": result["moving_averages"],
            "oscillators": result["oscillators"],
            "volatility": result["volatility"],
            "trend": result["trend"],
            "volume_profile": result["volume_profile"],
            "disclaimer": result["disclaimer"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indicator computation error: {str(exc)}",
        )
