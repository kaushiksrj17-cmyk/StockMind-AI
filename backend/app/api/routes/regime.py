"""Market Regime Detection API Routes."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Query, status
import pandas as pd

from ai_engine.regime.regime_engine import regime_engine
from backend.app.schemas.risk_anomaly_portfolio import RegimeResponse
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/regime", tags=["Market Regime Engine"])


@router.get(
    "/{symbol}",
    response_model=RegimeResponse,
    summary="Detect Market Regime",
)
async def get_market_regime(
    symbol: str,
    timeframe: str = Query(default="1d"),
    limit: int = Query(default=200, ge=30, le=500),
):
    """Classify instrument into Bull, Bear, Sideways, or High Volatility regimes with probabilistic confidence."""
    bars = await market_service.get_historical_ohlc(symbol=symbol, timeframe=timeframe, limit=limit)
    if not bars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical data unavailable for symbol '{symbol}'.",
        )

    records = [
        b.model_dump() if hasattr(b, "model_dump") else (b.dict() if hasattr(b, "dict") else b.__dict__)
        for b in bars
    ]
    df = pd.DataFrame(records)

    classification = regime_engine.detect_regime(df=df, symbol=symbol)
    return classification.to_dict()
