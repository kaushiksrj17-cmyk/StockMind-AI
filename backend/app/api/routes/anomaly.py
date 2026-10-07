"""Market Anomaly Detection API Routes."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Query, status
import pandas as pd

from ai_engine.anomaly.anomaly_engine import anomaly_engine
from backend.app.schemas.risk_anomaly_portfolio import AnomalyResponse
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/anomaly", tags=["Anomaly Detection Engine"])


@router.get(
    "/{symbol}",
    response_model=AnomalyResponse,
    summary="Detect Market Anomalies",
)
async def get_market_anomalies(
    symbol: str,
    timeframe: str = Query(default="1d"),
    limit: int = Query(default=150, ge=30, le=500),
):
    """Run Isolation Forest, statistical price Z-score, volume spike, and price-volume divergence detection."""
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

    report = anomaly_engine.detect_anomalies(df=df, symbol=symbol)
    return report.to_dict()
