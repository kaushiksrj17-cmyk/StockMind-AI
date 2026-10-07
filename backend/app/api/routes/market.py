"""Market Data REST API Routes.

Exposes endpoints for real-time and historical quotes, instruments,
and Indian market trading session status with explicit data provenance labeling.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from backend.app.services.market.market_service import market_service
from backend.app.services.market.market_status import MarketStatus
from backend.app.services.market.provider import Instrument, OHLCV, TickData

router = APIRouter(prefix="/market", tags=["Market Data"])


@router.get("/status", response_model=MarketStatus, summary="Indian Market Trading Session Status")
async def get_status(
    exchange: str = Query(default="NSE", description="Exchange: NSE or BSE"),
):
    """Retrieve current operating status of Indian exchanges (NSE / BSE).

    Evaluates regular trading hours (09:15 - 15:30 IST), pre-open, post-closing,
    weekends, and scheduled exchange holidays.
    """
    return market_service.get_market_status(exchange=exchange)


@router.get(
    "/instruments",
    response_model=List[Instrument],
    summary="List Indian Stock Market Instruments",
)
async def list_instruments(
    exchange: Optional[str] = Query(default=None, description="Filter by exchange (NSE/BSE)"),
):
    """Retrieve tradeable instruments across Indian equity and index segments."""
    return await market_service.get_instruments(exchange=exchange)


@router.get(
    "/quote/{symbol}",
    response_model=TickData,
    summary="Get Current Stock Quote Snapshot",
)
async def get_quote(symbol: str):
    """Fetch real-time or replay quote for a specific ticker symbol.

    Includes explicit data provenance (data_mode: LIVE or REPLAY),
    price change, day high/low, and disclaimer notice.
    """
    tick = await market_service.get_quote(symbol)
    if not tick:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quote not found for symbol '{symbol}'. Ensure it is a valid instrument.",
        )
    return tick


@router.get(
    "/quotes",
    response_model=List[TickData],
    summary="Get All Cached Quotes",
)
async def get_all_quotes():
    """Retrieve latest cached tick quotes for all monitored stock symbols."""
    quotes_dict = await market_service.get_all_quotes()
    return list(quotes_dict.values())


@router.get(
    "/historical/{symbol}",
    response_model=List[OHLCV],
    summary="Get Historical OHLCV Candles",
)
async def get_historical_ohlc(
    symbol: str,
    timeframe: str = Query(default="1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max number of candles to return"),
):
    """Retrieve historical open, high, low, close, and volume candles.

    Each candle is tagged with data_mode provenance.
    """
    bars = await market_service.get_historical_ohlc(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )
    if not bars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical data not available for symbol '{symbol}'.",
        )
    return bars
