"""Tests for Market REST API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_api_market_status(client: AsyncClient):
    """Verify /api/v1/market/status endpoint for NSE and BSE."""
    # NSE status
    res_nse = await client.get("/api/v1/market/status?exchange=NSE")
    assert res_nse.status_code == 200
    data_nse = res_nse.json()
    assert data_nse["exchange"] == "NSE"
    assert "status" in data_nse
    assert "is_trading_open" in data_nse
    assert data_nse["timezone"] == "Asia/Kolkata"

    # BSE status
    res_bse = await client.get("/api/v1/market/status?exchange=BSE")
    assert res_bse.status_code == 200
    data_bse = res_bse.json()
    assert data_bse["exchange"] == "BSE"


@pytest.mark.asyncio
async def test_api_market_instruments(client: AsyncClient):
    """Verify /api/v1/market/instruments endpoint."""
    response = await client.get("/api/v1/market/instruments")
    assert response.status_code == 200
    instruments = response.json()
    assert len(instruments) > 0

    symbols = [inst["symbol"] for inst in instruments]
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert "NIFTY 50" in symbols

    # Test filtering by exchange
    res_bse = await client.get("/api/v1/market/instruments?exchange=BSE")
    assert res_bse.status_code == 200
    bse_insts = res_bse.json()
    for inst in bse_insts:
        assert inst["exchange"] == "BSE"


@pytest.mark.asyncio
async def test_api_market_quote_success(client: AsyncClient):
    """Verify fetching quote snapshot with explicit provenance labeling."""
    response = await client.get("/api/v1/market/quote/RELIANCE")
    assert response.status_code == 200
    tick = response.json()
    assert tick["symbol"] == "RELIANCE"
    assert tick["exchange"] == "NSE"
    assert tick["price"] > 0
    assert tick["data_mode"] == "REPLAY"
    assert tick["is_live"] is False
    assert "SIMULATED / REPLAY" in tick["disclaimer"]


@pytest.mark.asyncio
async def test_api_market_quote_not_found(client: AsyncClient):
    """Verify 404 for unknown ticker symbol."""
    response = await client.get("/api/v1/market/quote/INVALID_TICKER_XYZ")
    assert response.status_code == 404
    assert "not found" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_api_market_quotes_list(client: AsyncClient):
    """Verify /api/v1/market/quotes returns collection of quotes."""
    # Warm up cache with a quote query first
    await client.get("/api/v1/market/quote/TCS")

    response = await client.get("/api/v1/market/quotes")
    assert response.status_code == 200
    quotes = response.json()
    assert isinstance(quotes, list)
    assert len(quotes) >= 1


@pytest.mark.asyncio
async def test_api_market_historical_ohlcv(client: AsyncClient):
    """Verify fetching historical OHLCV bars with limit and timeframe."""
    response = await client.get("/api/v1/market/historical/INFY?timeframe=1d&limit=25")
    assert response.status_code == 200
    bars = response.json()
    assert len(bars) == 25

    for candle in bars:
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle
        assert "volume" in candle
        assert candle["data_mode"] == "REPLAY"
