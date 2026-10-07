"""Unit tests for Replay and Demo Market Data Provider."""

import asyncio
import pytest
from backend.app.services.market.provider import DataMode, TickData
from backend.app.services.market.replay_provider import ReplayMarketDataProvider


@pytest.mark.asyncio
async def test_replay_provider_lifecycle():
    """Verify replay provider starts, connects, streams, and disconnects cleanly."""
    provider = ReplayMarketDataProvider(tick_interval_seconds=0.1)
    connected = await provider.connect()
    assert connected is True
    assert provider.is_connected is True
    assert provider.data_mode == DataMode.REPLAY

    # Top symbols subscribed by default
    assert "RELIANCE" in provider.subscribed_symbols

    await provider.disconnect()
    assert provider.is_connected is False


@pytest.mark.asyncio
async def test_replay_quote_provenance_and_data_integrity():
    """Verify replay quote is never labeled as live and carries explicit disclaimer."""
    provider = ReplayMarketDataProvider()

    quote = await provider.get_quote("RELIANCE")
    assert quote is not None
    assert quote.symbol == "RELIANCE"
    assert quote.exchange == "NSE"
    assert quote.price > 2000.0
    assert quote.open > 0
    assert quote.high >= quote.low
    assert quote.volume > 0

    # Strict provenance tests
    assert quote.data_mode == DataMode.REPLAY
    assert quote.is_live is False
    assert "SIMULATED / REPLAY" in quote.disclaimer.upper()
    assert "NOT LIVE MARKET" in quote.disclaimer.upper()


@pytest.mark.asyncio
async def test_replay_instruments_list():
    """Verify replay provider serves Indian benchmark instruments (NSE/BSE)."""
    provider = ReplayMarketDataProvider()

    instruments = await provider.get_instruments()
    symbols = [inst.symbol for inst in instruments]
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert "HDFCBANK" in symbols
    assert "NIFTY 50" in symbols
    assert "SENSEX" in symbols

    # Test exchange filter
    bse_only = await provider.get_instruments(exchange="BSE")
    for inst in bse_only:
        assert inst.exchange == "BSE"


@pytest.mark.asyncio
async def test_replay_historical_ohlcv():
    """Verify historical candle series generation for Indian stocks."""
    provider = ReplayMarketDataProvider()

    bars = await provider.get_historical_ohlc("TCS", timeframe="1d", limit=30)
    assert len(bars) == 30

    for bar in bars:
        assert bar.open > 0
        assert bar.high >= bar.low
        assert bar.high >= bar.open
        assert bar.high >= bar.close
        assert bar.low <= bar.open
        assert bar.low <= bar.close
        assert bar.volume > 0
        assert bar.data_mode == DataMode.REPLAY


@pytest.mark.asyncio
async def test_replay_tick_streaming():
    """Verify active streaming loop emits ticks to registered callback."""
    provider = ReplayMarketDataProvider(tick_interval_seconds=0.05)
    received = []

    async def on_tick(t: TickData):
        received.append(t)

    provider.register_tick_handler(on_tick)
    await provider.connect()
    await provider.subscribe(["INFY"])

    # Wait briefly for streaming loop to emit ticks
    await asyncio.sleep(0.2)
    await provider.disconnect()

    assert len(received) > 0
    for tick in received:
        assert tick.data_mode == DataMode.REPLAY
        assert tick.is_live is False
