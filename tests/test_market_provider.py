"""Unit tests for Market Data Provider Abstraction and Live Broker Connector."""

import pytest
from datetime import datetime, timezone
from backend.app.services.market.provider import (
    BaseMarketDataProvider,
    DataMode,
    Instrument,
    LiveBrokerMarketDataProvider,
    OHLCV,
    TickData,
)


@pytest.mark.asyncio
async def test_live_broker_credential_validation():
    """Verify live broker provider detects missing credentials and refuses live connection."""
    provider = LiveBrokerMarketDataProvider()
    # By default, test environment has empty broker credentials
    provider.api_key = ""
    provider.access_token = ""

    assert provider.validate_credentials() is False
    connected = await provider.connect()
    assert connected is False
    assert provider.is_connected is False
    assert provider.data_mode == DataMode.LIVE


@pytest.mark.asyncio
async def test_live_broker_credential_success_handshake():
    """Verify live broker succeeds handshake when credentials are provided."""
    provider = LiveBrokerMarketDataProvider()
    provider.api_key = "test_broker_api_key_valid"
    provider.access_token = "test_broker_daily_token_valid"

    assert provider.validate_credentials() is True
    connected = await provider.connect()
    assert connected is True
    assert provider.is_connected is True

    await provider.subscribe(["RELIANCE", "TCS"])
    assert "RELIANCE" in provider.subscribed_symbols
    assert "TCS" in provider.subscribed_symbols

    await provider.unsubscribe(["RELIANCE"])
    assert "RELIANCE" not in provider.subscribed_symbols
    assert "TCS" in provider.subscribed_symbols

    await provider.disconnect()
    assert provider.is_connected is False


@pytest.mark.asyncio
async def test_tick_emission_pipeline():
    """Verify tick handler callback receives emitted ticks."""
    provider = LiveBrokerMarketDataProvider()
    received_ticks = []

    async def sample_handler(tick: TickData):
        received_ticks.append(tick)

    provider.register_tick_handler(sample_handler)

    test_tick = TickData(
        symbol="INFY",
        exchange="NSE",
        price=1850.0,
        open=1840.0,
        high=1860.0,
        low=1835.0,
        close=1845.0,
        change=5.0,
        change_percent=0.27,
        volume=1200000,
        timestamp=datetime.now(timezone.utc),
        data_mode=DataMode.LIVE,
        is_live=True,
        disclaimer="VERIFIED LIVE MARKET DATA",
    )

    await provider.emit_tick(test_tick)
    assert len(received_ticks) == 1
    assert received_ticks[0].symbol == "INFY"
    assert received_ticks[0].data_mode == DataMode.LIVE
    assert received_ticks[0].is_live is True
