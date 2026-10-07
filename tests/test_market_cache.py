"""Unit tests for In-Memory Market Cache Layer."""

import pytest
from datetime import datetime, timezone
from backend.app.services.market.market_cache import MarketCache
from backend.app.services.market.provider import DataMode, OHLCV, TickData


@pytest.fixture
def cache():
    """Create fresh isolated cache instance for each test."""
    return MarketCache(max_tick_buffer_size=5)


@pytest.fixture
def sample_tick():
    """Sample tick packet fixture."""
    return TickData(
        symbol="SBIN",
        exchange="NSE",
        price=820.50,
        open=810.0,
        high=825.0,
        low=808.0,
        close=815.0,
        change=5.50,
        change_percent=0.67,
        volume=2500000,
        timestamp=datetime.now(timezone.utc),
        data_mode=DataMode.REPLAY,
        is_live=False,
        disclaimer="REPLAY",
    )


@pytest.mark.asyncio
async def test_cache_set_and_get_quote(cache: MarketCache, sample_tick: TickData):
    """Verify storing and retrieving latest quote snapshot."""
    await cache.set_quote(sample_tick)
    cached = await cache.get_quote("SBIN")
    assert cached is not None
    assert cached.symbol == "SBIN"
    assert cached.price == 820.50

    # Fast sync accessor
    assert cache.get_quote_sync("SBIN") is not None


@pytest.mark.asyncio
async def test_cache_rolling_buffer_and_limit(cache: MarketCache, sample_tick: TickData):
    """Verify rolling buffer stores ticks up to max buffer size."""
    for i in range(10):
        t = sample_tick.model_copy()
        t.price = 820.0 + i
        await cache.set_quote(t)

    recent = await cache.get_recent_ticks("SBIN", limit=10)
    # Cache was configured with max_tick_buffer_size=5
    assert len(recent) == 5
    assert recent[-1].price == 829.0


@pytest.mark.asyncio
async def test_cache_historical_bars(cache: MarketCache):
    """Verify historical candles can be cached and retrieved."""
    bars = [
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=50000,
            data_mode=DataMode.REPLAY,
        )
    ]
    await cache.set_historical("TATAMOTORS", "1d", bars)
    retrieved = await cache.get_historical("TATAMOTORS", "1d")
    assert retrieved is not None
    assert len(retrieved) == 1
    assert retrieved[0].open == 100.0


@pytest.mark.asyncio
async def test_cache_clear(cache: MarketCache, sample_tick: TickData):
    """Verify cache clear purges all quotes and buffers."""
    await cache.set_quote(sample_tick)
    await cache.clear()
    assert await cache.get_quote("SBIN") is None
    assert len(await cache.get_all_quotes()) == 0
