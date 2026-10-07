"""High-Speed In-Memory Market Cache Layer.

Stores real-time tick snapshots, rolling price buffers, and historical OHLCV candles
for low-latency retrieval by REST endpoints and WebSocket broadcasters.
"""

from collections import deque
from typing import Deque, Dict, List, Optional
import asyncio
from backend.app.services.market.provider import OHLCV, TickData


class MarketCache:
    """Thread-safe and async-compatible in-memory market cache."""

    def __init__(self, max_tick_buffer_size: int = 100) -> None:
        self.max_tick_buffer_size = max_tick_buffer_size
        self._quotes: Dict[str, TickData] = {}
        self._tick_history: Dict[str, Deque[TickData]] = {}
        self._historical_bars: Dict[str, List[OHLCV]] = {}
        self._lock = asyncio.Lock()

    async def set_quote(self, tick: TickData) -> None:
        """Cache the latest tick and append to symbol history buffer."""
        clean_symbol = tick.symbol.strip().upper()
        async with self._lock:
            self._quotes[clean_symbol] = tick
            if clean_symbol not in self._tick_history:
                self._tick_history[clean_symbol] = deque(maxlen=self.max_tick_buffer_size)
            self._tick_history[clean_symbol].append(tick)

    def get_quote_sync(self, symbol: str) -> Optional[TickData]:
        """Synchronous fast-path read for latest symbol snapshot."""
        return self._quotes.get(symbol.strip().upper())

    async def get_quote(self, symbol: str) -> Optional[TickData]:
        """Retrieve latest snapshot quote for symbol."""
        clean_symbol = symbol.strip().upper()
        async with self._lock:
            return self._quotes.get(clean_symbol)

    async def get_all_quotes(self) -> Dict[str, TickData]:
        """Return shallow copy of all cached latest quotes."""
        async with self._lock:
            return dict(self._quotes)

    async def get_recent_ticks(self, symbol: str, limit: int = 50) -> List[TickData]:
        """Retrieve the rolling buffer of recent ticks for a symbol."""
        clean_symbol = symbol.strip().upper()
        async with self._lock:
            buf = self._tick_history.get(clean_symbol)
            if not buf:
                return []
            ticks = list(buf)
            return ticks[-limit:] if limit < len(ticks) else ticks

    async def set_historical(
        self,
        symbol: str,
        timeframe: str,
        bars: List[OHLCV],
    ) -> None:
        """Cache historical OHLCV candle collection."""
        key = f"{symbol.strip().upper()}_{timeframe.lower()}"
        async with self._lock:
            self._historical_bars[key] = list(bars)

    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[List[OHLCV]]:
        """Retrieve cached historical bars if present."""
        key = f"{symbol.strip().upper()}_{timeframe.lower()}"
        async with self._lock:
            bars = self._historical_bars.get(key)
            return list(bars) if bars else None

    async def clear(self) -> None:
        """Flush all cached quotes and buffers."""
        async with self._lock:
            self._quotes.clear()
            self._tick_history.clear()
            self._historical_bars.clear()


# Global singleton cache instance
market_cache = MarketCache()
