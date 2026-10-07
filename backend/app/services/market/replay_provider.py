"""Replay & Demo Market Data Provider for Indian Stock Markets (NSE/BSE).

Provides deterministic and realistic historical simulation data for testing,
backtesting, and offline demonstration without live broker credentials.
STRICT RULE: All generated data is explicitly tagged as DataMode.REPLAY.
It is NEVER disguised or labeled as live market data.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import math
import random
from typing import Any, Dict, List, Optional

from backend.app.services.market.provider import (
    BaseMarketDataProvider,
    DataMode,
    Instrument,
    OHLCV,
    TickData,
)

# Standard benchmark Indian stocks and indices with realistic base prices (INR)
DEFAULT_INDIAN_INSTRUMENTS: List[Dict[str, Any]] = [
    {
        "symbol": "RELIANCE",
        "name": "Reliance Industries Limited",
        "exchange": "NSE",
        "segment": "EQUITY",
        "base_price": 2980.50,
        "lot_size": 1,
        "tick_size": 0.05,
    },
    {
        "symbol": "TCS",
        "name": "Tata Consultancy Services Limited",
        "exchange": "NSE",
        "segment": "EQUITY",
        "base_price": 4210.25,
        "lot_size": 1,
        "tick_size": 0.05,
    },
    {
        "symbol": "HDFCBANK",
        "name": "HDFC Bank Limited",
        "exchange": "NSE",
        "segment": "EQUITY",
        "base_price": 1640.80,
        "lot_size": 1,
        "tick_size": 0.05,
    },
    {
        "symbol": "INFY",
        "name": "Infosys Limited",
        "exchange": "NSE",
        "segment": "EQUITY",
        "base_price": 1890.15,
        "lot_size": 1,
        "tick_size": 0.05,
    },
    {
        "symbol": "ICICIBANK",
        "name": "ICICI Bank Limited",
        "exchange": "NSE",
        "segment": "EQUITY",
        "base_price": 1220.40,
        "lot_size": 1,
        "tick_size": 0.05,
    },
    {
        "symbol": "SBIN",
        "name": "State Bank of India",
        "exchange": "NSE",
        "segment": "EQUITY",
        "base_price": 815.60,
        "lot_size": 1,
        "tick_size": 0.05,
    },
    {
        "symbol": "TATAMOTORS",
        "name": "Tata Motors Limited",
        "exchange": "NSE",
        "segment": "EQUITY",
        "base_price": 965.30,
        "lot_size": 1,
        "tick_size": 0.05,
    },
    {
        "symbol": "NIFTY 50",
        "name": "NIFTY 50 Benchmark Index",
        "exchange": "NSE",
        "segment": "INDEX",
        "base_price": 24850.00,
        "lot_size": 25,
        "tick_size": 0.05,
    },
    {
        "symbol": "SENSEX",
        "name": "S&P BSE SENSEX Index",
        "exchange": "BSE",
        "segment": "INDEX",
        "base_price": 81200.00,
        "lot_size": 10,
        "tick_size": 0.05,
    },
]


class ReplayMarketDataProvider(BaseMarketDataProvider):
    """Simulated and historical replay market data provider."""

    def __init__(self, tick_interval_seconds: float = 1.0) -> None:
        super().__init__(data_mode=DataMode.REPLAY)
        self.tick_interval = tick_interval_seconds
        self._instruments_map: Dict[str, Dict[str, Any]] = {
            item["symbol"]: item for item in DEFAULT_INDIAN_INSTRUMENTS
        }
        self._current_prices: Dict[str, float] = {
            s: item["base_price"] for s, item in self._instruments_map.items()
        }
        self._stream_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Start replay stream engine."""
        self._is_connected = True
        # Subscribe to top 3 stocks by default for instant streaming demonstration
        if not self._subscribed_symbols:
            self._subscribed_symbols.update({"RELIANCE", "TCS", "NIFTY 50"})

        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._simulation_loop())
        return True

    async def disconnect(self) -> None:
        """Stop replay stream engine."""
        self._is_connected = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self._stream_task = None

    async def subscribe(self, symbols: List[str]) -> None:
        """Add symbols to active streaming replay."""
        for s in symbols:
            clean = s.strip().upper()
            if clean in self._instruments_map:
                self._subscribed_symbols.add(clean)

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Remove symbols from active streaming replay."""
        for s in symbols:
            clean = s.strip().upper()
            self._subscribed_symbols.discard(clean)

    def _generate_tick(self, symbol: str) -> TickData:
        """Generate a realistic replay tick packet with strict provenance metadata."""
        inst = self._instruments_map[symbol]
        base = inst["base_price"]
        current = self._current_prices[symbol]

        # Small stochastic price walk (±0.15% per tick)
        noise = (random.random() - 0.49) * 0.003
        new_price = round(max(current * (1 + noise), base * 0.7), 2)
        self._current_prices[symbol] = new_price

        change = round(new_price - base, 2)
        change_pct = round((change / base) * 100, 2)

        high = max(new_price, base * 1.015)
        low = min(new_price, base * 0.985)
        open_price = round(base * 0.998, 2)
        volume = random.randint(100000, 3500000)

        now = datetime.now(timezone.utc)
        return TickData(
            symbol=symbol,
            exchange=inst["exchange"],
            price=new_price,
            open=open_price,
            high=round(high, 2),
            low=round(low, 2),
            close=base,
            change=change,
            change_percent=change_pct,
            volume=volume,
            timestamp=now,
            data_mode=DataMode.REPLAY,
            is_live=False,
            disclaimer="SIMULATED / REPLAY DATA. NOT LIVE MARKET FEED.",
            last_trade_time=now,
        )

    async def _simulation_loop(self) -> None:
        """Background loop continuously emitting replay ticks for subscribed symbols."""
        while self._is_connected:
            try:
                for symbol in list(self._subscribed_symbols):
                    if symbol in self._instruments_map:
                        tick = self._generate_tick(symbol)
                        await self.emit_tick(tick)
                await asyncio.sleep(self.tick_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)

    async def get_quote(self, symbol: str) -> Optional[TickData]:
        """Generate on-demand tick snapshot for requested symbol."""
        clean_symbol = symbol.strip().upper()
        if clean_symbol not in self._instruments_map:
            return None
        return self._generate_tick(clean_symbol)

    async def get_historical_ohlc(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """Synthesize realistic historical OHLCV candles based on benchmark parameters."""
        clean_symbol = symbol.strip().upper()
        inst = self._instruments_map.get(clean_symbol)
        base_price = inst["base_price"] if inst else 1000.0

        if end_time is None:
            end_time = datetime.now(timezone.utc)

        delta = timedelta(days=1) if timeframe.lower() == "1d" else timedelta(hours=1)
        bars: List[OHLCV] = []
        current_time = end_time - (delta * limit)
        price = base_price * 0.90

        for i in range(limit):
            # Geometric random walk with slight upward drift
            drift = 0.0005
            shock = (random.random() - 0.48) * 0.02
            bar_open = price
            price = round(bar_open * (1 + drift + shock), 2)
            high_offset = abs(random.gauss(0, 0.008)) * price
            low_offset = abs(random.gauss(0, 0.008)) * price
            bar_high = round(max(bar_open, price) + high_offset, 2)
            bar_low = round(min(bar_open, price) - low_offset, 2)
            bar_volume = random.randint(500000, 8000000)

            bars.append(
                OHLCV(
                    timestamp=current_time,
                    open=bar_open,
                    high=bar_high,
                    low=bar_low,
                    close=price,
                    volume=bar_volume,
                    data_mode=DataMode.REPLAY,
                )
            )
            current_time += delta

        return bars

    async def get_instruments(self, exchange: Optional[str] = None) -> List[Instrument]:
        """Return Indian benchmark instruments (filtered by exchange if provided)."""
        instruments: List[Instrument] = []
        for item in DEFAULT_INDIAN_INSTRUMENTS:
            if exchange and item["exchange"].upper() != exchange.strip().upper():
                continue
            instruments.append(
                Instrument(
                    symbol=item["symbol"],
                    name=item["name"],
                    exchange=item["exchange"],
                    segment=item["segment"],
                    lot_size=item["lot_size"],
                    tick_size=item["tick_size"],
                    token=f"NSE_{item['symbol']}",
                )
            )
        return instruments
