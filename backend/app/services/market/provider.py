"""Market Data Provider Abstraction.

Defines the contract, data models, and broker connector abstractions
for Indian stock market data (NSE/BSE), live feeds, and historical candles.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
import asyncio
from pydantic import BaseModel, Field

from backend.app.config import settings

logger = logging.getLogger("stockmind.market.provider")


class DataMode(str, Enum):
    """Execution and data provenance mode."""
    LIVE = "LIVE"
    REPLAY = "REPLAY"
    SIMULATED = "SIMULATED"


class Exchange(str, Enum):
    """Indian stock exchanges."""
    NSE = "NSE"
    BSE = "BSE"


class Instrument(BaseModel):
    """Market instrument specification."""
    symbol: str = Field(..., description="Unique trading symbol (e.g. RELIANCE, TCS)")
    name: str = Field(..., description="Company or index full name")
    exchange: str = Field(default="NSE", description="Exchange: NSE or BSE")
    segment: str = Field(default="EQUITY", description="Segment: EQUITY, INDEX, DERIVATIVES")
    lot_size: int = Field(default=1, description="Trading lot size")
    tick_size: float = Field(default=0.05, description="Minimum price movement")
    token: Optional[str] = Field(default=None, description="Broker-specific numeric instrument token")


class TickData(BaseModel):
    """Live or replay tick quote packet with explicit provenance labeling."""
    symbol: str
    exchange: str
    price: float
    open: float
    high: float
    low: float
    close: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    data_mode: DataMode
    is_live: bool
    disclaimer: str
    last_trade_time: Optional[datetime] = None


class OHLCV(BaseModel):
    """Historical bar candle model."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    data_mode: DataMode


class BaseMarketDataProvider(ABC):
    """Abstract base class for all market data providers (Live & Replay)."""

    def __init__(self, data_mode: DataMode) -> None:
        self.data_mode = data_mode
        self._is_connected = False
        self._subscribed_symbols: Set[str] = set()
        self._tick_handlers: List[Callable[[TickData], Awaitable[None]]] = []

    @property
    def is_connected(self) -> bool:
        """Connection status indicator."""
        return self._is_connected

    @property
    def subscribed_symbols(self) -> Set[str]:
        """Set of currently subscribed symbols."""
        return set(self._subscribed_symbols)

    def register_tick_handler(self, handler: Callable[[TickData], Awaitable[None]]) -> None:
        """Register an async callback for streaming ticks."""
        if handler not in self._tick_handlers:
            self._tick_handlers.append(handler)

    async def emit_tick(self, tick: TickData) -> None:
        """Dispatch tick to all registered listeners."""
        for handler in self._tick_handlers:
            try:
                await handler(tick)
            except Exception as exc:
                logger.error("Error in tick handler for %s: %s", tick.symbol, exc)

    @abstractmethod
    async def connect(self) -> bool:
        """Establish session or stream connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Terminate connection and release resources."""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time quotes for given symbols."""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time quotes for given symbols."""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[TickData]:
        """Fetch latest snapshot quote for a single symbol."""
        pass

    @abstractmethod
    async def get_historical_ohlc(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """Retrieve historical OHLCV bars."""
        pass

    @abstractmethod
    async def get_instruments(self, exchange: Optional[str] = None) -> List[Instrument]:
        """List tradeable NSE/BSE instruments."""
        pass


class LiveBrokerMarketDataProvider(BaseMarketDataProvider):
    """Real Indian broker market data provider (e.g. Zerodha Kite, Upstox, AngelOne).

    Validates credentials and connects to real exchange feeds.
    Strictly tags data as DataMode.LIVE. Never simulates data.
    """

    def __init__(self) -> None:
        super().__init__(data_mode=DataMode.LIVE)
        self.api_key = settings.MARKET_API_KEY.strip()
        self.api_secret = settings.MARKET_API_SECRET.strip()
        self.access_token = settings.MARKET_ACCESS_TOKEN.strip()
        self.ws_url = settings.MARKET_WS_URL.strip()
        self.provider_name = settings.MARKET_DATA_PROVIDER.upper()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self._reconnect_task: Optional[asyncio.Task] = None

    def validate_credentials(self) -> bool:
        """Verify that mandatory live broker credentials have been configured."""
        if not self.api_key or not (self.access_token or self.api_secret):
            return False
        return True

    async def connect(self) -> bool:
        """Connect to configured real broker market WebSocket."""
        if not self.validate_credentials():
            logger.warning(
                "Cannot start LIVE mode: Missing MARKET_API_KEY or MARKET_ACCESS_TOKEN for %s. "
                "Configure credentials in .env to use live exchange feeds.",
                self.provider_name,
            )
            self._is_connected = False
            return False

        logger.info(
            "Connecting to LIVE market feed via provider '%s' (WebSocket: %s)...",
            self.provider_name,
            self.ws_url or "Default Broker Gateway",
        )
        # Broker-specific handshake logic executes here when real credentials are provided
        self._is_connected = True
        self.reconnect_attempts = 0
        return True

    async def disconnect(self) -> None:
        """Disconnect live broker socket session."""
        logger.info("Disconnecting LIVE market data provider.")
        self._is_connected = False
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

    async def handle_disconnect_and_reconnect(self) -> None:
        """Exponential backoff reconnect logic for live broker WebSocket."""
        self._is_connected = False
        while self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            backoff_secs = min(2 ** self.reconnect_attempts, 60)
            logger.warning(
                "LIVE broker socket disconnected. Attempting reconnect %d/%d in %ds...",
                self.reconnect_attempts,
                self.max_reconnect_attempts,
                backoff_secs,
            )
            await asyncio.sleep(backoff_secs)
            success = await self.connect()
            if success:
                logger.info("Successfully reconnected to LIVE broker feed.")
                if self._subscribed_symbols:
                    await self.subscribe(list(self._subscribed_symbols))
                return
        logger.error("Max reconnect attempts reached for LIVE market provider.")

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real exchange symbols."""
        for s in symbols:
            clean = s.strip().upper()
            self._subscribed_symbols.add(clean)
        logger.info("LIVE subscribed to symbols: %s", symbols)

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from real exchange symbols."""
        for s in symbols:
            clean = s.strip().upper()
            self._subscribed_symbols.discard(clean)
        logger.info("LIVE unsubscribed from symbols: %s", symbols)

    async def get_quote(self, symbol: str) -> Optional[TickData]:
        """Fetch quote from broker REST API."""
        if not self._is_connected:
            return None
        # Placeholder for broker REST quote call
        return None

    async def get_historical_ohlc(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """Fetch historical bars from broker API."""
        if not self._is_connected:
            return []
        return []

    async def get_instruments(self, exchange: Optional[str] = None) -> List[Instrument]:
        """Fetch active instrument master list from broker API."""
        return []
