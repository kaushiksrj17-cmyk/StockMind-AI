"""Market Service Orchestrator.

Integrates the active market data provider (LIVE or REPLAY), in-memory cache,
WebSocket manager, and Indian market status engine.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from backend.app.config import settings
from backend.app.services.market.market_cache import MarketCache, market_cache
from backend.app.services.market.market_status import MarketStatus, get_market_status
from backend.app.services.market.provider import (
    BaseMarketDataProvider,
    DataMode,
    Instrument,
    LiveBrokerMarketDataProvider,
    OHLCV,
    TickData,
)
from backend.app.services.market.replay_provider import ReplayMarketDataProvider
from backend.app.services.market.websocket_manager import (
    MarketWebSocketManager,
    market_ws_manager,
)

logger = logging.getLogger("stockmind.market.service")


class MarketService:
    """Central market data coordinator service."""

    def __init__(
        self,
        cache: MarketCache = market_cache,
        ws_manager: MarketWebSocketManager = market_ws_manager,
    ) -> None:
        self.cache = cache
        self.ws_manager = ws_manager
        self.provider: BaseMarketDataProvider = self._create_provider()
        self._setup_pipelines()

    def _create_provider(self) -> BaseMarketDataProvider:
        """Instantiate appropriate provider based on application configuration."""
        configured_mode = settings.MARKET_DATA_MODE.strip().upper()

        if configured_mode == "LIVE":
            logger.info("Initializing market service in LIVE mode...")
            live_prov = LiveBrokerMarketDataProvider()
            if not live_prov.validate_credentials():
                logger.warning(
                    "LIVE mode requested but broker credentials (MARKET_API_KEY / MARKET_ACCESS_TOKEN) "
                    "are not set in .env. Falling back safely to REPLAY mode. "
                    "Data will be labeled strictly as DataMode.REPLAY."
                )
                return ReplayMarketDataProvider()
            return live_prov

        logger.info("Initializing market service in REPLAY/DEMO mode.")
        return ReplayMarketDataProvider()

    def _setup_pipelines(self) -> None:
        """Bind tick processing and client subscription pipelines."""
        # Provider ticks -> cache -> WebSocket broadcast
        self.provider.register_tick_handler(self._on_tick_received)
        # Client socket symbol subscriptions -> provider.subscribe()
        self.ws_manager.register_provider_subscriber(self.provider.subscribe)

    async def _on_tick_received(self, tick: TickData) -> None:
        """Process an incoming tick: update in-memory cache, execute fast AI inference, and broadcast."""
        await self.cache.set_quote(tick)
        await self.ws_manager.broadcast_tick(tick)

        # Real-time tick-to-signal pipeline
        try:
            from ai_engine.signals.realtime_pipeline import realtime_signal_pipeline
            mode_label = f"{self.data_mode.value.upper()} DATA"
            signal = realtime_signal_pipeline.process_live_tick(
                symbol=tick.symbol,
                price=float(tick.last_price),
                volume=float(tick.volume),
                day_open=float(tick.open_price),
                timestamp=tick.timestamp.isoformat() if tick.timestamp else None,
                data_mode=mode_label,
            )
            await self.ws_manager.broadcast_signal(tick.symbol, signal.to_dict())
        except Exception as e:
            logger.debug("Real-time signal pipeline tick skipped: %s", str(e))

    async def start(self) -> bool:
        """Start the market data service and connect provider."""
        logger.info(
            "Starting Market Service [%s mode] (Provider: %s)...",
            self.provider.data_mode.value,
            self.provider.__class__.__name__,
        )
        return await self.provider.connect()

    async def stop(self) -> None:
        """Stop provider stream and clean up resources."""
        logger.info("Stopping Market Service...")
        await self.provider.disconnect()

    @property
    def data_mode(self) -> DataMode:
        """Current operational data mode."""
        return self.provider.data_mode

    @property
    def is_connected(self) -> bool:
        """Provider connection state."""
        return self.provider.is_connected

    async def get_quote(self, symbol: str) -> Optional[TickData]:
        """Fetch latest quote, trying cache first with fallback to provider."""
        clean_symbol = symbol.strip().upper()
        cached = await self.cache.get_quote(clean_symbol)
        if cached:
            return cached
        # Fallback to provider snapshot
        tick = await self.provider.get_quote(clean_symbol)
        if tick:
            await self.cache.set_quote(tick)
        return tick

    async def get_all_quotes(self) -> Dict[str, TickData]:
        """Return all currently cached quotes."""
        return await self.cache.get_all_quotes()

    async def get_historical_ohlc(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OHLCV]:
        """Fetch historical bars, checking cache before provider query."""
        clean_symbol = symbol.strip().upper()
        cached = await self.cache.get_historical(clean_symbol, timeframe)
        if cached and len(cached) >= limit:
            return cached[-limit:]

        bars = await self.provider.get_historical_ohlc(
            symbol=clean_symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        if bars:
            await self.cache.set_historical(clean_symbol, timeframe, bars)
        return bars

    async def get_instruments(self, exchange: Optional[str] = None) -> List[Instrument]:
        """List tradeable Indian stock instruments for specified exchange."""
        return await self.provider.get_instruments(exchange=exchange)

    def get_market_status(
        self,
        exchange: str = "NSE",
        reference_time: Optional[datetime] = None,
    ) -> MarketStatus:
        """Compute Indian market trading session status."""
        return get_market_status(exchange=exchange, reference_time=reference_time)


# Global singleton service
market_service = MarketService()
