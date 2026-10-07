"""Market WebSocket Manager.

Handles client connections, symbol-level real-time subscriptions,
and broadcasts live/replay tick feeds to frontend subscribers.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

from backend.app.services.market.provider import TickData

logger = logging.getLogger("stockmind.market.ws")


class MarketWebSocketManager:
    """Manages active WebSocket sessions subscribed to market data streams."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        # Mapping from symbol to set of subscribed WebSockets
        self.symbol_subscribers: Dict[str, Set[WebSocket]] = {}
        # Mapping from WebSocket to set of subscribed symbols
        self.client_subscriptions: Dict[WebSocket, Set[str]] = {}
        # Callback to inform market data provider of new symbols to stream
        self._provider_subscribe_cb: Optional[Callable[[List[str]], Any]] = None

    def register_provider_subscriber(self, cb: Callable[[List[str]], Any]) -> None:
        """Register provider callback triggered when new symbols are subscribed by clients."""
        self._provider_subscribe_cb = cb

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register incoming client socket."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_subscriptions[websocket] = set()
        logger.info("Market WS client connected (total active: %d)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Deregister disconnected socket and remove all its symbol subscriptions."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        subscribed_symbols = self.client_subscriptions.pop(websocket, set())
        for sym in subscribed_symbols:
            if sym in self.symbol_subscribers:
                self.symbol_subscribers[sym].discard(websocket)
                if not self.symbol_subscribers[sym]:
                    del self.symbol_subscribers[sym]

        logger.info("Market WS client disconnected (remaining: %d)", len(self.active_connections))

    async def subscribe(self, websocket: WebSocket, symbols: List[str]) -> None:
        """Subscribe client to one or more ticker symbols."""
        clean_symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        new_provider_symbols: List[str] = []

        for symbol in clean_symbols:
            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
                new_provider_symbols.append(symbol)

            self.symbol_subscribers[symbol].add(websocket)
            if websocket in self.client_subscriptions:
                self.client_subscriptions[websocket].add(symbol)

        # Notify provider to fetch/stream these symbols if new
        if new_provider_symbols and self._provider_subscribe_cb:
            try:
                res = self._provider_subscribe_cb(new_provider_symbols)
                if asyncio_is_coroutine(res):
                    await res
            except Exception as exc:
                logger.error("Error triggering provider subscribe for %s: %s", new_provider_symbols, exc)

        # Send acknowledgment back to client
        await self.send_personal_message(
            {
                "type": "subscription_ack",
                "symbols": clean_symbols,
                "symbol": clean_symbols[0] if clean_symbols else None,
                "status": "subscribed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            websocket,
        )

    async def unsubscribe(self, websocket: WebSocket, symbols: List[str]) -> None:
        """Unsubscribe client from one or more ticker symbols."""
        clean_symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        for symbol in clean_symbols:
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(websocket)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]

            if websocket in self.client_subscriptions:
                self.client_subscriptions[websocket].discard(symbol)

        await self.send_personal_message(
            {
                "type": "unsubscription_ack",
                "symbols": clean_symbols,
                "symbol": clean_symbols[0] if clean_symbols else None,
                "status": "unsubscribed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            websocket,
        )

    async def broadcast_tick(self, tick: TickData) -> None:
        """Push real-time tick to all clients subscribed to tick.symbol."""
        subscribers = self.symbol_subscribers.get(tick.symbol.upper(), set())
        if not subscribers:
            return

        payload = {
            "type": "tick",
            "data": tick.model_dump(mode="json"),
        }
        text = json.dumps(payload)

        for ws in list(subscribers):
            try:
                await ws.send_text(text)
            except Exception:
                self.disconnect(ws)

    async def broadcast_signal(self, symbol: str, signal_data: Dict[str, Any]) -> None:
        """Push real-time AI signal update to subscribers."""
        subscribers = self.symbol_subscribers.get(symbol.upper(), set())
        payload = {
            "type": "ai_signal",
            "symbol": symbol.upper(),
            "data": signal_data,
        }
        text = json.dumps(payload)
        targets = subscribers if subscribers else self.active_connections

        for ws in list(targets):
            try:
                await ws.send_text(text)
            except Exception:
                self.disconnect(ws)

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """Send direct payload to single client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)


def asyncio_is_coroutine(obj: Any) -> bool:
    """Helper to detect coroutine or awaitable returned by callback."""
    import inspect
    return inspect.isawaitable(obj)


# Global singleton instance
market_ws_manager = MarketWebSocketManager()
