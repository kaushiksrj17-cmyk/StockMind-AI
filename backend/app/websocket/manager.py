"""Real-Time WebSocket Connection & Topic Subscription Manager."""

from typing import Dict, List, Set, Any
from fastapi import WebSocket
import json


class ConnectionManager:
    """Manages active WebSocket connections and ticker/topic channel subscriptions."""

    def __init__(self) -> None:
        # All connected clients
        self.active_connections: List[WebSocket] = []
        # Symbol-specific subscriptions: { "AAPL": {ws1, ws2}, "NVDA": {ws2, ws3} }
        self.subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Deregister a disconnected client and purge all subscriptions."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        for symbol, subscribers in list(self.subscriptions.items()):
            subscribers.discard(websocket)
            if not subscribers:
                self.subscriptions.pop(symbol, None)

    def subscribe(self, websocket: WebSocket, symbol: str) -> None:
        """Subscribe a client socket to market updates for a specific ticker symbol."""
        clean_symbol = symbol.strip().upper()
        if clean_symbol not in self.subscriptions:
            self.subscriptions[clean_symbol] = set()
        self.subscriptions[clean_symbol].add(websocket)

    def unsubscribe(self, websocket: WebSocket, symbol: str) -> None:
        """Unsubscribe a client from a specific ticker symbol."""
        clean_symbol = symbol.strip().upper()
        if clean_symbol in self.subscriptions:
            self.subscriptions[clean_symbol].discard(websocket)
            if not self.subscriptions[clean_symbol]:
                self.subscriptions.pop(clean_symbol, None)

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """Send a JSON payload to a specific client."""
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        text = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(text)
            except Exception:
                self.disconnect(connection)

    async def broadcast_to_symbol(self, symbol: str, message: Dict[str, Any]) -> None:
        """Broadcast a message exclusively to subscribers of a particular ticker."""
        clean_symbol = symbol.strip().upper()
        subscribers = self.subscriptions.get(clean_symbol, set())
        text = json.dumps(message)
        for connection in list(subscribers):
            try:
                await connection.send_text(text)
            except Exception:
                self.disconnect(connection)


# Global singleton instance
ws_manager = ConnectionManager()
