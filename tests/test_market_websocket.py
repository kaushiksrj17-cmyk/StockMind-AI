"""Tests for Market WebSocket Manager and Real-Time WebSocket Streaming."""

import json
import pytest
from datetime import datetime, timezone
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.services.market.provider import DataMode, TickData
from backend.app.services.market.websocket_manager import MarketWebSocketManager


@pytest.fixture
def ws_manager():
    """Isolated manager fixture."""
    return MarketWebSocketManager()


class DummyWebSocket:
    """Mock WebSocket for unit testing MarketWebSocketManager."""

    def __init__(self):
        self.sent_messages = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, text: str):
        self.sent_messages.append(text)


@pytest.mark.asyncio
async def test_ws_manager_lifecycle_and_subscription(ws_manager: MarketWebSocketManager):
    """Verify socket connection, subscription, and unsubscription logic."""
    mock_ws = DummyWebSocket()
    await ws_manager.connect(mock_ws)
    assert mock_ws in ws_manager.active_connections
    assert mock_ws.accepted is True

    # Subscribe to RELIANCE and TCS
    await ws_manager.subscribe(mock_ws, ["RELIANCE", "TCS"])
    assert mock_ws in ws_manager.symbol_subscribers["RELIANCE"]
    assert mock_ws in ws_manager.symbol_subscribers["TCS"]

    # Verify subscription acknowledgment was sent
    assert len(mock_ws.sent_messages) == 1
    ack = json.loads(mock_ws.sent_messages[0])
    assert ack["type"] == "subscription_ack"
    assert "RELIANCE" in ack["symbols"]

    # Unsubscribe from TCS
    await ws_manager.unsubscribe(mock_ws, ["TCS"])
    assert "TCS" not in ws_manager.symbol_subscribers

    # Disconnect
    ws_manager.disconnect(mock_ws)
    assert mock_ws not in ws_manager.active_connections
    assert "RELIANCE" not in ws_manager.symbol_subscribers


@pytest.mark.asyncio
async def test_ws_manager_broadcast_tick(ws_manager: MarketWebSocketManager):
    """Verify tick is delivered only to clients subscribed to that specific symbol."""
    client_a = DummyWebSocket()
    client_b = DummyWebSocket()

    await ws_manager.connect(client_a)
    await ws_manager.connect(client_b)

    # Client A wants INFY, Client B wants HDFCBANK
    await ws_manager.subscribe(client_a, ["INFY"])
    await ws_manager.subscribe(client_b, ["HDFCBANK"])

    client_a.sent_messages.clear()
    client_b.sent_messages.clear()

    # Emit INFY tick
    infy_tick = TickData(
        symbol="INFY",
        exchange="NSE",
        price=1890.0,
        open=1880.0,
        high=1895.0,
        low=1875.0,
        close=1885.0,
        change=5.0,
        change_percent=0.26,
        volume=1500000,
        timestamp=datetime.now(timezone.utc),
        data_mode=DataMode.REPLAY,
        is_live=False,
        disclaimer="REPLAY",
    )

    await ws_manager.broadcast_tick(infy_tick)

    # Client A must receive INFY tick, Client B must not
    assert len(client_a.sent_messages) == 1
    assert "INFY" in client_a.sent_messages[0]
    assert len(client_b.sent_messages) == 0


def test_fastapi_websocket_market_integration():
    """Verify end-to-end WebSocket stream handshake and subscription in FastAPI."""
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        # 1. Connection acknowledgment
        raw_ack = ws.receive_text()
        ack = json.loads(raw_ack)
        assert ack["type"] == "connection_ack"
        assert "data_mode" in ack

        # 2. Ping / Pong
        ws.send_text(json.dumps({"action": "ping"}))
        raw_pong = ws.receive_text()
        pong = json.loads(raw_pong)
        assert pong["type"] == "pong"

        # 3. Subscribe to RELIANCE
        ws.send_text(json.dumps({"action": "subscribe", "symbol": "RELIANCE"}))
        sub_ack = json.loads(ws.receive_text())
        assert sub_ack["type"] == "subscription_ack"
        assert "RELIANCE" in sub_ack["symbols"]

        # 4. Instant tick snapshot received for RELIANCE
        tick_data = json.loads(ws.receive_text())
        assert tick_data["type"] == "tick"
        assert tick_data["data"]["symbol"] == "RELIANCE"
        assert tick_data["data"]["data_mode"] == "REPLAY"
