"""WebSocket connectivity and message subscription tests."""

import json
from starlette.testclient import TestClient
from backend.app.main import app


def test_websocket_connection_and_ping():
    """Verify WebSocket connection handshake and ping/pong response."""
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        # Receive welcome connection acknowledgment
        ack_data = websocket.receive_text()
        ack = json.loads(ack_data)
        assert ack["type"] == "connection_ack"

        # Send ping
        websocket.send_text(json.dumps({"action": "ping"}))
        pong_data = websocket.receive_text()
        pong = json.loads(pong_data)
        assert pong["type"] == "pong"


def test_websocket_symbol_subscription():
    """Verify subscribing and unsubscribing to stock tickers via WebSocket."""
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.receive_text()  # connection_ack

        # Subscribe to AAPL
        websocket.send_text(json.dumps({"action": "subscribe", "symbol": "AAPL"}))
        sub_data = websocket.receive_text()
        sub_ack = json.loads(sub_data)
        assert sub_ack["type"] == "subscription_ack"
        assert sub_ack["symbol"] == "AAPL"

        # Unsubscribe from AAPL
        websocket.send_text(json.dumps({"action": "unsubscribe", "symbol": "AAPL"}))
        unsub_data = websocket.receive_text()
        unsub_ack = json.loads(unsub_data)
        assert unsub_ack["type"] == "unsubscription_ack"
        assert unsub_ack["symbol"] == "AAPL"
