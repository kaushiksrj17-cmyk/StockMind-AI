"""Unit and Integration Tests for Frontend Services, Utils, and Data Provenance."""

import pytest
from unittest.mock import MagicMock, patch
from frontend.services.api_client import APIClient
from frontend.utils.formatters import (
    format_inr,
    format_percentage,
    format_volume,
    format_timestamp,
)
from frontend.components.badges import (
    render_data_mode_badge,
    render_market_status_badge,
    render_ws_badge,
)


class TestFrontendFormatters:
    """Test INR and market data formatting helpers."""

    def test_format_inr(self):
        assert format_inr(2850.5) == "₹2,850.50"
        assert format_inr(0) == "₹0.00"
        assert format_inr(-150.25) == "-₹150.25"

    def test_format_percentage(self):
        assert format_percentage(1.452) == "+1.45%"
        assert format_percentage(-2.3) == "-2.30%"
        assert format_percentage(0) == "+0.00%"

    def test_format_volume(self):
        assert format_volume(500) == "500"
        assert format_volume(25000) == "25.0K"
        assert format_volume(3400000) == "3.4M"
        assert format_volume(1500000000) == "1.5B"


class TestProvenanceBadges:
    """Verify that badges clearly distinguish LIVE, HISTORICAL, REPLAY, and AI PREDICTION."""

    def test_live_data_badge(self):
        badge = render_data_mode_badge("LIVE", is_live=True)
        assert "LIVE DATA" in badge
        assert "badge-live" in badge

    def test_historical_data_badge(self):
        badge = render_data_mode_badge("HISTORICAL")
        assert "HISTORICAL DATA" in badge
        assert "badge-historical" in badge

    def test_replay_data_badge(self):
        badge = render_data_mode_badge("REPLAY", is_live=False)
        assert "DEMO / REPLAY DATA" in badge
        assert "badge-replay" in badge

    def test_ai_prediction_badge(self):
        badge = render_data_mode_badge("PREDICTION")
        assert "AI PREDICTION" in badge
        assert "badge-prediction" in badge

    def test_market_status_badge(self):
        open_badge = render_market_status_badge("OPEN")
        assert "MARKET OPEN" in open_badge
        assert "badge-live" in open_badge

        closed_badge = render_market_status_badge("CLOSED")
        assert "MARKET CLOSED" in closed_badge
        assert "badge-replay" in closed_badge

    def test_ws_badge(self):
        connected = render_ws_badge(True, is_backend_online=True)
        assert "WS STREAM ONLINE" in connected
        assert "badge-live" in connected

        disconnected = render_ws_badge(False, is_backend_online=True)
        assert "WS POLLING" in disconnected
        assert "badge-replay" in disconnected

        offline = render_ws_badge(False, is_backend_online=False)
        assert "GATEWAY OFFLINE" in offline


class TestAPIClientUnit:
    """Test APIClient network failure fallbacks and parameter handling."""

    def test_api_client_initialization(self):
        client = APIClient("http://127.0.0.1:8000/")
        assert client.base_url == "http://127.0.0.1:8000"
        assert client.api_v1 == "http://127.0.0.1:8000/api/v1"

    def test_check_health_offline_fallback(self):
        client = APIClient("http://127.0.0.1:9999")  # Non-existent port
        health = client.check_health()
        assert health.get("status") == "offline"
        assert client.is_backend_reachable() is False

    def test_get_instruments_fallback(self):
        client = APIClient("http://127.0.0.1:9999")
        instruments = client.get_instruments()
        assert len(instruments) > 0
        symbols = [i["symbol"] for i in instruments]
        assert "RELIANCE" in symbols
        assert "TCS" in symbols
        assert "NIFTY 50" in symbols

    def test_get_market_status_fallback(self):
        client = APIClient("http://127.0.0.1:9999")
        status = client.get_market_status("NSE")
        assert status["exchange"] == "NSE"
        assert status["status"] == "UNKNOWN"
        assert status["is_trading_open"] is False
