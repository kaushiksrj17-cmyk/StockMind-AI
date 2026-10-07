"""Comprehensive System Audit & Final Verification Tests for StockMind-AI.

Audits and verifies:
- Backend configuration and USB storage isolation
- Detailed health check telemetry
- Database connectivity and local SQLite resolution
- Market provider abstraction (REPLAY mode without keys vs LIVE credential validation)
- WebSocket connection, subscription, and tick broadcast
- End-to-end Explainable AI (SHAP) and 8-Factor AI Signal Engine
- Model registry and AI subsystems availability
"""

import datetime
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.config import BASE_DIR, settings
from backend.app.database import check_db_connection
from backend.app.main import app
from backend.app.services.market.provider import LiveBrokerMarketDataProvider
from backend.app.services.market.replay_provider import ReplayMarketDataProvider
from ai_engine.explainability.shap_explainer import shap_explainer
from ai_engine.signals.signal_engine import ai_signal_engine
from ai_engine.signals.realtime_pipeline import realtime_signal_pipeline


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# 1. CONFIGURATION & USB ISOLATION AUDIT
# ============================================================================

def test_configuration_validation_and_usb_isolation():
    """Verify that settings are valid and all paths remain strictly confined to USB."""
    audit = settings.validate_configuration()
    assert audit["status"] in ("VALID", "WARNING")
    assert audit["is_usb_confined"] is True
    assert len(audit["issues"]) == 0

    # Verify USB root corresponds to D:\StockMind-AI
    usb_root = Path(audit["usb_root"])
    assert "StockMind-AI" in usb_root.name or "StockMind-AI" in str(usb_root)
    assert not str(audit["database_path"]).lower().startswith("c:\\users")


def test_detailed_health_check_endpoint(client: TestClient):
    """Verify GET /api/v1/health/detailed reports all subsystems and USB telemetry."""
    response = client.get("/api/v1/health/detailed")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert data["database"]["status"] == "connected"
    assert "market_engine" in data
    assert "ai_subsystems" in data
    assert "usb_storage" in data
    assert data["usb_storage"]["db_exists"] is True
    assert data["usb_storage"]["models_dir_exists"] is True
    assert "config_audit" in data
    assert data["config_audit"]["is_usb_confined"] is True


# ============================================================================
# 2. MARKET PROVIDER ABSTRACTION & CREDENTIAL VERIFICATION
# ============================================================================

from backend.app.services.market.provider import DataMode

@pytest.mark.asyncio
async def test_replay_mode_operates_without_credentials():
    """Verify Replay provider operates independently without requiring broker keys."""
    provider = ReplayMarketDataProvider()
    assert provider.data_mode == DataMode.REPLAY
    await provider.connect()
    assert provider.is_connected is True

    # Test historical bar retrieval
    bars = await provider.get_historical_ohlc("RELIANCE", limit=10)
    assert len(bars) > 0
    assert bars[0].close > 0
    assert bars[0].open > 0

    await provider.disconnect()
    assert provider.is_connected is False


def test_live_provider_validates_credentials():
    """Verify Live provider rejects invalid/missing credentials with explicit instructions."""
    live_provider = LiveBrokerMarketDataProvider()
    assert live_provider.data_mode == DataMode.LIVE
    live_provider.api_key = ""
    live_provider.access_token = ""
    assert live_provider.validate_credentials() is False


# ============================================================================
# 3. WEBSOCKET REAL-TIME COMMUNICATION AUDIT
# ============================================================================

def test_websocket_ping_pong_and_provenance():
    """Verify WebSocket protocol, ping-pong latency, and data provenance disclaimer."""
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws") as ws:
            # 1. Connection acknowledgement
            ack = ws.receive_json()
            assert ack["type"] == "connection_ack"
            assert "data_mode" in ack
            assert "disclaimer" in ack

            # 2. Ping-pong
            ws.send_json({"action": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"
            assert "timestamp" in pong

            # 3. Subscribe to ticker
            ws.send_json({"action": "subscribe", "symbols": ["RELIANCE"]})
            # Subscribed ticker acknowledgement
            ack_sub = ws.receive_json()
            assert ack_sub["type"] == "subscription_ack"
            # Subscribed ticker snapshot push
            tick_msg = ws.receive_json()
            assert tick_msg["type"] == "tick"
            assert tick_msg["data"]["symbol"] == "RELIANCE"


# ============================================================================
# 4. END-TO-END EXPLAINABLE AI & REAL-TIME SIGNAL SYNTHESIS
# ============================================================================

def test_end_to_end_ai_signal_synthesis_flow():
    """Verify synthesis of 8 dimensions into final institutional AI Signal with SHAP."""
    signal_res = ai_signal_engine.generate_signal(
        symbol="TCS",
        live_quote={"last_price": 3850.0, "change_percent": 1.25, "vwap": 3830.0},
        technical_data={"technical_score": 78.5, "rsi_14": 62.0, "trend_direction": "BULLISH"},
        ml_prediction={"direction": "UP", "probability": 0.74, "confidence": 0.74},
        deep_learning_forecast={"direction": "UP", "confidence": 0.70},
        sentiment_data={"sentiment_score": 0.42, "classification": "POSITIVE"},
        risk_metrics={"var_95_pct": 1.8, "sortino_ratio": 2.1, "volatility_annualized": 0.16},
        anomaly_report={"is_anomaly": False, "anomaly_score": 0.1},
        regime_data={"primary_regime": "BULL", "confidence_pct": 82.0},
    )

    assert signal_res.symbol == "TCS"
    assert signal_res.signal in ("STRONG BULLISH", "BULLISH")
    assert signal_res.composite_score >= 15.0
    assert signal_res.confidence_pct > 60.0
    assert len(signal_res.supporting_factors) > 0
    assert len(signal_res.shap_top_features) > 0
    assert "FINANCIAL ADVICE" in signal_res.compliance_disclaimer.upper()


def test_signals_api_routes(client: TestClient):
    """Verify GET /api/v1/signals/{symbol} and /explain endpoints."""
    # Test main signal endpoint
    res_sig = client.get("/api/v1/signals/RELIANCE")
    assert res_sig.status_code == 200
    sig_data = res_sig.json()
    assert sig_data["symbol"] == "RELIANCE"
    assert sig_data["signal"] in ("STRONG BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG BEARISH")
    assert "composite_score" in sig_data
    assert "confidence_pct" in sig_data
    assert "supporting_factors" in sig_data
    assert "opposing_factors" in sig_data
    assert "compliance_disclaimer" in sig_data

    # Test explainability endpoint
    res_exp = client.get("/api/v1/signals/RELIANCE/explain")
    assert res_exp.status_code == 200
    exp_data = res_exp.json()
    assert exp_data["symbol"] == "RELIANCE"
    assert "top_contributing_features" in exp_data
    assert "positive_impacts" in exp_data
    assert "negative_impacts" in exp_data
    assert "explanation_narrative" in exp_data
