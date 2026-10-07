"""Automated Audit & Verification for Streamlit Navigation Architecture.

Tests:
1. Registration of all 14 required pages via modern st.navigation & st.Page.
2. Clean title labels and exact URL path routing.
3. Exception-free rendering across all 14 pages.
4. HTTP 200 response on all 14 URL routes against the running Streamlit server.
5. Graceful offline degradation when backend is unreachable.
"""

import sys
from pathlib import Path
import pytest
import httpx
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

APP_PATH = str(ROOT_DIR / "frontend" / "app.py")


EXPECTED_PAGES = [
    ("Command Center", ["", "command_center"]),
    ("Live Market", ["live_market"]),
    ("Stock Analysis", ["stock_analysis"]),
    ("AI Prediction", ["ai_prediction"]),
    ("Technical Analysis", ["technical_analysis"]),
    ("News & Sentiment", ["news_sentiment"]),
    ("Risk Intelligence", ["risk_intelligence"]),
    ("Portfolio", ["portfolio"]),
    ("Backtesting Lab", ["backtesting_lab"]),
    ("Model Laboratory", ["model_laboratory"]),
    ("AI Market Copilot", ["ai_market_copilot"]),
    ("Market Heatmap", ["market_heatmap"]),
    ("Alerts", ["alerts"]),
    ("Settings", ["settings_page", "settings"]),
]


def test_navigation_registration():
    """Verify that st.navigation registers all 14 required pages with clean labels."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0, f"App execution failed with: {at.exception}"

    registered = at._registered_pages
    registered_names = {v["page_name"] for v in registered.values()}
    registered_urls = {v["url_pathname"] for v in registered.values()}

    for label, expected_urls in EXPECTED_PAGES:
        assert label in registered_names, f"Page label '{label}' not found in registered pages: {registered_names}"
        for u in expected_urls:
            assert u in registered_urls, f"URL path '{u}' for '{label}' not registered: {registered_urls}"


@pytest.mark.parametrize("route_slug", [
    "command_center",
    "live_market",
    "stock_analysis",
    "ai_prediction",
    "technical_analysis",
    "news_sentiment",
    "risk_intelligence",
    "portfolio",
    "backtesting_lab",
    "model_laboratory",
    "ai_market_copilot",
    "market_heatmap",
    "alerts",
    "settings_page",
])
def test_each_page_renders_without_crash(route_slug: str):
    """Verify that every single page opens and renders without unhandled exceptions."""
    at = AppTest.from_file(APP_PATH)
    at.query_params["nav"] = route_slug
    at.run(timeout=20)
    assert len(at.exception) == 0, f"Page '{route_slug}' raised exceptions: {[e.value for e in at.exception]}"


@pytest.mark.parametrize("url_path", [
    "/",
    "/command_center",
    "/live_market",
    "/stock_analysis",
    "/ai_prediction",
    "/technical_analysis",
    "/news_sentiment",
    "/risk_intelligence",
    "/portfolio",
    "/backtesting_lab",
    "/model_laboratory",
    "/ai_market_copilot",
    "/market_heatmap",
    "/alerts",
    "/settings_page",
    "/settings",
])
def test_live_streamlit_http_endpoints(url_path: str):
    """Verify HTTP status 200 on all routes against the running Streamlit instance."""
    try:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            resp = client.get(f"http://localhost:8501{url_path}")
            assert resp.status_code == 200, f"Expected 200 for {url_path}, got {resp.status_code}"
    except httpx.ConnectError:
        pytest.skip("Streamlit dev server not currently running on port 8501")


def test_offline_backend_graceful_handling():
    """Verify that when backend is offline, pages still render safely with a warning."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["backend_url"] = "http://127.0.0.1:9999"
    at.run(timeout=20)
    assert len(at.exception) == 0, f"App crashed in offline mode: {at.exception}"
    # Verify warning message appears
    warnings = [w.value for w in at.warning]
    assert any("Backend unavailable" in w or "FastAPI" in w for w in warnings), f"Expected offline notice in warnings: {warnings}"
