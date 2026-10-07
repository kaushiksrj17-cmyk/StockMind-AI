"""StockMind-AI — Institutional Financial Terminal Streamlit Application.

High-performance real-time market analysis, quantitative forecasting, and
portfolio intelligence platform connecting directly to the FastAPI backend.
"""

import streamlit as st

# 1. Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="StockMind AI · Financial Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path

# Ensure workspace root is always in sys.path when launched via `streamlit run frontend/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.utils.markdown_patch import patch_streamlit_markdown
patch_streamlit_markdown()

from frontend.utils.state import initialize_session_state, get_backend_url
from frontend.styles.theme import apply_terminal_theme
from frontend.services.api_client import get_api_client
from frontend.components.header import render_terminal_header
from frontend.components.sidebar import render_sidebar

# Import All 14 Page Modules
from frontend.pages.command_center import render_command_center
from frontend.pages.live_market import render_live_market
from frontend.pages.stock_analysis import render_stock_analysis
from frontend.pages.ai_prediction import render_ai_prediction
from frontend.pages.technical_analysis import render_technical_analysis
from frontend.pages.news_sentiment import render_news_sentiment
from frontend.pages.risk_intelligence import render_risk_intelligence
from frontend.pages.portfolio import render_portfolio
from frontend.pages.backtesting_lab import render_backtesting_lab
from frontend.pages.model_laboratory import render_model_laboratory
from frontend.pages.ai_market_copilot import render_ai_market_copilot
from frontend.pages.market_heatmap import render_market_heatmap_page
from frontend.pages.alerts import render_alerts
from frontend.pages.settings_page import render_settings_page


def _safe_render_page(page_name: str, render_callable, **kwargs) -> None:
    """Safe page execution wrapper ensuring resilience even when backend is offline."""
    try:
        render_callable(**kwargs)
    except Exception as exc:
        st.warning(
            f"⚠️ **Backend unavailable — start FastAPI to enable live features.** ({page_name})"
        )
        st.info("Feature unavailable until the required service/configuration is enabled.")
        st.markdown(
            f"""
            <div style="background: rgba(255, 179, 0, 0.05); border: 1px solid rgba(255, 179, 0, 0.2); border-radius: 8px; padding: 10px 14px; margin-top: 8px; font-size: 0.78rem; color: #94A3B8;">
                <div><strong>Notice:</strong> Temporary connection disruption to FastAPI Gateway (<code>{st.session_state.get('backend_url', get_backend_url())}</code>).</div>
                <div style="margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748B;">{type(exc).__name__}: {exc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Page Callables for st.Page
def page_command_center() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Command Center",
        render_command_center,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        timeframe=st.session_state.get("active_timeframe", "1d"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_live_market() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Live Market",
        render_live_market,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_stock_analysis() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Stock Analysis",
        render_stock_analysis,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        timeframe=st.session_state.get("active_timeframe", "1d"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_ai_prediction() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "AI Prediction",
        render_ai_prediction,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        timeframe=st.session_state.get("active_timeframe", "1d"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_technical_analysis() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Technical Analysis",
        render_technical_analysis,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        timeframe=st.session_state.get("active_timeframe", "1d"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_news_sentiment() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "News & Sentiment",
        render_news_sentiment,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_risk_intelligence() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Risk Intelligence",
        render_risk_intelligence,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_portfolio() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Portfolio",
        render_portfolio,
        api_client=api_client,
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_backtesting_lab() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Backtesting Lab",
        render_backtesting_lab,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_model_laboratory() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Model Laboratory",
        render_model_laboratory,
        api_client=api_client,
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_ai_market_copilot() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "AI Market Copilot",
        render_ai_market_copilot,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_market_heatmap() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Market Heatmap",
        render_market_heatmap_page,
        api_client=api_client,
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_alerts() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Alerts",
        render_alerts,
        api_client=api_client,
        symbol=st.session_state.get("active_symbol", "RELIANCE"),
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def page_settings() -> None:
    api_client = get_api_client()
    _safe_render_page(
        "Settings",
        render_settings_page,
        api_client=api_client,
        theme_mode=st.session_state.get("theme_mode", "Dark"),
    )


def main() -> None:
    """Application main orchestrator and navigation router."""
    # 1. Initialize session state variables
    initialize_session_state()

    # 2. Build 14 Navigation Pages grouped into clean functional categories
    page_cmd = st.Page(page_command_center, title="Command Center", icon="⚡", default=True)
    page_cmd_hidden = st.Page(page_command_center, title="Command Center", icon="⚡", url_path="command_center", visibility="hidden")
    page_live = st.Page(page_live_market, title="Live Market", icon="📈", url_path="live_market")
    page_heat = st.Page(page_market_heatmap, title="Market Heatmap", icon="🗺️", url_path="market_heatmap")
    page_stock = st.Page(page_stock_analysis, title="Stock Analysis", icon="🔍", url_path="stock_analysis")

    page_pred = st.Page(page_ai_prediction, title="AI Prediction", icon="🔮", url_path="ai_prediction")
    page_tech = st.Page(page_technical_analysis, title="Technical Analysis", icon="📊", url_path="technical_analysis")
    page_model = st.Page(page_model_laboratory, title="Model Laboratory", icon="🔬", url_path="model_laboratory")
    page_news = st.Page(page_news_sentiment, title="News & Sentiment", icon="📰", url_path="news_sentiment")
    page_copilot = st.Page(page_ai_market_copilot, title="AI Market Copilot", icon="🤖", url_path="ai_market_copilot")

    page_risk = st.Page(page_risk_intelligence, title="Risk Intelligence", icon="🛡️", url_path="risk_intelligence")
    page_port = st.Page(page_portfolio, title="Portfolio", icon="💼", url_path="portfolio")
    page_backtest = st.Page(page_backtesting_lab, title="Backtesting Lab", icon="🧪", url_path="backtesting_lab")
    page_alert = st.Page(page_alerts, title="Alerts", icon="🔔", url_path="alerts")

    page_set = st.Page(page_settings, title="Settings", icon="⚙️", url_path="settings_page")
    page_set_hidden = st.Page(page_settings, title="Settings", icon="⚙️", url_path="settings", visibility="hidden")

    pages = {
        "MARKET": [page_cmd, page_cmd_hidden, page_live, page_heat, page_stock],
        "AI INTELLIGENCE": [page_pred, page_tech, page_model, page_news, page_copilot],
        "RISK & PORTFOLIO": [page_risk, page_port, page_backtest, page_alert],
        "SYSTEM": [page_set, page_set_hidden],
    }

    # Register navigation system with Streamlit
    active_page = st.navigation(pages, position="sidebar")

    # Update active section in session state for state consistency
    st.session_state["section_name"] = active_page.title
    st.session_state["active_section"] = active_page.title

    # 3. Handle query-parameter routing fallback if provided
    query_nav = st.query_params.get("page") or st.query_params.get("section") or st.query_params.get("nav")
    if query_nav:
        clean_nav = query_nav.strip().lower()
        route_map = {
            "command_center": page_cmd,
            "live_market": page_live,
            "stock_analysis": page_stock,
            "ai_prediction": page_pred,
            "technical_analysis": page_tech,
            "news_sentiment": page_news,
            "risk_intelligence": page_risk,
            "portfolio": page_port,
            "backtesting_lab": page_backtest,
            "model_laboratory": page_model,
            "ai_market_copilot": page_copilot,
            "market_heatmap": page_heat,
            "alerts": page_alert,
            "settings": page_set,
            "settings_page": page_set,
        }
        target_page = route_map.get(clean_nav)
        if target_page and target_page != active_page:
            st.query_params.clear()
            st.switch_page(target_page)

    # 4. Resolve API client & check backend health
    api_client = get_api_client()
    theme_mode = st.session_state.get("theme_mode", "Dark")
    apply_terminal_theme(theme_mode=theme_mode)

    health_data = api_client.check_health()
    is_backend_online = health_data.get("status") in ("healthy", "degraded")
    st.session_state["backend_status"] = health_data.get("status", "offline")
    st.session_state["backend_healthy"] = is_backend_online

    # If backend gateway is offline, display informative notice
    if not is_backend_online:
        st.warning(
            f"⚠️ **Backend unavailable — start FastAPI to enable live features.** "
            f"(`{st.session_state.get('backend_url', get_backend_url())}`) — "
            "Terminal operating in sandbox **DEMO / REPLAY** mode."
        )

    # 5. Fetch market metadata & instruments
    instruments = api_client.get_instruments(exchange="NSE")
    market_status = api_client.get_market_status(exchange="NSE")

    # 6. Render Global Sidebar Controls
    render_sidebar(instruments=instruments)

    # 7. Active Quote & Header telemetry
    active_symbol = st.session_state.get("active_symbol", "RELIANCE")
    active_quote = api_client.get_quote(active_symbol)
    data_mode = active_quote.get("data_mode", "REPLAY") if active_quote else "REPLAY"
    is_live = active_quote.get("is_live", False) if active_quote else False
    is_ws_connected = is_backend_online and st.session_state.get("ws_connected", True)

    # Render Institutional Terminal Top Header
    render_terminal_header(
        market_status=market_status,
        data_mode=data_mode,
        is_live=is_live,
        is_ws_connected=is_ws_connected,
        is_backend_online=is_backend_online,
    )

    # 8. Render the currently selected page
    active_page.run()


if __name__ == "__main__":
    main()
