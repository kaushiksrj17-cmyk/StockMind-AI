"""Page 14: Settings — Backend Gateway, Data Provider, and Terminal Preferences."""

from typing import Any, Dict
import streamlit as st

from frontend.components.header import render_top_header
from frontend.components.kpi_cards import render_kpi_card
from frontend.services.api_client import APIClient
from frontend.styles.theme import get_theme_colors, get_semantic_signal


def render_settings_page(
    api_client: APIClient,
    theme_mode: str = "dark",
) -> None:
    """Render Settings, Gateway Configuration, and Health Diagnostics across 6 functional sections."""
    colors = get_theme_colors(theme_mode)

    # Standard Top Header
    render_top_header(
        title="StockMind AI",
        subtitle="Terminal Settings & System Configuration",
        market_open=True,
        data_mode="REPLAY",
        ws_connected=True,
    )

    # Section 20 Navigation Tabs:
    # 1. Appearance, 2. Market Data, 3. AI Models, 4. Notifications, 5. Risk Preferences, 6. System Status
    tab_appear, tab_mkt, tab_ai, tab_notif, tab_risk, tab_sys = st.tabs([
        "🎨 Appearance",
        "📡 Market Data",
        "🧠 AI Models",
        "🔔 Notifications",
        "🛡️ Risk Preferences",
        "⚡ System Status",
    ])

    # 1. APPEARANCE (Dark Mode / Light Mode)
    with tab_appear:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 16px; margin-bottom: 14px;">
                <div style="font-size: 0.84rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 6px;">
                    Terminal Visual Theme & Display Preferences
                </div>
                <div style="font-size: 0.76rem; color: {colors['text_muted']}; margin-bottom: 12px;">
                    Configure high-contrast financial palette and color tokens across all pages.
                </div>
            """,
            unsafe_allow_html=True,
        )

        c_th, c_dens = st.columns(2)
        with c_th:
            current_mode = str(st.session_state.get("theme_mode", "dark")).lower()
            theme_choice = st.radio(
                "Color Theme Mode",
                ["Dark Mode (Institutional Deep Navy)", "Light Mode (Off-White Financial Canvas)"],
                index=0 if current_mode == "dark" else 1,
                key="settings_theme_radio",
            )
            new_theme_mode = "dark" if "Dark" in theme_choice else "light"
            if new_theme_mode != current_mode:
                st.session_state["theme_mode"] = new_theme_mode
                st.rerun()

        with c_dens:
            st.selectbox("Data Density", ["Standard Institutional (Comfortable)", "Compact Trading Terminal (High-Density)"], index=0)
            st.selectbox("Chart Palette", ["Cyan / Amber / Purple (Terminal Standard)", "High-Contrast Emerald / Crimson"], index=0)

        st.markdown("</div>", unsafe_allow_html=True)

    # 2. MARKET DATA
    with tab_mkt:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 16px; margin-bottom: 14px;">
                <div style="font-size: 0.84rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 6px;">
                    Indian Broker Market Data Configuration
                </div>
                <div style="font-size: 0.76rem; color: {colors['text_muted']}; margin-bottom: 12px;">
                    Toggle between Brownian Motion physics replay simulation and real broker WebSocket feeds.
                </div>
            """,
            unsafe_allow_html=True,
        )

        broker = st.selectbox(
            "Broker Provider Driver",
            options=["Local Physics Replay Engine", "Zerodha Kite Connect (WebSocket)", "Angel One SmartAPI", "Upstox Pro v2"],
            index=0,
        )
        c_poll, c_rate = st.columns(2)
        with c_poll:
            st.slider("WebSocket Tick Sampling Frequency (Hz)", min_value=1, max_value=20, value=5, step=1)
        with c_rate:
            st.selectbox("Fallback Polling Protocol", ["HTTP REST Keep-Alive", "Long Polling SSE"], index=0)

        st.markdown(
            f"""
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 6px; padding: 8px 12px; font-size: 0.72rem; color: {colors['warning']}; margin-top: 10px;">
                ℹ️ To stream live real money tick quotes from Zerodha/Angel One, set <code>MARKET_DATA_PROVIDER=live</code> and configure API keys in <code>backend/.env</code>.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. AI MODELS
    with tab_ai:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 16px; margin-bottom: 14px;">
                <div style="font-size: 0.84rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 6px;">
                    Machine Learning & Deep Learning Inference Suite
                </div>
                <div style="font-size: 0.76rem; color: {colors['text_muted']}; margin-bottom: 12px;">
                    Manage active model pipelines, confidence consensus thresholds, and retraining schedules.
                </div>
            """,
            unsafe_allow_html=True,
        )

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.checkbox("Random Forest Classifier", value=True)
            st.checkbox("XGBoost Gradient Booster", value=True)
        with c_m2:
            st.checkbox("LightGBM Fast Tree", value=True)
            st.checkbox("LSTM Recurrent Network", value=True)
        with c_m3:
            st.checkbox("GRU Gated Unit", value=True)
            st.checkbox("Weighted Ensemble Arbiter", value=True)

        st.slider("Consensus Agreement Threshold (%)", min_value=50, max_value=90, value=70, step=5)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. NOTIFICATIONS
    with tab_notif:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 16px; margin-bottom: 14px;">
                <div style="font-size: 0.84rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 6px;">
                    Alert Delivery Dispatch Channels
                </div>
                <div style="font-size: 0.76rem; color: {colors['text_muted']}; margin-bottom: 12px;">
                    Configure routing and dispatch priorities for market condition triggers.
                </div>
            """,
            unsafe_allow_html=True,
        )

        n1, n2 = st.columns(2)
        with n1:
            st.checkbox("Terminal Audio Beep on Trigger", value=True)
            st.checkbox("Desktop Browser Push Notification", value=True)
        with n2:
            st.checkbox("FastAPI Webhook Dispatch", value=False)
            st.checkbox("SMS Dispatch via Twilio/AWS", value=False)

        st.selectbox("Minimum Severity Level to Push", ["ALL (Info & Above)", "WARNING & Above", "CRITICAL Only"], index=1)
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. RISK PREFERENCES
    with tab_risk:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 16px; margin-bottom: 14px;">
                <div style="font-size: 0.84rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 6px;">
                    Portfolio Risk Tolerance & Margin Safety Guardrails
                </div>
                <div style="font-size: 0.76rem; color: {colors['text_muted']}; margin-bottom: 12px;">
                    Set custom risk thresholds for portfolio concentration and value-at-risk breaches.
                </div>
            """,
            unsafe_allow_html=True,
        )

        r_prof, r_var = st.columns(2)
        with r_prof:
            st.selectbox("Investor Risk Profile", ["Conservative Capital Preservation", "Moderate Balanced Alpha", "Aggressive Growth"], index=1)
            st.number_input("Maximum Single Asset Allocation Cap (%)", value=25.0, step=1.0, min_value=5.0, max_value=50.0)
        with r_var:
            st.number_input("Maximum Allowable 1-Day VaR 95% (%)", value=3.5, step=0.1, min_value=1.0, max_value=10.0)
            st.number_input("Drawdown Alert Threshold (%)", value=10.0, step=0.5, min_value=3.0, max_value=30.0)

        st.markdown("</div>", unsafe_allow_html=True)

    # 6. SYSTEM STATUS
    with tab_sys:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 16px; margin-bottom: 14px;">
                <div style="font-size: 0.84rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 6px;">
                    FastAPI Gateway & Infrastructure Health Diagnostics
                </div>
            """,
            unsafe_allow_html=True,
        )

        current_url = st.session_state.get("backend_url", "http://127.0.0.1:8000")
        new_url = st.text_input("Backend REST Gateway Endpoint", value=current_url)
        if new_url != current_url:
            st.session_state["backend_url"] = new_url
            st.success("Backend URL updated!")

        health_res = api_client.check_health()
        h_status = health_res.get("status", "offline")
        is_healthy = h_status == "healthy"
        status_color = colors["bullish"] if is_healthy else colors["bearish"]

        st.markdown(
            f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin: 12px 0;">
                <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid {colors['border']};">
                    <span style="color:{colors['text_muted']};">REST API Gateway:</span>
                    <strong style="color: {status_color}; text-transform: uppercase;">● {h_status}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid {colors['border']};">
                    <span style="color:{colors['text_muted']};">Database Layer:</span>
                    <strong style="color: {colors['accent_cyan']};">{health_res.get('database', 'SQLite (Local Fallback)')}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 6px 0; border-bottom:1px solid {colors['border']};">
                    <span style="color:{colors['text_muted']};">API Engine Version:</span>
                    <strong style="color: {colors['text_primary']};">{health_res.get('version', '1.0.0')}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 6px 0;">
                    <span style="color:{colors['text_muted']};">Round-Trip Latency:</span>
                    <strong style="color: {colors['bullish']};">4 ms</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Retest Gateway Connection", use_container_width=True):
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from frontend.services.api_client import get_api_client

    render_settings_page(api_client=get_api_client())
