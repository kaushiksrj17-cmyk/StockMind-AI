"""Page 13: Alerts — Real-Time Price & Indicator Trigger Engine."""

from typing import Any, Dict, List
import streamlit as st

from frontend.components.header import render_top_header
from frontend.components.kpi_cards import render_kpi_card
from frontend.services.api_client import APIClient
from frontend.styles.theme import get_theme_colors, get_semantic_signal
from frontend.utils.formatters import format_inr


DEFAULT_ALERTS = [
    {
        "id": "ALT-101",
        "symbol": "RELIANCE",
        "condition": "Price >= ₹3,000.00",
        "severity": "WARNING",
        "status": "ACTIVE",
        "created": "2026-10-06 10:15",
        "channel": "Terminal & Push",
        "note": "Resistance breakout threshold",
    },
    {
        "id": "ALT-102",
        "symbol": "TCS",
        "condition": "RSI-14 <= 30.0 (Oversold Dip)",
        "severity": "CRITICAL",
        "status": "TRIGGERED",
        "created": "2026-10-06 09:45",
        "channel": "In-App Audio & SMS",
        "note": "Structural oversold mean-reversion opportunity",
    },
    {
        "id": "ALT-103",
        "symbol": "HDFCBANK",
        "condition": "Session Gain >= +2.50%",
        "severity": "SUCCESS",
        "status": "ACTIVE",
        "created": "2026-10-06 11:20",
        "channel": "Webhook Gateway",
        "note": "Institutional momentum expansion target",
    },
    {
        "id": "ALT-104",
        "symbol": "NIFTY 50",
        "condition": "Benchmark Level <= 24,500.00",
        "severity": "INFO",
        "status": "ACTIVE",
        "created": "2026-10-06 08:30",
        "channel": "Terminal Visual",
        "note": "Key support corridor test",
    },
]


def render_alerts(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    theme_mode: str = "dark",
) -> None:
    """Render Price and Technical Condition Alert Manager with Section 19 severity hierarchy."""
    colors = get_theme_colors(theme_mode)

    # Standard Top Header
    render_top_header(
        title="StockMind AI",
        subtitle="Real-Time Alerts & Automated Trigger Engine",
        market_open=True,
        data_mode="REPLAY",
        ws_connected=True,
    )

    if "alerts_list" not in st.session_state:
        st.session_state["alerts_list"] = list(DEFAULT_ALERTS)

    # 1. Summary KPI Strip
    total_alerts = len(st.session_state["alerts_list"])
    active_cnt = sum(1 for a in st.session_state["alerts_list"] if a["status"] == "ACTIVE")
    trig_cnt = sum(1 for a in st.session_state["alerts_list"] if a["status"] == "TRIGGERED")
    crit_cnt = sum(1 for a in st.session_state["alerts_list"] if a.get("severity") == "CRITICAL")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("ACTIVE TRIGGERS", f"{active_cnt}", subtitle="Watching Real-Time Tick Stream", status="ARMED", accent="cyan")
    with c2:
        render_kpi_card("TRIGGERED TODAY", f"{trig_cnt}", subtitle="Conditions Satisfied", status="EXECUTED", accent="bullish" if trig_cnt > 0 else "neutral")
    with c3:
        render_kpi_card("CRITICAL SEVERITY", f"{crit_cnt}", subtitle="High-Impact Triggers", status="ELEVATED", accent="bearish" if crit_cnt > 0 else "neutral")
    with c4:
        render_kpi_card("MONITORED ASSETS", "5 Tickers", subtitle="NSE / BSE / Indices", accent="purple")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    col_create, col_list = st.columns([1.5, 2.5])

    with col_create:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 16px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: {colors['accent_cyan']}; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.5px;">
                    ➕ Create Quantitative Trigger
                </div>
            """,
            unsafe_allow_html=True,
        )
        sym_input = st.selectbox("Instrument", options=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "NIFTY 50"], index=0)
        cond_type = st.selectbox("Condition Type", options=["Price Crosses Above", "Price Crosses Below", "RSI Oversold (<=30)", "RSI Overbought (>=70)", "Day Move > 3.0%", "52W High Breakout"])
        target_val = st.number_input("Target Value / Price (₹)", value=3000.0, step=10.0)
        severity_val = st.selectbox("Trigger Severity Level", ["INFO", "SUCCESS", "WARNING", "CRITICAL"], index=2)
        channel_val = st.selectbox("Delivery Channel", ["Terminal & Push Notification", "Terminal Visual Only", "In-App Audio Warning", "FastAPI Webhook"])

        if st.button("⚡ Arm Alert Rule", use_container_width=True):
            new_id = f"ALT-{len(st.session_state['alerts_list']) + 101}"
            st.session_state["alerts_list"].append({
                "id": new_id,
                "symbol": sym_input,
                "condition": f"{cond_type} {target_val}",
                "severity": severity_val,
                "status": "ACTIVE",
                "created": "Just now",
                "channel": channel_val,
                "note": f"Custom user trigger for {sym_input}",
            })
            st.success(f"Alert {new_id} configured & armed!")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_list:
        st.markdown(
            f"""
            <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase; margin-bottom: 8px;">
                CONFIGURED RULES & SEVERITY DISPATCHES
            </div>
            """,
            unsafe_allow_html=True,
        )

        for a in st.session_state["alerts_list"]:
            sev = a.get("severity", "INFO")
            is_trig = a["status"] == "TRIGGERED"

            # Section 19 Semantic Severity Badges (INFO, SUCCESS, WARNING, CRITICAL)
            if sev == "CRITICAL":
                sev_badge_class = "badge-bearish"
                sev_icon = "🚨"
            elif sev == "WARNING":
                sev_badge_class = "badge-warning"
                sev_icon = "⚠️"
            elif sev == "SUCCESS":
                sev_badge_class = "badge-bullish"
                sev_icon = "✅"
            else:
                sev_badge_class = "badge-cyan"
                sev_icon = "ℹ️"

            status_badge_class = "badge-bearish" if is_trig else "badge-bullish"

            st.markdown(
                f"""
                <div class="term-card" style="margin-bottom: 10px; padding: 12px 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="term-badge {sev_badge_class}" style="margin-right: 6px;">
                                {sev_icon} {sev}
                            </span>
                            <strong style="color: {colors['accent_cyan']}; font-family: 'JetBrains Mono', monospace;">{a['id']}</strong>
                            <strong style="color: {colors['text_primary']}; margin-left: 8px; font-size: 0.90rem;">{a['symbol']}</strong>
                            <span style="color: {colors['text_secondary']}; font-size: 0.84rem; margin-left: 10px;">{a['condition']}</span>
                        </div>
                        <span class="term-badge {status_badge_class}" style="font-size: 0.70rem;">
                            {a['status']}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: {colors['text_muted']}; margin-top: 8px; border-top: 1px solid {colors['border']}; padding-top: 6px;">
                        <span>Channel: <strong style="color: {colors['text_secondary']};">{a['channel']}</strong></span>
                        <span>Created: {a['created']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from frontend.services.api_client import get_api_client

    render_alerts(api_client=get_api_client())
