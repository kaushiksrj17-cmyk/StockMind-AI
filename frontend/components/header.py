"""Institutional Terminal Top Header Component."""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
import streamlit as st
from frontend.components.badges import (
    render_data_mode_badge,
    render_market_status_badge,
    render_ws_badge,
)
from frontend.styles.theme import ACCENT_CYAN, TEXT_PRIMARY, TEXT_MUTED, BORDER


def render_terminal_header(
    market_status: Optional[Dict[str, Any]] = None,
    data_mode: str = "REPLAY",
    is_live: bool = False,
    is_ws_connected: bool = True,
    is_backend_online: bool = True,
) -> None:
    """Render top header bar with branding, live indicators, and exchange time."""
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist_tz).strftime("%H:%M:%S IST · %d %b %Y")

    status_str = "CLOSED"
    if market_status:
        status_str = market_status.get("status", "CLOSED")

    mode_badge_html = render_data_mode_badge(data_mode, is_live)
    market_badge_html = render_market_status_badge(status_str)
    ws_badge_html = render_ws_badge(is_connected=is_ws_connected, is_backend_online=is_backend_online)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: linear-gradient(135deg, #22D3EE 0%, #3B82F6 100%); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.05rem; color: #070B14; box-shadow: 0 0 14px rgba(34, 211, 238, 0.35);">
                    ⚡
                </div>
                <div>
                    <div style="font-size: 1.22rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary); line-height: 1.1;">
                        StockMind <span style="color: var(--accent-cyan);">AI</span>
                    </div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;">
                        Real-Time AI Market Intelligence
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--text-muted); background: var(--panel); padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border);">
                    ⏱ {now_ist}
                </div>
                {market_badge_html}
                {mode_badge_html}
                {ws_badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_header(
    title: str = "StockMind AI",
    subtitle: str = "Real-Time AI Market Intelligence",
    market_open: bool = True,
    data_mode: str = "REPLAY",
    is_live: bool = False,
    ws_connected: bool = True,
    backend_online: bool = True,
) -> None:
    """Page section header displaying page title and description."""
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border);">
            <div>
                <div style="font-size: 1.22rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; display: flex; align-items: center; gap: 8px;">
                    <span>{title}</span>
                </div>
                <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">
                    {subtitle}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
