"""Sidebar Navigation & Global Controls Component for StockMind-AI."""

from typing import Any, Dict, List
import streamlit as st
from frontend.utils.state import get_backend_url

DEFAULT_INSTRUMENTS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "TATAMOTORS",
    "NIFTY 50",
    "SENSEX",
]


def render_sidebar(instruments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render the sidebar with brand header, global ticker selection, and terminal preferences."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 6px 10px 6px; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); margin: 10px 0 14px 0; background: rgba(34, 211, 238, 0.03); border-radius: 6px;">
                <div style="font-size: 0.72rem; font-weight: 800; color: var(--accent-cyan); letter-spacing: 0.08em; text-transform: uppercase;">
                    ⚙️ TERMINAL CONTROLS
                </div>
                <div style="font-size: 0.68rem; color: var(--text-muted); font-weight: 600; margin-top: 2px;">
                    Active Instrument & Parameters
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Global Ticker / Instrument Selection
        symbol_list = DEFAULT_INSTRUMENTS
        if instruments:
            extracted = [i.get("symbol") for i in instruments if i.get("symbol")]
            if extracted:
                symbol_list = extracted

        current_symbol = st.session_state.get("active_symbol", "RELIANCE")
        symbol_idx = 0
        if current_symbol in symbol_list:
            symbol_idx = symbol_list.index(current_symbol)

        selected_symbol = st.selectbox(
            "ACTIVE INSTRUMENT",
            options=symbol_list,
            index=symbol_idx,
            help="Select Indian stock or benchmark index to analyze.",
        )
        st.session_state["active_symbol"] = selected_symbol

        # 2. Timeframe Selection
        timeframes = ["1m", "5m", "15m", "1h", "1d"]
        current_tf = st.session_state.get("active_timeframe", "1d")
        tf_idx = timeframes.index(current_tf) if current_tf in timeframes else 4

        selected_tf = st.select_slider(
            "TIMEFRAME",
            options=timeframes,
            value=timeframes[tf_idx],
        )
        st.session_state["active_timeframe"] = selected_tf

        st.markdown("<hr style='border: none; border-top: 1px solid var(--border); margin: 16px 0;'>", unsafe_allow_html=True)

        # 3. Stream & UI Controls
        col_ref, col_theme = st.columns(2)
        with col_ref:
            auto_refresh = st.checkbox(
                "Live Poll",
                value=st.session_state.get("auto_refresh", False),
                help="Auto-poll backend quote cache periodically.",
            )
            st.session_state["auto_refresh"] = auto_refresh

        with col_theme:
            cur_th = str(st.session_state.get("theme_mode", "dark")).lower()
            theme_choice = st.selectbox(
                "Theme",
                options=["dark", "light"],
                format_func=lambda x: "Dark" if x == "dark" else "Light",
                index=0 if cur_th == "dark" else 1,
                label_visibility="collapsed",
            )
            st.session_state["theme_mode"] = theme_choice

        # 4. Connection Info
        st.markdown(
            f"""
            <div style="background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-top: 18px;">
                <div style="font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">BACKEND GATEWAY</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--accent-cyan); word-break: break-all; margin-top: 2px;">
                    {st.session_state.get('backend_url') or get_backend_url()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {
            "symbol": selected_symbol,
            "timeframe": selected_tf,
            "auto_refresh": auto_refresh,
            "theme_mode": theme_choice,
        }
