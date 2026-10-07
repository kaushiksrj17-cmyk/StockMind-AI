"""Streamlit Session State Initializer and Manager."""

import streamlit as st


def init_session_state() -> None:
    """Initialize global session state variables for StockMind-AI."""
    defaults = {
        "backend_url": "http://127.0.0.1:8000",
        "theme_mode": "dark",
        "selected_symbol": "RELIANCE",
        "selected_exchange": "NSE",
        "selected_timeframe": "1d",
        "auth_token": None,
        "current_user": None,
        "ws_connected": False,
        "active_section": "Command Center",
        "auto_refresh": False,
        "refresh_interval": 5,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# Compatibility alias
initialize_session_state = init_session_state
