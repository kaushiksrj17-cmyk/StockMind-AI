import os
from typing import Optional
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def get_backend_url() -> str:
    """Resolve backend gateway URL in strict priority order:
    1. BACKEND_API_URL environment variable
    2. BACKEND_URL environment variable
    3. Streamlit secrets BACKEND_API_URL
    4. Streamlit secrets BACKEND_URL
    5. Local fallback http://127.0.0.1:8000
    """
    # 1. BACKEND_API_URL environment variable
    env_api = os.getenv("BACKEND_API_URL")
    if env_api and env_api.strip():
        return env_api.strip().rstrip("/")

    # 2. BACKEND_URL environment variable
    env_backend = os.getenv("BACKEND_URL")
    if env_backend and env_backend.strip():
        return env_backend.strip().rstrip("/")

    # 3. Streamlit secrets BACKEND_API_URL
    try:
        secret_api = st.secrets.get("BACKEND_API_URL")
        if secret_api and str(secret_api).strip():
            return str(secret_api).strip().rstrip("/")
    except Exception:
        pass

    # 4. Streamlit secrets BACKEND_URL
    try:
        secret_backend = st.secrets.get("BACKEND_URL")
        if secret_backend and str(secret_backend).strip():
            return str(secret_backend).strip().rstrip("/")
    except Exception:
        pass

    # 5. Local fallback
    return DEFAULT_BACKEND_URL


def init_session_state() -> None:
    """Initialize global session state variables for StockMind-AI."""
    defaults = {
        "backend_url": get_backend_url(),
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
