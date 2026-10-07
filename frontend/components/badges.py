"""Visual Badges and Status Pills for StockMind-AI.

Renders high-visibility provenance indicators distinguishing LIVE DATA,
REPLAY DATA, HISTORICAL DATA, and AI PREDICTIONS using the unified financial palette.
"""

from frontend.styles.theme import (
    BULLISH,
    BEARISH,
    WARNING,
    ACCENT_BLUE,
    ACCENT_PURPLE,
    ACCENT_CYAN,
    NEUTRAL,
)


def render_data_mode_badge(data_mode: str, is_live: bool = False) -> str:
    """Return HTML string for data provenance badge."""
    mode_str = str(data_mode).upper()
    if mode_str == "LIVE" or is_live:
        return (
            '<span class="badge-live">'
            '<span class="pulse-dot pulse-green"></span>'
            'LIVE DATA'
            '</span>'
        )
    elif mode_str == "HISTORICAL":
        return (
            '<span class="badge-historical">'
            f'<span class="pulse-dot" style="background:{ACCENT_BLUE};"></span>'
            'HISTORICAL DATA'
            '</span>'
        )
    elif mode_str in ("PREDICTION", "AI PREDICTION"):
        return (
            '<span class="badge-prediction">'
            f'<span class="pulse-dot" style="background:{ACCENT_PURPLE};"></span>'
            'AI PREDICTION'
            '</span>'
        )
    else:
        return (
            '<span class="badge-replay">'
            '<span class="pulse-dot pulse-amber"></span>'
            'DEMO / REPLAY DATA'
            '</span>'
        )


def render_market_status_badge(market_status: str) -> str:
    """Return HTML badge for exchange operational status."""
    st_upper = str(market_status).upper()
    if st_upper == "OPEN":
        return f'<span class="badge-live"><span class="pulse-dot pulse-green"></span> MARKET OPEN</span>'
    elif st_upper == "PRE_OPEN":
        return f'<span class="badge-prediction"><span class="pulse-dot pulse-cyan"></span> PRE-OPEN</span>'
    else:
        return f'<span class="badge-replay" style="color:{WARNING}; border-color:rgba(245,158,11,0.4);"><span class="pulse-dot pulse-amber"></span> MARKET CLOSED</span>'


def render_ws_badge(is_connected: bool, is_backend_online: bool = True) -> str:
    """Return HTML badge for real-time WebSocket connection state."""
    if not is_backend_online:
        return f'<span class="badge-replay" style="color:{BEARISH}; border-color:rgba(239,68,68,0.4); background:rgba(239,68,68,0.12);"><span class="pulse-dot pulse-red"></span> GATEWAY OFFLINE</span>'
    if is_connected:
        return f'<span class="badge-live"><span class="pulse-dot pulse-green"></span> WS STREAM ONLINE</span>'
    return f'<span class="badge-replay"><span class="pulse-dot pulse-amber"></span> WS POLLING</span>'


def render_signal_pill(signal: str) -> str:
    """Render a standalone semantic signal pill."""
    sig_upper = str(signal).upper().strip()
    if sig_upper in ("STRONG BULLISH", "STRONG_BULLISH"):
        return f'<span style="background:rgba(34,197,94,0.15); color:{BULLISH}; border:1px solid {BULLISH}; border-radius:4px; padding:2px 8px; font-weight:700; font-size:0.72rem; box-shadow:0 0 10px rgba(34,197,94,0.3);">▲▲ STRONG BULLISH</span>'
    elif sig_upper in ("BULLISH", "BUY"):
        return f'<span style="background:rgba(34,197,94,0.12); color:{BULLISH}; border:1px solid rgba(34,197,94,0.4); border-radius:4px; padding:2px 8px; font-weight:700; font-size:0.72rem;">▲ BULLISH</span>'
    elif sig_upper in ("STRONG BEARISH", "STRONG_BEARISH"):
        return f'<span style="background:rgba(239,68,68,0.15); color:{BEARISH}; border:1px solid {BEARISH}; border-radius:4px; padding:2px 8px; font-weight:700; font-size:0.72rem; box-shadow:0 0 10px rgba(239,68,68,0.3);">▼▼ STRONG BEARISH</span>'
    elif sig_upper in ("BEARISH", "SELL"):
        return f'<span style="background:rgba(239,68,68,0.12); color:{BEARISH}; border:1px solid rgba(239,68,68,0.4); border-radius:4px; padding:2px 8px; font-weight:700; font-size:0.72rem;">▼ BEARISH</span>'
    else:
        return f'<span style="background:rgba(148,163,184,0.12); color:{NEUTRAL}; border:1px solid rgba(148,163,184,0.4); border-radius:4px; padding:2px 8px; font-weight:700; font-size:0.72rem;">◆ NEUTRAL</span>'
