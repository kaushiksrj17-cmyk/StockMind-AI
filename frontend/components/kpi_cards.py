"""Reusable Institutional KPI Cards for Terminal Dashboard."""

from typing import Any, Dict, Optional
import streamlit as st
from frontend.utils.formatters import format_inr, format_volume


def render_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "neutral",
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    status: Optional[str] = None,
    border_accent: Optional[str] = None,
    accent: Optional[str] = None,
    color: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Render a single financial terminal style KPI card with strict design system styling."""
    effective_accent = accent or border_accent
    delta_class = "kpi-delta-neutral"
    accent_class = ""

    if delta_type in ("positive", "bullish"):
        delta_class = "kpi-delta-pos"
        if not effective_accent:
            accent_class = "kpi-bullish"
    elif delta_type in ("negative", "bearish"):
        delta_class = "kpi-delta-neg"
        if not effective_accent:
            accent_class = "kpi-bearish"

    if effective_accent:
        accent_class = f"kpi-{effective_accent}"

    icon_html = f'<span style="margin-right: 6px; font-size: 0.85rem;">{icon}</span>' if icon else ""
    status_html = (
        f'<span style="font-size: 0.65rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; background: rgba(255,255,255,0.06); color: var(--text-muted);">{status}</span>'
        if status
        else ""
    )

    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    sub_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    val_style = f' style="color: {color};"' if color else ""

    card_html = (
        f'<div class="kpi-container {accent_class}">'
        f'<div class="kpi-title"><span>{icon_html}{title}</span>{status_html}</div>'
        f'<div class="kpi-value"{val_style}>{value}</div>'
        f'{delta_html}'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_quote_kpi_row(quote: Optional[Dict[str, Any]], symbol: str = "RELIANCE") -> None:
    """Render a 4-column KPI metric row for the active instrument quote."""
    c1, c2, c3, c4 = st.columns(4)
    if not quote:
        with c1:
            render_kpi_card(f"LTP · {symbol}", "₹0.00", subtitle="No Live Feed", accent="neutral")
        with c2:
            render_kpi_card("SESSION VWAP", "₹0.00", subtitle="Open: ₹0.00", accent="neutral")
        with c3:
            render_kpi_card("DAY HIGH / LOW", "₹0.00", subtitle="Low: ₹0.00", accent="neutral")
        with c4:
            render_kpi_card("SESSION VOLUME", "0", subtitle="Data: N/A", accent="neutral")
        return

    price = float(quote.get("price", 0.0))
    change = float(quote.get("change", 0.0))
    pct = float(quote.get("percent_change", 0.0))
    vwap = float(quote.get("vwap", price))
    open_p = float(quote.get("open", price))
    high_p = float(quote.get("high", price))
    low_p = float(quote.get("low", price))
    vol = int(quote.get("volume", 0))
    mode = quote.get("data_mode", "REPLAY")

    is_pos = pct >= 0
    sign = "+" if is_pos else ""
    delta_str = f"{sign}₹{abs(change):,.2f} ({sign}{pct:.2f}%)"
    delta_type = "positive" if is_pos else "negative"

    with c1:
        render_kpi_card(
            title=f"LTP · {symbol}",
            value=format_inr(price),
            delta=delta_str,
            delta_type=delta_type,
            subtitle="NSE · Segment: EQUITY",
            icon="⚡",
            accent="bullish" if is_pos else "bearish",
        )
    with c2:
        render_kpi_card(
            title="SESSION VWAP",
            value=format_inr(vwap),
            subtitle=f"Open: {format_inr(open_p)}",
            icon="📊",
            accent="cyan",
        )
    with c3:
        render_kpi_card(
            title="DAY HIGH / LOW",
            value=format_inr(high_p),
            subtitle=f"Low: {format_inr(low_p)}",
            icon="📈",
            accent="purple",
        )
    with c4:
        render_kpi_card(
            title="SESSION VOLUME",
            value=format_volume(vol),
            subtitle=f"Data Mode: {mode}",
            icon="📦",
            accent="warning",
        )
