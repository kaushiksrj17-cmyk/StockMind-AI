"""Page 12: Market Heatmap — Sector & Equity Treemap Visualization."""

from typing import Any, Dict, List
import streamlit as st

from frontend.components.header import render_top_header
from frontend.components.kpi_cards import render_kpi_card
from frontend.charts.heatmap import render_market_heatmap
from frontend.services.api_client import APIClient
from frontend.styles.theme import get_theme_colors, get_semantic_signal


SECTOR_CATALOG = {
    "RELIANCE": ("Energy & Oil", 1950000, 4200000),
    "TCS": ("Information Technology", 1420000, 1800000),
    "HDFCBANK": ("Banking & Finance", 1280000, 6800000),
    "INFY": ("Information Technology", 710000, 3200000),
    "ICICIBANK": ("Banking & Finance", 840000, 5100000),
    "SBIN": ("Banking & Finance", 720000, 7500000),
    "TATAMOTORS": ("Automotive", 360000, 4900000),
    "BHARTIARTL": ("Telecommunications", 950000, 2100000),
    "ITC": ("Consumer Goods", 610000, 3900000),
    "LT": ("Infrastructure", 480000, 1600000),
}


def render_market_heatmap_page(
    api_client: APIClient,
    theme_mode: str = "dark",
) -> None:
    """Render Indian stock sectoral performance treemap heatmap with filters."""
    colors = get_theme_colors(theme_mode)
    active_quotes = api_client.get_all_quotes() or []

    # Standard Top Header
    render_top_header(
        title="StockMind AI",
        subtitle="Indian Equity Sectoral Heatmap & Breadth Treemap",
        market_open=True,
        data_mode="REPLAY",
        ws_connected=True,
    )

    # 1. Filters & Configuration Bar (Section 18: NIFTY 50, Sector, Market Cap, Performance, Volume)
    col_idx, col_sec, col_sort = st.columns([1.5, 1.8, 1.5])
    with col_idx:
        index_filter = st.selectbox("Benchmark Universe", ["NIFTY 50 Constituents", "NIFTY BANK", "NIFTY IT", "ALL EQUITIES"], index=0)
    with col_sec:
        sector_filter = st.selectbox(
            "Sector Filter",
            ["ALL SECTORS", "Banking & Finance", "Information Technology", "Energy & Oil", "Automotive", "Infrastructure", "Consumer Goods", "Telecommunications"],
            index=0,
        )
    with col_sort:
        sort_by = st.selectbox("Sort / Tile Sizing", ["Market Capitalization", "Session Volume", "Absolute Return Magnitude"], index=0)

    # Build Heatmap Items
    data_items = []
    advances = 0
    declines = 0

    # Build lookup from active quotes
    quote_map = {q.get("symbol", ""): q for q in active_quotes if q.get("symbol")}

    for sym, (sec, m_cap, vol) in SECTOR_CATALOG.items():
        if sector_filter != "ALL SECTORS" and sec != sector_filter:
            continue
        if index_filter == "NIFTY BANK" and "Banking" not in sec:
            continue
        if index_filter == "NIFTY IT" and "Technology" not in sec:
            continue

        q = quote_map.get(sym, {})
        chg = float(q.get("change_percent", 0.85 if "BANK" in sym or sym == "RELIANCE" else -0.45))
        if chg >= 0:
            advances += 1
        else:
            declines += 1

        weight_val = m_cap if sort_by == "Market Capitalization" else (vol if sort_by == "Session Volume" else int(abs(chg) * 100000 + 50000))

        data_items.append({
            "symbol": sym,
            "name": sym,
            "sector": sec,
            "market_cap": weight_val,
            "change_pct": chg,
            "volume": vol,
        })

    # Summary Breadth KPI Row
    ad_ratio = (advances / (declines or 1))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("ADVANCES", f"▲ {advances}", subtitle="Equities Trading Positive", status="BULLISH", accent="bullish")
    with c2:
        render_kpi_card("DECLINES", f"▼ {declines}", subtitle="Equities Trading Negative", status="BEARISH", accent="bearish")
    with c3:
        render_kpi_card("A/D RATIO", f"{ad_ratio:.2f}", subtitle="Market Breadth Multiplier", accent="cyan")
    with c4:
        breadth_state = "BULLISH EXPANSION" if ad_ratio > 1.2 else ("BEARISH CONTRACTION" if ad_ratio < 0.8 else "NEUTRAL CHURN")
        render_kpi_card("BREADTH REGIME", breadth_state, subtitle="Institutional Flow Bias", accent="bullish" if ad_ratio > 1.2 else "warning")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Treemap Chart
    fig_heat = render_market_heatmap(data_items, theme_mode=theme_mode)
    st.plotly_chart(fig_heat, use_container_width=True)

    # 3. Top Gainers & Decliners Table
    st.markdown(
        f"""
        <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase; margin: 12px 0 8px 0;">
            TOP MOVERS IN ACTIVE UNIVERSE
        </div>
        """,
        unsafe_allow_html=True,
    )

    sorted_gainers = sorted(data_items, key=lambda x: x["change_pct"], reverse=True)
    c_gain, c_loss = st.columns(2)

    with c_gain:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 12px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: {colors['bullish']}; margin-bottom: 8px;">
                    ▲ TOP ADVANCING CONSTITUENTS
                </div>
            """,
            unsafe_allow_html=True,
        )
        for g in sorted_gainers[:3]:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid {colors['border']}; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;">
                    <span style="color: {colors['text_primary']}; font-weight: 600;">{g['symbol']} ({g['sector']})</span>
                    <span style="color: {colors['bullish']}; font-weight: 700;">+{g['change_pct']:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_loss:
        st.markdown(
            f"""
            <div class="term-card" style="padding: 12px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: {colors['bearish']}; margin-bottom: 8px;">
                    ▼ TOP DECLINING CONSTITUENTS
                </div>
            """,
            unsafe_allow_html=True,
        )
        for l in sorted_gainers[-3:]:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid {colors['border']}; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;">
                    <span style="color: {colors['text_primary']}; font-weight: 600;">{l['symbol']} ({l['sector']})</span>
                    <span style="color: {colors['bearish']}; font-weight: 700;">{l['change_pct']:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from frontend.services.api_client import get_api_client

    render_market_heatmap_page(api_client=get_api_client())
