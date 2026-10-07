"""Page 3: Stock Analysis — Comprehensive Equity Fundamental & Technical Breakdown."""

from typing import Any, Dict
import streamlit as st
from frontend.components.status_banner import render_provenance_banner
from frontend.components.kpi_cards import render_quote_kpi_row
from frontend.charts.candlestick import render_candlestick_chart
from frontend.services.api_client import APIClient
from frontend.utils.formatters import format_inr, format_volume


FUNDAMENTALS_DB = {
    "RELIANCE": {"pe": 26.4, "pb": 2.1, "div_yield": 0.35, "mcap_cr": 1950000, "roe": 9.4, "de": 0.42, "high52": 3217.90, "low52": 2220.30},
    "TCS": {"pe": 29.8, "pb": 13.2, "div_yield": 2.10, "mcap_cr": 1420000, "roe": 48.2, "de": 0.05, "high52": 4585.00, "low52": 3313.00},
    "HDFCBANK": {"pe": 18.2, "pb": 2.6, "div_yield": 1.15, "mcap_cr": 1280000, "roe": 16.5, "de": 0.88, "high52": 1794.00, "low52": 1363.55},
    "INFY": {"pe": 27.5, "pb": 8.4, "div_yield": 2.30, "mcap_cr": 710000, "roe": 31.8, "de": 0.08, "high52": 1991.45, "low52": 1358.35},
    "ICICIBANK": {"pe": 17.8, "pb": 3.1, "div_yield": 0.75, "mcap_cr": 840000, "roe": 18.2, "de": 0.72, "high52": 1332.00, "low52": 934.00},
    "SBIN": {"pe": 9.8, "pb": 1.4, "div_yield": 1.65, "mcap_cr": 720000, "roe": 17.1, "de": 1.12, "high52": 912.00, "low52": 555.00},
    "TATAMOTORS": {"pe": 10.4, "pb": 3.8, "div_yield": 0.60, "mcap_cr": 360000, "roe": 34.6, "de": 0.65, "high52": 1179.05, "low52": 615.00},
}


def render_stock_analysis(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    timeframe: str = "1d",
    theme_mode: str = "dark",
) -> None:
    """Render in-depth stock analysis breakdown with unified design system."""
    active_quote = api_client.get_quote(symbol)
    data_mode = active_quote.get("data_mode", "REPLAY") if active_quote else "REPLAY"
    is_live = active_quote.get("is_live", False) if active_quote else False

    render_provenance_banner(data_mode=data_mode, is_live=is_live)

    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">
                🔍 EQUITY VALUATION & METRICS · <span style="color: var(--accent-cyan);">{symbol}</span>
            </div>
            <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">
                Fundamental Ratios · 52-Week Corridor · Support / Resistance Pivots
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_quote_kpi_row(active_quote, symbol=symbol)
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    fund = FUNDAMENTALS_DB.get(symbol, {"pe": 22.0, "pb": 3.0, "div_yield": 1.2, "mcap_cr": 500000, "roe": 18.0, "de": 0.5, "high52": 3000.0, "low52": 2000.0})
    price = active_quote.get("price", 2800.0) if active_quote else 2800.0

    # 52-Week Range Bar
    high52 = fund["high52"]
    low52 = fund["low52"]
    pct_pos = max(0.0, min(100.0, ((price - low52) / (high52 - low52)) * 100)) if high52 > low52 else 50.0

    st.markdown(
        f"""
        <div class="term-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">52-WEEK RANGE CORRIDOR</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--accent-cyan); font-weight: 600;">Current: {format_inr(price)} ({pct_pos:.1f}% of Range)</span>
            </div>
            <div style="background: var(--bg-tertiary); border: 1px solid var(--border); height: 8px; border-radius: 4px; position: relative;">
                <div style="position: absolute; left: 0; width: {pct_pos}%; background: linear-gradient(90deg, #3B82F6 0%, #22D3EE 100%); height: 8px; border-radius: 4px;"></div>
                <div style="position: absolute; left: calc({pct_pos}% - 4px); top: -3px; width: 8px; height: 14px; background: #FFFFFF; border-radius: 2px; box-shadow: 0 0 8px var(--accent-cyan);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted);">
                <span>52W Low: {format_inr(low52)}</span>
                <span>52W High: {format_inr(high52)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fundamental Valuation Grid & Pivot Points
    col_fund, col_pivots = st.columns([1.8, 1.2])

    with col_fund:
        st.markdown(
            f"""
            <div class="term-card">
                <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-primary); margin-bottom: 12px; text-transform: uppercase;">
                    Valuation & Key Financial Metrics · {symbol}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">Price to Earnings (P/E)</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--text-primary); font-weight: 700;">{fund['pe']:.1f}x</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">Price to Book (P/B)</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--text-primary); font-weight: 700;">{fund['pb']:.1f}x</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">Return on Equity (ROE)</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--bullish); font-weight: 700;">{fund['roe']:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">Dividend Yield</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--warning); font-weight: 700;">{fund['div_yield']:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">Debt / Equity Ratio</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--text-primary); font-weight: 700;">{fund['de']:.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">Market Cap (INR)</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: var(--accent-cyan); font-weight: 700;">₹{fund['mcap_cr']:,.0f} Cr</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_pivots:
        high = active_quote.get("high", price * 1.01) if active_quote else price * 1.01
        low = active_quote.get("low", price * 0.99) if active_quote else price * 0.99
        close = price
        pivot = (high + low + close) / 3.0
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)

        st.markdown(
            f"""
            <div class="term-card">
                <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-primary); margin-bottom: 12px; text-transform: uppercase;">
                    Floor Pivot Support & Resistance
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
                    <div style="display:flex; justify-content:space-between; padding: 4px 0;"><span style="color:var(--bearish);">Resistance 2 (R2):</span><strong style="color:var(--bearish);">{format_inr(r2)}</strong></div>
                    <div style="display:flex; justify-content:space-between; padding: 4px 0;"><span style="color:var(--bearish); opacity:0.85;">Resistance 1 (R1):</span><strong style="color:var(--bearish); opacity:0.85;">{format_inr(r1)}</strong></div>
                    <div style="display:flex; justify-content:space-between; padding: 6px 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border);"><span style="color:var(--accent-cyan);">Central Pivot (P):</span><strong style="color:var(--accent-cyan);">{format_inr(pivot)}</strong></div>
                    <div style="display:flex; justify-content:space-between; padding: 4px 0;"><span style="color:var(--bullish); opacity:0.85;">Support 1 (S1):</span><strong style="color:var(--bullish); opacity:0.85;">{format_inr(s1)}</strong></div>
                    <div style="display:flex; justify-content:space-between; padding: 4px 0;"><span style="color:var(--bullish);">Support 2 (S2):</span><strong style="color:var(--bullish);">{format_inr(s2)}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Historical OHLCV Chart
    candles = api_client.get_historical_ohlc(symbol=symbol, timeframe=timeframe, limit=80)
    fig = render_candlestick_chart(candles, symbol=symbol, timeframe=timeframe, theme_mode=theme_mode)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from frontend.services.api_client import get_api_client

    render_stock_analysis(api_client=get_api_client())
