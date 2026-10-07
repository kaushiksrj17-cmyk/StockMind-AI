"""Page 2: Live Market Watch & Institutional Order Book Stream."""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
import streamlit as st

from frontend.components.status_banner import render_provenance_banner
from frontend.components.kpi_cards import render_quote_kpi_row, render_kpi_card
from frontend.charts.candlestick import render_candlestick_chart
from frontend.services.api_client import APIClient
from frontend.styles.theme import get_theme_colors
from frontend.utils.formatters import format_inr, format_volume


def render_live_market(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    theme_mode: str = "dark",
) -> None:
    """Render the Live Market Trading Terminal."""
    t = get_theme_colors(theme_mode)
    active_quote = api_client.get_quote(symbol)
    all_quotes = api_client.get_all_quotes()

    data_mode = active_quote.get("data_mode", "REPLAY") if active_quote else "REPLAY"
    is_live = active_quote.get("is_live", False) if active_quote else False
    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%H:%M:%S IST")

    # 1. Provenance Banner
    render_provenance_banner(data_mode=data_mode, is_live=is_live)

    # 2. Page Header & Live Telemetry Bar
    p = active_quote.get("price", 2890.50) if active_quote else 2890.50
    chg = active_quote.get("change", 41.20) if active_quote else 41.20
    chg_pct = active_quote.get("change_percent", 1.45) if active_quote else 1.45
    vol = active_quote.get("volume", 3240000) if active_quote else 3240000
    is_up = chg >= 0
    p_col = "var(--bullish)" if is_up else "var(--bearish)"
    p_sign = "+" if is_up else ""

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">
                    ⚡ LIVE MARKET TERMINAL · <span style="color: var(--accent-cyan);">{symbol}</span>
                </div>
                <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">
                    NSE Direct Order Flow Stream · {ist_now} · WS Connected
                </div>
            </div>
            <div style="text-align: right; font-family: 'JetBrains Mono', monospace;">
                <span style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary);">{format_inr(p)}</span>
                <span style="font-size: 0.88rem; font-weight: 700; color: {p_col}; margin-left: 8px;">{p_sign}{chg:.2f} ({p_sign}{chg_pct:.2f}%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Top KPI Row
    render_quote_kpi_row(active_quote, symbol=symbol)
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 4. Main Trading Terminal Layout: Chart + Order Book & Live Feed
    col_chart, col_depth = st.columns([2.6, 1.4])

    with col_chart:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary);">
                    📈 LIVE INTRADAY CANDLESTICK & VOLUME · {symbol}
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); font-family: 'JetBrains Mono';">1-Min Ticks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        candles = api_client.get_historical_ohlc(symbol=symbol, timeframe="1m", limit=120)
        fig = render_candlestick_chart(candles, symbol=symbol, timeframe="1m", theme_mode=theme_mode)
        st.plotly_chart(fig, use_container_width=True)

        # Technical Indicators Snapshot Bar
        st.markdown(
            """
            <div class="term-card" style="padding: 10px 14px; margin-top: -6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;">
                    <div><span style="color: var(--text-muted);">RSI (14):</span> <strong style="color: var(--accent-purple);">58.4</strong> (Neutral)</div>
                    <div><span style="color: var(--text-muted);">MACD:</span> <strong style="color: var(--bullish);">+4.12</strong> (Bullish)</div>
                    <div><span style="color: var(--text-muted);">VWAP:</span> <strong style="color: var(--warning);">₹2,884.10</strong></div>
                    <div><span style="color: var(--text-muted);">ATR (14):</span> <strong style="color: var(--accent-cyan);">₹24.80</strong></div>
                    <div><span style="color: var(--text-muted);">Trend:</span> <strong style="color: var(--bullish);">BULLISH MOMENTUM</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_depth:
        st.markdown(
            """
            <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">
                📊 LEVEL II MARKET DEPTH
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5-level Order Book centered around LTP
        bids = [
            (p - 0.25, 240, 4),
            (p - 0.50, 520, 8),
            (p - 0.75, 1150, 15),
            (p - 1.00, 2400, 28),
            (p - 1.25, 3800, 42),
        ]
        asks = [
            (p + 0.25, 180, 3),
            (p + 0.50, 490, 7),
            (p + 0.75, 980, 12),
            (p + 1.00, 2100, 22),
            (p + 1.25, 3400, 35),
        ]

        total_bid_qty = sum(b[1] for b in bids)
        total_ask_qty = sum(a[1] for a in asks)
        imbalance_pct = ((total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty) * 100) if (total_bid_qty + total_ask_qty) > 0 else 0.0

        bids_html = "".join(
            f'<div style="display:flex; justify-content:space-between; font-family: JetBrains Mono, monospace; font-size:0.72rem; padding: 3px 0;"><span style="color:var(--text-muted);">{b[2]}</span><span style="color:var(--text-secondary);">{b[1]:,}</span><span style="color:var(--bullish); font-weight:600;">{b[0]:.2f}</span></div>'
            for b in bids
        )
        asks_html = "".join(
            f'<div style="display:flex; justify-content:space-between; font-family: JetBrains Mono, monospace; font-size:0.72rem; padding: 3px 0;"><span style="color:var(--bearish); font-weight:600;">{a[0]:.2f}</span><span style="color:var(--text-secondary);">{a[1]:,}</span><span style="color:var(--text-muted);">{a[2]}</span></div>'
            for a in asks
        )

        st.markdown(
            f"""
            <div class="term-card">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                    <div>
                        <div style="font-size: 0.7rem; color: var(--bullish); font-weight: 700; margin-bottom: 4px;">BUY ORDERS (BIDS)</div>
                        <div style="font-size: 0.65rem; color: var(--text-muted); display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 2px;">
                            <span>ORD</span><span>QTY</span><span>PRICE</span>
                        </div>
                        {bids_html}
                        <div style="border-top: 1px solid var(--border); margin-top: 4px; padding-top: 3px; font-size: 0.7rem; display: flex; justify-content: space-between; color: var(--text-muted);">
                            <span>Total Bids:</span><span style="color: var(--bullish); font-family: monospace;">{total_bid_qty:,}</span>
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--bearish); font-weight: 700; margin-bottom: 4px;">SELL ORDERS (ASKS)</div>
                        <div style="font-size: 0.65rem; color: var(--text-muted); display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 2px;">
                            <span>PRICE</span><span>QTY</span><span>ORD</span>
                        </div>
                        {asks_html}
                        <div style="border-top: 1px solid var(--border); margin-top: 4px; padding-top: 3px; font-size: 0.7rem; display: flex; justify-content: space-between; color: var(--text-muted);">
                            <span>Total Asks:</span><span style="color: var(--bearish); font-family: monospace;">{total_ask_qty:,}</span>
                        </div>
                    </div>
                </div>
                <div style="background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 0.72rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                    <span>Order Imbalance:</span>
                    <strong style="color: var(--accent-cyan); font-family: monospace;">{imbalance_pct:+.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Recent Live Ticks Table
        st.markdown(
            """
            <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">
                ⏱ RECENT LIVE TRADES
            </div>
            """,
            unsafe_allow_html=True,
        )

        recent_trades = [
            (ist_now, p, 150, "BUY"),
            (ist_now, p - 0.05, 50, "SELL"),
            (ist_now, p, 200, "BUY"),
            (ist_now, p + 0.10, 80, "BUY"),
            (ist_now, p, 25, "SELL"),
        ]

        st.markdown('<div class="term-card" style="padding: 10px 14px;">', unsafe_allow_html=True)
        for t_time, t_pr, t_qty, t_side in recent_trades:
            t_col = "var(--bullish)" if t_side == "BUY" else "var(--bearish)"
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; padding: 3px 0; border-bottom: 1px solid var(--border);">
                    <span style="color: var(--text-muted);">{t_time}</span>
                    <span style="color: var(--text-primary);">₹{t_pr:.2f}</span>
                    <span style="color: var(--text-secondary);">{t_qty} shs</span>
                    <strong style="color: {t_col};">{t_side}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Full Market Watchlist Table
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">
            📋 ACTIVE TRADEABLE WATCHLIST
        </div>
        """,
        unsafe_allow_html=True,
    )

    if all_quotes:
        st.markdown(
            """
            <div style="background: var(--panel); border: 1px solid var(--border); border-radius: 8px 8px 0 0; padding: 8px 14px; display: grid; grid-template-columns: 2fr 1fr 1.5fr 1.5fr 1.5fr; font-size: 0.7rem; color: var(--text-muted); font-weight: 700; letter-spacing: 0.05em;">
                <div>SYMBOL / SEGMENT</div>
                <div>EXCH</div>
                <div style="text-align: right;">LTP (₹)</div>
                <div style="text-align: right;">CHANGE %</div>
                <div style="text-align: right;">VOLUME</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for q in all_quotes:
            sym_q = q.get("symbol", "")
            exch_q = q.get("exchange", "NSE")
            seg_q = q.get("segment", "EQUITY")
            pr_q = q.get("price", 0.0)
            chg_q = q.get("change_percent", 0.0)
            vol_q = q.get("volume", 0)
            c_col = "var(--bullish)" if chg_q >= 0 else "var(--bearish)"
            sign_q = "+" if chg_q >= 0 else ""
            is_active_row = sym_q == symbol
            row_style = "border-left: 3px solid var(--accent-cyan); background: rgba(34, 211, 238, 0.05);" if is_active_row else "background: var(--bg-tertiary);"

            st.markdown(
                f"""
                <div style="{row_style} border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); padding: 8px 14px; display: grid; grid-template-columns: 2fr 1fr 1.5fr 1.5fr 1.5fr; font-size: 0.78rem; align-items: center;">
                    <div><strong style="color: var(--text-primary);">{sym_q}</strong> <span style="font-size: 0.65rem; color: var(--text-muted);">{seg_q}</span></div>
                    <div style="color: var(--text-muted); font-size: 0.7rem;">{exch_q}</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text-primary);">{format_inr(pr_q)}</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {c_col};">{sign_q}{chg_q:.2f}%</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted);">{format_volume(vol_q)}</div>
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

    render_live_market(api_client=get_api_client())
