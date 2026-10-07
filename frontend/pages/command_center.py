"""Page 1: Command Center — Executive Financial Terminal Overview."""

from typing import Any, Dict, List
import streamlit as st
from frontend.components.status_banner import render_provenance_banner
from frontend.components.kpi_cards import render_quote_kpi_row, render_kpi_card
from frontend.components.badges import render_signal_pill
from frontend.charts.candlestick import render_candlestick_chart
from frontend.charts.heatmap import render_market_heatmap
from frontend.services.api_client import APIClient
from frontend.styles.theme import get_theme_colors, get_semantic_signal
from frontend.utils.formatters import format_inr, format_volume


def render_command_center(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    timeframe: str = "1d",
    theme_mode: str = "dark",
) -> None:
    """Render Command Center executive dashboard with unified design system."""
    t = get_theme_colors(theme_mode)

    # Fetch live data
    status_data = api_client.get_market_status(exchange="NSE")
    active_quote = api_client.get_quote(symbol)
    all_quotes = api_client.get_all_quotes()

    data_mode = active_quote.get("data_mode", "REPLAY") if active_quote else "REPLAY"
    is_live = active_quote.get("is_live", False) if active_quote else False

    # 1. Provenance Banner
    render_provenance_banner(data_mode=data_mode, is_live=is_live)

    # 2. Executive Page Header
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">
                STOCKMIND <span style="color: var(--accent-cyan);">AI</span>
            </div>
            <div style="font-size: 0.76rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                Market Intelligence Command Center
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Market Overview Benchmark Ribbon (NIFTY 50, SENSEX, BANK NIFTY, INDIA VIX)
    st.markdown(
        """
        <div style="font-size: 0.7rem; color: var(--text-muted); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;">
            MARKET OVERVIEW · BENCHMARK INDICES
        </div>
        """,
        unsafe_allow_html=True,
    )

    benchmarks = [
        {"name": "NIFTY 50", "price": 24850.20, "change": 104.30, "pct": 0.42},
        {"name": "SENSEX", "price": 81420.50, "change": 308.15, "pct": 0.38},
        {"name": "BANK NIFTY", "price": 51240.10, "change": 315.40, "pct": 0.62},
        {"name": "INDIA VIX", "price": 13.42, "change": -0.29, "pct": -2.15},
    ]

    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    for idx, (col, b) in enumerate(zip([c_b1, c_b2, c_b3, c_b4], benchmarks)):
        with col:
            is_pos = b["pct"] >= 0
            # VIX inverted meaning
            if b["name"] == "INDIA VIX":
                delta_type = "positive" if b["pct"] < 0 else "negative"
            else:
                delta_type = "positive" if is_pos else "negative"
            sign = "+" if is_pos else ""
            render_kpi_card(
                title=b["name"],
                value=f"{b['price']:,.2f}",
                delta=f"{sign}{b['pct']:.2f}%",
                delta_type=delta_type,
                subtitle=f"{sign}₹{b['change']:,.2f}",
                icon="📊",
            )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 4. AI Market Signal & Risk Telemetry Row (Above the Fold)
    sig_data = api_client.get_ai_signal(symbol)
    risk_data = api_client.get_risk_metrics(symbol)
    regime_data = api_client.get_market_regime(symbol)

    sig_label = sig_data.get("signal", "NEUTRAL")
    comp_score = float(sig_data.get("composite_score", 0.0))
    sig_conf = float(sig_data.get("confidence_pct", 65.0))
    regime_label = regime_data.get("regime", "BULLISH").upper()
    risk_score = int(risk_data.get("var_95_1d_pct", 2.15) * 20)  # normalized 0-100 score

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_kpi_card(
            title=f"AI SIGNAL · {symbol}",
            value=sig_label,
            delta=f"Composite Score: {comp_score:+.1f}",
            delta_type="positive" if "BULLISH" in sig_label else ("negative" if "BEARISH" in sig_label else "neutral"),
            subtitle="Multi-Factor Consensus",
            icon="🤖",
            border_accent="cyan",
        )
    with s2:
        render_kpi_card(
            title="MARKET REGIME",
            value=regime_label,
            subtitle="HMM Dynamic State",
            delta="Trend-Following Active",
            delta_type="neutral",
            icon="🌐",
            border_accent="bullish" if "BULLISH" in regime_label else "warning",
        )
    with s3:
        render_kpi_card(
            title="AI CONFIDENCE",
            value=f"{sig_conf:.1f}%",
            subtitle="Ensemble Calibration",
            delta="HIGH CONVICTION" if sig_conf >= 70 else "MODERATE CONVICTION",
            delta_type="positive" if sig_conf >= 70 else "neutral",
            icon="🎯",
            border_accent="purple",
        )
    with s4:
        render_kpi_card(
            title="RISK SCORE",
            value=f"{risk_score} / 100",
            subtitle="Tail Risk & Volatility",
            delta="CONTROLLED EXPOSURE" if risk_score < 50 else "ELEVATED RISK",
            delta_type="positive" if risk_score < 50 else "negative",
            icon="🛡️",
            border_accent="warning" if risk_score >= 50 else "cyan",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 5. Active Symbol LTP & Depth Row
    render_quote_kpi_row(active_quote, symbol=symbol)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 6. Interactive Candlestick Chart & Sentinel Side Panel
    chart_col, intel_col = st.columns([2.8, 1.2])

    with chart_col:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary);">
                    📈 INTERACTIVE CHART · {symbol} ({timeframe.upper()})
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); font-family: 'JetBrains Mono';">MODE: {data_mode}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        candles = api_client.get_historical_ohlc(symbol=symbol, timeframe=timeframe, limit=100)
        fig = render_candlestick_chart(candles, symbol=symbol, timeframe=timeframe, theme_mode=theme_mode)
        st.plotly_chart(fig, use_container_width=True)

    with intel_col:
        st.markdown(
            """
            <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">
                ⚡ MARKET SENTINEL & PULSE
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Breadth
        st.markdown(
            """
            <div class="term-card">
                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">NSE Market Breadth</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace;">
                    <span style="color: var(--bullish); font-size: 0.8rem; font-weight: 700;">▲ 1,348 Advances</span>
                    <span style="color: var(--bearish); font-size: 0.8rem; font-weight: 700;">▼ 842 Declines</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.08); height: 6px; border-radius: 3px; overflow: hidden; display: flex;">
                    <div style="width: 61.5%; background: var(--bullish);"></div>
                    <div style="width: 38.5%; background: var(--bearish);"></div>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 6px;">A/D Ratio: 1.60 (Bullish Advance)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI Insights Card
        explanation_txt = sig_data.get("explanation", "Multi-factor consensus consolidating.")
        st.markdown(
            f"""
            <div class="term-card" style="border-left: 3px solid var(--accent-cyan);">
                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">
                    💡 AI REAL-TIME INSIGHT
                </div>
                <div style="font-size: 0.78rem; color: var(--text-secondary); line-height: 1.45;">
                    {explanation_txt}
                </div>
                <div style="margin-top: 8px; font-size: 0.65rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                    <span>MODEL: v3.2-prod</span>
                    <span style="color: var(--accent-cyan);">AI ANALYTICAL</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 7. Sector Heatmap & Top Movers Row
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    heat_col, movers_col = st.columns([2.0, 1.2])

    with heat_col:
        st.markdown(
            """
            <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">
                🗺️ MARKET SECTOR HEATMAP
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_heat = render_market_heatmap([], theme_mode=theme_mode)
        st.plotly_chart(fig_heat, use_container_width=True)

    with movers_col:
        st.markdown(
            """
            <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">
                🚀 TOP MARKET MOVERS
            </div>
            """,
            unsafe_allow_html=True,
        )

        top_movers = [
            ("TATAMOTORS", 1045.20, +3.40),
            ("ICICIBANK", 1210.00, +2.15),
            ("RELIANCE", 2890.50, +1.45),
            ("INFY", 1785.40, -1.20),
            ("TCS", 4120.00, -0.85),
        ]

        st.markdown('<div class="term-card" style="padding: 12px;">', unsafe_allow_html=True)
        for sym_m, pr, chg in top_movers:
            col_m = "var(--bullish)" if chg >= 0 else "var(--bearish)"
            sign_m = "+" if chg >= 0 else ""
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 7px 4px; border-bottom: 1px solid var(--border);">
                    <strong style="font-size: 0.82rem; color: var(--text-primary);">{sym_m}</strong>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                        <span style="color: var(--text-secondary); margin-right: 8px;">₹{pr:,.2f}</span>
                        <strong style="color: {col_m};">{sign_m}{chg:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # 8. News Sentiment Feed Preview
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">
            📰 LIVE NEWS SENTIMENT
        </div>
        """,
        unsafe_allow_html=True,
    )
    news_data = api_client.get_news_feed(symbol=symbol, limit=3)
    articles = news_data.get("articles", [])
    if articles:
        n_cols = st.columns(min(len(articles), 3))
        for idx, (col_n, art) in enumerate(zip(n_cols, articles[:3])):
            with col_n:
                sent_data = art.get("sentiment")
                if isinstance(sent_data, dict):
                    sent = str(sent_data.get("direction") or sent_data.get("label") or "NEUTRAL").upper()
                elif isinstance(sent_data, str):
                    sent = sent_data.upper()
                else:
                    sent = "NEUTRAL"
                sent_color = "var(--bullish)" if any(k in sent for k in ["POS", "BULL"]) else ("var(--bearish)" if any(k in sent for k in ["NEG", "BEAR"]) else "var(--neutral)")
                st.markdown(
                    f"""
                    <div class="term-card" style="height: 130px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: var(--text-muted); margin-bottom: 4px;">
                                <span>{art.get('source', 'Financial Express')}</span>
                                <span style="color: {sent_color}; font-weight: 700;">{sent}</span>
                            </div>
                            <div style="font-size: 0.78rem; font-weight: 600; color: var(--text-primary); line-height: 1.3; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">
                                {art.get('title', 'Market development')}
                            </div>
                        </div>
                        <div style="font-size: 0.66rem; color: var(--text-muted);">
                            {art.get('published_at', 'Recent')}
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

    render_command_center(api_client=get_api_client())
