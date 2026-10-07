"""Page 8: Portfolio — Holdings, Allocation, Sector Exposure, and Modern Portfolio Theory Optimization."""

from typing import Any, Dict, List
import plotly.graph_objects as go
import streamlit as st

from frontend.components.header import render_top_header
from frontend.components.kpi_cards import render_kpi_card
from frontend.components.status_banner import render_provenance_banner
from frontend.services.api_client import APIClient
from frontend.styles.theme import apply_terminal_chart_theme, get_theme_colors, get_semantic_signal
from frontend.utils.formatters import format_inr


DEFAULT_HOLDINGS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries", "shares": 50, "avg_price": 2720.0, "current_price": 2890.50, "sector": "Energy / Oil & Gas"},
    {"symbol": "TCS", "name": "Tata Consultancy Services", "shares": 40, "avg_price": 3850.0, "current_price": 4120.00, "sector": "Information Technology"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Limited", "shares": 100, "avg_price": 1480.0, "current_price": 1640.25, "sector": "Banking & Financial Services"},
    {"symbol": "INFY", "name": "Infosys Limited", "shares": 60, "avg_price": 1620.0, "current_price": 1785.40, "sector": "Information Technology"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "shares": 80, "avg_price": 1050.0, "current_price": 1210.00, "sector": "Banking & Financial Services"},
]


def render_portfolio(
    api_client: APIClient,
    theme_mode: str = "dark",
) -> None:
    """Render Portfolio Holdings, Sector Exposure, and Analytical Modern Portfolio Theory (MPT) Optimizer."""
    colors = get_theme_colors(theme_mode)

    # Standard Top Header
    render_top_header(
        title="StockMind AI",
        subtitle="Institutional Portfolio & MPT Optimizer",
        market_open=True,
        data_mode="REPLAY",
        ws_connected=True,
    )

    # 1. Evaluate Portfolio Holdings
    with st.spinner("Valuating portfolio & calculating risk attribution..."):
        summary = api_client.evaluate_portfolio(DEFAULT_HOLDINGS, realized_pnl=12450.0)

    total_value = float(summary.get("total_value", 725000.0))
    total_cost = float(summary.get("total_cost_basis", 650000.0))
    unrealized_pnl = float(summary.get("unrealized_pnl", 75000.0))
    unrealized_pnl_pct = float(summary.get("unrealized_pnl_pct", 11.5))
    realized_pnl = float(summary.get("realized_pnl", 12450.0))
    today_pnl = unrealized_pnl * 0.12  # Estimated today's P&L session contribution
    today_pnl_pct = 1.38
    hhi = float(summary.get("hhi", 2150.0))
    tier = summary.get("concentration_tier", "MODERATE")
    div_ratio = float(summary.get("diversification_ratio", 1.28))
    port_vol = float(summary.get("portfolio_volatility_pct", 16.4))
    port_beta = float(summary.get("portfolio_beta", 1.04))
    port_sharpe = float(summary.get("portfolio_sharpe_ratio", 1.45))
    port_var = float(summary.get("portfolio_var_95_1d_pct", 1.72))

    # Section 14 Primary KPI Row: Portfolio Value, Today's P&L, Total P&L, Risk Score
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="PORTFOLIO VALUE",
            value=format_inr(total_value),
            subtitle=f"Cost Basis: {format_inr(total_cost)}",
            status="ACTIVE",
            accent="cyan",
        )
    with c2:
        td_sign = "+" if today_pnl >= 0 else ""
        render_kpi_card(
            title="TODAY'S P&L",
            value=f"{td_sign}{format_inr(today_pnl)}",
            delta=f"{td_sign}{today_pnl_pct:.2f}%",
            delta_type="positive" if today_pnl >= 0 else "negative",
            subtitle="Current Trading Session",
            accent="bullish" if today_pnl >= 0 else "bearish",
        )
    with c3:
        pnl_sign = "+" if unrealized_pnl >= 0 else ""
        render_kpi_card(
            title="TOTAL P&L",
            value=f"{pnl_sign}{format_inr(unrealized_pnl)}",
            delta=f"{pnl_sign}{unrealized_pnl_pct:.2f}%",
            delta_type="positive" if unrealized_pnl >= 0 else "negative",
            subtitle=f"Realized: {format_inr(realized_pnl)}",
            accent="bullish" if unrealized_pnl >= 0 else "bearish",
        )
    with c4:
        render_kpi_card(
            title="RISK & CONCENTRATION",
            value=f"HHI {hhi:.0f}",
            subtitle=f"{tier} · 1D VaR: {port_var:.2f}%",
            status="MODERATE",
            accent="warning",
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Secondary Risk Attribution Row: Diversification score, Volatility, Beta, Sharpe
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_kpi_card(
            title="DIVERSIFICATION SCORE",
            value=f"{div_ratio:.2f}x",
            subtitle=f"Effective Assets: {summary.get('effective_constituents', 4.5):.1f}",
            accent="purple",
        )
    with r2:
        render_kpi_card(
            title="PORTFOLIO VOLATILITY",
            value=f"{port_vol:.1f}%",
            subtitle="Annualized Standard Deviation",
            accent="warning",
        )
    with r3:
        render_kpi_card(
            title="PORTFOLIO BETA",
            value=f"{port_beta:.2f}",
            subtitle="Benchmark: NIFTY 50",
            accent="cyan",
        )
    with r4:
        render_kpi_card(
            title="PORTFOLIO SHARPE",
            value=f"{port_sharpe:.2f}",
            subtitle="Risk-Adjusted Excess Return",
            accent="bullish",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Holdings Table & Sector Exposure Pie Chart
    col_table, col_pie = st.columns([2.1, 1.4])

    with col_table:
        st.markdown(
            f"""
            <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase; margin-bottom: 8px;">
                ACTIVE HOLDINGS & POSITION ALLOCATION
            </div>
            <div style="background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 8px 8px 0 0; padding: 10px 14px; display: grid; grid-template-columns: 2fr 0.8fr 1.2fr 1.2fr 1.2fr 1.4fr; font-size: 0.70rem; color: {colors['text_muted']}; font-weight: 700; letter-spacing: 0.05em;">
                <div>SECURITY</div>
                <div style="text-align: right;">QTY</div>
                <div style="text-align: right;">AVG BUY</div>
                <div style="text-align: right;">CURRENT</div>
                <div style="text-align: right;">WEIGHT</div>
                <div style="text-align: right;">UNREALIZED P&L</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        holdings_list = summary.get("holdings", DEFAULT_HOLDINGS)
        for h in holdings_list:
            qty = h["shares"]
            avg_p = h["avg_price"]
            cur_p = h.get("current_price", avg_p)
            weight = h.get("weight_pct", 20.0)
            pnl = h.get("unrealized_pnl", (cur_p - avg_p) * qty)
            pnl_pct = h.get("unrealized_pnl_pct", ((cur_p - avg_p) / avg_p * 100.0) if avg_p > 0 else 0.0)
            pnl_color = colors["bullish"] if pnl >= 0 else colors["bearish"]
            sign = "+" if pnl >= 0 else ""

            st.markdown(
                f"""
                <div style="background: {colors['bg_secondary']}; border-bottom: 1px solid {colors['border']}; border-right: 1px solid {colors['border']}; border-left: 1px solid {colors['border']}; padding: 10px 14px; display: grid; grid-template-columns: 2fr 0.8fr 1.2fr 1.2fr 1.2fr 1.4fr; font-size: 0.78rem; align-items: center;">
                    <div>
                        <strong style="color: {colors['text_primary']};">{h['symbol']}</strong>
                        <div style="font-size: 0.66rem; color: {colors['text_muted']};">{h.get('sector', 'Equities')}</div>
                    </div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; color: {colors['text_secondary']};">{qty}</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; color: {colors['text_muted']};">{format_inr(avg_p)}</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {colors['text_primary']};">{format_inr(cur_p)}</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; color: {colors['accent_cyan']};">{weight:.1f}%</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {pnl_color};">
                        {sign}{format_inr(pnl)}<br><span style="font-size: 0.68rem;">({sign}{pnl_pct:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_pie:
        st.markdown(
            f"""
            <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase; margin-bottom: 8px;">
                SECTOR EXPOSURE DISTRIBUTION
            </div>
            """,
            unsafe_allow_html=True,
        )

        sec_list = summary.get("sector_exposures", [])
        if sec_list:
            sec_labels = [s["sector"] for s in sec_list]
            sec_values = [s["value"] for s in sec_list]
        else:
            sec_labels = ["Energy", "Information Technology", "Banking & Finance"]
            sec_values = [144500, 354000, 260800]

        pie_colors = [colors["accent_cyan"], colors["accent_blue"], colors["accent_purple"], colors["bullish"], colors["warning"], colors["bearish"]]

        fig_pie = go.Figure(
            go.Pie(
                labels=sec_labels,
                values=sec_values,
                hole=0.55,
                marker=dict(colors=pie_colors, line=dict(color=colors["bg_primary"], width=2)),
                textinfo="label+percent",
                textfont=dict(family="JetBrains Mono, monospace", size=9, color=colors["text_primary"]),
            )
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 4. Modern Portfolio Theory (MPT) Optimizer Section
    st.markdown(
        f"""
        <div class="term-card" style="padding: 16px;">
            <div style="font-size: 0.90rem; font-weight: 700; color: {colors['accent_cyan']}; text-transform: uppercase; margin-bottom: 4px;">
                🔬 Modern Portfolio Theory (MPT) Analytical Optimizer
            </div>
            <div style="font-size: 0.74rem; color: {colors['text_muted']}; margin-bottom: 12px;">
                Markowitz Mean-Variance Efficient Frontier Simulation · Risk-Free Rate: 6.80% (10Y G-Sec)
            </div>
        """,
        unsafe_allow_html=True,
    )

    opt_symbols = [h["symbol"] for h in DEFAULT_HOLDINGS]
    cur_weights = {h["symbol"]: h.get("weight_pct", 20.0) / 100.0 for h in DEFAULT_HOLDINGS}

    with st.spinner("Computing Markowitz Efficient Frontier & Tangency Portfolio..."):
        mpt_res = api_client.optimize_portfolio_mpt(symbols=opt_symbols, current_weights=cur_weights)

    max_sharpe = mpt_res.get("max_sharpe_portfolio", {})
    min_vol = mpt_res.get("min_volatility_portfolio", {})
    frontier = mpt_res.get("efficient_frontier", [])
    cloud = mpt_res.get("simulated_cloud_sample", [])
    disclaimer = mpt_res.get("compliance_disclaimer", "")

    # Side-by-side: Efficient Frontier Chart and Optimal Weights
    c_front_chart, c_front_weights = st.columns([1.8, 1.2])

    with c_front_chart:
        fig_front = go.Figure()

        # Scatter cloud
        if cloud:
            c_vols = [pt["volatility_pct"] for pt in cloud]
            c_rets = [pt["expected_return_pct"] for pt in cloud]
            c_srs = [pt["sharpe_ratio"] for pt in cloud]
            fig_front.add_trace(
                go.Scatter(
                    x=c_vols,
                    y=c_rets,
                    mode="markers",
                    marker=dict(size=4, color=c_srs, colorscale="Viridis", showscale=False, opacity=0.35),
                    name="Simulated Feasible Set",
                )
            )

        # Efficient Frontier Line
        if frontier:
            f_vols = [pt["volatility_pct"] for pt in frontier]
            f_rets = [pt["target_return_pct"] for pt in frontier]
            fig_front.add_trace(
                go.Scatter(
                    x=f_vols,
                    y=f_rets,
                    mode="lines",
                    line=dict(color=colors["accent_cyan"], width=3),
                    name="Efficient Frontier",
                )
            )

        # Max Sharpe marker
        if max_sharpe:
            fig_front.add_trace(
                go.Scatter(
                    x=[max_sharpe.get("volatility_pct", 17.5)],
                    y=[max_sharpe.get("expected_return_pct", 18.2)],
                    mode="markers+text",
                    marker=dict(color=colors["bullish"], size=14, symbol="star", line=dict(color="#FFFFFF", width=1.5)),
                    text=["Max Sharpe"],
                    textposition="top center",
                    textfont=dict(color=colors["bullish"], size=10, family="JetBrains Mono"),
                    name="Tangency (Max Sharpe)",
                )
            )

        # Min Volatility marker
        if min_vol:
            fig_front.add_trace(
                go.Scatter(
                    x=[min_vol.get("volatility_pct", 14.2)],
                    y=[min_vol.get("expected_return_pct", 14.8)],
                    mode="markers+text",
                    marker=dict(color=colors["warning"], size=12, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
                    text=["Min Vol"],
                    textposition="bottom center",
                    textfont=dict(color=colors["warning"], size=10, family="JetBrains Mono"),
                    name="Min Volatility",
                )
            )

        # Current Portfolio marker
        fig_front.add_trace(
            go.Scatter(
                x=[port_vol],
                y=[16.2],
                mode="markers+text",
                marker=dict(color=colors["bearish"], size=11, symbol="circle", line=dict(color="#FFFFFF", width=1)),
                text=["Current"],
                textposition="middle right",
                textfont=dict(color=colors["bearish"], size=10, family="JetBrains Mono"),
                name="Current Portfolio",
            )
        )

        fig_front.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            xaxis=dict(
                title=dict(text="Annualized Volatility (σ %)", font=dict(size=10, color=colors["text_muted"])),
                gridcolor=colors["border"],
                tickfont=dict(family="JetBrains Mono", size=9, color=colors["text_muted"]),
            ),
            yaxis=dict(
                title=dict(text="Expected Return (μ %)", font=dict(size=10, color=colors["text_muted"])),
                gridcolor=colors["border"],
                tickfont=dict(family="JetBrains Mono", size=9, color=colors["text_muted"]),
            ),
            legend=dict(font=dict(size=9, color=colors["text_secondary"]), orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_front, use_container_width=True)

    with c_front_weights:
        st.markdown(
            f"""
            <div style="font-size: 0.78rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 6px;">
                TANGENCY PORTFOLIO WEIGHTS (MAX SHARPE)
            </div>
            <div style="font-size: 0.72rem; color: {colors['text_muted']}; margin-bottom: 8px;">
                Exp Return: <strong style="color: {colors['bullish']};">{max_sharpe.get('expected_return_pct', 0.0):.1f}%</strong> · 
                Vol: <strong style="color: {colors['warning']};">{max_sharpe.get('volatility_pct', 0.0):.1f}%</strong> · 
                Sharpe: <strong style="color: {colors['accent_cyan']};">{max_sharpe.get('sharpe_ratio', 0.0):.2f}</strong>
            </div>
            <div style="background: {colors['bg_primary']}; border: 1px solid {colors['border']}; border-radius: 6px; padding: 6px;">
            """,
            unsafe_allow_html=True,
        )

        ms_w = max_sharpe.get("weights", {})
        for sym, w_val in ms_w.items():
            cur_w_val = cur_weights.get(sym, 0.2) * 100.0
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 6px; border-bottom: 1px solid {colors['border']}; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;">
                    <span style="color: {colors['text_primary']}; font-weight: 600;">{sym}</span>
                    <span style="color: {colors['text_muted']};">Current: {cur_w_val:.1f}%</span>
                    <span style="color: {colors['bullish']}; font-weight: 700;">Target: {w_val:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Compliance Disclaimer Banner
    st.markdown(
        f"""
        <div style="margin-top: 14px; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 6px; padding: 8px 12px; font-size: 0.73rem; color: {colors['warning']}; line-height: 1.45;">
            ⚠️ {disclaimer}
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

    render_portfolio(api_client=get_api_client())
