"""Page 7: Risk Intelligence — Multi-Factor Risk, Anomaly Detection, Market Regime & Tail Metrics."""

from typing import Any, Dict, List
import plotly.graph_objects as go
import streamlit as st

from frontend.components.header import render_top_header
from frontend.components.kpi_cards import render_kpi_card
from frontend.components.status_banner import render_provenance_banner
from frontend.services.api_client import APIClient
from frontend.styles.theme import apply_terminal_chart_theme, get_theme_colors, get_semantic_signal
from frontend.utils.formatters import format_inr


def render_risk_intelligence(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    theme_mode: str = "dark",
) -> None:
    """Render institutional risk assessment, anomaly detection, regime classification, and stress testing."""
    colors = get_theme_colors(theme_mode)
    active_quote = api_client.get_quote(symbol) or {}
    data_mode = active_quote.get("data_mode", "REPLAY")
    is_live = active_quote.get("is_live", False)
    market_open = active_quote.get("is_market_open", True)

    # Standard Top Header & Provenance Banner
    render_top_header(
        title="StockMind AI",
        subtitle=f"Risk Intelligence & Tail Risk Radar · {symbol}",
        market_open=market_open,
        data_mode=data_mode,
        ws_connected=True,
    )
    render_provenance_banner(data_mode=data_mode, is_live=is_live)

    # 1. Fetch live metrics from API client
    with st.spinner("Executing quantitative risk models & anomaly scans..."):
        risk_data = api_client.get_risk_metrics(symbol)
        regime_data = api_client.get_market_regime(symbol)
        anomaly_data = api_client.get_anomalies(symbol)
        stress_data = api_client.get_stress_test(symbol)

    var_95 = float(risk_data.get("var_95_1d_pct", 2.15))
    var_99 = float(risk_data.get("var_99_1d_pct", 3.05))
    vol_ann = float(risk_data.get("annualized_volatility_pct", 18.4))
    vol_20d = float(risk_data.get("rolling_20d_volatility_pct", vol_ann))
    sharpe = float(risk_data.get("sharpe_ratio", 1.85))
    sortino = float(risk_data.get("sortino_ratio", 2.41))
    calmar = float(risk_data.get("calmar_ratio", 1.65))
    max_dd = float(risk_data.get("max_drawdown_pct", 11.2))
    beta = float(risk_data.get("beta", 1.12))
    alpha = float(risk_data.get("alpha_annualized_pct", 2.10))
    cvar_95 = float(risk_data.get("expected_shortfall_95_pct", 3.24))
    downside_dev = float(risk_data.get("downside_deviation_pct", 12.1))
    semi_var = float(risk_data.get("semi_variance", 0.00012))
    risk_level = risk_data.get("risk_level", "MODERATE")
    risk_score_val = 42 if risk_level == "MODERATE" else (78 if risk_level == "HIGH" else 24)

    # 2. Section 13 Required Metrics: Risk Score, Volatility, Max Drawdown, Sharpe, Sortino, VaR, CVaR, Beta
    st.markdown(
        f"""
        <div style="font-size: 0.78rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
            INSTITUTIONAL RISK EXPOSURE PROFILE (G-SEC RF: 6.80% · BENCHMARK: NIFTY 50)
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_kpi_card(
            title="RISK SCORE",
            value=f"{risk_score_val} / 100",
            subtitle=f"Tier: {risk_level}",
            status=risk_level,
            accent="warning" if risk_level == "MODERATE" else ("bearish" if risk_level == "HIGH" else "bullish"),
        )
    with r2:
        render_kpi_card(
            title="ANNUAL VOLATILITY",
            value=f"{vol_ann:.1f}%",
            subtitle=f"20D Rolling: {vol_20d:.1f}%",
            accent="warning",
        )
    with r3:
        render_kpi_card(
            title="MAX DRAWDOWN",
            value=f"-{max_dd:.1f}%",
            subtitle=f"Calmar: {calmar:.2f}",
            accent="bearish",
        )
    with r4:
        render_kpi_card(
            title="SHARPE RATIO",
            value=f"{sharpe:.2f}",
            subtitle=f"Sortino: {sortino:.2f}",
            accent="bullish" if sharpe >= 1.5 else "cyan",
        )

    r5, r6, r7, r8 = st.columns(4)
    with r5:
        render_kpi_card(
            title="SORTINO RATIO",
            value=f"{sortino:.2f}",
            subtitle="Downside-Penalized Alpha",
            accent="bullish",
        )
    with r6:
        render_kpi_card(
            title="PARAMETRIC VAR (95%)",
            value=f"{var_95:.2f}%",
            subtitle=f"99% 1-Day: {var_99:.2f}%",
            accent="warning",
        )
    with r7:
        render_kpi_card(
            title="EXPECTED SHORTFALL",
            value=f"{cvar_95:.2f}%",
            subtitle="CVaR 95% Tail Loss",
            accent="bearish",
        )
    with r8:
        render_kpi_card(
            title="ASSET BETA",
            value=f"{beta:.2f}",
            subtitle=f"Alpha (Ann.): {alpha:+.2f}%",
            accent="cyan",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Compact Horizontal Risk Bars (Replacing clunky circular gauges with sleek progress bars)
    st.markdown(
        f"""
        <div class="term-card" style="margin-bottom: 14px; padding: 14px 18px;">
            <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.5px;">
                ⚡ QUANTITATIVE RISK FACTOR BREAKDOWN
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;">
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: {colors['text_secondary']}; margin-bottom: 4px;">
                        <span>1-Day VaR (95% Parametric)</span>
                        <strong style="color: {colors['warning']}; font-family: 'JetBrains Mono', monospace;">{var_95:.2f}%</strong>
                    </div>
                    <div style="height: 6px; background: {colors['bg_primary']}; border-radius: 3px; overflow: hidden;">
                        <div style="width: {min(100.0, (var_95 / 6.0) * 100.0)}%; height: 100%; background: {colors['warning']};"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: {colors['text_secondary']}; margin-bottom: 4px;">
                        <span>Annualized Volatility (σ)</span>
                        <strong style="color: {colors['accent_cyan']}; font-family: 'JetBrains Mono', monospace;">{vol_ann:.1f}%</strong>
                    </div>
                    <div style="height: 6px; background: {colors['bg_primary']}; border-radius: 3px; overflow: hidden;">
                        <div style="width: {min(100.0, (vol_ann / 40.0) * 100.0)}%; height: 100%; background: {colors['accent_cyan']};"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: {colors['text_secondary']}; margin-bottom: 4px;">
                        <span>Tail CVaR Expected Shortfall</span>
                        <strong style="color: {colors['bearish']}; font-family: 'JetBrains Mono', monospace;">{cvar_95:.2f}%</strong>
                    </div>
                    <div style="height: 6px; background: {colors['bg_primary']}; border-radius: 3px; overflow: hidden;">
                        <div style="width: {min(100.0, (cvar_95 / 8.0) * 100.0)}%; height: 100%; background: {colors['bearish']};"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: {colors['text_secondary']}; margin-bottom: 4px;">
                        <span>Peak-to-Trough Drawdown</span>
                        <strong style="color: {colors['bearish']}; font-family: 'JetBrains Mono', monospace;">-{max_dd:.1f}%</strong>
                    </div>
                    <div style="height: 6px; background: {colors['bg_primary']}; border-radius: 3px; overflow: hidden;">
                        <div style="width: {min(100.0, (max_dd / 30.0) * 100.0)}%; height: 100%; background: {colors['bearish']};"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Market Regime & Anomaly Detection Dual Panel
    col_regime, col_anomaly = st.columns([1.2, 1.8])

    with col_regime:
        primary_regime = regime_data.get("primary_regime", "SIDEWAYS")
        conf_pct = float(regime_data.get("confidence_pct", 70.0))
        stance = regime_data.get("recommended_stance", "Tactical Range Trading")
        regime_expl = regime_data.get("institutional_explanation", "")
        probs = regime_data.get("regime_probabilities", {"BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.40, "HIGH_VOLATILITY": 0.10})

        reg_color = colors["bullish"] if primary_regime == "BULL" else (colors["bearish"] if primary_regime == "BEAR" else (colors["accent_purple"] if primary_regime == "HIGH_VOLATILITY" else colors["warning"]))

        st.markdown(
            f"""
            <div class="term-card" style="height: 100%; padding: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase;">
                        MARKET REGIME DETECTOR
                    </span>
                    <span class="term-badge" style="color: {reg_color}; border-color: {reg_color};">
                        {primary_regime} ({conf_pct:.0f}%)
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: {colors['text_secondary']}; line-height: 1.45; margin-bottom: 10px;">
                    {regime_expl}
                </div>
                <div style="font-size: 0.74rem; color: {colors['accent_cyan']}; font-weight: 600; background: rgba(34, 211, 238, 0.08); padding: 6px 10px; border-radius: 6px; margin-bottom: 12px; border: 1px solid rgba(34, 211, 238, 0.2);">
                    🎯 Tactical Stance: {stance}
                </div>
                <div style="font-size: 0.72rem; color: {colors['text_muted']}; font-weight: 700; margin-bottom: 6px;">REGIME PROBABILITY DECOMPOSITION</div>
            """,
            unsafe_allow_html=True,
        )

        # Mini Probability Horizontal Bar Chart
        p_keys = list(probs.keys())
        p_vals = [probs[k] * 100.0 for k in p_keys]
        p_colors = [
            colors["bullish"] if k == "BULL" else (colors["bearish"] if k == "BEAR" else (colors["accent_purple"] if k == "HIGH_VOLATILITY" else colors["warning"]))
            for k in p_keys
        ]

        fig_bar = go.Figure(
            go.Bar(
                x=p_vals,
                y=p_keys,
                orientation="h",
                marker=dict(color=p_colors, line=dict(color=colors["bg_primary"], width=1)),
                text=[f"{v:.1f}%" for v in p_vals],
                textposition="auto",
                textfont=dict(family="JetBrains Mono, monospace", size=10, color="#FFFFFF"),
            )
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=10, t=5, b=5),
            height=130,
            xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
            yaxis=dict(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=10, color=colors["text_secondary"])),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_anomaly:
        cur_sev = anomaly_data.get("current_severity", "NORMAL")
        pri_expl = anomaly_data.get("primary_explanation", "Normal statistical flow.")
        div_stat = anomaly_data.get("divergence_status", "SYNCHRONIZED")
        anom_list = anomaly_data.get("anomalies", [])

        sev_color = colors["bearish"] if cur_sev == "CRITICAL" else (colors["warning"] if cur_sev in ["HIGH", "MEDIUM"] else colors["bullish"])

        st.markdown(
            f"""
            <div class="term-card" style="height: 100%; padding: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase;">
                        ISOLATION FOREST & ANOMALY RADAR
                    </span>
                    <span class="term-badge" style="color: {sev_color}; border-color: {sev_color};">
                        STATUS: {cur_sev}
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: {colors['text_secondary']}; line-height: 1.45; margin-bottom: 8px;">
                    {pri_expl}
                </div>
                <div style="font-size: 0.74rem; color: {colors['text_muted']}; margin-bottom: 10px; font-family: 'JetBrains Mono', monospace;">
                    ⚡ Divergence Status: <strong style="color: {colors['text_primary']};">{div_stat}</strong>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if anom_list:
            st.markdown(
                f"""
                <div style="max-height: 150px; overflow-y: auto; border: 1px solid {colors['border']}; border-radius: 6px; padding: 6px; background: {colors['bg_primary']};">
                """,
                unsafe_allow_html=True,
            )
            for a in anom_list[-4:]:
                a_sev = a.get("severity", "LOW")
                badge_c = colors["bearish"] if a_sev in ["CRITICAL", "HIGH"] else colors["warning"]
                st.markdown(
                    f"""
                    <div style="padding: 6px 8px; border-bottom: 1px solid {colors['border']}; font-size: 0.74rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                            <span style="font-family: 'JetBrains Mono', monospace; color: {colors['text_muted']};">{a.get('timestamp', '')} · {a.get('anomaly_type', '').replace('_', ' ')}</span>
                            <span style="color: {badge_c}; font-weight: 700;">{a_sev} ({a.get('score', 0):.2f})</span>
                        </div>
                        <div style="color: {colors['text_secondary']};">{a.get('description', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No structural price or volume anomalies detected in the recent observation window.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 5. Drawdown Underwater Curve & Detailed Quantitative Risk Matrix
    c_dd_chart, c_tail_matrix = st.columns([1.7, 1.3])

    with c_dd_chart:
        st.markdown(
            f"""
            <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase; margin-bottom: 8px;">
                UNDERWATER DRAWDOWN CURVE (PEAK-TO-TROUGH EROSION)
            </div>
            """,
            unsafe_allow_html=True,
        )

        dd_metrics = risk_data.get("drawdown_metrics", {})
        underwater_vals = dd_metrics.get("underwater_series", [])
        if not underwater_vals:
            underwater_vals = [-0.5, -1.2, -3.4, -6.8, -11.2, -8.4, -4.2, -1.8, 0.0, -0.4, -2.1]

        x_bars = [f"T-{len(underwater_vals) - i}" for i in range(len(underwater_vals))]
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=x_bars,
                y=underwater_vals,
                mode="lines",
                line=dict(color=colors["bearish"], width=2),
                fill="tozeroy",
                fillcolor=colors["bearish_soft"],
                name="Drawdown %",
            )
        )
        fig_dd.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=250,
            xaxis=dict(showgrid=False, tickfont=dict(family="JetBrains Mono", size=9, color=colors["text_muted"])),
            yaxis=dict(
                ticksuffix="%",
                gridcolor=colors["border"],
                tickfont=dict(family="JetBrains Mono", size=9, color=colors["text_muted"]),
            ),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with c_tail_matrix:
        st.markdown(
            f"""
            <div class="term-card" style="height: 100%; padding: 14px;">
                <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_primary']}; text-transform: uppercase; margin-bottom: 10px;">
                    Institutional Risk Matrix · {symbol}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
                    <div style="display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {colors['border']};">
                        <span style="color:{colors['text_muted']};">Asset Beta (vs NIFTY 50):</span><strong style="color:{colors['accent_cyan']};">{beta:.2f}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {colors['border']};">
                        <span style="color:{colors['text_muted']};">Jensen's Alpha (Ann.):</span><strong style="color:{colors['bullish']};">{alpha:+.2f}%</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {colors['border']};">
                        <span style="color:{colors['text_muted']};">Parametric VaR (95% / 99%):</span><strong style="color:{colors['warning']};">{var_95:.2f}% / {var_99:.2f}%</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {colors['border']};">
                        <span style="color:{colors['text_muted']};">Expected Shortfall (CVaR 95%):</span><strong style="color:{colors['bearish']};">{cvar_95:.2f}%</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {colors['border']};">
                        <span style="color:{colors['text_muted']};">Sortino Ratio (Downside):</span><strong style="color:{colors['bullish']};">{sortino:.2f}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {colors['border']};">
                        <span style="color:{colors['text_muted']};">Calmar Ratio (CAGR / MaxDD):</span><strong style="color:{colors['accent_cyan']};">{calmar:.2f}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid {colors['border']};">
                        <span style="color:{colors['text_muted']};">Downside Deviation (Ann.):</span><strong style="color:{colors['text_primary']};">{downside_dev:.2f}%</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding: 5px 0;">
                        <span style="color:{colors['text_muted']};">Downside Semi-Variance:</span><strong style="color:{colors['text_primary']};">{semi_var:.6f}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 6. Macroeconomic Stress Testing Scenarios Table
    st.markdown(
        f"""
        <div style="font-size: 0.80rem; font-weight: 700; color: {colors['text_muted']}; text-transform: uppercase; margin-bottom: 8px;">
            MACROECONOMIC CRISIS STRESS SIMULATION (HYPOTHETICAL TAIL SHOCKS)
        </div>
        """,
        unsafe_allow_html=True,
    )

    scenarios = stress_data.get("scenarios", [])
    if scenarios:
        st.markdown(
            f"""
            <div style="background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 8px 8px 0 0; padding: 10px 14px; display: grid; grid-template-columns: 2.2fr 2fr 1fr 1.2fr 1.2fr; font-size: 0.72rem; color: {colors['text_muted']}; font-weight: 700; letter-spacing: 0.05em;">
                <div>CRISIS SCENARIO</div>
                <div>DYNAMIC DRIVER</div>
                <div style="text-align: right;">BENCHMARK</div>
                <div style="text-align: right;">PROJECTED LOSS</div>
                <div style="text-align: right;">PER ₹1,00,000</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for sc in scenarios:
            sev = sc.get("severity", "HIGH")
            col = colors["bearish"] if sev == "CRITICAL" else (colors["warning"] if sev == "HIGH" else colors["accent_cyan"])
            loss_pct = sc.get("projected_asset_impact_pct", 0.0)
            loss_val = sc.get("estimated_loss_per_100k", 0.0)
            st.markdown(
                f"""
                <div style="background: {colors['bg_secondary']}; border-bottom: 1px solid {colors['border']}; border-right: 1px solid {colors['border']}; border-left: 1px solid {colors['border']}; padding: 10px 14px; display: grid; grid-template-columns: 2.2fr 2fr 1fr 1.2fr 1.2fr; font-size: 0.78rem; align-items: center;">
                    <div>
                        <strong style="color: {colors['text_primary']};">{sc.get('scenario_name')}</strong>
                        <span style="font-size: 0.65rem; color: {col}; border: 1px solid {col}55; border-radius: 3px; padding: 1px 4px; margin-left: 6px; font-weight: 700;">{sev}</span>
                    </div>
                    <div style="color: {colors['text_muted']}; font-size: 0.74rem;">{sc.get('description')}</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; color: {colors['text_secondary']};">{sc.get('simulated_market_drop_pct')}%</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {colors['bearish']};">{loss_pct:.1f}%</div>
                    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {colors['bearish']};">-₹{loss_val:,.0f}</div>
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

    render_risk_intelligence(api_client=get_api_client())
