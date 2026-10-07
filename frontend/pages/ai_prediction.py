"""Page 4: AI Prediction — Machine Learning & Deep Learning Quantitative Forecasting Engine."""

from datetime import datetime, timedelta
from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components.status_banner import render_ai_disclaimer, render_provenance_banner
from frontend.services.api_client import APIClient
from frontend.utils.formatters import format_inr


def render_ai_prediction(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    timeframe: str = "1d",
    theme_mode: str = "dark",
) -> None:
    """Render comprehensive Machine Learning & Deep Learning forecasting dashboard."""
    active_quote = api_client.get_quote(symbol)
    data_mode = active_quote.get("data_mode", "REPLAY") if active_quote else "REPLAY"
    is_live = active_quote.get("is_live", False) if active_quote else False

    render_provenance_banner(data_mode=data_mode, is_live=is_live)
    render_ai_disclaimer()

    is_dark = theme_mode.lower() == "dark"
    bg_color = "#0A0E17" if is_dark else "#F8FAFC"
    card_bg = "rgba(255,255,255,0.03)" if is_dark else "#FFFFFF"
    text_color = "#F1F5F9" if is_dark else "#0F172A"
    sub_color = "#94A3B8" if is_dark else "#64748B"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.28rem; font-weight: 800; color: #F1F5F9; letter-spacing: -0.5px;">
                    🔮 QUANTITATIVE PREDICTION & DEEP LEARNING ENGINE · <span style="color: #00F0FF;">{symbol}</span>
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8;">
                    Hybrid Ensemble (Classical ML + PyTorch LSTM/GRU) · Sequence Generation · Controlled Retraining Schedules
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mode Selector Tabs
    tab_signal, tab_ensemble, tab_dl, tab_comp = st.tabs(
        [
            "🎯 Real-Time AI Signal & SHAP",
            "🤝 Hybrid Consensus Ensemble",
            "🧠 Deep Learning (LSTM & GRU)",
            "📊 Model Registry & Benchmarks",
        ]
    )

    # =========================================================================
    # TAB 1: REAL-TIME AI SIGNAL & SHAP EXPLAINABILITY
    # =========================================================================
    with tab_signal:
        st.caption(
            "Synthesizes 8 analytical dimensions: Live Market Data + Technicals + Classical ML + "
            "Deep Learning + News Sentiment + Risk Intelligence + Anomaly Radar + Market Regime."
        )

        with st.spinner("Executing real-time multi-factor signal synthesis & SHAP decomposition..."):
            sig_data = api_client.get_ai_signal(symbol)
            shap_data = api_client.get_signal_explanation(symbol)

        sig_label = sig_data.get("signal", "NEUTRAL")
        comp_score = float(sig_data.get("composite_score", 0.0))
        sig_conf = float(sig_data.get("confidence_pct", 65.0))
        sig_mode = sig_data.get("data_mode", data_mode)
        sig_ver = sig_data.get("model_version", "v3.2-institutional")
        sig_ts = sig_data.get("timestamp", datetime.now().isoformat())
        explanation_txt = sig_data.get("explanation", "")
        supp_factors = sig_data.get("supporting_factors", [])
        opp_factors = sig_data.get("opposing_factors", [])
        comp_scores = sig_data.get("component_scores", {})

        sig_colors = {
            "STRONG BULLISH": ("#00E676", "▲▲ STRONG BULLISH"),
            "BULLISH": ("#00F0FF", "▲ BULLISH"),
            "NEUTRAL": ("#FFB300", "◆ NEUTRAL"),
            "BEARISH": ("#FF7700", "▼ BEARISH"),
            "STRONG BEARISH": ("#FF3366", "▼▼ STRONG BEARISH"),
        }
        sig_col, sig_display = sig_colors.get(sig_label, ("#00F0FF", sig_label))

        # Provenance Tags Ribbon
        st.markdown(
            f"""
            <div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
                <span class="badge-replay" style="color: #00F0FF; border-color: rgba(0,240,255,0.3); background: rgba(0,240,255,0.08);">
                    DATA PROVENANCE: {sig_mode}
                </span>
                <span class="badge-replay" style="color: #A855F7; border-color: rgba(168,85,247,0.3); background: rgba(168,85,247,0.08);">
                    ENGINE: MULTI-FACTOR CONSENSUS
                </span>
                <span class="badge-replay" style="color: #CBD5E1; border-color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.04);">
                    MODEL: {sig_ver}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Signal Hero Card
        st.markdown(
            f"""
            <div style="background: rgba(18, 24, 38, 0.7); border: 2px solid {sig_col}66; border-radius: 10px; padding: 18px; margin-bottom: 16px; box-shadow: 0 0 20px {sig_col}15;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                            INSTITUTIONAL COMPOSITE AI SIGNAL
                        </div>
                        <div style="font-size: 1.8rem; font-weight: 900; color: {sig_col}; letter-spacing: -0.5px;">
                            {sig_display}
                        </div>
                    </div>
                    <div style="display: flex; gap: 20px; font-family: 'JetBrains Mono', monospace;">
                        <div style="text-align: right;">
                            <div style="font-size: 0.70rem; color: #64748B;">COMPOSITE SCORE</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: {sig_col};">{comp_score:+.1f} / 100</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.70rem; color: #64748B;">CONFIDENCE</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #F1F5F9;">{sig_conf:.1f}%</div>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
                    {explanation_txt}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 8-Factor Deconstruction Grid
        st.markdown(
            """
            <div style="font-size: 0.85rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 8px;">
                8-Factor Analytical Dimension Attribution (-100 to +100)
            </div>
            """,
            unsafe_allow_html=True,
        )
        f_cols = st.columns(4)
        factor_items = [
            ("Live Market Flow", comp_scores.get("market_flow", 0.0), "VWAP & Intraday Momentum"),
            ("Technical AI Score", comp_scores.get("technical", 0.0), "SMA/EMA, RSI, MACD, Patterns"),
            ("Classical ML Ensemble", comp_scores.get("machine_learning", 0.0), "RandomForest, XGBoost, LogReg"),
            ("Deep Learning (LSTM/GRU)", comp_scores.get("deep_learning", 0.0), "Sequential Temporal Flow"),
            ("News Sentiment (FinBERT)", comp_scores.get("sentiment", 0.0), "Financial Lexicon Score"),
            ("Quantitative Risk", comp_scores.get("risk", 0.0), "VaR, Sharpe & Volatility Sizing"),
            ("Anomaly Radar", comp_scores.get("anomaly", 0.0), "Isolation Forest & Flow Divergence"),
            ("Market Macro Regime", comp_scores.get("regime", 0.0), "Bull, Bear, Sideways, High Vol"),
        ]
        for idx, (f_title, f_val, f_sub) in enumerate(factor_items):
            with f_cols[idx % 4]:
                f_c = "#00E676" if f_val > 15 else ("#FF3366" if f_val < -15 else "#FFB300")
                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 8px 10px; margin-bottom: 8px;">
                        <div style="font-size: 0.70rem; color: #94A3B8; font-weight: 600;">{f_title}</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; color: {f_c}; margin: 2px 0;">
                            {f_val:+.1f}
                        </div>
                        <div style="font-size: 0.62rem; color: #64748B;">{f_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Supporting vs Opposing Factors Columns
        c_supp, c_opp = st.columns(2)
        with c_supp:
            st.markdown(
                """
                <div style="background: rgba(0, 230, 118, 0.05); border: 1px solid rgba(0, 230, 118, 0.2); border-radius: 8px; padding: 12px; height: 100%;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #00E676; text-transform: uppercase; margin-bottom: 8px;">
                        ✅ Supporting Factors (Bullish Drivers)
                    </div>
                """,
                unsafe_allow_html=True,
            )
            for sf in supp_factors:
                st.markdown(f"<div style='font-size: 0.76rem; color: #E2E8F0; margin-bottom: 4px;'>• {sf}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_opp:
            st.markdown(
                """
                <div style="background: rgba(255, 51, 102, 0.05); border: 1px solid rgba(255, 51, 102, 0.2); border-radius: 8px; padding: 12px; height: 100%;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #FF3366; text-transform: uppercase; margin-bottom: 8px;">
                        ⚠️ Opposing Factors (Headwinds & Risk)
                    </div>
                """,
                unsafe_allow_html=True,
            )
            for of in opp_factors:
                st.markdown(f"<div style='font-size: 0.76rem; color: #E2E8F0; margin-bottom: 4px;'>• {of}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # SHAP Explainability Studio
        st.markdown(
            """
            <div style="background: rgba(18, 24, 38, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px;">
                <div style="font-size: 0.88rem; font-weight: 700; color: #00F0FF; text-transform: uppercase; margin-bottom: 4px;">
                    🧬 SHAP Model Interpretability Studio
                </div>
                <div style="font-size: 0.74rem; color: #94A3B8; margin-bottom: 12px;">
                    Shapley Feature Attribution: Decomposes exact mathematical contributions pushing the model forecast up or down.
                </div>
            """,
            unsafe_allow_html=True,
        )

        top_feats = shap_data.get("top_contributing_features", [])
        if top_feats:
            f_names = [f["feature_name"] for f in top_feats]
            f_shaps = [f["shap_value"] for f in top_feats]
            f_colors = ["#00E676" if s >= 0 else "#FF3366" for s in f_shaps]

            fig_shap = go.Figure(
                go.Bar(
                    x=f_shaps,
                    y=f_names,
                    orientation="h",
                    marker=dict(color=f_colors, line=dict(color="#0A0E17", width=1)),
                    text=[f"{s:+.3f}" for s in f_shaps],
                    textposition="auto",
                    textfont=dict(family="JetBrains Mono, monospace", size=9, color="#FFFFFF"),
                )
            )
            fig_shap.update_layout(
                paper_bgcolor="#0A0E17" if is_dark else "#F8FAFC",
                plot_bgcolor="#0A0E17" if is_dark else "#F8FAFC",
                margin=dict(l=10, r=10, t=10, b=10),
                height=240,
                xaxis=dict(
                    title=dict(text="SHAP Marginal Contribution (ϕ)", font=dict(size=10, color="#94A3B8")),
                    gridcolor="rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.05)",
                    tickfont=dict(family="JetBrains Mono", size=9, color="#64748B"),
                ),
                yaxis=dict(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=9, color="#CBD5E1")),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        narrative = shap_data.get("explanation_narrative", "")
        if narrative:
            st.markdown(
                f"""
                <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.45; background: rgba(0,0,0,0.3); border-radius: 6px; padding: 10px; margin-top: 8px;">
                    📝 <strong>Interpretability Narrative:</strong> {narrative}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: HYBRID CONSENSUS ENSEMBLE
    # =========================================================================
    with tab_ensemble:
        col_ctrl1, col_btn1 = st.columns([3, 1])
        with col_ctrl1:
            st.caption(
                "Aggregates Classical ML (Random Forest, Logistic Regression, SVM, Gradient Boosting) "
                "with Deep Learning Sequence Models (LSTM, GRU) using calibrated confidence weights."
            )
        with col_btn1:
            if st.button("🔄 Refresh Ensemble", key=f"refresh_ens_{symbol}", use_container_width=True):
                st.rerun()

        with st.spinner("Synthesizing multi-model consensus prediction..."):
            consensus = api_client.get_ensemble_consensus(symbol, timeframe)

        current_price = float(consensus.get("current_price", active_quote.get("price", 2850.0)))
        cons_dir = consensus.get("consensus_direction", "UP")
        cons_conf = float(consensus.get("consensus_confidence", 65.0))
        agree_pct = float(consensus.get("model_agreement_pct", 75.0))
        agree_summary = consensus.get("agreement_summary", "4 of 6 models agree")
        target_price = float(consensus.get("target_price", current_price * 1.015))
        exp_return_pct = float(consensus.get("expected_return_pct", 1.5))
        range_low = float(consensus.get("expected_range_low", current_price * 0.98))
        range_high = float(consensus.get("expected_range_high", current_price * 1.02))
        classical_dir = consensus.get("classical_direction", "UP")
        dl_dir = consensus.get("dl_direction", "UP")
        votes = consensus.get("votes", [])

        is_bullish = cons_dir == "UP"
        dir_color = "#00E676" if is_bullish else "#FF3366"
        dir_arrow = "▲ BULLISH" if is_bullish else "▼ BEARISH"

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">Consensus Direction</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem; font-weight:800; color:{dir_color}; margin-top:4px;">
                        {dir_arrow}
                    </div>
                    <div style="font-size:0.72rem; color:{sub_color};">Confidence: <strong>{cons_conf:.1f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">Model Agreement</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem; font-weight:800; color:#00F0FF; margin-top:4px;">
                        {agree_pct:.1f}%
                    </div>
                    <div style="font-size:0.72rem; color:#A78BFA;">{agree_summary}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            ret_color = "#00E676" if exp_return_pct >= 0 else "#FF3366"
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">Expected Target</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem; font-weight:800; color:#F1F5F9; margin-top:4px;">
                        {format_inr(target_price)}
                    </div>
                    <div style="font-size:0.72rem; color:{ret_color};"><strong>{exp_return_pct:+.2f}%</strong> expected movement</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">Statistical 95% Range</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.05rem; font-weight:700; color:#F1F5F9; margin-top:8px;">
                        {format_inr(range_low)} – {format_inr(range_high)}
                    </div>
                    <div style="font-size:0.72rem; color:{sub_color};">Volatility range (±2σ)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Consensus Sub-Category Banner
        st.markdown(
            f"""
            <div style="margin: 12px 0; padding: 8px 14px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; display: flex; justify-content: space-between; font-size: 0.78rem;">
                <div>Classical ML Sub-Consensus: <strong style="color: {'#00E676' if classical_dir=='UP' else '#FF3366'};">{classical_dir}</strong></div>
                <div>Deep Learning Sub-Consensus: <strong style="color: {'#00E676' if dl_dir=='UP' else '#FF3366'};">{dl_dir}</strong></div>
                <div>Agreement Status: <strong style="color: {'#00E676' if classical_dir==dl_dir else '#FFB300'};">{'ALIGNED' if classical_dir==dl_dir else 'DIVERGENT'}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Historical Trajectory Chart
        candles = api_client.get_historical_ohlc(symbol=symbol, timeframe=timeframe, limit=60)
        hist_closes = [float(c.get("close", current_price)) for c in candles] if candles else [current_price] * 30
        hist_timestamps = [c.get("timestamp", f"Day {i}") for i, c in enumerate(candles)] if candles else [f"T-{30-i}" for i in range(30)]

        future_dates = [(datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(5)]
        forecast_path = [current_price + (target_price - current_price) * ((i + 1) / 5) for i in range(5)]
        upper_bounds = [range_high] * 5
        lower_bounds = [range_low] * 5

        fig_ens = go.Figure()
        fig_ens.add_trace(
            go.Scatter(
                x=hist_timestamps,
                y=hist_closes,
                name="Observed History",
                line=dict(color="#94A3B8", width=2),
            )
        )
        fig_ens.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper_bounds,
                name="Range High (+2σ)",
                line=dict(color="rgba(0, 240, 255, 0.35)", width=1, dash="dash"),
            )
        )
        fig_ens.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower_bounds,
                name="Range Low (-2σ)",
                fill="tonexty",
                fillcolor="rgba(0, 240, 255, 0.08)",
                line=dict(color="rgba(0, 240, 255, 0.35)", width=1, dash="dash"),
            )
        )
        fig_ens.add_trace(
            go.Scatter(
                x=[hist_timestamps[-1]] + future_dates,
                y=[hist_closes[-1]] + forecast_path,
                name="Ensemble Consensus Path",
                line=dict(color="#00F0FF", width=2.5, dash="dot"),
                mode="lines+markers",
            )
        )

        fig_ens.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> · Multi-Model Consensus Projected Trajectory",
                font=dict(size=12, color=text_color, family="Plus Jakarta Sans"),
                x=0.01,
                xanchor="left",
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(family="JetBrains Mono, monospace", color=text_color, size=11),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title=dict(text="Price (₹)", font=dict(size=9, color=sub_color))),
            margin=dict(l=10, r=10, t=35, b=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_ens, use_container_width=True)

        # Model Agreement Breakdown
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #F1F5F9; margin: 16px 0 10px 0;">
                🗳 Individual Model Voting & Agreement Matrix
            </div>
            """,
            unsafe_allow_html=True,
        )

        if votes:
            vote_cols = st.columns(len(votes))
            for idx, v in enumerate(votes):
                v_name = v.get("model_name", f"Model {idx+1}")
                v_cat = v.get("category", "ML")
                v_dir = v.get("direction", "UP")
                v_conf = float(v.get("confidence", 50.0))
                v_wt = float(v.get("weight", 0.16))
                is_v_up = v_dir == "UP"
                v_color = "#00E676" if is_v_up else "#FF3366"
                cat_badge = "#A78BFA" if v_cat == "deep_learning" else "#38BDF8"

                with vote_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:10px; text-align:center;">
                            <div style="font-size:0.65rem; color:{cat_badge}; font-weight:700; text-transform:uppercase;">
                                { 'NEURAL DL' if v_cat == 'deep_learning' else 'CLASSICAL ML' }
                            </div>
                            <div style="font-size:0.8rem; font-weight:800; color:#F1F5F9; margin:2px 0;">
                                {v_name}
                            </div>
                            <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; font-weight:800; color:{v_color}; margin:4px 0;">
                                { '▲ UP' if is_v_up else '▼ DOWN' }
                            </div>
                            <div style="font-size:0.7rem; color:{sub_color};">Conf: <strong>{v_conf:.1f}%</strong></div>
                            <div style="font-size:0.65rem; color:#64748B;">Weight: {v_wt:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Individual model voting details loading from ensemble...")

    # =========================================================================
    # TAB 2: DEEP LEARNING (LSTM & GRU)
    # =========================================================================
    with tab_dl:
        col_dl_left, col_dl_right = st.columns([2, 1])
        with col_dl_left:
            arch_choice = st.radio(
                "Neural Architecture:",
                ["LSTM", "GRU"],
                horizontal=True,
                key=f"dl_arch_sel_{symbol}",
            )
        with col_dl_right:
            st.markdown(
                """
                <div style="text-align: right; padding-top: 14px;">
                    <span style="font-size:0.72rem; color:#94A3B8; background:rgba(0,240,255,0.08); padding:4px 8px; border-radius:4px; border:1px solid rgba(0,240,255,0.2);">
                        ⏱ Controlled Schedule: 60-min cooldown
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Trigger controlled training button
        col_tinfo, col_tbtn = st.columns([3, 1])
        with col_tinfo:
            st.caption(
                f"Multi-task recurrent neural network ({arch_choice}) with dual output heads: "
                "BCEWithLogits direction classifier & MSE expected return regressor + Monte Carlo Dropout."
            )
        with col_tbtn:
            if st.button(f"🧠 Train {arch_choice} Model", key=f"train_dl_btn_{symbol}", use_container_width=True):
                with st.spinner(f"Training {arch_choice} network with early stopping and checkpoints..."):
                    train_res = api_client.train_dl_models(
                        symbol=symbol,
                        model_type=arch_choice,
                        epochs=25,
                        sequence_length=30,
                    )
                    if train_res.get("models_trained"):
                        st.success(f"{arch_choice} model successfully trained and registered in model registry!")
                    else:
                        st.info("Controlled schedule active: Model is up-to-date within the cooldown interval.")

        with st.spinner(f"Running {arch_choice} sequence inference..."):
            dl_pred = api_client.get_dl_prediction(symbol, timeframe, model_type=arch_choice)

        dl_price = float(dl_pred.get("current_price", active_quote.get("price", 2850.0)))
        dl_direction = dl_pred.get("predicted_direction", "UP")
        dl_conf = float(dl_pred.get("confidence_score", 65.0))
        dl_prob_up = float(dl_pred.get("probability_up", 65.0))
        dl_prob_down = float(dl_pred.get("probability_down", 35.0))
        dl_target = float(dl_pred.get("target_price", dl_price * 1.015))
        dl_ret_pct = float(dl_pred.get("expected_return_pct", 1.5))
        dl_low = float(dl_pred.get("expected_range_low", dl_price * 0.98))
        dl_high = float(dl_pred.get("expected_range_high", dl_price * 1.02))
        dl_std = float(dl_pred.get("uncertainty_std", 0.012))
        dl_seq_len = int(dl_pred.get("sequence_length", 30))
        dl_ver = dl_pred.get("version", "v1.0")

        is_dl_up = dl_direction == "UP"
        dl_dir_color = "#00E676" if is_dl_up else "#FF3366"
        dl_dir_arrow = "▲ UP" if is_dl_up else "▼ DOWN"

        # DL KPI Cards
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">{arch_choice} Direction</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem; font-weight:800; color:{dl_dir_color}; margin-top:4px;">
                        {dl_dir_arrow}
                    </div>
                    <div style="font-size:0.72rem; color:{sub_color};">Confidence: <strong>{dl_conf:.1f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">Sequence Window</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem; font-weight:800; color:#00F0FF; margin-top:4px;">
                        {dl_seq_len} BARS
                    </div>
                    <div style="font-size:0.72rem; color:#A78BFA;">Lookback history</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d3:
            dl_ret_color = "#00E676" if dl_ret_pct >= 0 else "#FF3366"
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">Expected Return</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem; font-weight:800; color:{dl_ret_color}; margin-top:4px;">
                        {dl_ret_pct:+.2f}%
                    </div>
                    <div style="font-size:0.72rem; color:#F1F5F9;">Target: <strong>{format_inr(dl_target)}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                    <div style="font-size:0.7rem; color:{sub_color}; font-weight:600; text-transform:uppercase;">MC Uncertainty (σ)</div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:1.35rem; font-weight:800; color:#F1F5F9; margin-top:4px;">
                        ±{(dl_std*100):.2f}%
                    </div>
                    <div style="font-size:0.72rem; color:{sub_color};">Monte Carlo Dropout</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probability Distribution Bar
        st.markdown(
            f"""
            <div style="margin: 14px 0 10px 0; background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:6px;">
                    <span style="color:#00E676; font-weight:700;">▲ PROBABILITY UP: {dl_prob_up:.1f}%</span>
                    <span style="color:#FF3366; font-weight:700;">PROBABILITY DOWN: {dl_prob_down:.1f}% ▼</span>
                </div>
                <div style="height:10px; width:100%; background:rgba(255,51,102,0.3); border-radius:5px; overflow:hidden;">
                    <div style="height:100%; width:{dl_prob_up:.1f}%; background:#00E676; transition:width 0.5s ease;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # DL Model Architecture Summary
        c_arch1, c_arch2 = st.columns(2)
        with c_arch1:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;">
                    <div style="font-size:0.8rem; font-weight:700; color:#F1F5F9; text-transform:uppercase; margin-bottom:8px;">
                        🔬 Network Architecture ({arch_choice})
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:{sub_color}; line-height:1.6;">
                        • Input Features: Close, High, Low, Volume, Returns, SMA20, EMA20, RSI, MACD<br/>
                        • Recurrent Layers: 2 Stacked {'LSTM' if arch_choice=='LSTM' else 'GRU'} Layers (Hidden dim: 64)<br/>
                        • Dropout Regularization: 0.20 (active during inference for MC uncertainty)<br/>
                        • Dual Heads: Linear Direction Head + Linear Return Head<br/>
                        • Checkpoint Path: <code>models/{symbol.lower()}_{arch_choice.lower()}_best.pt</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_arch2:
            st.markdown(
                f"""
                <div style="background:{card_bg}; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:14px;">
                    <div style="font-size:0.8rem; font-weight:700; color:#F1F5F9; text-transform:uppercase; margin-bottom:8px;">
                        🛡️ Anti-Leakage & Controlled Schedule Protocol
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:{sub_color}; line-height:1.6;">
                        • Sequence Normalization: <code>SequenceScaler</code> fit strictly on Train split<br/>
                        • Chronological Split: 70% Train · 15% Validation · 15% Out-of-sample Test<br/>
                        • Early Stopping: Patience 7 epochs on Validation Multi-Task Loss<br/>
                        • Tick Thrash Protection: Minimum 60-minute retraining cooldown<br/>
                        • Inference Latency: Fast PyTorch forward pass (avg &lt; 2ms)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 3: MODEL REGISTRY & BENCHMARKS
    # =========================================================================
    with tab_comp:
        st.markdown(
            """
            <div style="font-size: 1.05rem; font-weight: 700; color: #F1F5F9; margin: 10px 0 6px 0;">
                📊 Comprehensive Benchmark Matrix: Classical ML vs Deep Learning
            </div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-bottom: 14px;">
                All models evaluated strictly out-of-sample on identical chronological test periods.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Fetching model comparison registry..."):
            dl_comp = api_client.get_dl_model_comparison(symbol)

        comp_models = dl_comp.get("models", [])
        if comp_models:
            rows = []
            for m in comp_models:
                m_type = m.get("model_type", "Unknown")
                is_dl = m_type.upper() in ["LSTM", "GRU"]
                cat_label = "Deep Learning" if is_dl else "Classical ML"

                rows.append(
                    {
                        "Architecture": m_type.upper(),
                        "Category": cat_label,
                        "Version": m.get("version", "v1.0"),
                        "Accuracy": f"{float(m.get('accuracy', 0.0))*100:.1f}%",
                        "Precision": f"{float(m.get('precision', 0.0))*100:.1f}%",
                        "Recall": f"{float(m.get('recall', 0.0))*100:.1f}%",
                        "F1 Score": f"{float(m.get('f1', 0.0)):.3f}",
                        "MAE": f"{float(m.get('mae', 0.0)):.4f}",
                        "RMSE": f"{float(m.get('rmse', 0.0)):.4f}",
                        "R²": f"{float(m.get('r2', 0.0)):.3f}",
                        "Status": "ACTIVE" if m.get("is_active") else "Trained",
                    }
                )
            df_comp = pd.DataFrame(rows)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        else:
            st.info("Registry initializing. Train models to populate benchmark comparison matrix.")

        # Storage directory note
        st.caption("All trained weights, checkpoints, and registry records stored locally in D:\\StockMind-AI\\models.")

    # Statutory Disclaimers
    st.markdown(
        """
        <div style="margin-top: 24px; padding: 14px; background: rgba(255, 179, 0, 0.05); border: 1px solid rgba(255, 179, 0, 0.2); border-radius: 8px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #FFB300; margin-bottom: 4px;">
                ⚠️ STATUTORY NOTICE: MODEL-GENERATED ANALYTICAL OUTPUT
            </div>
            <div style="font-size: 0.74rem; color: #94A3B8; line-height: 1.45;">
                Deep learning and machine learning predictions are algorithmic analytical estimates and not guaranteed outcomes.
                Financial markets carry inherent risks. Historical performance does not guarantee future results.
                Outputs are provided for educational and decision-support purposes only and do not constitute financial advice.
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

    render_ai_prediction(api_client=get_api_client())
