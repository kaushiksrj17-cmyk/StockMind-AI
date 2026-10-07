"""Page 5: Technical Analysis — Institutional Indicator Studio & AI Score."""

from typing import Any, Dict, List
import streamlit as st
import pandas as pd

from frontend.components.status_banner import render_provenance_banner
from frontend.charts.technical_indicators import (
    render_rsi_chart,
    render_macd_chart,
    render_bollinger_bands,
    render_ai_score_gauge,
    render_comprehensive_indicator_chart,
)
from frontend.services.api_client import APIClient


def render_technical_analysis(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    timeframe: str = "1d",
    theme_mode: str = "dark",
) -> None:
    """Render comprehensive technical indicator studio and AI score."""
    # 1. Fetch real quote snapshot and technical analysis
    active_quote = api_client.get_quote(symbol)
    data_mode = active_quote.get("data_mode", "REPLAY") if active_quote else "REPLAY"
    is_live = active_quote.get("is_live", False) if active_quote else False

    render_provenance_banner(data_mode=data_mode, is_live=is_live)

    # 2. Fetch full technical analysis
    with st.spinner(f"Computing mathematical technical indicators for {symbol}..."):
        analysis = api_client.get_technical_analysis(symbol=symbol, timeframe=timeframe, limit=200)

    if not analysis:
        st.error(f"Unable to compute technical analysis for '{symbol}'. Please verify ticker or data availability.")
        return

    current_price = analysis.get("current_price", 0.0)
    price_change = analysis.get("price_change", 0.0)
    price_change_pct = analysis.get("price_change_pct", 0.0)
    score_data = analysis.get("ai_technical_score", {})
    score = score_data.get("score", 50.0)
    classification = score_data.get("classification", "NEUTRAL")
    trend_dir = score_data.get("trend_direction", "SIDEWAYS")
    trend_str = score_data.get("trend_strength", "MODERATE")
    ma_data = analysis.get("moving_averages", {})
    golden_cross = ma_data.get("golden_cross", False)
    death_cross = ma_data.get("death_cross", False)
    is_bullish_stack = ma_data.get("is_bullish_stack", False)
    is_bearish_stack = ma_data.get("is_bearish_stack", False)

    chg_color = "#00E676" if price_change >= 0 else "#FF3366"
    chg_sign = "+" if price_change >= 0 else ""

    # Header Bar
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">📈 TECHNICAL ANALYSIS STUDIO</span>
                <span style="margin-left: 10px; font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--accent-cyan);">{symbol}</span>
                <span style="margin-left: 6px; font-size: 0.76rem; color: var(--text-muted); background: var(--panel); padding: 3px 8px; border-radius: 4px; border: 1px solid var(--border);">{timeframe.upper()}</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 800; color: var(--text-primary);">
                ₹{current_price:,.2f} <span style="font-size: 0.92rem; color: {'var(--bullish)' if price_change >= 0 else 'var(--bearish)'}; font-weight: 700;">({chg_sign}{price_change:,.2f} / {chg_sign}{price_change_pct:.2f}%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Executive Technical Metrics (4 Columns)
    c_gauge, c_trend, c_cross, c_stack = st.columns([1.5, 1.1, 1.1, 1.3])

    with c_gauge:
        fig_gauge = render_ai_score_gauge(score=score, classification=classification, theme_mode=theme_mode)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c_trend:
        t_color = "var(--bullish)" if "UP" in trend_dir else ("var(--bearish)" if "DOWN" in trend_dir else "var(--warning)")
        st.markdown(
            f"""
            <div class="term-card" style="text-align: center; height: 210px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">TREND REGIME</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 800; color: {t_color}; margin: 8px 0 4px 0;">
                    {trend_dir}
                </div>
                <div style="font-size: 0.76rem; color: var(--text-secondary);">Strength: <strong style="color: var(--text-primary);">{trend_str.replace('_', ' ').title()}</strong></div>
                <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 8px;">ADX: {analysis.get('trend', {}).get('adx', 0.0)} · +DI: {analysis.get('trend', {}).get('plus_di', 0.0)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_cross:
        if golden_cross:
            cross_title = "GOLDEN CROSS"
            cross_color = "var(--bullish)"
            cross_desc = "50-SMA crossed above 200-SMA"
        elif death_cross:
            cross_title = "DEATH CROSS"
            cross_color = "var(--bearish)"
            cross_desc = "50-SMA crossed below 200-SMA"
        else:
            sma50 = ma_data.get("values", {}).get("sma_50", 0.0)
            sma200 = ma_data.get("values", {}).get("sma_200", 0.0)
            if sma50 >= sma200:
                cross_title = "50 > 200 SMA"
                cross_color = "var(--accent-cyan)"
                cross_desc = "Bullish medium-term trend"
            else:
                cross_title = "50 < 200 SMA"
                cross_color = "var(--warning)"
                cross_desc = "Bearish medium-term trend"

        st.markdown(
            f"""
            <div class="term-card" style="text-align: center; height: 210px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">CROSSOVER STATUS</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 800; color: {cross_color}; margin: 8px 0 4px 0;">
                    {cross_title}
                </div>
                <div style="font-size: 0.74rem; color: var(--text-secondary);">{cross_desc}</div>
                <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 8px;">SMA 50: ₹{ma_data.get('values', {}).get('sma_50', 0):,.1f}<br>SMA 200: ₹{ma_data.get('values', {}).get('sma_200', 0):,.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_stack:
        if is_bullish_stack:
            stack_label = "BULLISH STACK"
            stack_color = "var(--bullish)"
            stack_desc = "Price > SMA20 > SMA50 > SMA200"
        elif is_bearish_stack:
            stack_label = "BEARISH STACK"
            stack_color = "var(--bearish)"
            stack_desc = "Price < SMA20 < SMA50 < SMA200"
        else:
            stack_label = "MIXED ALIGNMENT"
            stack_color = "var(--warning)"
            stack_desc = "Consolidation / Transitional Stack"

        st.markdown(
            f"""
            <div class="term-card" style="text-align: center; height: 210px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">MA STACK BIAS</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 800; color: {stack_color}; margin: 8px 0 4px 0;">
                    {stack_label}
                </div>
                <div style="font-size: 0.74rem; color: var(--text-secondary);">{stack_desc}</div>
                <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 8px;">Bias Score: {ma_data.get('bias_score', 0)} / 7</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Primary Master Candlestick Chart with Indicators Overlay
    fig_master = render_comprehensive_indicator_chart(analysis=analysis, theme_mode=theme_mode)
    st.plotly_chart(fig_master, use_container_width=True)

    # Oscillators Subplots (RSI & MACD)
    candles = api_client.get_historical_ohlc(symbol=symbol, timeframe=timeframe, limit=100)
    c_rsi, c_macd = st.columns(2)
    with c_rsi:
        fig_rsi = render_rsi_chart(candles, symbol=symbol, period=14, theme_mode=theme_mode)
        st.plotly_chart(fig_rsi, use_container_width=True)
    with c_macd:
        fig_macd = render_macd_chart(candles, theme_mode=theme_mode)
        st.plotly_chart(fig_macd, use_container_width=True)

    # Deep Technical Indicators Matrix (Tabs)
    tab_mas, tab_osc, tab_vol, tab_levels, tab_patterns = st.tabs([
        "📊 Moving Averages",
        "🌊 Oscillators & Momentum",
        "⚡ Volatility & Volume",
        "🎯 Support/Resistance & Fibonacci",
        "🕯 Candlestick Patterns",
    ])

    with tab_mas:
        ma_vals = ma_data.get("values", {})
        ma_df = pd.DataFrame([
            {"Indicator": "SMA 20", "Value": f"₹{ma_vals.get('sma_20', 0):,.2f}", "Distance": f"{ma_data.get('price_vs_sma20_pct', 0):+.2f}%", "Interpretation": "Above" if ma_data.get('price_vs_sma20_pct', 0) >= 0 else "Below"},
            {"Indicator": "SMA 50", "Value": f"₹{ma_vals.get('sma_50', 0):,.2f}", "Distance": f"{ma_data.get('price_vs_sma50_pct', 0):+.2f}%", "Interpretation": "Above" if ma_data.get('price_vs_sma50_pct', 0) >= 0 else "Below"},
            {"Indicator": "SMA 100", "Value": f"₹{ma_vals.get('sma_100', 0):,.2f}", "Distance": f"{((current_price - ma_vals.get('sma_100', 1)) / ma_vals.get('sma_100', 1)) * 100:+.2f}%", "Interpretation": "Medium-Term Benchmark"},
            {"Indicator": "SMA 200", "Value": f"₹{ma_vals.get('sma_200', 0):,.2f}", "Distance": f"{ma_data.get('price_vs_sma200_pct', 0):+.2f}%", "Interpretation": "Institutional Baseline (Bullish)" if ma_data.get('price_vs_sma200_pct', 0) >= 0 else "Institutional Baseline (Bearish)"},
            {"Indicator": "EMA 9", "Value": f"₹{ma_vals.get('ema_9', 0):,.2f}", "Distance": f"{((current_price - ma_vals.get('ema_9', 1)) / ma_vals.get('ema_9', 1)) * 100:+.2f}%", "Interpretation": "Fast Tactical Trigger"},
            {"Indicator": "EMA 21", "Value": f"₹{ma_vals.get('ema_21', 0):,.2f}", "Distance": f"{((current_price - ma_vals.get('ema_21', 1)) / ma_vals.get('ema_21', 1)) * 100:+.2f}%", "Interpretation": "Short-Term Trend Guide"},
            {"Indicator": "EMA 50", "Value": f"₹{ma_vals.get('ema_50', 0):,.2f}", "Distance": f"{((current_price - ma_vals.get('ema_50', 1)) / ma_vals.get('ema_50', 1)) * 100:+.2f}%", "Interpretation": "Intermediate Trend EMA"},
            {"Indicator": "EMA 200", "Value": f"₹{ma_vals.get('ema_200', 0):,.2f}", "Distance": f"{((current_price - ma_vals.get('ema_200', 1)) / ma_vals.get('ema_200', 1)) * 100:+.2f}%", "Interpretation": "Long-Term Trend EMA"},
        ])
        st.dataframe(ma_df, hide_index=True, use_container_width=True)

    with tab_osc:
        osc = analysis.get("oscillators", {})
        macd_sub = osc.get("macd", {})
        stoch_sub = osc.get("stochastic", {})
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown(
                f"""<div class="term-card">
<div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; margin-bottom: 8px;">MOMENTUM OSCILLATORS</div>
<table style="width: 100%; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
<tr><td style="color: #94A3B8;">RSI (14):</td><td><strong>{osc.get('rsi', 50)}</strong> ({osc.get('rsi_condition', 'NEUTRAL')})</td></tr>
<tr><td style="color: #94A3B8;">ROC (12):</td><td><strong>{osc.get('roc_12', 0):+.2f}%</strong></td></tr>
<tr><td style="color: #94A3B8;">Momentum (10):</td><td><strong>{osc.get('momentum_10', 0):+.2f}</strong></td></tr>
<tr><td style="color: #94A3B8;">Oscillator Composite:</td><td><strong>{osc.get('oscillator_score', 0)} / 5</strong></td></tr>
</table>
</div>""",
                unsafe_allow_html=True,
            )
        with col_o2:
            st.markdown(
                f"""<div class="term-card">
<div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; margin-bottom: 8px;">MACD & STOCHASTIC</div>
<table style="width: 100%; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
<tr><td style="color: #94A3B8;">MACD Line:</td><td><strong>{macd_sub.get('macd_line', 0)}</strong></td></tr>
<tr><td style="color: #94A3B8;">Signal Line:</td><td><strong>{macd_sub.get('signal_line', 0)}</strong></td></tr>
<tr><td style="color: #94A3B8;">Histogram:</td><td><strong style="color: {'#00E676' if macd_sub.get('histogram', 0) >= 0 else '#FF3366'};">{macd_sub.get('histogram', 0)}</strong> ({macd_sub.get('condition', '')})</td></tr>
<tr><td style="color: #94A3B8;">Stochastic %K / %D:</td><td><strong>{stoch_sub.get('stoch_k', 50)} / {stoch_sub.get('stoch_d', 50)}</strong> ({stoch_sub.get('condition', '')})</td></tr>
</table>
</div>""",
                unsafe_allow_html=True,
            )

    with tab_vol:
        volat = analysis.get("volatility", {})
        bb = volat.get("bollinger_bands", {})
        vol_prof = analysis.get("volume_profile", {})
        div_data = vol_prof.get("divergence", {})
        div_type = div_data.get("divergence_type", "CONFIRMING") if isinstance(div_data, dict) else str(div_data)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(
                f"""<div class="term-card">
<div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; margin-bottom: 8px;">VOLATILITY & BOLLINGER BANDS</div>
<table style="width: 100%; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
<tr><td style="color: #94A3B8;">BB Upper / Lower:</td><td><strong>₹{bb.get('upper', 0):,.2f} / ₹{bb.get('lower', 0):,.2f}</strong></td></tr>
<tr><td style="color: #94A3B8;">BB %B Position:</td><td><strong>{bb.get('pct_b', 0.5):.2f}</strong> ({bb.get('position', 'INSIDE_BANDS')})</td></tr>
<tr><td style="color: #94A3B8;">Bandwidth:</td><td><strong>{bb.get('bandwidth_pct', 0):.2f}%</strong> {'⚠️ (Squeeze Active)' if bb.get('is_squeeze') else ''}</td></tr>
<tr><td style="color: #94A3B8;">ATR (14):</td><td><strong>₹{volat.get('atr', 0):,.2f} ({volat.get('atr_pct_of_price', 0):.2f}%)</strong></td></tr>
<tr><td style="color: #94A3B8;">Annualized Realized Vol:</td><td><strong>{volat.get('annualized_volatility_pct', 0):.1f}%</strong></td></tr>
</table>
</div>""",
                unsafe_allow_html=True,
            )
        with col_v2:
            div_color = "#00E676" if "BULLISH" in div_type else ("#FF3366" if "BEARISH" in div_type else "#94A3B8")
            st.markdown(
                f"""<div class="term-card">
<div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; margin-bottom: 8px;">VOLUME FLOW & VWAP</div>
<table style="width: 100%; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
<tr><td style="color: #94A3B8;">VWAP:</td><td><strong>₹{vol_prof.get('vwap', 0):,.2f}</strong> ({vol_prof.get('price_vs_vwap_pct', 0):+.2f}% vs Price)</td></tr>
<tr><td style="color: #94A3B8;">Relative Volume (RVOL):</td><td><strong>{vol_prof.get('rvol_20', 1.0):.2f}x</strong> {'🔥 (Volume Surge)' if vol_prof.get('is_volume_surge') else ''}</td></tr>
<tr><td style="color: #94A3B8;">OBV (On-Balance Volume):</td><td><strong>{vol_prof.get('obv', 0):,d}</strong></td></tr>
<tr><td style="color: #94A3B8;">Price-Volume Divergence:</td><td><strong style="color: {div_color};">{div_type.replace('_', ' ').title()}</strong></td></tr>
</table>
</div>""",
                unsafe_allow_html=True,
            )

    with tab_levels:
        sr = analysis.get("support_resistance", {})
        fib = analysis.get("fibonacci_levels", {})
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            pivots = sr.get("pivot_points", {})
            st.markdown(
                f"""<div class="term-card">
<div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; margin-bottom: 8px;">SWING PIVOTS & KEY LEVELS</div>
<div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 6px;">
Nearest Support: <strong style="color: #00E676;">₹{sr.get('nearest_support', 0):,.2f}</strong> ·
Nearest Resistance: <strong style="color: #FF3366;">₹{sr.get('nearest_resistance', 0):,.2f}</strong>
</div>
<table style="width: 100%; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
<tr><td style="color: #FF3366;">Resistance 2 (R2):</td><td>₹{pivots.get('r2', 0):,.2f}</td></tr>
<tr><td style="color: #FF8000;">Resistance 1 (R1):</td><td>₹{pivots.get('r1', 0):,.2f}</td></tr>
<tr><td style="color: #00F0FF;">Central Pivot (P):</td><td>₹{pivots.get('pivot', 0):,.2f}</td></tr>
<tr><td style="color: #00E676;">Support 1 (S1):</td><td>₹{pivots.get('s1', 0):,.2f}</td></tr>
<tr><td style="color: #059669;">Support 2 (S2):</td><td>₹{pivots.get('s2', 0):,.2f}</td></tr>
</table>
</div>""",
                unsafe_allow_html=True,
            )
        with col_l2:
            fib_levels = fib.get("levels", {})
            fib_rows = [{"Ratio": k, "Price": f"₹{v:,.2f}"} for k, v in fib_levels.items()]
            st.markdown(
                f"""<div class="term-card">
<div style="font-size: 0.8rem; font-weight: 700; color: #00F0FF; margin-bottom: 8px;">FIBONACCI RETRACEMENTS (Lookback Swing)</div>
<div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 6px;">
High: ₹{fib.get('swing_high', 0):,.2f} · Low: ₹{fib.get('swing_low', 0):,.2f} · Nearest: <strong>{fib.get('nearest_level', '')}</strong>
</div>
</div>""",
                unsafe_allow_html=True,
            )
            if fib_rows:
                st.dataframe(pd.DataFrame(fib_rows), hide_index=True, use_container_width=True)

    with tab_patterns:
        patterns = analysis.get("candlestick_patterns", [])
        if patterns:
            pat_records = []
            for p in patterns:
                pat_records.append({
                    "Pattern": p.get("name"),
                    "Bias": p.get("bias"),
                    "Significance": p.get("significance"),
                    "Recency": f"{p.get('bars_ago', 0)} bar(s) ago",
                    "Description": p.get("description"),
                })
            st.dataframe(pd.DataFrame(pat_records), hide_index=True, use_container_width=True)
        else:
            st.info("No prominent Japanese candlestick reversal patterns recognized within recent bars.")

    # Plain-English Explanations & Technical Score Rationale
    st.markdown(
        """<div style="margin-top: 16px; margin-bottom: 6px; font-size: 0.95rem; font-weight: 700; color: #F1F5F9;">
🧠 AI TECHNICAL SCORE EXPLANATION & SUB-COMPONENT ATTRIBUTION
</div>""",
        unsafe_allow_html=True,
    )

    c_exp1, c_exp2 = st.columns([1.4, 1.0])
    with c_exp1:
        st.markdown(
            f"""<div class="term-card">
<div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 6px;">EXECUTIVE REASONING</div>
<div style="font-size: 0.88rem; line-height: 1.5; color: #F1F5F9;">
{score_data.get('score_explanation', '')}
</div>
</div>""",
            unsafe_allow_html=True,
        )
    with c_exp2:
        comp_scores = score_data.get("component_scores", {})
        comp_df = pd.DataFrame([
            {"Pillar": "Trend & MAs (30%)", "Score": f"{comp_scores.get('trend', 50)}/100"},
            {"Pillar": "Momentum (30%)", "Score": f"{comp_scores.get('momentum', 50)}/100"},
            {"Pillar": "Volume (15%)", "Score": f"{comp_scores.get('volume', 50)}/100"},
            {"Pillar": "Volatility (15%)", "Score": f"{comp_scores.get('volatility', 50)}/100"},
            {"Pillar": "Price Action (10%)", "Score": f"{comp_scores.get('price_action', 50)}/100"},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Detailed Indicator Explanations Expanders
    with st.expander("📖 Indicator Pillar Summaries (Plain English)", expanded=False):
        explanations = score_data.get("indicator_explanations", {})
        for cat, expl in explanations.items():
            st.markdown(f"**{cat.replace('_', ' ').title()}**: {expl}")

    # Mandatory Statutory Regulatory Disclaimer Box
    disclaimer_text = analysis.get("disclaimer", "")
    st.markdown(
        f"""<div style="margin-top: 20px; padding: 12px 16px; background: rgba(255, 179, 0, 0.08); border-left: 3px solid #FFB300; border-radius: 6px; font-size: 0.75rem; color: #CBD5E1; line-height: 1.45;">
<strong>⚖️ STATISTICAL & REGULATORY NOTICE:</strong> {disclaimer_text}
</div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from frontend.services.api_client import get_api_client

    render_technical_analysis(api_client=get_api_client())
