"""Page 11: AI Market Copilot — Institutional Quantitative Terminal Assistant.

Synthesizes multi-factor market telemetry with strict epistemic categorization:
- FACT
- CURRENT MARKET DATA
- MODEL PREDICTION
- INFERENCE
- UNCERTAINTY

Statutory Invariant: NEVER claims guaranteed returns.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.components.header import render_top_header
from frontend.components.status_banner import render_ai_disclaimer
from frontend.services.api_client import APIClient
from frontend.styles.theme import get_theme_colors, get_semantic_signal


DEFAULT_COPILOT_HISTORY = [
    {
        "role": "assistant",
        "content": (
            "### Market Context\n"
            "Indian equity benchmark NSE NIFTY 50 is trading with moderate institutional liquidity. Active constituent volatility remains within normal historical bands.\n\n"
            "### Technical View\n"
            "Primary trend indicators (20 EMA / 50 SMA) reflect neutral-to-bullish continuation with momentum oscillators confirming positive divergence.\n\n"
            "### AI Prediction\n"
            "Ensemble machine learning consensus (Random Forest + XGBoost + LightGBM + Neural Networks) projects a mild upward drift over the upcoming 5-day horizon.\n\n"
            "### Risk\n"
            "1-Day parametric Value-at-Risk (95%) stands at 2.15%. Tail risk metrics indicate controlled drawdown exposure.\n\n"
            "### News Sentiment\n"
            "FinBERT and domain lexicons indicate net positive sentiment across institutional wires, driven by operational expansion dispatches.\n\n"
            "### Conclusion\n"
            "Tactical bias favors disciplined trend participation with strict risk limits. *Analytical prediction only — not guaranteed financial advice.*"
        ),
        "epistemic_breakdown": None,
        "citations": [],
    }
]

PROMPT_SUGGESTIONS = [
    ("📊 Executive Market Brief", "Provide a comprehensive multi-factor quantitative briefing for {symbol}."),
    ("🔮 5-Day ML & Neural Forecast", "What is the hybrid machine-learning and deep-learning consensus projection and target range for {symbol}?"),
    ("🛡️ Tail Risk & 1D VaR", "Analyze the parametric Value-at-Risk, annualized volatility, and drawdown sensitivity for {symbol}."),
    ("📰 News & Catalysts", "Summarize recent financial news headlines and sentiment catalysts affecting {symbol}."),
    ("📈 Technicals & Score", "Break down the AI Technical Score (0-100), moving averages, and oscillator momentum for {symbol}."),
]


def render_ai_market_copilot(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    theme_mode: str = "dark",
) -> None:
    """Render interactive institutional AI Market Copilot terminal."""
    colors = get_theme_colors(theme_mode)
    quote = api_client.get_quote(symbol) or {}
    data_mode = quote.get("data_mode", "REPLAY")
    is_live = quote.get("is_live", False)
    market_open = quote.get("is_market_open", True)

    # Standard Top Header
    render_top_header(
        title="StockMind AI",
        subtitle="AI Market Copilot · Quantitative Terminal Assistant",
        market_open=market_open,
        data_mode=data_mode,
        ws_connected=True,
    )
    render_ai_disclaimer()

    # Session State Initialization
    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = list(DEFAULT_COPILOT_HISTORY)

    # 1. Page Header (Section 17: AI MARKET COPILOT - Your Real-Time Market Intelligence Assistant)
    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 1.3rem; font-weight: 800; color: {colors['text_primary']}; letter-spacing: -0.3px;">
                        🤖 AI MARKET COPILOT
                    </span>
                    <span class="term-badge badge-cyan" style="margin-left: 8px;">
                        REAL-TIME ASSISTANT
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: {colors['accent_cyan']}; font-family: 'JetBrains Mono', monospace;">
                    ⚡ Powered by Multi-Factor Synthesis Engine
                </div>
            </div>
            <div style="font-size: 0.84rem; color: {colors['text_secondary']}; margin-top: 2px;">
                Your Real-Time Market Intelligence Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Section 17 Requirement: Small "AI Context" Panel showing what the AI is using
    # (Live Data, Technical Analysis, ML Models, News, Risk Engine)
    st.markdown(
        f"""
        <div class="term-card" style="margin-bottom: 14px; padding: 12px 16px; border: 1px solid {colors['border']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.74rem; font-weight: 700; color: {colors['accent_purple']}; text-transform: uppercase; letter-spacing: 0.5px;">
                    🧠 ACTIVE AI CONTEXT PIPELINE
                </span>
                <span style="font-size: 0.70rem; color: {colors['text_muted']};">5 Analytical Engines Integrated</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;">
                <div style="background: {colors['bg_primary']}; border: 1px solid {colors['border']}; border-radius: 6px; padding: 8px 10px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 700; color: {colors['bullish']};">
                        <span class="pulse-dot dot-live" style="width: 7px; height: 7px;"></span> LIVE DATA
                    </div>
                    <div style="font-size: 0.70rem; color: {colors['text_muted']}; margin-top: 2px;">Tick Stream & VWAP Engine</div>
                </div>
                <div style="background: {colors['bg_primary']}; border: 1px solid {colors['border']}; border-radius: 6px; padding: 8px 10px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 700; color: {colors['accent_cyan']};">
                        <span class="pulse-dot dot-ai" style="width: 7px; height: 7px;"></span> TECHNICAL ANALYSIS
                    </div>
                    <div style="font-size: 0.70rem; color: {colors['text_muted']}; margin-top: 2px;">Oscillators & Trend Bias (0-100)</div>
                </div>
                <div style="background: {colors['bg_primary']}; border: 1px solid {colors['border']}; border-radius: 6px; padding: 8px 10px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 700; color: {colors['accent_purple']};">
                        <span class="pulse-dot dot-ai" style="width: 7px; height: 7px;"></span> ML MODELS
                    </div>
                    <div style="font-size: 0.70rem; color: {colors['text_muted']}; margin-top: 2px;">6-Model Hybrid Ensemble</div>
                </div>
                <div style="background: {colors['bg_primary']}; border: 1px solid {colors['border']}; border-radius: 6px; padding: 8px 10px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 700; color: {colors['bullish']};">
                        <span class="pulse-dot dot-live" style="width: 7px; height: 7px;"></span> NEWS & SENTIMENT
                    </div>
                    <div style="font-size: 0.70rem; color: {colors['text_muted']}; margin-top: 2px;">FinBERT NLP Catalyst Feed</div>
                </div>
                <div style="background: {colors['bg_primary']}; border: 1px solid {colors['border']}; border-radius: 6px; padding: 8px 10px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 700; color: {colors['warning']};">
                        <span class="pulse-dot dot-replay" style="width: 7px; height: 7px;"></span> RISK ENGINE
                    </div>
                    <div style="font-size: 0.70rem; color: {colors['text_muted']}; margin-top: 2px;">Parametric VaR & Anomaly Radar</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Active Instrument Selector & Quick Prompts
    c_sym, c_prompt_label = st.columns([1.5, 4.5])
    with c_sym:
        active_symbol = st.selectbox(
            "ACTIVE TICKER",
            ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "TATAMOTORS", "LT", "NIFTY 50"],
            index=0 if symbol not in ["TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "TATAMOTORS", "LT", "NIFTY 50"] else ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "TATAMOTORS", "LT", "NIFTY 50"].index(symbol),
            label_visibility="collapsed",
            key="copilot_symbol_selector",
        )

    p_cols = st.columns(len(PROMPT_SUGGESTIONS))
    trigger_prompt: Optional[str] = None

    for i, (btn_label, prompt_template) in enumerate(PROMPT_SUGGESTIONS):
        with p_cols[i]:
            if st.button(btn_label, key=f"quick_prompt_{i}", use_container_width=True):
                trigger_prompt = prompt_template.format(symbol=active_symbol)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Main Conversational Message History Container
    chat_box = st.container()

    with chat_box:
        for idx, msg in enumerate(st.session_state["copilot_history"]):
            is_user = msg["role"] == "user"

            if is_user:
                st.markdown(
                    f"""
                    <div style="margin-left: 15%; background: rgba(34, 211, 238, 0.08); border: 1px solid rgba(34, 211, 238, 0.25); border-left: 4px solid {colors['accent_cyan']}; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 0.72rem; font-weight: 800; color: {colors['accent_cyan']}; font-family: 'JetBrains Mono', monospace;">👤 USER INQUIRY</span>
                        </div>
                        <div style="font-size: 0.88rem; color: {colors['text_primary']}; line-height: 1.5;">
                            {msg['content']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Assistant response card with subtle cyan/purple accent
                st.markdown(
                    f"""
                    <div style="margin-right: 5%; background: {colors['panel']}; border: 1px solid {colors['border']}; border-left: 4px solid {colors['accent_purple']}; border-radius: 8px; padding: 16px 20px; margin-bottom: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                            <span style="font-size: 0.74rem; font-weight: 800; color: {colors['accent_cyan']}; font-family: 'JetBrains Mono', monospace;">
                                🤖 STOCKMIND AI MARKET COPILOT
                            </span>
                            <span class="term-badge badge-purple" style="font-size: 0.65rem;">
                                MULTI-FACTOR SYNTHESIS
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(msg["content"])

                # Expandable Epistemic Truth Matrix & Citations
                ep_breakdown = msg.get("epistemic_breakdown")
                citations = msg.get("citations", [])

                if ep_breakdown or citations:
                    with st.expander("🔍 Epistemic Grounding & Source Citations", expanded=False):
                        tab_fact, tab_live, tab_ml, tab_inf, tab_unc, tab_src = st.tabs([
                            "📌 Fact",
                            "⏱️ Market Data",
                            "🔮 Model Forecast",
                            "💡 Inference",
                            "⚠️ Uncertainty",
                            "📑 Sources & Citations",
                        ])

                        with tab_fact:
                            facts = ep_breakdown.get("fact", []) if ep_breakdown else []
                            if facts:
                                for f in facts:
                                    st.markdown(f"- `{f}`")
                            else:
                                st.caption("No statutory fact assertions logged for this turn.")

                        with tab_live:
                            live_pts = ep_breakdown.get("current_market_data", []) if ep_breakdown else []
                            if live_pts:
                                for lp in live_pts:
                                    st.markdown(f"- `{lp}`")
                            else:
                                st.caption("No real-time market data logged.")

                        with tab_ml:
                            preds = ep_breakdown.get("model_prediction", []) if ep_breakdown else []
                            if preds:
                                for pr in preds:
                                    st.markdown(f"- `{pr}`")
                            else:
                                st.caption("No algorithmic forecast logged.")

                        with tab_inf:
                            infs = ep_breakdown.get("inference", []) if ep_breakdown else []
                            if infs:
                                for inf in infs:
                                    st.markdown(f"- `{inf}`")
                            else:
                                st.caption("No analytical inference logged.")

                        with tab_unc:
                            uncs = ep_breakdown.get("uncertainty", []) if ep_breakdown else []
                            if uncs:
                                for u in uncs:
                                    st.markdown(f"- `{u}`")
                            else:
                                st.caption("Standard market volatility applies.")

                        with tab_src:
                            if citations:
                                for c in citations:
                                    st.markdown(
                                        f"**[{c.get('source', 'Market Wire')}]** {c.get('title', '')} "
                                        f"· *{c.get('time', '')}* ({c.get('impact', '')})"
                                    )
                            else:
                                st.caption("No external news citations referenced in this analytical pass.")

    # 5. Chat Input Box & Submission Handler
    user_input = st.chat_input("Ask StockMind Copilot anything about markets, valuation, technicals, or predictions...")

    # Process either chat input or quick prompt button
    active_query = trigger_prompt or user_input

    if active_query:
        st.session_state["copilot_history"].append({
            "role": "user",
            "content": active_query,
            "epistemic_breakdown": None,
            "citations": [],
        })

        with st.spinner("🤖 Consulting quantitative models, technical engine, and sentiment news flow..."):
            history_tuples = [
                {"role": m["role"], "content": m["content"][:200]}
                for m in st.session_state["copilot_history"][:-1]
            ]
            response_dict = api_client.ask_copilot(
                query=active_query,
                symbol=active_symbol,
                history=history_tuples,
            )

        st.session_state["copilot_history"].append({
            "role": "assistant",
            "content": response_dict.get("reply_markdown", "No response generated."),
            "epistemic_breakdown": response_dict.get("epistemic_breakdown"),
            "citations": response_dict.get("citations", []),
        })

        st.rerun()

    # Clear chat history button in footer
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    c_left, c_right = st.columns([8, 2])
    with c_right:
        if st.button("🗑️ Reset Chat History", use_container_width=True):
            st.session_state["copilot_history"] = list(DEFAULT_COPILOT_HISTORY)
            st.rerun()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from frontend.services.api_client import get_api_client

    render_ai_market_copilot(api_client=get_api_client())
