"""Data Provenance & Statutory Disclaimer Banners.

Renders high-visibility banners clearly distinguishing:
- LIVE DATA
- HISTORICAL DATA
- REPLAY / SIMULATED DATA
- AI PREDICTION
"""

import streamlit as st


def render_provenance_banner(data_mode: str = "REPLAY", is_live: bool = False) -> None:
    """Render prominent banner detailing current market data provenance using unified theme variables."""
    mode_str = str(data_mode).upper()
    if is_live or mode_str == "LIVE":
        st.markdown(
            """
            <div style="background: var(--bullish-soft); border: 1px solid rgba(34, 197, 94, 0.35); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--bullish); box-shadow:0 0 10px var(--bullish);"></span>
                    <strong style="color: var(--bullish); font-size: 0.82rem; letter-spacing: 0.04em;">LIVE BROKER FEED ACTIVE</strong>
                    <span style="color: var(--text-secondary); font-size: 0.78rem;">| Streaming tick data from authorized Indian exchange gateway</span>
                </div>
                <span style="color: var(--text-muted); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">REAL-TIME EXECUTION READY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode_str == "HISTORICAL":
        st.markdown(
            """
            <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--accent-blue); box-shadow:0 0 10px var(--accent-blue);"></span>
                    <strong style="color: var(--accent-blue); font-size: 0.82rem; letter-spacing: 0.04em;">HISTORICAL MARKET DATA MODE</strong>
                    <span style="color: var(--text-secondary); font-size: 0.78rem;">| Official end-of-day OHLCV session records archived from exchange</span>
                </div>
                <span style="color: var(--accent-blue); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;">HISTORICAL ARCHIVE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode_str in ("PREDICTION", "AI PREDICTION"):
        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--accent-purple); box-shadow:0 0 10px var(--accent-purple);"></span>
                    <strong style="color: var(--accent-purple); font-size: 0.82rem; letter-spacing: 0.04em;">AI PREDICTION MODE</strong>
                    <span style="color: var(--text-secondary); font-size: 0.78rem;">| Probabilistic quantitative forecasting models</span>
                </div>
                <span style="color: var(--accent-purple); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;">STATISTICAL ESTIMATE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: var(--warning-soft); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--warning); box-shadow:0 0 10px var(--warning);"></span>
                    <strong style="color: var(--warning); font-size: 0.82rem; letter-spacing: 0.04em;">DEMO / REPLAY DATA MODE</strong>
                    <span style="color: var(--text-secondary); font-size: 0.78rem;">| Prices generated via realistic Brownian replay engine for sandbox testing</span>
                </div>
                <span style="color: var(--warning); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;">DEMO / REPLAY - NOT REAL MONEY</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ai_disclaimer() -> None:
    """Render statutory AI prediction disclaimer banner."""
    st.markdown(
        """
        <div style="background: rgba(34, 211, 238, 0.04); border: 1px solid rgba(34, 211, 238, 0.2); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.95rem;">🔮</span>
                <strong style="color: var(--accent-cyan); font-size: 0.8rem; letter-spacing: 0.03em;">AI STATISTICAL ESTIMATION DISCLAIMER:</strong>
                <span style="color: var(--text-muted); font-size: 0.76rem;">
                    Projections represent probabilistic quantitative models (LSTM, XGBoost, Ensembles). Models carry inherent forecast uncertainty. Never constitute financial or investment advice.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
