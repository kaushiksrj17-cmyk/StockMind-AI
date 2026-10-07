"""Page 6: News & Sentiment Intelligence — FinBERT & Financial NLP Feed."""

from datetime import datetime
from typing import Any, Dict, List
import streamlit as st

from frontend.components.header import render_top_header
from frontend.components.kpi_cards import render_kpi_card
from frontend.components.status_banner import render_ai_disclaimer
from frontend.services.api_client import APIClient
from frontend.styles.theme import get_theme_colors, get_semantic_signal


def render_news_sentiment(
    api_client: APIClient,
    symbol: str = "RELIANCE",
    theme_mode: str = "dark",
) -> None:
    """Render FinBERT NLP News Sentiment Intelligence."""
    colors = get_theme_colors(theme_mode)
    quote = api_client.get_quote(symbol) or {}
    data_mode = quote.get("data_mode", "REPLAY")
    is_live = quote.get("is_live", False)
    market_open = quote.get("is_market_open", True)

    # Standard Top Header & AI Disclaimer
    render_top_header(
        title="StockMind AI",
        subtitle="News & Sentiment Intelligence",
        market_open=market_open,
        data_mode=data_mode,
        ws_connected=True,
    )
    render_ai_disclaimer()

    # 1. Ticker Filter & Feed Fetching Toolbar
    c_hdr, c_sel, c_lim = st.columns([2.5, 1.5, 1.0])
    with c_hdr:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0;">
                <span style="font-size: 1.1rem; font-weight: 700; color: {colors['text_primary']};">
                    📰 FINANCIAL NLP NEWSWIRE
                </span>
                <span class="term-badge badge-cyan">FinBERT & Loughran-McDonald</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_sel:
        target_sym = st.selectbox(
            "FILTER BY INSTRUMENT",
            ["ALL MARKET", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "TATAMOTORS", "LT", "NIFTY 50"],
            index=0,
            label_visibility="collapsed",
            key="news_symbol_filter",
        )
    with c_lim:
        article_limit = st.slider("Max Articles", min_value=5, max_value=30, value=15, step=5, label_visibility="collapsed")

    query_sym = "" if target_sym == "ALL MARKET" else target_sym

    # Query API / fallback
    feed_data = api_client.get_news_feed(symbol=query_sym or "MARKET", limit=article_limit)
    articles = feed_data.get("articles", [])
    summary = feed_data.get("sentiment_summary", {})

    agg_score = summary.get("aggregate_score", 0.64)
    regime = summary.get("sentiment_regime", "BULLISH")
    total_arts = summary.get("total_articles", len(articles))
    bullish_cnt = summary.get("bullish_count", 0)
    bearish_cnt = summary.get("bearish_count", 0)
    neutral_cnt = summary.get("neutral_count", 0)
    avg_impact = summary.get("average_impact_score", 58.0)

    # 2. KPI Cards Strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        regime_sig = "BULLISH" if agg_score > 0.15 else ("BEARISH" if agg_score < -0.15 else "NEUTRAL")
        regime_color = colors["bullish"] if regime_sig == "BULLISH" else (colors["bearish"] if regime_sig == "BEARISH" else colors["neutral"])
        render_kpi_card(
            title="SENTIMENT REGIME",
            value=f"{agg_score:+.2f} · {regime}",
            subtitle=f"{bullish_cnt} Bullish · {neutral_cnt} Neutral · {bearish_cnt} Bearish",
            status=regime_sig,
            color=regime_color,
            accent="bullish" if regime_sig == "BULLISH" else ("bearish" if regime_sig == "BEARISH" else "neutral"),
        )
    with c2:
        render_kpi_card(
            title="NEWS VOLUME",
            value=f"{total_arts} Articles",
            subtitle="Deduplicated & Time-Normalized",
            status="INDEXED",
            accent="cyan",
        )
    with c3:
        render_kpi_card(
            title="AVERAGE IMPACT",
            value=f"{avg_impact:.1f} / 100",
            subtitle="Tier-Weighted Catalyst Score",
            accent="purple",
        )
    with c4:
        render_kpi_card(
            title="NLP ARCHITECTURE",
            value="FinBERT + Lexicon",
            subtitle="Loughran-McDonald Institutional",
            status="OPERATIONAL",
            accent="cyan",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Main Views: Feed, Synthesis & Live Tester
    tab_feed, tab_synth, tab_tester = st.tabs([
        "📰 Processed News Feed & Impact",
        "📋 Multi-Event Executive Synthesis",
        "🧪 Ad-Hoc Financial Sentiment Analyzer",
    ])

    with tab_feed:
        if not articles:
            st.info("No matching news articles located for current filter.")
        else:
            for item in articles:
                sent_data = item.get("sentiment") or {}
                label = sent_data.get("label", "NEUTRAL").upper()
                score = float(sent_data.get("score", 0.0))
                impact = float(item.get("impact_score", 50.0))

                # Section 12 rule: Sentiment badge communicates sentiment (Positive=green, Neutral=slate, Negative=red)
                if label in ["POSITIVE", "BULLISH"]:
                    badge_class = "badge-bullish"
                    badge_label = f"▲ POSITIVE ({score:+.2f})"
                elif label in ["NEGATIVE", "BEARISH"]:
                    badge_class = "badge-bearish"
                    badge_label = f"▼ NEGATIVE ({score:+.2f})"
                else:
                    badge_class = "badge-neutral"
                    badge_label = f"● NEUTRAL ({score:+.2f})"

                tier = item.get("source_tier", 2)
                tier_label = "Tier 1 Wire" if tier == 1 else ("Mainstream Press" if tier == 2 else "Desk Re-Syndication")
                item_sym = item.get("symbol", "NIFTY50")

                st.markdown(
                    f"""
                    <div class="term-card" style="margin-bottom: 12px; padding: 14px 18px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 8px;">
                            <div style="font-size: 0.96rem; font-weight: 700; color: {colors['text_primary']}; line-height: 1.4;">
                                {item.get('title', '')}
                            </div>
                            <span class="term-badge {badge_class}" style="white-space: nowrap;">
                                {badge_label}
                            </span>
                        </div>
                        <div style="font-size: 0.84rem; color: {colors['text_secondary']}; margin-bottom: 10px; line-height: 1.5;">
                            {item.get('summary', '')}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; color: {colors['text_muted']}; border-top: 1px solid {colors['border']}; padding-top: 8px;">
                            <div>
                                <strong style="color: {colors['text_secondary']};">{item.get('source', '')}</strong> ({tier_label})
                                &nbsp;·&nbsp; ⏱ {item.get('time_ago', '')}
                                &nbsp;·&nbsp; Ticker: <span style="color: {colors['accent_cyan']}; font-family: 'JetBrains Mono', monospace; font-weight: 600;">#{item_sym}</span>
                            </div>
                            <div style="color: {colors['accent_purple']}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                                Catalyst Impact: {impact:.1f}/100
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_synth:
        synth_sym = query_sym or "RELIANCE"
        st.markdown(f"#### Executive Multi-Event Synthesis · **{synth_sym}**")
        with st.spinner("Synthesizing recent dispatches..."):
            exec_brief = api_client.get_executive_summary(symbol=synth_sym)

        st.markdown(
            f"""
            <div class="term-card" style="margin-bottom: 14px; border-left: 3px solid {colors['accent_cyan']};">
                <div style="font-size: 0.74rem; font-weight: 700; color: {colors['accent_cyan']}; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">
                    EXECUTIVE DISPATCH BRIEFING
                </div>
                <div style="font-size: 0.90rem; color: {colors['text_primary']}; line-height: 1.55;">
                    {exec_brief.get('headline_overview', 'No summary generated.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_cat, col_risk = st.columns(2)
        with col_cat:
            st.markdown(
                f"""
                <div class="term-card" style="height: 100%;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: {colors['bullish']}; margin-bottom: 8px; text-transform: uppercase;">
                        ▲ KEY POSITIVE CATALYSTS
                    </div>
                """,
                unsafe_allow_html=True,
            )
            for cat in exec_brief.get("key_catalysts", []):
                st.markdown(f"- {cat}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_risk:
            st.markdown(
                f"""
                <div class="term-card" style="height: 100%;">
                    <div style="font-size: 0.80rem; font-weight: 700; color: {colors['bearish']}; margin-bottom: 8px; text-transform: uppercase;">
                        ▼ IDENTIFIED RISKS & HEADWINDS
                    </div>
                """,
                unsafe_allow_html=True,
            )
            for r in exec_brief.get("identified_risks", []):
                st.markdown(f"- {r}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: {colors['text_secondary']};">
                <strong style="color: {colors['text_primary']};">Sentiment Analytical Rationale:</strong> {exec_brief.get('sentiment_rationale', '')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_tester:
        st.markdown("#### Live Financial Text Sentiment Tester")
        st.caption("Test the institutional Loughran-McDonald & FinBERT sentiment analyzer on custom text, earnings call transcripts, or breaking headlines.")

        sample_text = st.text_area(
            "Enter Financial Headline or Filing Excerpt:",
            value="Tata Consultancy Services bags $450M multi-year digital transformation deal with European logistics consortium, boosting operating margins.",
            height=90,
        )

        if st.button("⚡ Analyze Financial Sentiment", use_container_width=True):
            with st.spinner("Analyzing text via domain lexicons and FinBERT pipeline..."):
                res = api_client.analyze_sentiment(sample_text)

            res_label = res.get("label", "NEUTRAL").upper()
            res_score = float(res.get("score", 0.0))
            res_conf = float(res.get("confidence", 0.5))
            res_impact = float(res.get("impact_score", 50.0))
            res_phrases = res.get("key_phrases", [])

            t1, t2, t3, t4 = st.columns(4)
            with t1:
                render_kpi_card(
                    "CLASSIFICATION",
                    res_label,
                    accent="bullish" if res_label in ["POSITIVE", "BULLISH"] else ("bearish" if res_label in ["NEGATIVE", "BEARISH"] else "neutral"),
                )
            with t2:
                render_kpi_card("POLARITY SCORE", f"{res_score:+.2f}", accent="cyan")
            with t3:
                render_kpi_card("CONFIDENCE", f"{res_conf * 100.0:.1f}%", accent="purple")
            with t4:
                render_kpi_card("IMPACT SCORE", f"{res_impact:.1f}/100", accent="warning")

            if res_phrases:
                st.markdown(
                    f"""
                    <div style="margin-top: 10px; font-size: 0.82rem; color: {colors['text_secondary']};">
                        <strong>Detected Domain Catalysts / Idioms:</strong> <code>{', '.join(res_phrases)}</code>
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

    render_news_sentiment(api_client=get_api_client())
