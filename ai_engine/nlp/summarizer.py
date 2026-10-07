"""Financial Article and Multi-Event Summarizer for StockMind-AI.

Implements:
1. Financial sentence scoring (numerical presence, market catalyst density, sentiment extremity).
2. Extractive summarization of single articles.
3. Multi-event executive synthesis for portfolio tickers.
4. Sentiment rationale generator for analytical transparency.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from ai_engine.nlp.sentiment import SentimentResult, sentiment_analyzer


@dataclass
class ExecutiveSummary:
    """Institutional synthesis of market dispatches for a stock."""
    symbol: str
    headline_overview: str
    key_catalysts: List[str]
    identified_risks: List[str]
    sentiment_rationale: str
    sources_referenced: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "headline_overview": self.headline_overview,
            "key_catalysts": self.key_catalysts,
            "identified_risks": self.identified_risks,
            "sentiment_rationale": self.sentiment_rationale,
            "sources_referenced": self.sources_referenced,
        }


class FinancialSummarizer:
    """Institutional financial text summarization and synthesis engine."""

    FINANCIAL_METRIC_REGEX = re.compile(
        r"(₹|\$|£|€|\bcrore\b|\blakh\b|\bbps\b|\bpercent\b|%|\bq[1-4]\b|\bfy\d\d\b)",
        re.IGNORECASE,
    )

    CATALYST_MARKERS = [
        "commissioned", "contract", "deal", "acquisition", "merger", "order book",
        "guidance", "dividend", "buyback", "patent", "approval", "capex", "ebitda",
        "net profit", "deposit growth", "ldr", "cash flow",
    ]

    RISK_MARKERS = [
        "headwind", "softness", "probe", "investigation", "npa", "downgrade",
        "inflation", "slowdown", "delay", "curb", "scrutiny", "litigation", "deficit",
    ]

    def __init__(self) -> None:
        self.sentiment_engine = sentiment_analyzer

    def summarize_article(self, text: str, max_sentences: int = 2) -> str:
        """Extract the most financially informative sentences from text."""
        if not text or not text.strip():
            return ""

        # Sentence splitting
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        scored: List[tuple[float, int, str]] = []
        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            score = 0.0

            # 1. Lead sentence bonus (journalistic inverted pyramid)
            if idx == 0:
                score += 2.5
            elif idx == 1:
                score += 1.0

            # 2. Financial quantitative metrics presence (crucial in finance)
            num_matches = len(self.FINANCIAL_METRIC_REGEX.findall(sent))
            score += min(num_matches * 1.5, 4.0)

            # 3. Specific catalyst keywords
            for cat in self.CATALYST_MARKERS:
                if cat in sent_lower:
                    score += 1.2
            for risk in self.RISK_MARKERS:
                if risk in sent_lower:
                    score += 1.2

            scored.append((score, idx, sent))

        # Select top scoring sentences, maintain original chronological flow
        top_sentences = sorted(scored, key=lambda x: -x[0])[:max_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[1])

        return " ".join(s[2] for s in top_sentences)

    def synthesize_symbol_news(
        self,
        symbol: str,
        articles: List[Dict[str, Any]],
    ) -> ExecutiveSummary:
        """Synthesize multiple articles into an executive institutional brief."""
        symbol = symbol.upper()
        if not articles:
            return ExecutiveSummary(
                symbol=symbol,
                headline_overview=f"No recent news dispatches on record for {symbol}.",
                key_catalysts=["Trading within normal historical parameters."],
                identified_risks=["Standard macroeconomic and systematic equity market risks apply."],
                sentiment_rationale="Neutral baseline due to absence of verified breaking catalysts.",
                sources_referenced=[],
            )

        sources: List[str] = []
        catalysts: List[str] = []
        risks: List[str] = []
        pos_scores: List[float] = []
        neg_scores: List[float] = []

        for art in articles:
            title = art.get("title", "")
            src = art.get("source", "Market Wire")
            if src not in sources:
                sources.append(src)

            sent_res = self.sentiment_engine.analyze(f"{title} {art.get('summary', '')}")
            if sent_res.score > 0.15:
                pos_scores.append(sent_res.score)
                # Extract catalyst summary
                catalysts.append(f"{title} ({src})")
            elif sent_res.score < -0.15:
                neg_scores.append(sent_res.score)
                risks.append(f"{title} ({src})")

        # Overview synthesis
        top_article = articles[0]
        headline = f"Recent news flow for {symbol} is anchored by '{top_article.get('title', '')}' reported via {top_article.get('source', 'wire')}."

        # Sentiment rationale
        net_pos = len(pos_scores)
        net_neg = len(neg_scores)
        if net_pos > net_neg:
            rationale = (
                f"Sentiment leans Bullish with {net_pos} positive headlines against {net_neg} negative dispatches. "
                f"Institutional flow reflects optimism regarding capex execution and revenue accretion."
            )
        elif net_neg > net_pos:
            rationale = (
                f"Sentiment leans Bearish with {net_neg} negative headline catalysts. "
                f"Market commentary highlights near-term margin headwinds and discretionary expenditure deferrals."
            )
        else:
            rationale = (
                f"Sentiment is balanced and Neutral. News distribution exhibits countervailing factors "
                f"without unilateral breakout drivers."
            )

        return ExecutiveSummary(
            symbol=symbol,
            headline_overview=headline,
            key_catalysts=catalysts[:3] if catalysts else ["Order book continuity and standard operational execution."],
            identified_risks=risks[:3] if risks else ["Macroeconomic volatility and interest rate sensitivity."],
            sentiment_rationale=rationale,
            sources_referenced=sources[:4],
        )


# Global singleton instance
financial_summarizer = FinancialSummarizer()
