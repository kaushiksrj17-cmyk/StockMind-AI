"""Unit tests for the Financial NLP Engine and AI Market Copilot in StockMind-AI.

Tests:
1. Sentiment Analysis: Positive, Negative, Neutral polarity, negation handling, impact scores.
2. Ticker Entity Matcher: Extraction of NSE equities and indices.
3. Timestamp Normalizer: ISO, Unix epoch, and relative time expressions.
4. News Deduplicator: Exact and near-duplicate re-syndication detection.
5. News Processor: Grounded article ingestion, impact score, and aggregate summary.
6. Financial Summarizer: Sentence ranking, multi-event executive synthesis.
7. AI Market Copilot: Multi-factor context integration, epistemic categorization (FACT,
   CURRENT MARKET DATA, MODEL PREDICTION, INFERENCE, UNCERTAINTY), non-guarantee invariant,
   and source references.
8. FastAPI Endpoints: REST routes for sentiment, news feed, summary, and Copilot chat.
"""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from ai_engine.copilot.copilot_engine import (
    AIMarketCopilot,
    STATUTORY_COPILOT_DISCLAIMER,
    ai_copilot,
)
from ai_engine.nlp.sentiment import (
    FinancialLexiconSentimentModel,
    FinancialSentimentAnalyzer,
    SentimentLabel,
    sentiment_analyzer,
)
from ai_engine.nlp.news_processor import (
    NewsArticle,
    NewsDeduplicator,
    NewsProcessor,
    TickerEntityMatcher,
    TimestampNormalizer,
    news_processor,
)
from ai_engine.nlp.summarizer import (
    FinancialSummarizer,
    financial_summarizer,
)
from backend.app.main import app


# ============================================================================
# 1. SENTIMENT ANALYSIS TESTS
# ============================================================================

def test_financial_sentiment_positive():
    """Verify bullish/positive financial statements yield positive polarity."""
    text = "Reliance Industries reports record net profit growth and commissioning of new green hydrogen capacity."
    res = sentiment_analyzer.analyze(text)
    assert res.label in [SentimentLabel.POSITIVE.value, SentimentLabel.BULLISH.value]
    assert res.score > 0.20
    assert res.positive_prob > res.negative_prob
    assert 50.0 <= res.impact_score <= 100.0


def test_financial_sentiment_negative():
    """Verify bearish/negative financial statements yield negative polarity."""
    text = "Company faces severe debt default risks, margin compression, and an active regulatory probe by SEBI."
    res = sentiment_analyzer.analyze(text)
    assert res.label in [SentimentLabel.NEGATIVE.value, SentimentLabel.BEARISH.value]
    assert res.score < -0.20
    assert res.negative_prob > res.positive_prob
    assert 50.0 <= res.impact_score <= 100.0


def test_financial_sentiment_neutral():
    """Verify neutral macroeconomic in-line dispatches yield near-zero polarity."""
    text = "RBI MPC maintains benchmark repo rate unchanged at 6.50% in-line with broader consensus expectations."
    res = sentiment_analyzer.analyze(text)
    assert res.label == SentimentLabel.NEUTRAL.value
    assert abs(res.score) <= 0.20
    assert res.neutral_prob >= 0.30


def test_financial_sentiment_negation_handling():
    """Verify negation alters polarity appropriately."""
    text_pos = "The company reported profit growth and expansion."
    text_negated = "The company reported no profit growth and failed to achieve expansion."

    res_pos = sentiment_analyzer.analyze(text_pos)
    res_negated = sentiment_analyzer.analyze(text_negated)

    assert res_pos.score > res_negated.score
    assert res_negated.score < 0.10


# ============================================================================
# 2. TICKER ENTITY MATCHER TESTS
# ============================================================================

def test_ticker_entity_matching():
    """Verify ticker extraction for key Indian equities."""
    matcher = TickerEntityMatcher()

    # 1. Reliance
    p1, m1 = matcher.resolve_primary_symbol("Reliance Industries announces mega capex for solar gigafactory.")
    assert p1 == "RELIANCE"
    assert "RELIANCE" in m1

    # 2. TCS
    p2, m2 = matcher.resolve_primary_symbol("Tata Consultancy Services bags multi-million cloud deal.")
    assert p2 == "TCS"
    assert "TCS" in m2

    # 3. HDFC Bank
    p3, m3 = matcher.resolve_primary_symbol("HDFC Bank expands branches across rural Maharashtra.")
    assert p3 == "HDFCBANK"

    # 4. Fallback to NIFTY 50 for broad macro
    p4, m4 = matcher.resolve_primary_symbol("Reserve Bank of India monetary policy meeting concludes today.")
    assert p4 == "NIFTY 50"


# ============================================================================
# 3. TIMESTAMP NORMALIZATION TESTS
# ============================================================================

def test_timestamp_normalizer():
    """Verify ISO, epoch, and relative string conversions."""
    now = datetime.now(timezone.utc)

    # Relative minutes
    dt_rel = TimestampNormalizer.parse("15 mins ago")
    assert abs((now - dt_rel).total_seconds() - 900) < 5

    # Relative hours
    dt_hr = TimestampNormalizer.parse("2 hours ago")
    assert abs((now - dt_hr).total_seconds() - 7200) < 5

    # ISO string
    iso_str = "2026-10-06T12:00:00Z"
    dt_iso = TimestampNormalizer.parse(iso_str)
    assert dt_iso.year == 2026
    assert dt_iso.month == 10
    assert dt_iso.day == 6


# ============================================================================
# 4. DEDUPLICATION TESTS
# ============================================================================

def test_news_deduplicator():
    """Verify exact and syndicated near-duplicate detection."""
    dedup = NewsDeduplicator(jaccard_threshold=0.65)

    art1 = NewsArticle(
        article_id="ART_001",
        title="Reliance Industries commissions green hydrogen electrolyzer in Gujarat",
        summary="Rollout proceeds on schedule with zero debt escalation.",
        source="Mint Financial",
        source_tier=1,
        published_at=datetime.now(timezone.utc),
        symbol="RELIANCE",
    )

    art2_exact = NewsArticle(
        article_id="ART_002",
        title="Reliance Industries commissions green hydrogen electrolyzer in Gujarat",
        summary="Different summary but exact title and source.",
        source="Mint Financial",
        source_tier=1,
        published_at=datetime.now(timezone.utc),
        symbol="RELIANCE",
    )

    art3_syndicated = NewsArticle(
        article_id="ART_003",
        title="Reliance Industries green hydrogen electrolyzer commissioned in Gujarat state",
        summary="Syndicated wire report.",
        source="Wire Desk Re-Syndication",
        source_tier=3,
        published_at=datetime.now(timezone.utc),
        symbol="RELIANCE",
    )

    is_dup1, _ = dedup.check_duplicate(art1)
    assert is_dup1 is False  # First article is novel

    # Same article queried again should not flag as duplicate of itself
    is_dup1_repeat, _ = dedup.check_duplicate(art1)
    assert is_dup1_repeat is False

    is_dup2, orig_id2 = dedup.check_duplicate(art2_exact)
    assert is_dup2 is True
    assert orig_id2 == "ART_001"

    is_dup3, orig_id3 = dedup.check_duplicate(art3_syndicated)
    assert is_dup3 is True
    assert orig_id3 == "ART_001"


# ============================================================================
# 5. NEWS PROCESSOR PIPELINE TESTS
# ============================================================================

def test_news_processor_pipeline():
    """Verify news processor retrieves and aggregates grounded market news."""
    articles = news_processor.get_processed_news(symbol="RELIANCE", limit=5)
    assert len(articles) > 0

    art = articles[0]
    assert art.symbol == "RELIANCE"
    assert art.sentiment is not None
    assert 0.0 <= art.impact_score <= 100.0
    assert art.time_ago_str() is not None

    summary = news_processor.get_market_sentiment_summary(symbol="RELIANCE")
    assert summary["symbol"] == "RELIANCE"
    assert summary["sentiment_regime"] in ["BULLISH", "NEUTRAL", "BEARISH"]
    assert "total_articles" in summary


# ============================================================================
# 6. SUMMARIZER TESTS
# ============================================================================

def test_financial_summarizer_extractive():
    """Verify extractive article summarization ranks numerical and catalyst sentences."""
    article_text = (
        "Reliance Industries announced commissioning of its new electrolyzer today. "
        "The project incurred capex of ₹4,500 crore and is expected to drive 18% EBITDA margin accretion by FY27. "
        "The weather in Ahmedabad was partly cloudy throughout the afternoon session. "
        "Management reaffirmed its target to achieve net carbon zero operations."
    )
    summary = financial_summarizer.summarize_article(article_text, max_sentences=2)
    assert "₹4,500 crore" in summary or "18% EBITDA" in summary
    assert "partly cloudy" not in summary  # Non-financial sentence filtered out


def test_financial_summarizer_synthesis():
    """Verify multi-event executive synthesis creates structured brief."""
    raw_articles = [
        {
            "title": "Reliance Industries commissions green hydrogen electrolyzer in Gujarat",
            "source": "Mint Financial",
            "summary": "Capex rollout proceeds ahead of schedule.",
        },
        {
            "title": "Crude oil volatility may present slight near-term margin headwind",
            "source": "Reuters",
            "summary": "Refining margins fluctuate with Brent swings.",
        },
    ]

    exec_brief = financial_summarizer.synthesize_symbol_news(symbol="RELIANCE", articles=raw_articles)
    assert exec_brief.symbol == "RELIANCE"
    assert len(exec_brief.key_catalysts) > 0
    assert len(exec_brief.identified_risks) > 0
    assert "Mint Financial" in exec_brief.sources_referenced


# ============================================================================
# 7. AI MARKET COPILOT ENGINE TESTS
# ============================================================================

def test_ai_market_copilot_epistemic_taxonomy():
    """Verify AI Market Copilot strictly outputs all 5 required epistemic categories."""
    copilot = AIMarketCopilot()
    resp = copilot.answer_query(
        query="What is the outlook, forecast, and risk profile for RELIANCE?",
        symbol="RELIANCE",
    )

    # 1. Epistemic Buckets Check
    assert len(resp.fact) >= 2
    assert len(resp.current_market_data) >= 2
    assert len(resp.model_prediction) >= 2
    assert len(resp.inference) >= 2
    assert len(resp.uncertainty) >= 2

    # 2. Strict Content Invariants
    # Fact should mention listing or exchange fundamentals
    assert any("listed" in f.lower() or "mega-cap" in f.lower() or "sector" in f.lower() for f in resp.fact)

    # Current market data should mention price or volume
    assert any("price" in cmd.lower() or "volume" in cmd.lower() for cmd in resp.current_market_data)

    # Model prediction should include consensus direction or target
    assert any("consensus" in mp.lower() or "target" in mp.lower() for mp in resp.model_prediction)

    # Uncertainty must include Value-at-Risk or volatility and non-guarantee notice
    assert any("var" in u.lower() or "volatility" in u.lower() for u in resp.uncertainty)
    assert any("guarantee" in u.lower() for u in resp.uncertainty)

    # 3. Never claim guaranteed returns invariant
    assert "guaranteed returns" not in resp.reply_markdown.lower() or "never" in resp.reply_markdown.lower() or "not" in resp.reply_markdown.lower()
    assert "STATUTORY FINANCIAL NOTICE" in resp.disclaimer

    # 4. Citations & References
    assert isinstance(resp.citations, list)
    if resp.citations:
        assert "source" in resp.citations[0]
        assert "title" in resp.citations[0]


# ============================================================================
# 8. REST API ENDPOINT TESTS
# ============================================================================

def test_nlp_sentiment_analyze_api():
    """Test POST /api/v1/nlp/sentiment/analyze."""
    client = TestClient(app)
    resp = client.post(
        "/api/v1/nlp/sentiment/analyze",
        json={"text": "TCS bags $450M multi-year digital transformation deal with European logistics consortium."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ["POSITIVE", "BULLISH"]
    assert data["score"] > 0
    assert "probabilities" in data


def test_nlp_news_feed_api():
    """Test GET /api/v1/nlp/news/{symbol}."""
    client = TestClient(app)
    resp = client.get("/api/v1/nlp/news/RELIANCE?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert "articles" in data
    assert len(data["articles"]) > 0
    assert "sentiment_summary" in data


def test_nlp_executive_summary_api():
    """Test GET /api/v1/nlp/summary/{symbol}."""
    client = TestClient(app)
    resp = client.get("/api/v1/nlp/summary/RELIANCE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert "key_catalysts" in data
    assert "identified_risks" in data


def test_copilot_chat_api():
    """Test POST /api/v1/copilot/chat."""
    client = TestClient(app)
    payload = {
        "query": "What is the 5-day statistical forecast and downside risk for RELIANCE?",
        "symbol": "RELIANCE",
        "history": [],
    }
    resp = client.post("/api/v1/copilot/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert "reply_markdown" in data
    assert "epistemic_breakdown" in data
    eb = data["epistemic_breakdown"]
    assert "fact" in eb
    assert "current_market_data" in eb
    assert "model_prediction" in eb
    assert "inference" in eb
    assert "uncertainty" in eb
    assert "disclaimer" in data


def test_copilot_context_api():
    """Test GET /api/v1/copilot/context/{symbol}."""
    client = TestClient(app)
    resp = client.get("/api/v1/copilot/context/RELIANCE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert "current_market_data" in data
    assert "technicals" in data
    assert "predictions" in data
    assert "risk_metrics" in data
    assert "fundamentals" in data
