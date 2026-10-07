"""Financial NLP and Sentiment API routes."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from ai_engine.nlp.sentiment import sentiment_analyzer
from ai_engine.nlp.news_processor import news_processor
from ai_engine.nlp.summarizer import financial_summarizer
from backend.app.schemas.nlp import (
    ExecutiveSummaryResponse,
    NewsArticleResponse,
    NewsFeedResponse,
    SentimentAnalyzeRequest,
    SentimentResponse,
)

router = APIRouter(prefix="/nlp", tags=["Financial NLP & Sentiment"])


@router.post(
    "/sentiment/analyze",
    response_model=SentimentResponse,
    summary="Analyze financial text sentiment",
    description="Classifies sentiment using FinBERT or the institutional financial lexicon.",
)
async def analyze_sentiment(payload: SentimentAnalyzeRequest) -> SentimentResponse:
    """Analyze ad-hoc financial text or headline."""
    try:
        res = sentiment_analyzer.analyze(payload.text)
        return SentimentResponse(**res.to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@router.get(
    "/news/{symbol}",
    response_model=NewsFeedResponse,
    summary="Get processed financial news feed for a symbol",
    description="Retrieves deduplicated, sentiment-scored news articles with impact metrics.",
)
async def get_symbol_news(
    symbol: str,
    limit: int = Query(10, ge=1, le=50, description="Max articles to retrieve"),
    include_duplicates: bool = Query(False, description="Whether to include syndication duplicates"),
) -> NewsFeedResponse:
    """Retrieve processed news articles for a specific ticker symbol."""
    sym = symbol.strip().upper()
    try:
        articles = news_processor.get_processed_news(
            symbol=sym,
            limit=limit,
            exclude_duplicates=not include_duplicates,
        )
        sentiment_summary = news_processor.get_market_sentiment_summary(symbol=sym)

        article_dicts = [a.to_dict() for a in articles]
        sources = list(dict.fromkeys(a.source for a in articles))

        return NewsFeedResponse(
            symbol=sym,
            total_articles=len(articles),
            articles=article_dicts,
            sentiment_summary=sentiment_summary,
            sources_referenced=sources,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch news feed for {sym}: {str(e)}",
        )


@router.get(
    "/sentiment/summary/{symbol}",
    summary="Get aggregated market sentiment summary",
    description="Returns aggregate polarity, sentiment regime, and top key phrase catalysts.",
)
async def get_sentiment_summary(symbol: str) -> Dict[str, Any]:
    """Retrieve macro sentiment summary for a symbol."""
    sym = symbol.strip().upper()
    try:
        return news_processor.get_market_sentiment_summary(symbol=sym)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate sentiment summary for {sym}: {str(e)}",
        )


@router.get(
    "/summary/{symbol}",
    response_model=ExecutiveSummaryResponse,
    summary="Get executive multi-event synthesis",
    description="Synthesizes recent news flow into an institutional executive brief.",
)
async def get_executive_summary(symbol: str) -> ExecutiveSummaryResponse:
    """Generate executive multi-event synthesis for a symbol."""
    sym = symbol.strip().upper()
    try:
        articles = news_processor.get_processed_news(symbol=sym, limit=8, exclude_duplicates=True)
        raw_list = [a.to_dict() for a in articles]
        summary = financial_summarizer.synthesize_symbol_news(symbol=sym, articles=raw_list)
        return ExecutiveSummaryResponse(**summary.to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate executive summary for {sym}: {str(e)}",
        )
