"""Pydantic schemas for Financial NLP and Sentiment API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SentimentAnalyzeRequest(BaseModel):
    """Ad-hoc text sentiment analysis request."""
    text: str = Field(..., min_length=2, max_length=5000, description="Financial text or headline to analyze")


class SentimentResponse(BaseModel):
    """Sentiment classification payload."""
    label: str
    direction: str
    score: float
    confidence: float
    confidence_pct: float
    impact_score: float
    probabilities: Dict[str, float]
    model_name: str
    key_phrases: List[str] = Field(default_factory=list)


class NewsArticleResponse(BaseModel):
    """Processed news article with sentiment and impact."""
    article_id: str
    title: str
    summary: str
    source: str
    source_tier: int
    published_at: str
    time_ago: str
    symbol: str
    matched_symbols: List[str]
    url: Optional[str] = None
    sentiment: Optional[SentimentResponse] = None
    impact_score: float
    is_duplicate: bool
    duplicate_of_id: Optional[str] = None


class NewsFeedResponse(BaseModel):
    """List of news articles with summary metrics."""
    symbol: str
    total_articles: int
    articles: List[NewsArticleResponse]
    sentiment_summary: Dict[str, Any]
    sources_referenced: List[str]


class ExecutiveSummaryResponse(BaseModel):
    """Executive multi-event brief."""
    symbol: str
    headline_overview: str
    key_catalysts: List[str]
    identified_risks: List[str]
    sentiment_rationale: str
    sources_referenced: List[str]
