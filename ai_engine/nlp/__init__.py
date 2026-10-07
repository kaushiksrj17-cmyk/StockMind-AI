"""StockMind-AI Financial NLP & Market Sentiment System.

Exposes:
- FinancialSentimentAnalyzer, SentimentResult, SentimentLabel
- NewsProcessor, NewsArticle, FinancialNewsProvider
- FinancialSummarizer, ExecutiveSummary
"""

from ai_engine.nlp.sentiment import (
    FinancialLexiconSentimentModel,
    FinancialSentimentAnalyzer,
    FinBERTSentimentModel,
    SentimentLabel,
    SentimentResult,
    sentiment_analyzer,
)
from ai_engine.nlp.news_processor import (
    FinancialNewsProvider,
    InstitutionalGroundedNewsProvider,
    NewsArticle,
    NewsDeduplicator,
    NewsProcessor,
    TickerEntityMatcher,
    TimestampNormalizer,
    news_processor,
)
from ai_engine.nlp.summarizer import (
    ExecutiveSummary,
    FinancialSummarizer,
    financial_summarizer,
)

__all__ = [
    "SentimentLabel",
    "SentimentResult",
    "FinancialLexiconSentimentModel",
    "FinBERTSentimentModel",
    "FinancialSentimentAnalyzer",
    "sentiment_analyzer",
    "NewsArticle",
    "FinancialNewsProvider",
    "InstitutionalGroundedNewsProvider",
    "TickerEntityMatcher",
    "TimestampNormalizer",
    "NewsDeduplicator",
    "NewsProcessor",
    "news_processor",
    "ExecutiveSummary",
    "FinancialSummarizer",
    "financial_summarizer",
]
