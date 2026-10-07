"""Pydantic schemas for AI Market Copilot API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CopilotChatRequest(BaseModel):
    """User prompt to the AI Market Copilot."""
    query: str = Field(..., min_length=2, max_length=1500, description="Natural language question or request")
    symbol: str = Field("RELIANCE", description="Ticker symbol to ground the analysis in (e.g. RELIANCE, TCS, INFY, NIFTY 50)")
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Prior conversation messages")


class EpistemicBreakdown(BaseModel):
    """Categorized factual and probabilistic evidence layers."""
    fact: List[str] = Field(description="Verified corporate facts, fundamentals, and exchange filings")
    current_market_data: List[str] = Field(description="Real-time quotes, OHLCV, and volume telemetry")
    model_prediction: List[str] = Field(description="Probabilistic ML and PyTorch neural network forecasts")
    inference: List[str] = Field(description="Analytical deductions from technical and sentiment indicators")
    uncertainty: List[str] = Field(description="Risk boundaries, variance, and non-guarantee notices")


class CitationItem(BaseModel):
    """Source reference for news and verified dispatches."""
    source: str
    title: str
    time: str
    url: str
    impact: str


class CopilotChatResponse(BaseModel):
    """Structured response from the AI Market Copilot."""
    query: str
    symbol: str
    reply_markdown: str
    epistemic_breakdown: EpistemicBreakdown
    citations: List[CitationItem]
    telemetry_summary: Dict[str, Any]
    disclaimer: str
