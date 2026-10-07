"""Pydantic Schemas for Technical Analysis Engine."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CandlestickPatternItem(BaseModel):
    name: str = Field(..., description="Pattern identifier, e.g. Hammer, Bullish Engulfing")
    bias: str = Field(..., description="BULLISH, BEARISH, or NEUTRAL")
    significance: str = Field(..., description="MODERATE, HIGH, or VERY_HIGH")
    bars_ago: int = Field(..., description="How many bars ago this pattern occurred")
    description: str = Field(..., description="Detailed technical pattern explanation")


class AITechnicalScoreSummary(BaseModel):
    score: float = Field(..., description="Normalized AI Technical Score between 0.0 and 100.0")
    classification: str = Field(..., description="STRONG_BULLISH, BULLISH, NEUTRAL, BEARISH, or STRONG_BEARISH")
    summary_label: str = Field(..., description="Human-readable summary label")
    trend_direction: str = Field(..., description="UPTREND, DOWNTREND, or SIDEWAYS")
    trend_strength: str = Field(..., description="VERY_STRONG, STRONG, MODERATE, or WEAK")
    component_scores: Dict[str, float] = Field(..., description="Sub-scores across 5 technical pillars")
    component_weights: Dict[str, float] = Field(..., description="Relative weights used in final composite calculation")
    indicator_explanations: Dict[str, str] = Field(..., description="Plain-English summaries for each indicator group")
    score_explanation: str = Field(..., description="Detailed explanation of the technical score drivers")
    disclaimer: str = Field(..., description="Mandatory statistical and regulatory disclaimer")


class MovingAveragesSummary(BaseModel):
    values: Dict[str, float] = Field(..., description="SMA and EMA values (20, 50, 100, 200, etc.)")
    price_vs_sma20_pct: float
    price_vs_sma50_pct: float
    price_vs_sma200_pct: float
    golden_cross: bool = Field(..., description="Golden Cross active (50-SMA > 200-SMA)")
    death_cross: bool = Field(..., description="Death Cross active (50-SMA < 200-SMA)")
    golden_cross_50_200: Optional[bool] = None
    death_cross_50_200: Optional[bool] = None
    tactical_cross_20_50: Optional[bool] = None
    classical_50_200_status: str
    tactical_20_50_status: str
    is_bullish_stack: bool
    is_bearish_stack: bool
    bias_score: int


class SupportResistanceSummary(BaseModel):
    current_price: float
    support_levels: List[float]
    resistance_levels: List[float]
    nearest_support: float
    nearest_resistance: float
    pivot_points: Dict[str, float]


class FibonacciSummary(BaseModel):
    swing_high: float
    swing_low: float
    current_price: float
    levels: Dict[str, float]
    nearest_level: str
    nearest_level_price: float


class TechnicalAnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    as_of: str
    current_price: float
    price_change: float
    price_change_pct: float
    ai_technical_score: AITechnicalScoreSummary
    moving_averages: Dict[str, Any]
    oscillators: Dict[str, Any]
    volatility: Dict[str, Any]
    trend: Dict[str, Any]
    volume_profile: Dict[str, Any]
    support_resistance: SupportResistanceSummary
    fibonacci_levels: FibonacciSummary
    candlestick_patterns: List[CandlestickPatternItem]
    chart_data: Dict[str, List[Any]]
    disclaimer: str


class TechnicalScoreOnlyResponse(BaseModel):
    symbol: str
    current_price: float
    ai_technical_score: AITechnicalScoreSummary
    disclaimer: str


class TechnicalIndicatorsOnlyResponse(BaseModel):
    symbol: str
    current_price: float
    moving_averages: Dict[str, Any]
    oscillators: Dict[str, Any]
    volatility: Dict[str, Any]
    trend: Dict[str, Any]
    volume_profile: Dict[str, Any]
    disclaimer: str
