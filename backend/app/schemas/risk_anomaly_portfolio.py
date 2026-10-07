"""Pydantic Schemas for Risk, Anomaly, Regime, Portfolio MPT, and Backtesting APIs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 1. RISK ENGINE SCHEMAS
# ============================================================================

class DrawdownMetricsSchema(BaseModel):
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    current_drawdown_pct: float
    peak_value: float
    trough_value: float
    underwater_series: List[float] = Field(default_factory=list)


class RiskMetricsResponse(BaseModel):
    symbol: str
    annualized_volatility_pct: float
    rolling_20d_volatility_pct: float
    max_drawdown_pct: float
    drawdown_metrics: DrawdownMetricsSchema
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    alpha_annualized_pct: float
    var_95_1d_pct: float
    var_99_1d_pct: float
    var_95_historical_pct: float
    var_95_monte_carlo_pct: float
    expected_shortfall_95_pct: float
    downside_deviation_pct: float
    semi_variance: float
    risk_level: str


class ConcentrationRiskRequest(BaseModel):
    weights: List[float] = Field(..., description="List of portfolio weights")


class ConcentrationRiskResponse(BaseModel):
    hhi: float
    top_1_holding_pct: float
    top_3_holdings_pct: float
    effective_constituents: float
    concentration_tier: str


class StressTestScenarioResult(BaseModel):
    scenario_name: str
    description: str
    simulated_market_drop_pct: float
    projected_asset_impact_pct: float
    estimated_loss_per_100k: float
    severity: str


class StressTestResponse(BaseModel):
    symbol: str
    beta: float
    annual_volatility_pct: float
    scenarios: List[StressTestScenarioResult]


# ============================================================================
# 2. ANOMALY ENGINE SCHEMAS
# ============================================================================

class AnomalyItemSchema(BaseModel):
    index: int
    timestamp: str
    anomaly_type: str
    severity: str
    score: float
    description: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class AnomalyResponse(BaseModel):
    symbol: str
    has_anomalies: bool
    total_detected: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    is_current_bar_anomalous: bool
    current_anomaly_score: float
    current_severity: str
    primary_explanation: str
    divergence_status: str
    anomalies: List[AnomalyItemSchema] = Field(default_factory=list)


# ============================================================================
# 3. MARKET REGIME SCHEMAS
# ============================================================================

class RegimeResponse(BaseModel):
    symbol: str
    primary_regime: str
    confidence_pct: float
    regime_probabilities: Dict[str, float]
    trend_score: float
    volatility_score: float
    trend_strength_adx: float
    stability_score: float
    bars_in_regime: int
    recommended_stance: str
    institutional_explanation: str
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 4. PORTFOLIO & MPT SCHEMAS
# ============================================================================

class HoldingInput(BaseModel):
    symbol: str
    shares: int = Field(..., gt=0)
    avg_price: float = Field(..., gt=0)
    sector: Optional[str] = None
    name: Optional[str] = None


class HoldingItemSchema(BaseModel):
    symbol: str
    name: str
    shares: int
    avg_price: float
    current_price: float
    sector: str
    cost_basis: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_pct: float


class SectorExposureSchema(BaseModel):
    sector: str
    value: float
    weight_pct: float
    holding_count: int


class PortfolioEvaluationRequest(BaseModel):
    holdings: List[HoldingInput]
    realized_pnl: float = 0.0


class PortfolioEvaluationResponse(BaseModel):
    total_value: float
    total_cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    holding_count: int
    holdings: List[HoldingItemSchema]
    sector_exposures: List[SectorExposureSchema]
    hhi: float
    concentration_tier: str
    top_1_weight_pct: float
    top_3_weight_pct: float
    effective_constituents: float
    diversification_ratio: float
    portfolio_volatility_pct: float
    portfolio_beta: float
    portfolio_var_95_1d_pct: float
    portfolio_sharpe_ratio: float
    risk_level: str


class MPTOptimizationRequest(BaseModel):
    symbols: List[str] = Field(default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"])
    current_weights: Optional[Dict[str, float]] = None
    max_asset_weight: float = Field(default=0.45, ge=0.15, le=1.0)


class OptimizedPortfolioPointSchema(BaseModel):
    label: str
    weights: Dict[str, float]
    expected_return_pct: float
    volatility_pct: float
    sharpe_ratio: float


class FrontierPointSchema(BaseModel):
    target_return_pct: float
    volatility_pct: float
    sharpe_ratio: float
    weights: Dict[str, float]


class MPTOptimizationResponse(BaseModel):
    symbols: List[str]
    max_sharpe_portfolio: OptimizedPortfolioPointSchema
    min_volatility_portfolio: OptimizedPortfolioPointSchema
    current_portfolio: Optional[OptimizedPortfolioPointSchema] = None
    efficient_frontier: List[FrontierPointSchema]
    simulated_cloud_sample: List[Dict[str, float]]
    risk_free_rate_pct: float
    compliance_disclaimer: str


# ============================================================================
# 5. BACKTESTING SCHEMAS
# ============================================================================

class BacktestRequest(BaseModel):
    symbol: str = "RELIANCE"
    strategy: str = Field(default="ma_cross", description="ma_cross, rsi, bollinger, macd, multi_factor")
    initial_capital: float = Field(default=500000.0, ge=10000.0)
    slippage_pct: float = Field(default=0.0005, ge=0.0, le=0.02)
    transaction_fee_pct: float = Field(default=0.0003, ge=0.0, le=0.02)
    timeframe: str = "1d"
    limit: int = Field(default=250, ge=50, le=1000)


class TradeRecordSchema(BaseModel):
    trade_id: int
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    shares: int
    gross_pnl: float
    net_pnl: float
    net_return_pct: float
    fees_paid: float
    duration_bars: int
    exit_reason: str


class BacktestResponse(BaseModel):
    strategy_name: str
    symbol: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    payoff_ratio: float
    total_fees_paid: float
    equity_curve: List[Dict[str, Any]]
    drawdown_curve: List[Dict[str, Any]]
    trades: List[TradeRecordSchema]
    compliance_disclaimer: str
