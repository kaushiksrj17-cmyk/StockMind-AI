"""Risk Intelligence API Routes."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
import numpy as np
import pandas as pd

from ai_engine.risk.risk_engine import risk_engine
from backend.app.schemas.risk_anomaly_portfolio import (
    ConcentrationRiskRequest,
    ConcentrationRiskResponse,
    RiskMetricsResponse,
    StressTestResponse,
    StressTestScenarioResult,
)
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/risk", tags=["Risk Intelligence"])


async def _get_ohlcv_df(symbol: str, limit: int = 250) -> pd.DataFrame:
    """Helper to fetch historical bars for an instrument."""
    bars = await market_service.get_historical_ohlc(symbol=symbol, timeframe="1d", limit=limit)
    if not bars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical data unavailable for symbol '{symbol}'.",
        )
    records = [
        b.model_dump() if hasattr(b, "model_dump") else (b.dict() if hasattr(b, "dict") else b.__dict__)
        for b in bars
    ]
    return pd.DataFrame(records)


@router.get(
    "/{symbol}",
    response_model=RiskMetricsResponse,
    summary="Comprehensive Risk Evaluation",
)
async def get_risk_metrics(
    symbol: str,
    limit: int = Query(default=250, ge=30, le=1000),
):
    """Compute institutional volatility, Max Drawdown, Sharpe, Sortino, Beta, VaR, and Expected Shortfall."""
    df = await _get_ohlcv_df(symbol, limit=limit)
    close_prices = df["close"].astype(float).values

    # Benchmark series (NIFTY 50)
    bench_prices = None
    try:
        bench_df = await _get_ohlcv_df("NIFTY 50", limit=limit)
        bench_prices = bench_df["close"].astype(float).values
    except Exception:
        bench_prices = None

    metrics = risk_engine.analyze_asset(
        prices=close_prices,
        symbol=symbol,
        benchmark_prices=bench_prices,
    )
    return metrics.to_dict()


@router.post(
    "/concentration",
    response_model=ConcentrationRiskResponse,
    summary="Concentration Risk Assessment",
)
async def compute_concentration(request: ConcentrationRiskRequest):
    """Compute Herfindahl-Hirschman Index (HHI) and concentration percentiles for asset weights."""
    res = risk_engine.compute_concentration_risk(request.weights)
    return res


@router.get(
    "/{symbol}/stress-test",
    response_model=StressTestResponse,
    summary="Macro Scenario Stress Testing",
)
async def get_stress_test(
    symbol: str,
):
    """Simulate asset drawdown under historical macroeconomic crisis scenarios."""
    df = await _get_ohlcv_df(symbol, limit=200)
    close_prices = df["close"].astype(float).values
    bench_prices = None
    try:
        bench_df = await _get_ohlcv_df("NIFTY 50", limit=200)
        bench_prices = bench_df["close"].astype(float).values
    except Exception:
        bench_prices = None

    metrics = risk_engine.analyze_asset(prices=close_prices, symbol=symbol, benchmark_prices=bench_prices)
    beta = metrics.beta
    vol = metrics.annualized_volatility_pct

    # Predefined institutional macro stress scenarios
    historical_scenarios = [
        {
            "scenario_name": "Global Financial Crisis Shock (2008)",
            "description": "Systemic credit crisis with severe equity liquidation.",
            "market_drop_pct": -52.0,
            "severity": "CRITICAL",
        },
        {
            "scenario_name": "Pandemic Flash Liquidation (March 2020)",
            "description": "Rapid global lockdown panic and circuit-breaker trigger.",
            "market_drop_pct": -38.4,
            "severity": "CRITICAL",
        },
        {
            "scenario_name": "Aggressive Central Bank Rate Shock",
            "description": "250 bps unexpected policy tightening cycle and liquidity crunch.",
            "market_drop_pct": -16.5,
            "severity": "HIGH",
        },
        {
            "scenario_name": "Emerging Market Currency Contagion",
            "description": "Sharp FII capital outflow and currency depreciation.",
            "market_drop_pct": -12.0,
            "severity": "MEDIUM",
        },
    ]

    results: List[StressTestScenarioResult] = []
    for s in historical_scenarios:
        m_drop = s["market_drop_pct"]
        # Projected impact = Market Drop * Beta * Non-linear tail factor
        tail_factor = 1.0 + (vol / 100.0) * 0.25
        asset_impact = m_drop * beta * tail_factor
        loss_100k = abs(asset_impact / 100.0) * 100000.0

        results.append(
            StressTestScenarioResult(
                scenario_name=s["scenario_name"],
                description=s["description"],
                simulated_market_drop_pct=round(m_drop, 2),
                projected_asset_impact_pct=round(asset_impact, 2),
                estimated_loss_per_100k=round(loss_100k, 2),
                severity=s["severity"],
            )
        )

    return StressTestResponse(
        symbol=symbol.upper(),
        beta=round(beta, 2),
        annual_volatility_pct=round(vol, 2),
        scenarios=results,
    )
