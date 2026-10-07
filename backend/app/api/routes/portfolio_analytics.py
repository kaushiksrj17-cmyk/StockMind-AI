"""Institutional Portfolio Analytics and MPT Optimization API Routes."""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status

from ai_engine.portfolio.optimizer import mpt_optimizer
from ai_engine.portfolio.portfolio_engine import portfolio_engine
from backend.app.schemas.risk_anomaly_portfolio import (
    MPTOptimizationRequest,
    MPTOptimizationResponse,
    PortfolioEvaluationRequest,
    PortfolioEvaluationResponse,
)
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/portfolio-analytics", tags=["Portfolio Analytics & Optimization"])


@router.post(
    "/evaluate",
    response_model=PortfolioEvaluationResponse,
    summary="Evaluate Portfolio Holdings, P&L, and Risk",
)
async def evaluate_portfolio(request: PortfolioEvaluationRequest):
    """Evaluate multi-asset holdings, current market value, sector exposure, HHI, and risk attribution."""
    holdings_data = [h.model_dump() for h in request.holdings]
    if not holdings_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one portfolio holding must be provided.",
        )

    # Fetch latest quotes for all symbols to update current prices
    symbols = list(set([h["symbol"].upper() for h in holdings_data]))
    asset_prices: Dict[str, float] = {}
    for sym in symbols:
        try:
            quote = await market_service.get_quote(sym)
            if quote and getattr(quote, "last_price", 0.0) > 0:
                asset_prices[sym] = float(quote.last_price)
        except Exception:
            pass

    summary = portfolio_engine.evaluate_portfolio(
        raw_holdings=holdings_data,
        realized_pnl=request.realized_pnl,
        asset_prices=asset_prices,
    )
    return summary.to_dict()


@router.post(
    "/optimize-mpt",
    response_model=MPTOptimizationResponse,
    summary="Analytical Modern Portfolio Theory Optimization",
)
async def optimize_portfolio_mpt(request: MPTOptimizationRequest):
    """Run Markowitz Mean-Variance Optimization for Tangency (Max Sharpe) and Min Volatility portfolios.
    
    Contains explicit compliance disclaimer indicating simulation status.
    """
    if len(request.symbols) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 unique symbols are required for portfolio optimization.",
        )

    result = mpt_optimizer.optimize_portfolio(
        symbols=request.symbols,
        current_weights=request.current_weights,
        max_asset_weight=request.max_asset_weight,
    )
    return result.to_dict()
