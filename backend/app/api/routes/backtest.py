"""Institutional Strategy Backtesting API Routes."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
import pandas as pd

from ai_engine.backtesting.backtest_engine import BacktestEngine
from ai_engine.backtesting.strategies import (
    BaseStrategy,
    BollingerBreakoutStrategy,
    MACDStrategy,
    MovingAverageCrossStrategy,
    MultiFactorConsensusStrategy,
    RSIMeanReversionStrategy,
)
from backend.app.schemas.risk_anomaly_portfolio import BacktestRequest, BacktestResponse
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/backtest", tags=["Algorithmic Backtesting Lab"])


def _resolve_strategy(strategy_key: str) -> BaseStrategy:
    """Map request string to strategy instance."""
    key = strategy_key.strip().lower()
    if key in ["ma_cross", "golden_cross", "moving_average"]:
        return MovingAverageCrossStrategy(fast_period=20, slow_period=50, use_ema=True)
    elif key in ["rsi", "rsi_mean_reversion", "mean_reversion"]:
        return RSIMeanReversionStrategy(rsi_period=14, oversold=30.0, overbought=70.0)
    elif key in ["bollinger", "bollinger_breakout", "bb"]:
        return BollingerBreakoutStrategy(period=20, num_std=2.0)
    elif key in ["macd", "macd_trend"]:
        return MACDStrategy(fast_span=12, slow_span=26, signal_span=9)
    elif key in ["multi_factor", "consensus", "ml_consensus"]:
        return MultiFactorConsensusStrategy()
    else:
        return MovingAverageCrossStrategy(fast_period=20, slow_period=50, use_ema=True)


@router.post(
    "/run",
    response_model=BacktestResponse,
    summary="Execute Algorithmic Strategy Backtest",
)
async def run_strategy_backtest(request: BacktestRequest):
    """Execute chronological walk-forward historical simulation with realistic slippage and transaction fees.
    
    Prevents look-ahead bias and future data leakage by executing bar t signals at bar t+1 Open.
    """
    bars = await market_service.get_historical_ohlc(
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=request.limit,
    )
    if not bars or len(bars) < 20:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insufficient historical data available for '{request.symbol}' to run simulation.",
        )

    records = [
        b.model_dump() if hasattr(b, "model_dump") else (b.dict() if hasattr(b, "dict") else b.__dict__)
        for b in bars
    ]
    df = pd.DataFrame(records)

    # Fetch benchmark data for alpha and relative performance comparison
    bench_df = None
    try:
        bench_bars = await market_service.get_historical_ohlc(
            symbol="NIFTY 50",
            timeframe=request.timeframe,
            limit=request.limit,
        )
        if bench_bars:
            b_records = [
                b.model_dump() if hasattr(b, "model_dump") else (b.dict() if hasattr(b, "dict") else b.__dict__)
                for b in bench_bars
            ]
            bench_df = pd.DataFrame(b_records)
    except Exception:
        bench_df = None

    strategy_instance = _resolve_strategy(request.strategy)

    engine = BacktestEngine(
        initial_capital=request.initial_capital,
        slippage_pct=request.slippage_pct,
        transaction_fee_pct=request.transaction_fee_pct,
    )

    result = engine.run_backtest(
        df=df,
        strategy=strategy_instance,
        symbol=request.symbol,
        benchmark_df=bench_df,
    )
    return result.to_dict()
