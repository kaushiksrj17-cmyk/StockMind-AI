"""Institutional Quantitative Backtesting Engine for StockMind-AI.

Guarantees:
1. Strict walk-forward chronological simulation (Zero Look-Ahead Bias).
2. Future data leakage prevention (Signals at bar t executed at bar t+1 Open).
3. Realistic execution assumptions:
   - Configurable slippage penalties (default 0.05% / 5 bps).
   - Exchange brokerage & turnover transaction costs (default 0.03% / 3 bps).
   - Whole share integer fills (no fractional share magic).
4. Full performance attribution:
   - Total Return, CAGR, Alpha vs Benchmark.
   - Max Drawdown with full underwater time-series.
   - Sharpe, Sortino, and Calmar ratios.
   - Win Rate, Profit Factor, and Payoff Ratio.
5. Explicit regulatory compliance disclaimer.
"""

from dataclasses import dataclass, field
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ai_engine.backtesting.strategies import (
    BaseStrategy,
    BollingerBreakoutStrategy,
    MACDStrategy,
    MovingAverageCrossStrategy,
    MultiFactorConsensusStrategy,
    RSIMeanReversionStrategy,
)
from ai_engine.risk.risk_engine import DEFAULT_RISK_FREE_RATE


BACKTEST_DISCLAIMER = (
    "ANALYTICAL SIMULATION NOTICE: Historical backtesting is a mathematical simulation "
    "subject to inherent modeling limitations. It does not reflect live execution realities, "
    "guarantee future profits, or constitute investment advice. Past performance is no guarantee of future returns."
)


@dataclass
class TradeRecord:
    """Individual trade execution audit."""
    trade_id: int
    entry_bar: int
    entry_time: str
    entry_price: float
    exit_bar: int
    exit_time: str
    exit_price: float
    shares: int
    gross_pnl: float
    net_pnl: float
    net_return_pct: float
    fees_paid: float
    duration_bars: int
    exit_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time,
            "entry_price": round(self.entry_price, 2),
            "exit_time": self.exit_time,
            "exit_price": round(self.exit_price, 2),
            "shares": self.shares,
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "net_return_pct": round(self.net_return_pct, 2),
            "fees_paid": round(self.fees_paid, 2),
            "duration_bars": self.duration_bars,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    """Institutional backtesting performance report."""
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
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    drawdown_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    compliance_disclaimer: str = BACKTEST_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "benchmark_return_pct": round(self.benchmark_return_pct, 2),
            "alpha_pct": round(self.alpha_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_drawdown_duration_bars": self.max_drawdown_duration_bars,
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "profit_factor": round(self.profit_factor, 2),
            "payoff_ratio": round(self.payoff_ratio, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "equity_curve": self.equity_curve[-100:],  # lightweight sample for responses
            "drawdown_curve": self.drawdown_curve[-100:],
            "trades": [t.to_dict() for t in self.trades[-50:]],
            "compliance_disclaimer": self.compliance_disclaimer,
        }


class BacktestEngine:
    """Rigorous Event-Driven Walk-Forward Backtester."""

    def __init__(
        self,
        initial_capital: float = 500000.0,
        slippage_pct: float = 0.0005,  # 0.05% slippage on entry/exit
        transaction_fee_pct: float = 0.0003,  # 0.03% brokerage & regulatory fee
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> None:
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.transaction_fee_pct = transaction_fee_pct
        self.risk_free_rate = risk_free_rate

    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "RELIANCE",
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Execute historical simulation with strictly verified timing."""
        if df is None or len(df) < 10:
            return self._empty_result(strategy.name, symbol)

        data = df.copy()
        data.columns = [str(c).strip().lower() for c in data.columns]
        close = data["close"].astype(float).values
        open_p = data["open"].astype(float).values if "open" in data.columns else close
        n = len(close)

        timestamps = []
        if "timestamp" in data.columns:
            timestamps = [str(t) for t in data["timestamp"]]
        elif "date" in data.columns:
            timestamps = [str(t) for t in data["date"]]
        else:
            timestamps = [f"Bar-{i}" for i in range(n)]

        # 1. Generate Strategy Signals strictly on information available at Close of bar t
        raw_signals = strategy.generate_signals(data).values

        # 2. Walk-forward execution loop
        cash = self.initial_capital
        shares = 0
        current_entry_price = 0.0
        current_entry_bar = 0
        current_entry_time = ""
        total_fees_paid = 0.0
        trades: List[TradeRecord] = []
        trade_id_counter = 1

        equity_curve: List[Dict[str, Any]] = []
        daily_equity: List[float] = []

        # Benchmark setup (Buy & Hold baseline starting at bar 0)
        bench_start = close[0] if benchmark_df is None else float(benchmark_df["close"].iloc[0])
        bench_close = close if benchmark_df is None else benchmark_df["close"].astype(float).values

        # We step bar by bar:
        # At bar t (starting at t = 1):
        # The desired position from bar t-1 signal is enacted at bar t OPEN.
        for t in range(n):
            curr_open = open_p[t]
            curr_close = close[t]
            curr_time = timestamps[t]

            # Signal from PREVIOUS bar dictates action at current bar Open
            desired_pos = raw_signals[t - 1] if t > 0 else 0

            # Execute Rebalancing at OPEN
            if desired_pos == 1 and shares == 0:
                # BUY: Long entry with slippage
                fill_price = curr_open * (1.0 + self.slippage_pct)
                # Position sizing: allocate 98% of available cash to maintain buffer for fees
                alloc_cash = cash * 0.98
                qty = int(alloc_cash / fill_price)
                if qty > 0:
                    trade_cost = qty * fill_price
                    fee = trade_cost * self.transaction_fee_pct
                    cash -= (trade_cost + fee)
                    total_fees_paid += fee
                    shares = qty
                    current_entry_price = fill_price
                    current_entry_bar = t
                    current_entry_time = curr_time

            elif desired_pos == 0 and shares > 0:
                # SELL: Exit with slippage
                fill_price = curr_open * (1.0 - self.slippage_pct)
                gross_proceeds = shares * fill_price
                fee = gross_proceeds * self.transaction_fee_pct
                cash += (gross_proceeds - fee)
                total_fees_paid += fee

                gross_pnl = (fill_price - current_entry_price) * shares
                net_pnl = gross_pnl - (fee + (shares * current_entry_price * self.transaction_fee_pct))
                net_ret_pct = (net_pnl / (shares * current_entry_price) * 100.0) if current_entry_price > 0 else 0.0

                trades.append(
                    TradeRecord(
                        trade_id=trade_id_counter,
                        entry_bar=current_entry_bar,
                        entry_time=current_entry_time,
                        entry_price=current_entry_price,
                        exit_bar=t,
                        exit_time=curr_time,
                        exit_price=fill_price,
                        shares=shares,
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        net_return_pct=net_ret_pct,
                        fees_paid=fee,
                        duration_bars=t - current_entry_bar,
                        exit_reason="STRATEGY_SIGNAL_EXIT",
                    )
                )
                trade_id_counter += 1
                shares = 0
                current_entry_price = 0.0

            # Mark to Market at current bar CLOSE
            pos_val = shares * curr_close
            total_equity = cash + pos_val
            daily_equity.append(total_equity)

            # Benchmark mark to market
            bench_idx = min(t, len(bench_close) - 1)
            bench_eq = self.initial_capital * (bench_close[bench_idx] / bench_start)

            equity_curve.append({
                "bar": t,
                "time": curr_time,
                "portfolio_equity": round(total_equity, 2),
                "benchmark_equity": round(bench_eq, 2),
                "cash": round(cash, 2),
                "in_position": bool(shares > 0),
            })

        # Close open position at end of backtest if still open
        if shares > 0:
            final_close = close[-1]
            fill_price = final_close * (1.0 - self.slippage_pct)
            gross_proceeds = shares * fill_price
            fee = gross_proceeds * self.transaction_fee_pct
            cash += (gross_proceeds - fee)
            total_fees_paid += fee
            gross_pnl = (fill_price - current_entry_price) * shares
            net_pnl = gross_pnl - fee
            net_ret_pct = (net_pnl / (shares * current_entry_price) * 100.0) if current_entry_price > 0 else 0.0
            trades.append(
                TradeRecord(
                    trade_id=trade_id_counter,
                    entry_bar=current_entry_bar,
                    entry_time=current_entry_time,
                    entry_price=current_entry_price,
                    exit_bar=n - 1,
                    exit_time=timestamps[-1],
                    exit_price=fill_price,
                    shares=shares,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    net_return_pct=net_ret_pct,
                    fees_paid=fee,
                    duration_bars=n - 1 - current_entry_bar,
                    exit_reason="SIMULATION_END_FLATTEN",
                )
            )
            shares = 0

        # 3. Compute Risk and Performance Statistics
        final_equity = cash
        total_ret_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100.0

        # Benchmark return
        final_bench = bench_close[min(n - 1, len(bench_close) - 1)]
        benchmark_ret_pct = ((final_bench - bench_start) / bench_start) * 100.0
        alpha_pct = total_ret_pct - benchmark_ret_pct

        # CAGR
        years = max(n / 252.0, 0.1)
        cagr_pct = ((final_equity / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

        # Drawdown profile
        eq_arr = np.array(daily_equity)
        peaks = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - peaks) / peaks * 100.0
        max_dd_pct = abs(float(np.min(drawdowns)))

        # Max drawdown duration
        max_duration = 0
        curr_dur = 0
        for dd in drawdowns:
            if dd < 0.0:
                curr_dur += 1
                if curr_dur > max_duration:
                    max_duration = curr_dur
            else:
                curr_dur = 0

        drawdown_curve = [
            {"time": timestamps[i], "drawdown_pct": round(float(drawdowns[i]), 2)}
            for i in range(len(drawdowns))
        ]

        # Returns series for Sharpe and Sortino
        eq_returns = np.diff(eq_arr) / eq_arr[:-1]
        eq_returns = np.nan_to_num(eq_returns, nan=0.0)

        vol_ann = float(np.std(eq_returns, ddof=1) * math.sqrt(252)) if len(eq_returns) > 2 else 0.0
        mean_ret_ann = float(np.mean(eq_returns) * 252) if len(eq_returns) > 2 else 0.0
        sharpe = (mean_ret_ann - self.risk_free_rate) / vol_ann if vol_ann > 1e-4 else 0.0

        downside_diff = np.minimum(eq_returns - (self.risk_free_rate / 252.0), 0.0)
        downside_vol = float(math.sqrt(np.mean(downside_diff ** 2)) * math.sqrt(252)) if len(eq_returns) > 2 else 0.0
        sortino = (mean_ret_ann - self.risk_free_rate) / downside_vol if downside_vol > 1e-4 else 0.0

        calmar = (cagr_pct / max_dd_pct) if max_dd_pct > 0.1 else 0.0

        # Trade metrics
        total_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-4 else (99.0 if gross_profit > 0 else 0.0)

        avg_win = (gross_profit / len(wins)) if len(wins) > 0 else 0.0
        avg_loss = (gross_loss / len(losses)) if len(losses) > 0 else 1.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 1e-4 else 0.0

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol.upper(),
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return_pct=total_ret_pct,
            cagr_pct=cagr_pct,
            benchmark_return_pct=benchmark_ret_pct,
            alpha_pct=alpha_pct,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration_bars=max_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_rate_pct=win_rate,
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            total_fees_paid=total_fees_paid,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trades=trades,
            compliance_disclaimer=BACKTEST_DISCLAIMER,
        )

    def _empty_result(self, strategy_name: str, symbol: str) -> BacktestResult:
        """Return empty result on insufficient data."""
        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
            total_return_pct=0.0,
            cagr_pct=0.0,
            benchmark_return_pct=0.0,
            alpha_pct=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_bars=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            win_rate_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            profit_factor=0.0,
            payoff_ratio=0.0,
            total_fees_paid=0.0,
            equity_curve=[],
            drawdown_curve=[],
            trades=[],
            compliance_disclaimer=BACKTEST_DISCLAIMER,
        )


# Global singleton instance
backtest_engine = BacktestEngine()
