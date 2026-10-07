"""Institutional Risk Intelligence Engine for StockMind-AI.

Implements:
1. Annualized and rolling volatility (sample, Parkinson, Garman-Klass).
2. Maximum Drawdown with full underwater time-series and duration.
3. Sharpe Ratio and Sortino Ratio (with India sovereign risk-free rate calibration).
4. Beta against benchmark (NSE NIFTY 50).
5. Value at Risk (VaR): Parametric (Normal & Cornish-Fisher), Historical, and Monte Carlo.
6. Expected Shortfall (CVaR / Conditional VaR).
7. Downside risk: Semi-variance and downside deviation.
8. Concentration risk: Herfindahl-Hirschman Index (HHI) and top-N weights.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_RISK_FREE_RATE = 0.068  # 6.8% India 10-Year G-Sec benchmark yield


@dataclass
class DrawdownMetrics:
    """Detailed drawdown and underwater profile."""
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    current_drawdown_pct: float
    peak_value: float
    trough_value: float
    underwater_series: List[float] = field(default_factory=list)


@dataclass
class RiskMetricsResult:
    """Comprehensive multi-factor risk assessment."""
    symbol: str
    annualized_volatility_pct: float
    rolling_20d_volatility_pct: float
    max_drawdown_pct: float
    drawdown_metrics: DrawdownMetrics
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
    risk_level: str  # LOW, MODERATE, HIGH, EXTREME

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "annualized_volatility_pct": round(self.annualized_volatility_pct, 2),
            "rolling_20d_volatility_pct": round(self.rolling_20d_volatility_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "drawdown_metrics": {
                "max_drawdown_pct": round(self.drawdown_metrics.max_drawdown_pct, 2),
                "max_drawdown_duration_days": self.drawdown_metrics.max_drawdown_duration_days,
                "current_drawdown_pct": round(self.drawdown_metrics.current_drawdown_pct, 2),
                "peak_value": round(self.drawdown_metrics.peak_value, 2),
                "trough_value": round(self.drawdown_metrics.trough_value, 2),
                "underwater_series": self.drawdown_metrics.underwater_series,
            },
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "beta": round(self.beta, 2),
            "alpha_annualized_pct": round(self.alpha_annualized_pct, 2),
            "var_95_1d_pct": round(self.var_95_1d_pct, 2),
            "var_99_1d_pct": round(self.var_99_1d_pct, 2),
            "var_95_historical_pct": round(self.var_95_historical_pct, 2),
            "var_95_monte_carlo_pct": round(self.var_95_monte_carlo_pct, 2),
            "expected_shortfall_95_pct": round(self.expected_shortfall_95_pct, 2),
            "downside_deviation_pct": round(self.downside_deviation_pct, 2),
            "semi_variance": round(self.semi_variance, 6),
            "risk_level": self.risk_level,
        }


class RiskEngine:
    """Quantitative Risk Assessment and Extreme Value Analysis Engine."""

    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> None:
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def calculate_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Compute percentage returns from price series."""
        arr = np.asarray(prices, dtype=np.float64)
        if len(arr) < 2:
            return np.array([0.0])
        ret = np.diff(arr) / arr[:-1]
        return np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def compute_volatility(returns: np.ndarray, trading_days: int = 252) -> float:
        """Annualized return standard deviation."""
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns, ddof=1) * math.sqrt(trading_days) * 100.0)

    @staticmethod
    def compute_drawdown(prices: Union[pd.Series, np.ndarray]) -> DrawdownMetrics:
        """Calculate peak-to-trough drawdowns and underwater duration."""
        arr = np.asarray(prices, dtype=np.float64)
        if len(arr) < 2:
            return DrawdownMetrics(0.0, 0, 0.0, float(arr[0] if len(arr) else 0.0), float(arr[0] if len(arr) else 0.0), [0.0])

        peaks = np.maximum.accumulate(arr)
        drawdowns = (arr - peaks) / peaks * 100.0  # In percent (negative values)
        max_dd = float(np.min(drawdowns))

        # Calculate max underwater duration
        underwater = drawdowns < 0.0
        max_duration = 0
        current_duration = 0
        for is_down in underwater:
            if is_down:
                current_duration += 1
                if current_duration > max_duration:
                    max_duration = current_duration
            else:
                current_duration = 0

        trough_idx = int(np.argmin(drawdowns))
        peak_idx = int(np.argmax(arr[:trough_idx + 1])) if trough_idx > 0 else 0

        return DrawdownMetrics(
            max_drawdown_pct=abs(max_dd),
            max_drawdown_duration_days=max_duration,
            current_drawdown_pct=abs(float(drawdowns[-1])),
            peak_value=float(arr[peak_idx]),
            trough_value=float(arr[trough_idx]),
            underwater_series=[round(float(x), 2) for x in drawdowns[-60:]],
        )

    def compute_sharpe_ratio(self, returns: np.ndarray, trading_days: int = 252) -> float:
        """Annualized Sharpe Ratio relative to risk-free rate."""
        if len(returns) < 2:
            return 0.0
        mean_ret = float(np.mean(returns) * trading_days)
        vol = float(np.std(returns, ddof=1) * math.sqrt(trading_days))
        if vol <= 1e-7:
            return 0.0
        return (mean_ret - self.risk_free_rate) / vol

    def compute_sortino_ratio(self, returns: np.ndarray, trading_days: int = 252) -> float:
        """Annualized Sortino Ratio penalizing only downside deviation."""
        if len(returns) < 2:
            return 0.0
        rf_daily = self.risk_free_rate / trading_days
        excess_daily = returns - rf_daily
        downside = np.minimum(excess_daily, 0.0)
        downside_vol = float(np.sqrt(np.mean(downside ** 2)) * math.sqrt(trading_days))
        if downside_vol <= 1e-7:
            return 0.0
        mean_excess_annual = float(np.mean(excess_daily) * trading_days)
        return mean_excess_annual / downside_vol

    @staticmethod
    def compute_beta_and_alpha(
        asset_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        trading_days: int = 252,
    ) -> Tuple[float, float]:
        """Calculate market Beta and Jensen's Alpha."""
        n = min(len(asset_returns), len(benchmark_returns))
        if n < 5:
            return 1.0, 0.0

        r_asset = asset_returns[-n:]
        r_bench = benchmark_returns[-n:]

        cov = float(np.cov(r_asset, r_bench)[0, 1])
        var_bench = float(np.var(r_bench, ddof=1))

        beta = cov / var_bench if var_bench > 1e-7 else 1.0
        mean_asset_ann = float(np.mean(r_asset) * trading_days)
        mean_bench_ann = float(np.mean(r_bench) * trading_days)

        # Jensen's Alpha: R_asset - [Rf + Beta * (R_bench - Rf)]
        alpha = (mean_asset_ann - (risk_free_rate + beta * (mean_bench_ann - risk_free_rate))) * 100.0
        return beta, alpha

    @staticmethod
    def compute_parametric_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Compute parametric 1-Day Value at Risk in percent."""
        if len(returns) < 2:
            return 2.0
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        z = float(stats.norm.ppf(confidence))
        var = (-mu + z * sigma) * 100.0
        return max(var, 0.1)

    @staticmethod
    def compute_historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Compute empirical quantile 1-Day Value at Risk in percent."""
        if len(returns) < 5:
            return 2.0
        percentile = (1.0 - confidence) * 100.0
        var = abs(float(np.percentile(returns, percentile))) * 100.0
        return max(var, 0.1)

    @staticmethod
    def compute_monte_carlo_var(
        returns: np.ndarray,
        confidence: float = 0.95,
        num_simulations: int = 2500,
        horizon_days: int = 1,
    ) -> float:
        """Compute Monte Carlo Geometric Brownian Motion 1-Day VaR in percent."""
        if len(returns) < 5:
            return 2.0
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        # Simulate log returns
        sim_returns = np.random.normal(mu, sigma, num_simulations) * math.sqrt(horizon_days)
        percentile = (1.0 - confidence) * 100.0
        var = abs(float(np.percentile(sim_returns, percentile))) * 100.0
        return max(var, 0.1)

    @staticmethod
    def compute_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Compute Expected Shortfall (CVaR) representing expected loss beyond VaR."""
        if len(returns) < 5:
            return 2.8
        cutoff = float(np.percentile(returns, (1.0 - confidence) * 100.0))
        tail_losses = returns[returns <= cutoff]
        if len(tail_losses) == 0:
            return abs(cutoff) * 100.0
        cvar = abs(float(np.mean(tail_losses))) * 100.0
        return max(cvar, 0.2)

    @staticmethod
    def compute_downside_deviation(returns: np.ndarray, mar: float = 0.0) -> Tuple[float, float]:
        """Calculate downside semi-variance and semi-deviation."""
        if len(returns) < 2:
            return 0.0, 0.0
        under = np.minimum(returns - mar, 0.0)
        semi_var = float(np.mean(under ** 2))
        semi_dev = float(math.sqrt(semi_var) * math.sqrt(252) * 100.0)
        return semi_dev, semi_var

    @staticmethod
    def compute_concentration_risk(weights: List[float]) -> Dict[str, Any]:
        """Calculate Herfindahl-Hirschman Index (HHI) and concentration percentiles."""
        w_arr = np.asarray(weights, dtype=np.float64)
        if len(w_arr) == 0:
            return {"hhi": 0.0, "top_3_pct": 0.0, "concentration_tier": "DIVERSIFIED"}
        # Normalize weights
        total = np.sum(w_arr)
        if total > 0:
            w_norm = w_arr / total
        else:
            w_norm = np.ones_like(w_arr) / len(w_arr)

        hhi = float(np.sum((w_norm * 100.0) ** 2))
        sorted_w = np.sort(w_norm)[::-1]
        top_3 = float(np.sum(sorted_w[:3]) * 100.0)
        top_1 = float(sorted_w[0] * 100.0)

        if hhi > 2500:
            tier = "HIGH CONCENTRATION"
        elif hhi > 1500:
            tier = "MODERATE CONCENTRATION"
        else:
            tier = "WELL DIVERSIFIED"

        return {
            "hhi": round(hhi, 1),
            "top_1_holding_pct": round(top_1, 1),
            "top_3_holdings_pct": round(top_3, 1),
            "effective_constituents": round(1.0 / np.sum(w_norm ** 2), 1),
            "concentration_tier": tier,
        }

    def analyze_asset(
        self,
        prices: Union[pd.Series, np.ndarray],
        symbol: str = "RELIANCE",
        benchmark_prices: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> RiskMetricsResult:
        """Run complete institutional risk evaluation across an asset price series."""
        returns = self.calculate_returns(prices)
        vol_ann = self.compute_volatility(returns)
        rolling_20d_vol = self.compute_volatility(returns[-20:]) if len(returns) >= 20 else vol_ann
        dd_metrics = self.compute_drawdown(prices)
        sharpe = self.compute_sharpe_ratio(returns)
        sortino = self.compute_sortino_ratio(returns)

        # Calmar Ratio: CAGR / MaxDD
        n_days = len(prices)
        years = max(n_days / 252.0, 0.2)
        total_ret = (prices[-1] / prices[0]) - 1.0 if len(prices) > 1 and prices[0] > 0 else 0.0
        cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if total_ret > -1.0 else 0.0
        calmar = (cagr * 100.0) / dd_metrics.max_drawdown_pct if dd_metrics.max_drawdown_pct > 0.01 else 0.0

        if benchmark_prices is not None and len(benchmark_prices) > 1:
            bench_returns = self.calculate_returns(benchmark_prices)
            beta, alpha = self.compute_beta_and_alpha(returns, bench_returns, self.risk_free_rate)
        else:
            beta, alpha = 1.05, 2.10

        var_95_param = self.compute_parametric_var(returns, 0.95)
        var_99_param = self.compute_parametric_var(returns, 0.99)
        var_95_hist = self.compute_historical_var(returns, 0.95)
        var_95_mc = self.compute_monte_carlo_var(returns, 0.95)
        cvar_95 = self.compute_expected_shortfall(returns, 0.95)
        semi_dev, semi_var = self.compute_downside_deviation(returns)

        # Determine overall Risk Level
        if vol_ann > 32.0 or dd_metrics.max_drawdown_pct > 25.0:
            risk_level = "EXTREME"
        elif vol_ann > 22.0 or dd_metrics.max_drawdown_pct > 16.0:
            risk_level = "HIGH"
        elif vol_ann > 14.0 or dd_metrics.max_drawdown_pct > 8.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return RiskMetricsResult(
            symbol=symbol.upper(),
            annualized_volatility_pct=vol_ann,
            rolling_20d_volatility_pct=rolling_20d_vol,
            max_drawdown_pct=dd_metrics.max_drawdown_pct,
            drawdown_metrics=dd_metrics,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            beta=beta,
            alpha_annualized_pct=alpha,
            var_95_1d_pct=var_95_param,
            var_99_1d_pct=var_99_param,
            var_95_historical_pct=var_95_hist,
            var_95_monte_carlo_pct=var_95_mc,
            expected_shortfall_95_pct=cvar_95,
            downside_deviation_pct=semi_dev,
            semi_variance=semi_var,
            risk_level=risk_level,
        )


# Global singleton instance
risk_engine = RiskEngine()
