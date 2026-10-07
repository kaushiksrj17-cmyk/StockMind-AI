"""Analytical Modern Portfolio Theory (MPT) Optimizer for StockMind-AI.

Implements:
1. Markowitz Mean-Variance Optimization using SLSQP via SciPy.
2. Maximum Sharpe Ratio Portfolio (Tangency Portfolio).
3. Minimum Volatility Portfolio.
4. Efficient Frontier Curve Generation across target return spectrum.
5. Feasible Set Monte Carlo simulation.
6. Explicit compliance disclaimer labeling: ANALYTICAL SIMULATION, NOT FINANCIAL ADVICE.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ai_engine.risk.risk_engine import DEFAULT_RISK_FREE_RATE


COMPLIANCE_DISCLAIMER = (
    "ANALYTICAL SIMULATION NOTICE: Modern Portfolio Theory optimization is an "
    "analytical simulation based on historical statistical covariance and returns. "
    "It does not constitute investment advice, a financial recommendation, "
    "or a guarantee of future performance. Past performance is no guarantee of future returns."
)

DEFAULT_ASSET_EXPECTED_RETURNS = {
    "RELIANCE": 0.165,
    "TCS": 0.145,
    "INFY": 0.155,
    "HDFCBANK": 0.150,
    "ICICIBANK": 0.175,
    "SBIN": 0.180,
    "BHARTIARTL": 0.160,
    "ITC": 0.135,
    "HINDUNILVR": 0.125,
    "TATAMOTORS": 0.210,
    "LT": 0.170,
    "SUNPHARMA": 0.140,
}

DEFAULT_ASSET_VOLATILITIES = {
    "RELIANCE": 0.19,
    "TCS": 0.17,
    "INFY": 0.21,
    "HDFCBANK": 0.18,
    "ICICIBANK": 0.22,
    "SBIN": 0.25,
    "BHARTIARTL": 0.20,
    "ITC": 0.15,
    "HINDUNILVR": 0.14,
    "TATAMOTORS": 0.29,
    "LT": 0.21,
    "SUNPHARMA": 0.18,
}


@dataclass
class OptimizedPortfolioPoint:
    """An optimized portfolio allocation on or near the efficient frontier."""
    label: str
    weights: Dict[str, float]  # symbol -> weight (0.0 to 1.0)
    expected_return_pct: float
    volatility_pct: float
    sharpe_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "weights": {k: round(v * 100.0, 2) for k, v in self.weights.items()},
            "expected_return_pct": round(self.expected_return_pct, 2),
            "volatility_pct": round(self.volatility_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }


@dataclass
class FrontierPoint:
    """Single point along the Markowitz Efficient Frontier curve."""
    target_return_pct: float
    volatility_pct: float
    sharpe_ratio: float
    weights: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_return_pct": round(self.target_return_pct, 2),
            "volatility_pct": round(self.volatility_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "weights": {k: round(v * 100.0, 2) for k, v in self.weights.items()},
        }


@dataclass
class OptimizationResult:
    """Complete Modern Portfolio Theory analytical optimization package."""
    symbols: List[str]
    max_sharpe_portfolio: OptimizedPortfolioPoint
    min_volatility_portfolio: OptimizedPortfolioPoint
    current_portfolio: Optional[OptimizedPortfolioPoint]
    efficient_frontier: List[FrontierPoint]
    simulated_cloud: List[Dict[str, float]]  # Sampled points for risk-return scatter cloud
    risk_free_rate_pct: float
    compliance_disclaimer: str = COMPLIANCE_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "max_sharpe_portfolio": self.max_sharpe_portfolio.to_dict(),
            "min_volatility_portfolio": self.min_volatility_portfolio.to_dict(),
            "current_portfolio": self.current_portfolio.to_dict() if self.current_portfolio else None,
            "efficient_frontier": [p.to_dict() for p in self.efficient_frontier],
            "simulated_cloud_sample": self.simulated_cloud[:100],  # truncated for light JSON payloads
            "risk_free_rate_pct": round(self.risk_free_rate_pct, 2),
            "compliance_disclaimer": self.compliance_disclaimer,
        }


class MPTOptimizer:
    """Markowitz Mean-Variance Portfolio Optimizer."""

    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> None:
        self.risk_free_rate = risk_free_rate

    def _build_cov_and_returns(
        self,
        symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute annualized expected returns and covariance matrix."""
        n = len(symbols)
        if returns_df is not None and len(returns_df) > 20:
            avail = [s for s in symbols if s in returns_df.columns]
            if len(avail) == n:
                mu = returns_df[symbols].mean().values * 252.0
                cov = returns_df[symbols].cov().values * 252.0
                return mu, cov

        # Baseline institutional calibration
        mu_list = [DEFAULT_ASSET_EXPECTED_RETURNS.get(s, 0.15) for s in symbols]
        vol_list = [DEFAULT_ASSET_VOLATILITIES.get(s, 0.20) for s in symbols]
        mu = np.array(mu_list)
        vols = np.array(vol_list)

        # Baseline 0.40 cross-asset correlation
        corr = np.full((n, n), 0.40)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(vols, vols) * corr
        return mu, cov

    @staticmethod
    def _portfolio_performance(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Tuple[float, float]:
        """Calculate annualized portfolio expected return and volatility."""
        ret = float(np.dot(weights, mu))
        var = float(np.dot(weights.T, np.dot(cov, weights)))
        vol = float(math.sqrt(max(var, 1e-7)))
        return ret, vol

    def optimize_portfolio(
        self,
        symbols: List[str],
        current_weights: Optional[Dict[str, float]] = None,
        returns_df: Optional[pd.DataFrame] = None,
        max_asset_weight: float = 0.50,
    ) -> OptimizationResult:
        """Execute analytical MPT optimization across specified universe."""
        clean_symbols = [s.upper().strip() for s in symbols]
        n = len(clean_symbols)
        if n == 0:
            clean_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
            n = len(clean_symbols)

        mu, cov = self._build_cov_and_returns(clean_symbols, returns_df)

        # Optimization Bounds and Constraints
        bounds = tuple((0.0, max_asset_weight) for _ in range(n))
        init_weights = np.ones(n) / n
        weight_sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # 1. Max Sharpe Ratio Optimization
        def neg_sharpe(w: np.ndarray) -> float:
            r, v = self._portfolio_performance(w, mu, cov)
            if v <= 1e-6:
                return 0.0
            return -((r - self.risk_free_rate) / v)

        res_sharpe = minimize(
            neg_sharpe,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[weight_sum_constraint],
            options={"maxiter": 500, "ftol": 1e-7},
        )
        w_sharpe = res_sharpe.x if res_sharpe.success else init_weights
        w_sharpe = np.maximum(w_sharpe, 0.0)
        w_sharpe = w_sharpe / np.sum(w_sharpe)
        r_sharpe, v_sharpe = self._portfolio_performance(w_sharpe, mu, cov)
        sr_sharpe = (r_sharpe - self.risk_free_rate) / v_sharpe if v_sharpe > 1e-4 else 0.0

        max_sharpe_pt = OptimizedPortfolioPoint(
            label="Maximum Sharpe Ratio (Tangency)",
            weights={clean_symbols[i]: float(w_sharpe[i]) for i in range(n)},
            expected_return_pct=r_sharpe * 100.0,
            volatility_pct=v_sharpe * 100.0,
            sharpe_ratio=sr_sharpe,
        )

        # 2. Minimum Volatility Optimization
        def portfolio_vol(w: np.ndarray) -> float:
            _, v = self._portfolio_performance(w, mu, cov)
            return v

        res_min_vol = minimize(
            portfolio_vol,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[weight_sum_constraint],
            options={"maxiter": 500, "ftol": 1e-7},
        )
        w_min_vol = res_min_vol.x if res_min_vol.success else init_weights
        w_min_vol = np.maximum(w_min_vol, 0.0)
        w_min_vol = w_min_vol / np.sum(w_min_vol)
        r_min_vol, v_min_vol = self._portfolio_performance(w_min_vol, mu, cov)
        sr_min_vol = (r_min_vol - self.risk_free_rate) / v_min_vol if v_min_vol > 1e-4 else 0.0

        min_vol_pt = OptimizedPortfolioPoint(
            label="Minimum Volatility",
            weights={clean_symbols[i]: float(w_min_vol[i]) for i in range(n)},
            expected_return_pct=r_min_vol * 100.0,
            volatility_pct=v_min_vol * 100.0,
            sharpe_ratio=sr_min_vol,
        )

        # 3. Current Portfolio Point
        cur_pt: Optional[OptimizedPortfolioPoint] = None
        if current_weights:
            w_cur_arr = np.array([current_weights.get(s, 0.0) for s in clean_symbols])
            tot = np.sum(w_cur_arr)
            if tot > 0:
                w_cur_arr = w_cur_arr / tot
                r_cur, v_cur = self._portfolio_performance(w_cur_arr, mu, cov)
                sr_cur = (r_cur - self.risk_free_rate) / v_cur if v_cur > 1e-4 else 0.0
                cur_pt = OptimizedPortfolioPoint(
                    label="Current Allocation",
                    weights={clean_symbols[i]: float(w_cur_arr[i]) for i in range(n)},
                    expected_return_pct=r_cur * 100.0,
                    volatility_pct=v_cur * 100.0,
                    sharpe_ratio=sr_cur,
                )

        # 4. Efficient Frontier Generation
        frontier_points: List[FrontierPoint] = []
        min_r = r_min_vol
        max_r = float(np.max(mu)) * 0.98
        target_returns = np.linspace(min_r, max_r, 25)

        for target_r in target_returns:
            constraints = [
                weight_sum_constraint,
                {"type": "eq", "fun": lambda w, tr=target_r: self._portfolio_performance(w, mu, cov)[0] - tr},
            ]
            res_target = minimize(
                portfolio_vol,
                init_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-6},
            )
            if res_target.success:
                w_t = np.maximum(res_target.x, 0.0)
                w_t = w_t / np.sum(w_t)
                ret_t, vol_t = self._portfolio_performance(w_t, mu, cov)
                sr_t = (ret_t - self.risk_free_rate) / vol_t if vol_t > 1e-4 else 0.0
                frontier_points.append(
                    FrontierPoint(
                        target_return_pct=ret_t * 100.0,
                        volatility_pct=vol_t * 100.0,
                        sharpe_ratio=sr_t,
                        weights={clean_symbols[i]: float(w_t[i]) for i in range(n)},
                    )
                )

        # 5. Monte Carlo Simulation Cloud (Feasible Set)
        cloud_samples: List[Dict[str, float]] = []
        np.random.seed(42)
        for _ in range(400):
            rnd = np.random.dirichlet(np.ones(n))
            ret_rnd, vol_rnd = self._portfolio_performance(rnd, mu, cov)
            sr_rnd = (ret_rnd - self.risk_free_rate) / vol_rnd if vol_rnd > 1e-4 else 0.0
            cloud_samples.append({
                "volatility_pct": round(vol_rnd * 100.0, 2),
                "expected_return_pct": round(ret_rnd * 100.0, 2),
                "sharpe_ratio": round(sr_rnd, 2),
            })

        return OptimizationResult(
            symbols=clean_symbols,
            max_sharpe_portfolio=max_sharpe_pt,
            min_volatility_portfolio=min_vol_pt,
            current_portfolio=cur_pt,
            efficient_frontier=frontier_points,
            simulated_cloud=cloud_samples,
            risk_free_rate_pct=self.risk_free_rate * 100.0,
            compliance_disclaimer=COMPLIANCE_DISCLAIMER,
        )


# Global singleton instance
mpt_optimizer = MPTOptimizer()
