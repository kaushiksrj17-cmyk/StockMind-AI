"""Institutional Portfolio Evaluation and Risk Attribution Engine for StockMind-AI.

Implements:
1. Holdings aggregation and cost-basis accounting.
2. Weight and asset allocation breakdown.
3. Total portfolio valuation, realized/unrealized P&L.
4. Sector exposure aggregation.
5. Herfindahl-Hirschman Index (HHI) concentration analysis.
6. Diversification Ratio and effective number of constituents.
7. Portfolio Risk Attribution: Portfolio Volatility, Beta, 1-Day VaR, Sharpe.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ai_engine.risk.risk_engine import DEFAULT_RISK_FREE_RATE, RiskEngine


SECTOR_MAP = {
    "RELIANCE": "Energy / Oil & Gas",
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HDFCBANK": "Banking & Financial Services",
    "ICICIBANK": "Banking & Financial Services",
    "SBIN": "Banking & Financial Services",
    "BHARTIARTL": "Telecommunications",
    "ITC": "Fast Moving Consumer Goods",
    "HINDUNILVR": "Fast Moving Consumer Goods",
    "TATAMOTORS": "Automotive",
    "MARUTI": "Automotive",
    "LT": "Infrastructure & Capital Goods",
    "SUNPHARMA": "Pharmaceuticals & Healthcare",
    "BAJFINANCE": "Non-Banking Financial",
    "AXISBANK": "Banking & Financial Services",
    "KOTAKBANK": "Banking & Financial Services",
    "TITAN": "Consumer Discretionary",
    "ASIANPAINT": "Consumer Discretionary",
    "WIPRO": "Information Technology",
    "HCLTECH": "Information Technology",
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

DEFAULT_ASSET_BETAS = {
    "RELIANCE": 1.05,
    "TCS": 0.85,
    "INFY": 0.95,
    "HDFCBANK": 1.10,
    "ICICIBANK": 1.20,
    "SBIN": 1.35,
    "BHARTIARTL": 0.90,
    "ITC": 0.65,
    "HINDUNILVR": 0.60,
    "TATAMOTORS": 1.45,
    "LT": 1.15,
    "SUNPHARMA": 0.70,
}


@dataclass
class HoldingItem:
    """Individual portfolio holding."""
    symbol: str
    name: str
    shares: int
    avg_price: float
    current_price: float
    sector: str
    cost_basis: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "shares": self.shares,
            "avg_price": round(self.avg_price, 2),
            "current_price": round(self.current_price, 2),
            "sector": self.sector,
            "cost_basis": round(self.cost_basis, 2),
            "current_value": round(self.current_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "weight_pct": round(self.weight_pct, 2),
        }


@dataclass
class SectorExposure:
    """Sector weight and capital exposure."""
    sector: str
    value: float
    weight_pct: float
    holding_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector": self.sector,
            "value": round(self.value, 2),
            "weight_pct": round(self.weight_pct, 2),
            "holding_count": self.holding_count,
        }


@dataclass
class PortfolioSummary:
    """Comprehensive portfolio valuation and risk report."""
    total_value: float
    total_cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    holding_count: int
    holdings: List[HoldingItem]
    sector_exposures: List[SectorExposure]
    # Risk & Concentration
    hhi: float
    concentration_tier: str  # WELL DIVERSIFIED, MODERATE CONCENTRATION, HIGH CONCENTRATION
    top_1_weight_pct: float
    top_3_weight_pct: float
    effective_constituents: float
    diversification_ratio: float
    portfolio_volatility_pct: float
    portfolio_beta: float
    portfolio_var_95_1d_pct: float
    portfolio_sharpe_ratio: float
    risk_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_value": round(self.total_value, 2),
            "total_cost_basis": round(self.total_cost_basis, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "holding_count": self.holding_count,
            "holdings": [h.to_dict() for h in self.holdings],
            "sector_exposures": [s.to_dict() for s in self.sector_exposures],
            "hhi": round(self.hhi, 1),
            "concentration_tier": self.concentration_tier,
            "top_1_weight_pct": round(self.top_1_weight_pct, 2),
            "top_3_weight_pct": round(self.top_3_weight_pct, 2),
            "effective_constituents": round(self.effective_constituents, 1),
            "diversification_ratio": round(self.diversification_ratio, 2),
            "portfolio_volatility_pct": round(self.portfolio_volatility_pct, 2),
            "portfolio_beta": round(self.portfolio_beta, 2),
            "portfolio_var_95_1d_pct": round(self.portfolio_var_95_1d_pct, 2),
            "portfolio_sharpe_ratio": round(self.portfolio_sharpe_ratio, 2),
            "risk_level": self.risk_level,
        }


class PortfolioEngine:
    """Quantitative evaluation and risk engine for multi-asset portfolios."""

    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> None:
        self.risk_free_rate = risk_free_rate

    def evaluate_portfolio(
        self,
        raw_holdings: List[Dict[str, Any]],
        realized_pnl: float = 0.0,
        asset_prices: Optional[Dict[str, float]] = None,
        asset_returns_df: Optional[pd.DataFrame] = None,
    ) -> PortfolioSummary:
        """Calculate complete valuation, sector breakdown, and portfolio risk."""
        if not raw_holdings:
            return PortfolioSummary(
                total_value=0.0,
                total_cost_basis=0.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                realized_pnl=realized_pnl,
                holding_count=0,
                holdings=[],
                sector_exposures=[],
                hhi=0.0,
                concentration_tier="WELL DIVERSIFIED",
                top_1_weight_pct=0.0,
                top_3_weight_pct=0.0,
                effective_constituents=0.0,
                diversification_ratio=1.0,
                portfolio_volatility_pct=0.0,
                portfolio_beta=1.0,
                portfolio_var_95_1d_pct=0.0,
                portfolio_sharpe_ratio=0.0,
                risk_level="LOW",
            )

        prices = asset_prices or {}
        items: List[HoldingItem] = []
        total_value = 0.0
        total_cost = 0.0

        # Pass 1: compute valuations
        for h in raw_holdings:
            sym = str(h.get("symbol", "")).upper()
            shares = int(h.get("shares", 0))
            avg_p = float(h.get("avg_price", 0.0))
            cur_p = float(prices.get(sym, h.get("current_price", avg_p)))
            if cur_p <= 0.0:
                cur_p = avg_p

            name = str(h.get("name", sym))
            sector = str(h.get("sector") or SECTOR_MAP.get(sym, "General Industry"))

            cost = shares * avg_p
            val = shares * cur_p
            unreal_pnl = val - cost
            unreal_pct = (unreal_pnl / cost * 100.0) if cost > 0 else 0.0

            total_value += val
            total_cost += cost

            items.append(
                HoldingItem(
                    symbol=sym,
                    name=name,
                    shares=shares,
                    avg_price=avg_p,
                    current_price=cur_p,
                    sector=sector,
                    cost_basis=cost,
                    current_value=val,
                    unrealized_pnl=unreal_pnl,
                    unrealized_pnl_pct=unreal_pct,
                )
            )

        # Pass 2: compute weights
        for item in items:
            item.weight_pct = (item.current_value / total_value * 100.0) if total_value > 0 else (100.0 / len(items))

        # Overall P&L
        tot_unrealized = total_value - total_cost
        tot_unrealized_pct = (tot_unrealized / total_cost * 100.0) if total_cost > 0 else 0.0

        # Sector breakdown
        sectors: Dict[str, Dict[str, Any]] = {}
        for item in items:
            sec = item.sector
            if sec not in sectors:
                sectors[sec] = {"value": 0.0, "count": 0}
            sectors[sec]["value"] += item.current_value
            sectors[sec]["count"] += 1

        sector_exposures = [
            SectorExposure(
                sector=s,
                value=d["value"],
                weight_pct=(d["value"] / total_value * 100.0) if total_value > 0 else 0.0,
                holding_count=d["count"],
            )
            for s, d in sorted(sectors.items(), key=lambda x: x[1]["value"], reverse=True)
        ]

        # Concentration & HHI
        weights = np.array([item.weight_pct / 100.0 for item in items])
        hhi = float(np.sum((weights * 100.0) ** 2))
        sorted_weights = np.sort(weights)[::-1]
        top_1 = float(sorted_weights[0] * 100.0) if len(sorted_weights) > 0 else 0.0
        top_3 = float(np.sum(sorted_weights[:3]) * 100.0) if len(sorted_weights) > 0 else 0.0
        effective_n = float(1.0 / np.sum(weights ** 2)) if np.sum(weights ** 2) > 0 else len(items)

        if hhi > 2500:
            concentration_tier = "HIGH CONCENTRATION"
        elif hhi > 1500:
            concentration_tier = "MODERATE CONCENTRATION"
        else:
            concentration_tier = "WELL DIVERSIFIED"

        # Risk Attribution
        symbols = [item.symbol for item in items]
        cov_matrix = None
        asset_vols = []
        asset_betas = []

        for sym in symbols:
            asset_vols.append(DEFAULT_ASSET_VOLATILITIES.get(sym, 0.20))
            asset_betas.append(DEFAULT_ASSET_BETAS.get(sym, 1.0))

        vols = np.array(asset_vols)
        betas = np.array(asset_betas)

        # Portfolio Beta: weighted sum of betas
        portfolio_beta = float(np.dot(weights, betas))

        # Check if empirical returns matrix is provided
        if asset_returns_df is not None and len(asset_returns_df) > 10:
            sub_df = asset_returns_df[[c for c in symbols if c in asset_returns_df.columns]]
            if len(sub_df.columns) == len(symbols):
                cov_matrix = sub_df.cov().values * 252.0
                port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
                port_vol = math.sqrt(max(port_var, 1e-6))
            else:
                # Default correlation of 0.45 between stocks
                corr = np.full((len(weights), len(weights)), 0.45)
                np.fill_diagonal(corr, 1.0)
                cov_matrix = np.outer(vols, vols) * corr
                port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
                port_vol = math.sqrt(max(port_var, 1e-6))
        else:
            # Synthetic standard institutional covariance model with 0.45 average cross-correlation
            corr = np.full((len(weights), len(weights)), 0.45)
            np.fill_diagonal(corr, 1.0)
            cov_matrix = np.outer(vols, vols) * corr
            port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
            port_vol = math.sqrt(max(port_var, 1e-6))

        # Diversification Ratio: Weighted individual volatility / Portfolio Volatility
        sum_weighted_vols = float(np.dot(weights, vols))
        div_ratio = sum_weighted_vols / port_vol if port_vol > 1e-4 else 1.0

        # 1-Day 95% Parametric VaR = 1.645 * (port_vol / sqrt(252))
        port_var_95_1d = (1.645 * (port_vol / math.sqrt(252))) * 100.0

        # Portfolio Sharpe Ratio estimation
        # Assuming expected market equity risk premium of 8.0% over Rf
        exp_return = self.risk_free_rate + portfolio_beta * 0.08
        port_sharpe = (exp_return - self.risk_free_rate) / port_vol if port_vol > 1e-4 else 1.0

        # Risk Level
        if port_vol > 0.25 or hhi > 3000:
            risk_level = "HIGH"
        elif port_vol > 0.16 or hhi > 1800:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return PortfolioSummary(
            total_value=total_value,
            total_cost_basis=total_cost,
            unrealized_pnl=tot_unrealized,
            unrealized_pnl_pct=tot_unrealized_pct,
            realized_pnl=realized_pnl,
            holding_count=len(items),
            holdings=items,
            sector_exposures=sector_exposures,
            hhi=hhi,
            concentration_tier=concentration_tier,
            top_1_weight_pct=top_1,
            top_3_weight_pct=top_3,
            effective_constituents=effective_n,
            diversification_ratio=div_ratio,
            portfolio_volatility_pct=port_vol * 100.0,
            portfolio_beta=portfolio_beta,
            portfolio_var_95_1d_pct=port_var_95_1d,
            portfolio_sharpe_ratio=port_sharpe,
            risk_level=risk_level,
        )


# Global singleton instance
portfolio_engine = PortfolioEngine()
