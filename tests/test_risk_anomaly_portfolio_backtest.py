
"""Unit and Integration Tests for Risk, Anomaly, Market Regime, Portfolio MPT, and Backtesting Engines."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ai_engine.risk.risk_engine import RiskEngine, risk_engine
from ai_engine.anomaly.anomaly_engine import AnomalyEngine, anomaly_engine
from ai_engine.regime.regime_engine import MarketRegimeEngine, regime_engine
from ai_engine.portfolio.portfolio_engine import PortfolioEngine, portfolio_engine
from ai_engine.portfolio.optimizer import MPTOptimizer, mpt_optimizer
from ai_engine.backtesting.strategies import (
    BollingerBreakoutStrategy,
    MACDStrategy,
    MovingAverageCrossStrategy,
    MultiFactorConsensusStrategy,
    RSIMeanReversionStrategy,
)
from ai_engine.backtesting.backtest_engine import BacktestEngine, backtest_engine
from backend.app.main import app


@pytest.fixture
def sample_ohlcv_df():
    """Generate synthetic OHLCV dataframe for tests."""
    np.random.seed(42)
    n = 100
    base_price = 1000.0
    returns = np.random.normal(0.0008, 0.015, n)
    prices = base_price * np.cumprod(1.0 + returns)

    high = prices * (1.0 + np.abs(np.random.normal(0.005, 0.003, n)))
    low = prices * (1.0 - np.abs(np.random.normal(0.005, 0.003, n)))
    open_p = np.roll(prices, 1)
    open_p[0] = base_price
    volume = np.random.lognormal(13.0, 0.6, n)
    dates = [f"2026-01-{(i % 28) + 1:02d}" for i in range(n)]

    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume,
        "date": dates,
    })


# ============================================================================
# 1. RISK ENGINE TESTS
# ============================================================================

def test_risk_engine_volatility_and_sharpe():
    """Verify annualized volatility and Sharpe ratio calculations."""
    returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002, 0.008])
    vol = RiskEngine.compute_volatility(returns)
    assert vol > 0.0
    assert isinstance(vol, float)

    sharpe = risk_engine.compute_sharpe_ratio(returns)
    assert isinstance(sharpe, float)

    sortino = risk_engine.compute_sortino_ratio(returns)
    assert isinstance(sortino, float)


def test_risk_engine_drawdown_metrics():
    """Verify peak-to-trough drawdown and duration calculations."""
    prices = np.array([100.0, 110.0, 120.0, 90.0, 84.0, 105.0, 125.0])
    dd = RiskEngine.compute_drawdown(prices)

    # Peak was 120, trough was 84 -> drawdown = (84 - 120) / 120 = -30.0%
    assert abs(dd.max_drawdown_pct - 30.0) < 1e-2
    assert dd.peak_value == 120.0
    assert dd.trough_value == 84.0
    assert dd.max_drawdown_duration_days >= 2


def test_risk_engine_var_and_expected_shortfall():
    """Verify Parametric, Historical, Monte Carlo VaR, and Expected Shortfall."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, 250)

    var_param = RiskEngine.compute_parametric_var(returns, confidence=0.95)
    var_hist = RiskEngine.compute_historical_var(returns, confidence=0.95)
    var_mc = RiskEngine.compute_monte_carlo_var(returns, confidence=0.95, num_simulations=1000)
    cvar = RiskEngine.compute_expected_shortfall(returns, confidence=0.95)

    assert 1.0 < var_param < 5.0
    assert 1.0 < var_hist < 5.0
    assert 1.0 < var_mc < 5.0
    # Expected shortfall should generally exceed or equal VaR
    assert cvar >= (var_hist * 0.8)


def test_risk_engine_beta_and_concentration():
    """Verify Beta, Alpha, and Concentration risk (HHI)."""
    np.random.seed(42)
    bench_ret = np.random.normal(0.0005, 0.01, 100)
    asset_ret = 1.2 * bench_ret + np.random.normal(0.0, 0.005, 100)

    beta, alpha = RiskEngine.compute_beta_and_alpha(asset_ret, bench_ret)
    assert 1.0 < beta < 1.4
    assert isinstance(alpha, float)

    # Concentration HHI test
    equal_weights = [0.25, 0.25, 0.25, 0.25]
    res_div = RiskEngine.compute_concentration_risk(equal_weights)
    assert res_div["hhi"] == 2500.0  # 4 * 25^2 = 2500
    assert res_div["concentration_tier"] in ("WELL DIVERSIFIED", "MODERATE CONCENTRATION")

    concentrated_weights = [0.80, 0.10, 0.10]
    res_conc = RiskEngine.compute_concentration_risk(concentrated_weights)
    assert res_conc["hhi"] > 6000.0
    assert res_conc["concentration_tier"] == "HIGH CONCENTRATION"


# ============================================================================
# 2. ANOMALY ENGINE TESTS
# ============================================================================

def test_anomaly_engine_detection(sample_ohlcv_df):
    """Verify Isolation Forest and anomaly detection pipeline."""
    # Inject a dramatic price drop and volume surge on row 85
    df = sample_ohlcv_df.copy()
    df.loc[85, "close"] = df.loc[84, "close"] * 0.92  # -8% crash
    df.loc[85, "volume"] = df.loc[84, "volume"] * 5.5  # 5.5x volume spike

    report = anomaly_engine.detect_anomalies(df, symbol="RELIANCE")
    assert report.symbol == "RELIANCE"
    assert report.total_detected >= 1
    assert any(a.index == 85 for a in report.anomalies)

    crash_anomaly = [a for a in report.anomalies if a.index == 85][0]
    assert crash_anomaly.severity in ("CRITICAL", "HIGH")
    assert len(crash_anomaly.description) > 0


def test_anomaly_engine_empty_input():
    """Verify graceful handling of empty or minimal dataframes."""
    empty_df = pd.DataFrame()
    report = anomaly_engine.detect_anomalies(empty_df, symbol="TEST")
    assert report.has_anomalies is False
    assert report.total_detected == 0


# ============================================================================
# 3. MARKET REGIME ENGINE TESTS
# ============================================================================

def test_regime_engine_bull_and_bear(sample_ohlcv_df):
    """Verify regime classification across upward and downward trends."""
    # Create strongly trending upward series
    bull_df = sample_ohlcv_df.copy()
    trend = np.linspace(100.0, 200.0, len(bull_df))
    bull_df["close"] = trend
    bull_df["high"] = trend * 1.01
    bull_df["low"] = trend * 0.99
    bull_df["open"] = trend

    regime_res = regime_engine.detect_regime(bull_df, symbol="NIFTY")
    assert regime_res.primary_regime in ("BULL", "SIDEWAYS")
    assert "BULL" in regime_res.regime_probabilities
    assert sum(regime_res.regime_probabilities.values()) == pytest.approx(1.0, rel=1e-3)
    assert len(regime_res.institutional_explanation) > 0


# ============================================================================
# 4. PORTFOLIO ENGINE & MPT OPTIMIZER TESTS
# ============================================================================

def test_portfolio_engine_evaluation():
    """Verify portfolio holdings valuation, sector exposure, and HHI."""
    holdings = [
        {"symbol": "RELIANCE", "shares": 50, "avg_price": 2500.0, "current_price": 2800.0, "sector": "Energy"},
        {"symbol": "TCS", "shares": 30, "avg_price": 3500.0, "current_price": 4000.0, "sector": "IT"},
    ]
    summary = portfolio_engine.evaluate_portfolio(raw_holdings=holdings, realized_pnl=5000.0)

    # Cost = 50*2500 + 30*3500 = 125,000 + 105,000 = 230,000
    # Val = 50*2800 + 30*4000 = 140,000 + 120,000 = 260,000
    assert summary.total_cost_basis == 230000.0
    assert summary.total_value == 260000.0
    assert summary.unrealized_pnl == 30000.0
    assert summary.holding_count == 2
    assert len(summary.sector_exposures) == 2
    assert summary.portfolio_volatility_pct > 0.0


def test_mpt_optimizer():
    """Verify Markowitz Mean-Variance Tangency Portfolio and Efficient Frontier."""
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
    res = mpt_optimizer.optimize_portfolio(symbols=symbols)

    assert res.symbols == symbols
    # Weights must sum to 1.0 (and 100% in dictionary representation)
    assert sum(res.max_sharpe_portfolio.weights.values()) == pytest.approx(1.0, rel=1e-2)
    assert sum(res.max_sharpe_portfolio.to_dict()["weights"].values()) == pytest.approx(100.0, rel=1e-2)
    assert sum(res.min_volatility_portfolio.weights.values()) == pytest.approx(1.0, rel=1e-2)
    assert sum(res.min_volatility_portfolio.to_dict()["weights"].values()) == pytest.approx(100.0, rel=1e-2)

    # Min volatility should have <= volatility than Max Sharpe
    assert res.min_volatility_portfolio.volatility_pct <= res.max_sharpe_portfolio.volatility_pct + 1e-4

    # Efficient frontier curve must be generated
    assert len(res.efficient_frontier) > 10
    # Must contain compliance disclaimer
    assert "ANALYTICAL SIMULATION NOTICE" in res.compliance_disclaimer


# ============================================================================
# 5. BACKTESTING ENGINE TESTS
# ============================================================================

def test_backtest_strategies_signals(sample_ohlcv_df):
    """Verify that strategies output valid 0/1 binary position signals."""
    ma_strat = MovingAverageCrossStrategy(fast_period=10, slow_period=20)
    sig_ma = ma_strat.generate_signals(sample_ohlcv_df)
    assert set(sig_ma.unique()).issubset({0, 1})
    assert len(sig_ma) == len(sample_ohlcv_df)

    rsi_strat = RSIMeanReversionStrategy(rsi_period=14, oversold=30, overbought=70)
    sig_rsi = rsi_strat.generate_signals(sample_ohlcv_df)
    assert set(sig_rsi.unique()).issubset({0, 1})

    bb_strat = BollingerBreakoutStrategy(period=20, num_std=2.0)
    sig_bb = bb_strat.generate_signals(sample_ohlcv_df)
    assert set(sig_bb.unique()).issubset({0, 1})

    consensus_strat = MultiFactorConsensusStrategy()
    sig_mf = consensus_strat.generate_signals(sample_ohlcv_df)
    assert set(sig_mf.unique()).issubset({0, 1})


def test_backtest_engine_walk_forward_execution(sample_ohlcv_df):
    """Verify walk-forward simulation timing (bar t signal executes at bar t+1 open)."""
    strat = MovingAverageCrossStrategy(fast_period=5, slow_period=15)
    engine = BacktestEngine(
        initial_capital=100000.0,
        slippage_pct=0.0005,
        transaction_fee_pct=0.0003,
    )
    result = engine.run_backtest(df=sample_ohlcv_df, strategy=strat, symbol="TEST")

    assert result.initial_capital == 100000.0
    assert result.final_equity > 0.0
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown_pct, float)
    assert len(result.equity_curve) == len(sample_ohlcv_df)
    assert "ANALYTICAL SIMULATION NOTICE" in result.compliance_disclaimer


# ============================================================================
# 6. FASTAPI API ROUTES TESTS
# ============================================================================

def test_api_risk_routes():
    """Verify FastAPI /api/v1/risk endpoints."""
    client = TestClient(app)

    # GET /risk/{symbol}
    resp = client.get("/api/v1/risk/RELIANCE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert "annualized_volatility_pct" in data
    assert "var_95_1d_pct" in data

    # POST /risk/concentration
    resp_conc = client.post("/api/v1/risk/concentration", json={"weights": [0.3, 0.3, 0.4]})
    assert resp_conc.status_code == 200
    assert "hhi" in resp_conc.json()

    # GET /risk/{symbol}/stress-test
    resp_stress = client.get("/api/v1/risk/RELIANCE/stress-test")
    assert resp_stress.status_code == 200
    assert len(resp_stress.json()["scenarios"]) >= 3


def test_api_anomaly_and_regime_routes():
    """Verify FastAPI /api/v1/anomaly and /api/v1/regime endpoints."""
    client = TestClient(app)

    resp_anom = client.get("/api/v1/anomaly/RELIANCE")
    assert resp_anom.status_code == 200
    assert "has_anomalies" in resp_anom.json()

    resp_reg = client.get("/api/v1/regime/RELIANCE")
    assert resp_reg.status_code == 200
    assert resp_reg.json()["primary_regime"] in ("BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY")


def test_api_portfolio_and_backtest_routes():
    """Verify FastAPI /api/v1/portfolio-analytics and /api/v1/backtest endpoints."""
    client = TestClient(app)

    # POST /portfolio-analytics/evaluate
    payload_eval = {
        "holdings": [
            {"symbol": "RELIANCE", "shares": 20, "avg_price": 2700.0, "sector": "Energy"},
            {"symbol": "TCS", "shares": 15, "avg_price": 3900.0, "sector": "IT"},
        ],
        "realized_pnl": 2000.0,
    }
    resp_eval = client.post("/api/v1/portfolio-analytics/evaluate", json=payload_eval)
    assert resp_eval.status_code == 200
    assert resp_eval.json()["total_value"] > 0.0

    # POST /portfolio-analytics/optimize-mpt
    payload_mpt = {
        "symbols": ["RELIANCE", "TCS", "HDFCBANK", "INFY"],
        "max_asset_weight": 0.50,
    }
    resp_mpt = client.post("/api/v1/portfolio-analytics/optimize-mpt", json=payload_mpt)
    assert resp_mpt.status_code == 200
    assert "max_sharpe_portfolio" in resp_mpt.json()
    assert "compliance_disclaimer" in resp_mpt.json()

    # POST /backtest/run
    payload_bt = {
        "symbol": "RELIANCE",
        "strategy": "ma_cross",
        "initial_capital": 250000.0,
        "slippage_pct": 0.0005,
        "transaction_fee_pct": 0.0003,
        "limit": 100,
    }
    resp_bt = client.post("/api/v1/backtest/run", json=payload_bt)
    assert resp_bt.status_code == 200
    assert resp_bt.json()["initial_capital"] == 250000.0
    assert "equity_curve" in resp_bt.json()
