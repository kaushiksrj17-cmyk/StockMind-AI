"""API Routes initialization and aggregation."""

from fastapi import APIRouter
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.protected import router as protected_router
from backend.app.api.routes.watchlist import router as watchlist_router
from backend.app.api.routes.portfolio import router as portfolio_router
from backend.app.api.routes.market import router as market_router
from backend.app.api.routes.technical import router as technical_router
from backend.app.api.routes.ml import router as ml_router
from backend.app.api.routes.nlp import router as nlp_router
from backend.app.api.routes.copilot import router as copilot_router
from backend.app.api.routes.risk import router as risk_router
from backend.app.api.routes.anomaly import router as anomaly_router
from backend.app.api.routes.regime import router as regime_router
from backend.app.api.routes.portfolio_analytics import router as portfolio_analytics_router
from backend.app.api.routes.backtest import router as backtest_router
from backend.app.api.routes.signals import router as signals_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(protected_router)
api_router.include_router(watchlist_router)
api_router.include_router(portfolio_router)
api_router.include_router(market_router)
api_router.include_router(technical_router)
api_router.include_router(ml_router)
api_router.include_router(nlp_router)
api_router.include_router(copilot_router)
api_router.include_router(risk_router)
api_router.include_router(anomaly_router)
api_router.include_router(regime_router)
api_router.include_router(portfolio_analytics_router)
api_router.include_router(backtest_router)
api_router.include_router(signals_router)

__all__ = ["api_router"]
