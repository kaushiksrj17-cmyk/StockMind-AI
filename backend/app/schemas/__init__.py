"""Schemas package initialization."""

from backend.app.schemas.auth import Token, TokenData, LoginRequest
from backend.app.schemas.user import UserBase, UserCreate, UserUpdate, UserResponse
from backend.app.schemas.watchlist import (
    WatchlistBase,
    WatchlistCreate,
    WatchlistUpdate,
    WatchlistResponse,
)
from backend.app.schemas.portfolio import (
    PortfolioBase,
    PortfolioCreate,
    PortfolioResponse,
    PortfolioPositionBase,
    PortfolioPositionCreate,
    PortfolioPositionResponse,
)
from backend.app.schemas.technical import (
    CandlestickPatternItem,
    AITechnicalScoreSummary,
    MovingAveragesSummary,
    SupportResistanceSummary,
    FibonacciSummary,
    TechnicalAnalysisResponse,
    TechnicalScoreOnlyResponse,
    TechnicalIndicatorsOnlyResponse,
)
from backend.app.schemas.ml import (
    ExpectedMovement,
    ModelMeta,
    MLPredictionResponse,
    MLModelComparisonItem,
    MLModelComparisonResponse,
    MLModelStatusResponse,
    MLTrainRequest,
    MLTrainResponse,
)

__all__ = [
    "Token",
    "TokenData",
    "LoginRequest",
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "WatchlistBase",
    "WatchlistCreate",
    "WatchlistUpdate",
    "WatchlistResponse",
    "PortfolioBase",
    "PortfolioCreate",
    "PortfolioResponse",
    "PortfolioPositionBase",
    "PortfolioPositionCreate",
    "PortfolioPositionResponse",
    "CandlestickPatternItem",
    "AITechnicalScoreSummary",
    "MovingAveragesSummary",
    "SupportResistanceSummary",
    "FibonacciSummary",
    "TechnicalAnalysisResponse",
    "TechnicalScoreOnlyResponse",
    "TechnicalIndicatorsOnlyResponse",
    "ExpectedMovement",
    "ModelMeta",
    "MLPredictionResponse",
    "MLModelComparisonItem",
    "MLModelComparisonResponse",
    "MLModelStatusResponse",
    "MLTrainRequest",
    "MLTrainResponse",
]
