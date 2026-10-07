"""Portfolio and PortfolioPosition schemas."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PortfolioPositionBase(BaseModel):
    """Base position attributes."""
    symbol: str = Field(..., max_length=20, description="Asset ticker symbol")
    quantity: float = Field(..., ge=0.0, description="Number of shares held")
    average_buy_price: float = Field(..., ge=0.0, description="Cost basis per share")


class PortfolioPositionCreate(PortfolioPositionBase):
    """Payload to add or modify a position."""
    pass


class PortfolioPositionResponse(PortfolioPositionBase):
    """Position response representation."""
    id: int
    portfolio_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PortfolioBase(BaseModel):
    """Base portfolio attributes."""
    name: str = Field(..., min_length=1, max_length=100, description="Portfolio name")
    description: Optional[str] = Field(default=None, max_length=255, description="Portfolio description")
    initial_cash: float = Field(default=100000.0, ge=0.0, description="Starting cash capital")
    currency: str = Field(default="USD", max_length=10, description="Portfolio base currency")


class PortfolioCreate(PortfolioBase):
    """Payload for creating a new portfolio."""
    pass


class PortfolioResponse(PortfolioBase):
    """Portfolio public response schema."""
    id: int
    user_id: int
    cash_balance: float
    positions: List[PortfolioPositionResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
