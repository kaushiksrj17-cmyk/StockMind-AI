"""Watchlist schemas."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class WatchlistBase(BaseModel):
    """Base attributes for stock watchlist."""
    name: str = Field(..., min_length=1, max_length=100, description="Watchlist name")
    description: Optional[str] = Field(default=None, max_length=255, description="Watchlist description")
    symbols: List[str] = Field(default_factory=list, description="List of stock ticker symbols")


class WatchlistCreate(WatchlistBase):
    """Payload for creating a new watchlist."""
    pass


class WatchlistUpdate(BaseModel):
    """Payload for updating an existing watchlist."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=255)
    symbols: Optional[List[str]] = None


class WatchlistResponse(WatchlistBase):
    """Public response representation of a watchlist."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
