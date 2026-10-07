"""Portfolio management API routes."""

from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.models.portfolio import Portfolio, PortfolioPosition
from backend.app.schemas.portfolio import (
    PortfolioCreate,
    PortfolioPositionCreate,
    PortfolioPositionResponse,
    PortfolioResponse,
)

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


@router.get("", response_model=List[PortfolioResponse], summary="List user portfolios")
async def list_portfolios(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Retrieve all portfolios belonging to the authenticated user."""
    stmt = select(Portfolio).where(Portfolio.user_id == current_user.id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post(
    "",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new portfolio",
)
async def create_portfolio(
    portfolio_in: PortfolioCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new investment portfolio."""
    portfolio = Portfolio(
        user_id=current_user.id,
        name=portfolio_in.name,
        description=portfolio_in.description,
        initial_cash=portfolio_in.initial_cash,
        cash_balance=portfolio_in.initial_cash,
        currency=portfolio_in.currency,
    )
    db.add(portfolio)
    await db.flush()
    await db.refresh(portfolio)
    return portfolio


@router.get("/{portfolio_id}", response_model=PortfolioResponse, summary="Get portfolio by ID")
async def get_portfolio(
    portfolio_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Retrieve detailed portfolio metrics and positions."""
    stmt = select(Portfolio).where(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found.",
        )
    return portfolio


@router.post(
    "/{portfolio_id}/positions",
    response_model=PortfolioPositionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a stock position to portfolio",
)
async def add_position(
    portfolio_id: int,
    position_in: PortfolioPositionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Add or increment a stock holding in the specified portfolio."""
    stmt = select(Portfolio).where(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found.",
        )

    position = PortfolioPosition(
        portfolio_id=portfolio.id,
        symbol=position_in.symbol.strip().upper(),
        quantity=position_in.quantity,
        average_buy_price=position_in.average_buy_price,
    )
    db.add(position)
    await db.flush()
    await db.refresh(position)
    return position


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete portfolio")
async def delete_portfolio(
    portfolio_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Delete a portfolio and all associated positions."""
    stmt = select(Portfolio).where(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found.",
        )
    await db.delete(portfolio)
    return None
