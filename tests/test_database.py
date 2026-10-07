"""Database connection and model persistence tests."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import check_db_connection
from backend.app.models.user import User
from backend.app.models.watchlist import Watchlist
from backend.app.models.portfolio import Portfolio, PortfolioPosition
from backend.app.security import hash_password


@pytest.mark.asyncio
async def test_database_connection():
    """Verify that database connection check returns True."""
    connected = await check_db_connection()
    assert connected is True


@pytest.mark.asyncio
async def test_create_and_query_user(db_session: AsyncSession):
    """Test inserting and retrieving a User record."""
    user = User(
        email="trader@stockmind.ai",
        username="trader_alpha",
        hashed_password=hash_password("SecureTraderPass123!"),
        full_name="Alpha Trader",
    )
    db_session.add(user)
    await db_session.flush()

    assert user.id is not None
    assert user.is_active is True
    assert user.created_at is not None

    stmt = select(User).where(User.username == "trader_alpha")
    result = await db_session.execute(stmt)
    queried_user = result.scalar_one_or_none()

    assert queried_user is not None
    assert queried_user.email == "trader@stockmind.ai"


@pytest.mark.asyncio
async def test_watchlist_relationship(db_session: AsyncSession):
    """Test Watchlist creation linked to a User with JSON symbols."""
    user = User(
        email="investor@stockmind.ai",
        username="investor_joe",
        hashed_password=hash_password("Pass123456!"),
    )
    db_session.add(user)
    await db_session.flush()

    watchlist = Watchlist(
        user_id=user.id,
        name="Tech Giants",
        description="High momentum tech stocks",
    )
    watchlist.symbols = ["AAPL", "NVDA", "MSFT", "GOOGL"]
    db_session.add(watchlist)
    await db_session.flush()

    assert watchlist.id is not None
    assert "AAPL" in watchlist.symbols
    assert len(watchlist.symbols) == 4


@pytest.mark.asyncio
async def test_portfolio_and_positions(db_session: AsyncSession):
    """Test Portfolio and associated PortfolioPosition cascade."""
    user = User(
        email="fund@stockmind.ai",
        username="fund_manager",
        hashed_password=hash_password("Secret12345!"),
    )
    db_session.add(user)
    await db_session.flush()

    portfolio = Portfolio(
        user_id=user.id,
        name="Growth Fund",
        initial_cash=500000.0,
        cash_balance=450000.0,
        currency="USD",
    )
    db_session.add(portfolio)
    await db_session.flush()

    pos = PortfolioPosition(
        portfolio_id=portfolio.id,
        symbol="NVDA",
        quantity=50.0,
        average_buy_price=800.0,
    )
    db_session.add(pos)
    await db_session.flush()

    assert pos.id is not None
    assert pos.portfolio_id == portfolio.id
    assert pos.symbol == "NVDA"
