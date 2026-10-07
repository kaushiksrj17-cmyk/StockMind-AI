"""Portfolio and Position SQLAlchemy ORM Models."""

from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base
from backend.app.models.base import TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.user import User


class Portfolio(Base, TimestampMixin):
    """Investment portfolio entity tracking capital and asset allocations."""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    initial_cash: Mapped[float] = mapped_column(Float, default=100000.0, nullable=False)
    cash_balance: Mapped[float] = mapped_column(Float, default=100000.0, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="portfolios")
    positions: Mapped[List["PortfolioPosition"]] = relationship(
        "PortfolioPosition",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class PortfolioPosition(Base, TimestampMixin):
    """Stock holding position within a specific portfolio."""

    __tablename__ = "portfolio_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    average_buy_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Relationship
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="positions")
