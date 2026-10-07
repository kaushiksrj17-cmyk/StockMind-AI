"""Watchlist SQLAlchemy ORM Model."""

import json
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base
from backend.app.models.base import TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.user import User


class Watchlist(Base, TimestampMixin):
    """User-created watchlist for monitoring specific stock tickers."""

    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    _symbols: Mapped[str] = mapped_column("symbols", Text, default="[]", nullable=False)

    # Relationship
    user: Mapped["User"] = relationship("User", back_populates="watchlists")

    @property
    def symbols(self) -> List[str]:
        """Deserialize JSON-encoded ticker symbols list."""
        try:
            return json.loads(self._symbols)
        except Exception:
            return []

    @symbols.setter
    def symbols(self, val: List[str]) -> None:
        """Serialize list of tickers to JSON string."""
        clean_symbols = sorted(list({s.strip().upper() for s in val if s and s.strip()}))
        self._symbols = json.dumps(clean_symbols)
