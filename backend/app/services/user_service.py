"""User persistence and query services."""

from typing import Optional
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user import User
from backend.app.schemas.user import UserCreate
from backend.app.security import hash_password


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Retrieve user by primary key ID."""
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Retrieve user by normalized email address."""
    stmt = select(User).where(User.email == email.lower().strip())
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Retrieve user by username."""
    stmt = select(User).where(User.username == username.strip())
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_username_or_email(
    db: AsyncSession,
    identifier: str,
) -> Optional[User]:
    """Retrieve user by username or email address."""
    clean_id = identifier.strip().lower()
    stmt = select(User).where(
        or_(
            User.email == clean_id,
            User.username == identifier.strip(),
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, user_in: UserCreate) -> User:
    """Create and persist a new user record with hashed password."""
    user = User(
        email=user_in.email.lower().strip(),
        username=user_in.username.strip(),
        full_name=user_in.full_name,
        hashed_password=hash_password(user_in.password),
        is_active=True,
        is_superuser=False,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user
