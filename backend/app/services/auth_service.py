"""Authentication business logic services."""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user import User
from backend.app.security import verify_password
from backend.app.services.user_service import get_user_by_username_or_email


async def authenticate_user(
    db: AsyncSession,
    username_or_email: str,
    password: str,
) -> Optional[User]:
    """Validate user credentials against stored password hash.

    Args:
        db: Active async database session.
        username_or_email: User-supplied username or email string.
        password: User-supplied plaintext password.

    Returns:
        User object if credentials valid, None otherwise.
    """
    user = await get_user_by_username_or_email(db, identifier=username_or_email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
