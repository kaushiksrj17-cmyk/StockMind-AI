"""Protected test routes verifying authentication and security scopes."""

from datetime import datetime, timezone
from typing import Annotated
from fastapi import APIRouter, Depends

from backend.app.dependencies import get_current_user
from backend.app.models.user import User

router = APIRouter(prefix="/protected", tags=["Protected Verification"])


@router.get("/test", summary="Verify JWT Authentication")
async def protected_test(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Protected endpoint requiring a valid JWT Bearer token."""
    return {
        "message": f"Hello, {current_user.username}! Authentication successful.",
        "authenticated": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "is_active": current_user.is_active,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
