"""Authentication request and response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class Token(BaseModel):
    """JWT Token response schema."""
    access_token: str = Field(..., description="Signed JWT access token")
    token_type: str = Field(default="bearer", description="Token scheme type")


class TokenData(BaseModel):
    """Extracted JWT claims payload."""
    sub: Optional[str] = None
    email: Optional[str] = None


class LoginRequest(BaseModel):
    """User login credential payload."""
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., min_length=6, description="User password")
