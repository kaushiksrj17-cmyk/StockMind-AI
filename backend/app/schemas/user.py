"""User request and response schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    """Base user attributes."""
    email: EmailStr = Field(..., description="User unique email address")
    username: str = Field(..., min_length=3, max_length=64, description="Unique username")
    full_name: Optional[str] = Field(default=None, max_length=128, description="User full display name")


class UserCreate(UserBase):
    """Payload for registering a new user."""
    password: str = Field(..., min_length=8, description="User password (minimum 8 characters)")


class UserUpdate(BaseModel):
    """Payload for updating user profile."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=8)


class UserResponse(UserBase):
    """Public user response schema."""
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
