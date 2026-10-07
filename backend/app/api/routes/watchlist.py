"""Watchlist management API routes."""

from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.models.watchlist import Watchlist
from backend.app.schemas.watchlist import (
    WatchlistCreate,
    WatchlistResponse,
    WatchlistUpdate,
)

router = APIRouter(prefix="/watchlists", tags=["Watchlists"])


@router.get("", response_model=List[WatchlistResponse], summary="List user watchlists")
async def list_watchlists(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Retrieve all watchlists owned by the authenticated user."""
    stmt = select(Watchlist).where(Watchlist.user_id == current_user.id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post(
    "",
    response_model=WatchlistResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new watchlist",
)
async def create_watchlist(
    watchlist_in: WatchlistCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new watchlist of ticker symbols for the current user."""
    watchlist = Watchlist(
        user_id=current_user.id,
        name=watchlist_in.name,
        description=watchlist_in.description,
    )
    watchlist.symbols = watchlist_in.symbols
    db.add(watchlist)
    await db.flush()
    await db.refresh(watchlist)
    return watchlist


@router.get("/{watchlist_id}", response_model=WatchlistResponse, summary="Get watchlist by ID")
async def get_watchlist(
    watchlist_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Retrieve details for a specific watchlist."""
    stmt = select(Watchlist).where(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    watchlist = result.scalar_one_or_none()
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found.",
        )
    return watchlist


@router.put("/{watchlist_id}", response_model=WatchlistResponse, summary="Update watchlist")
async def update_watchlist(
    watchlist_id: int,
    watchlist_in: WatchlistUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update a watchlist's name, description, or monitored tickers."""
    stmt = select(Watchlist).where(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    watchlist = result.scalar_one_or_none()
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found.",
        )

    if watchlist_in.name is not None:
        watchlist.name = watchlist_in.name
    if watchlist_in.description is not None:
        watchlist.description = watchlist_in.description
    if watchlist_in.symbols is not None:
        watchlist.symbols = watchlist_in.symbols

    await db.flush()
    await db.refresh(watchlist)
    return watchlist


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete watchlist")
async def delete_watchlist(
    watchlist_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Delete a watchlist."""
    stmt = select(Watchlist).where(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    watchlist = result.scalar_one_or_none()
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found.",
        )
    await db.delete(watchlist)
    return None
