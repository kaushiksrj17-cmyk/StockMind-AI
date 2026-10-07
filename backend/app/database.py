"""Database Connection & Session Management Layer.

Provides asynchronous SQLAlchemy 2.0 engine, declarative base, session factory,
and connectivity verification with PostgreSQL readiness and local SQLite fallback.
"""

from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.app.config import settings


class Base(AsyncAttrs, DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


# Database engine configuration
engine_kwargs = {"echo": False}
if "sqlite" in settings.async_database_url:
    engine_kwargs["connect_args"] = {"check_same_thread": False}

async_engine: AsyncEngine = create_async_engine(
    settings.async_database_url,
    **engine_kwargs,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for obtaining an asynchronous database session.

    Yields:
        AsyncSession: Active database session with automated rollback on exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables from registered SQLAlchemy models."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_db_connection() -> bool:
    """Execute a simple query to verify database connectivity.

    Returns:
        bool: True if connection succeeds, False otherwise.
    """
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception:
        return False
