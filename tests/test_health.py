"""System and API health endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_top_level_health(client: AsyncClient):
    """Test top-level /health endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_versioned_health(client: AsyncClient):
    """Test versioned /api/v1/health endpoint."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root / metadata endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "StockMind-AI"
    assert "/docs" in data["docs_url"]
