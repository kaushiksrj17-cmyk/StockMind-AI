"""Watchlist and Portfolio endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_watchlist_and_portfolio_lifecycle(client: AsyncClient):
    """Test full lifecycle of watchlists and portfolios under authenticated user."""
    # 1. Register and Login
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "investor_x@stockmind.ai",
            "username": "investor_x",
            "password": "SecurePassword123!",
        },
    )
    login_res = await client.post(
        "/api/v1/auth/login",
        json={
            "username_or_email": "investor_x",
            "password": "SecurePassword123!",
        },
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create Watchlist
    wl_payload = {
        "name": "Semiconductors",
        "description": "Leading chip designers and foundries",
        "symbols": ["NVDA", "TSM", "AMD"],
    }
    wl_res = await client.post("/api/v1/watchlists", json=wl_payload, headers=headers)
    assert wl_res.status_code == 201
    wl_data = wl_res.json()
    assert wl_data["name"] == "Semiconductors"
    assert "NVDA" in wl_data["symbols"]
    watchlist_id = wl_data["id"]

    # 3. List Watchlists
    list_wl_res = await client.get("/api/v1/watchlists", headers=headers)
    assert list_wl_res.status_code == 200
    assert len(list_wl_res.json()) >= 1

    # 4. Create Portfolio
    pf_payload = {
        "name": "Alpha Quantum Fund",
        "description": "Quantitative momentum strategy",
        "initial_cash": 250000.0,
        "currency": "USD",
    }
    pf_res = await client.post("/api/v1/portfolios", json=pf_payload, headers=headers)
    assert pf_res.status_code == 201
    pf_data = pf_res.json()
    assert pf_data["name"] == "Alpha Quantum Fund"
    assert pf_data["cash_balance"] == 250000.0
    portfolio_id = pf_data["id"]

    # 5. Add Position to Portfolio
    pos_payload = {
        "symbol": "NVDA",
        "quantity": 100.0,
        "average_buy_price": 850.50,
    }
    pos_res = await client.post(
        f"/api/v1/portfolios/{portfolio_id}/positions",
        json=pos_payload,
        headers=headers,
    )
    assert pos_res.status_code == 201
    pos_data = pos_res.json()
    assert pos_data["symbol"] == "NVDA"
    assert pos_data["quantity"] == 100.0
