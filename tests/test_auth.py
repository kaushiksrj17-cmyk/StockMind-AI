"""Authentication and protected endpoint tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_user_success(client: AsyncClient):
    """Test successful user registration."""
    payload = {
        "email": "newuser@stockmind.ai",
        "username": "newtrader",
        "password": "StrongPassword123!",
        "full_name": "New Trader",
    }
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@stockmind.ai"
    assert data["username"] == "newtrader"
    assert data["is_active"] is True
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    """Test rejection when registering with an existing email."""
    payload = {
        "email": "dupe@stockmind.ai",
        "username": "user_one",
        "password": "Password123!",
    }
    res1 = await client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    payload_dupe = {
        "email": "dupe@stockmind.ai",
        "username": "user_two",
        "password": "Password123!",
    }
    res2 = await client.post("/api/v1/auth/register", json=payload_dupe)
    assert res2.status_code == 400
    assert "email address already exists" in res2.json()["message"]


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient):
    """Test rejection when registering with an existing username."""
    payload = {
        "email": "first@stockmind.ai",
        "username": "same_username",
        "password": "Password123!",
    }
    res1 = await client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    payload_dupe = {
        "email": "second@stockmind.ai",
        "username": "same_username",
        "password": "Password123!",
    }
    res2 = await client.post("/api/v1/auth/register", json=payload_dupe)
    assert res2.status_code == 400
    assert "username already exists" in res2.json()["message"]


@pytest.mark.asyncio
async def test_login_and_access_protected(client: AsyncClient):
    """Test registration, login, and accessing protected endpoint with Bearer token."""
    # 1. Register
    reg_payload = {
        "email": "authuser@stockmind.ai",
        "username": "authuser",
        "password": "SecretPassword123!",
    }
    await client.post("/api/v1/auth/register", json=reg_payload)

    # 2. Login via JSON
    login_payload = {
        "username_or_email": "authuser",
        "password": "SecretPassword123!",
    }
    login_res = await client.post("/api/v1/auth/login", json=login_payload)
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    token = token_data["access_token"]

    # 3. Access /auth/me
    headers = {"Authorization": f"Bearer {token}"}
    me_res = await client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["username"] == "authuser"
    assert me_data["email"] == "authuser@stockmind.ai"

    # 4. Access protected test endpoint
    protected_res = await client.get("/api/v1/protected/test", headers=headers)
    assert protected_res.status_code == 200
    prot_data = protected_res.json()
    assert prot_data["authenticated"] is True
    assert prot_data["username"] == "authuser"


@pytest.mark.asyncio
async def test_protected_endpoint_without_token(client: AsyncClient):
    """Verify that protected endpoints reject unauthenticated requests with 401."""
    response = await client.get("/api/v1/protected/test")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_invalid_password(client: AsyncClient):
    """Verify that login fails with wrong password."""
    reg_payload = {
        "email": "wrongpwd@stockmind.ai",
        "username": "wrongpwd",
        "password": "CorrectPassword123!",
    }
    await client.post("/api/v1/auth/register", json=reg_payload)

    login_payload = {
        "username_or_email": "wrongpwd",
        "password": "IncorrectPassword999!",
    }
    res = await client.post("/api/v1/auth/login", json=login_payload)
    assert res.status_code == 401
