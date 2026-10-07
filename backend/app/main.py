"""StockMind-AI FastAPI Application Entrypoint.

Configures application lifecycle, CORS, structured error handling, logging,
versioned API routing, and WebSocket foundation.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import api_router
from backend.app.config import settings
from backend.app.database import check_db_connection, init_db
from backend.app.services.market import market_service, market_ws_manager


# ------------------------------------------------------------------------------
# 1. Structured Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stockmind.backend")


# ------------------------------------------------------------------------------
# 2. Application Lifespan (Startup & Shutdown)
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle management."""
    logger.info("Initializing StockMind-AI Backend (version %s)...", settings.APP_VERSION)
    logger.info("Environment: %s | Debug: %s", settings.ENVIRONMENT, settings.DEBUG)
    logger.info("Database URL configured: %s", settings.async_database_url.split("@")[-1])

    # Ensure database schema is initialized
    try:
        await init_db()
        logger.info("Database schema verified and initialized.")
    except Exception as exc:
        logger.error("Failed to initialize database tables: %s", exc)

    db_ok = await check_db_connection()
    logger.info("Database connectivity check: %s", "SUCCESS" if db_ok else "FAILED")

    # Start Market Data Engine
    try:
        await market_service.start()
        logger.info(
            "Market Data Engine running in %s mode (Provider: %s).",
            market_service.data_mode.value,
            market_service.provider.__class__.__name__,
        )
    except Exception as exc:
        logger.error("Failed to start market service: %s", exc)

    yield

    logger.info("Shutting down StockMind-AI Backend services...")
    try:
        await market_service.stop()
        logger.info("Market data service stopped cleanly.")
    except Exception as exc:
        logger.error("Error stopping market service: %s", exc)


# ------------------------------------------------------------------------------
# 3. FastAPI Application Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "StockMind-AI Institutional Market Intelligence & Quantitative Analytics Platform API. "
        "Provides authentication, real-time tickers, portfolio tracking, and AI modeling services."
    ),
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan,
)


# ------------------------------------------------------------------------------
# 4. CORS Middleware
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# 5. Structured Error Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions with structured response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request payload validation failures."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "message": "Validation Error",
            "errors": exc.errors(),
            "path": request.url.path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Fallback handler for unhandled exceptions."""
    logger.exception("Unhandled server exception occurred: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Internal server error occurred.",
            "path": request.url.path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ------------------------------------------------------------------------------
# 6. Global & Versioned Routes
# ------------------------------------------------------------------------------
@app.get("/", summary="Root Index", tags=["General"])
async def root():
    """Service metadata and documentation pointers."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "market_mode": market_service.data_mode.value,
        "is_live_data": market_service.data_mode == "LIVE",
        "docs_url": f"{settings.API_V1_PREFIX}/docs",
        "health_url": "/health",
        "api_v1_url": settings.API_V1_PREFIX,
        "ws_url": "/ws",
    }


@app.get("/health", summary="Top-level Health Endpoint", tags=["General"])
async def top_level_health():
    """Convenience alias for top-level health checking."""
    db_ok = await check_db_connection()
    payload = {
        "status": "healthy" if db_ok else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "connected" if db_ok else "disconnected",
        "market_engine": {
            "mode": market_service.data_mode.value,
            "connected": market_service.is_connected,
            "is_live": market_service.data_mode == "LIVE",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    status_code = status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)


# Mount Versioned API Routes (/api/v1)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# ------------------------------------------------------------------------------
# 7. Real-Time Market WebSocket Foundation
# ------------------------------------------------------------------------------
@app.websocket("/ws")
@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint for ticker streams, market updates, and alerts."""
    await market_ws_manager.connect(websocket)

    # Send welcome / connection acknowledgment with explicit provenance
    await market_ws_manager.send_personal_message(
        {
            "type": "connection_ack",
            "message": "Connected to StockMind-AI real-time market stream.",
            "data_mode": market_service.data_mode.value,
            "is_live": market_service.data_mode == "LIVE",
            "disclaimer": (
                "VERIFIED LIVE MARKET DATA"
                if market_service.data_mode == "LIVE"
                else "SIMULATED / REPLAY DATA. NOT LIVE MARKET FEED."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        websocket,
    )

    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                await market_ws_manager.send_personal_message(
                    {"type": "error", "message": "Payload must be valid JSON."},
                    websocket,
                )
                continue

            action = data.get("action", "").lower()

            if action == "ping":
                await market_ws_manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()},
                    websocket,
                )
            elif action == "subscribe":
                raw_symbols = data.get("symbols", [])
                if not raw_symbols and "symbol" in data:
                    raw_symbols = [data["symbol"]]

                clean_symbols = [s.strip().upper() for s in raw_symbols if s and s.strip()]
                if clean_symbols:
                    await market_ws_manager.subscribe(websocket, clean_symbols)
                    # Instantly push current cached snapshots for subscribed tickers
                    for s in clean_symbols:
                        snapshot = await market_service.get_quote(s)
                        if snapshot:
                            await market_ws_manager.send_personal_message(
                                {
                                    "type": "tick",
                                    "data": snapshot.model_dump(mode="json"),
                                },
                                websocket,
                            )
            elif action == "unsubscribe":
                raw_symbols = data.get("symbols", [])
                if not raw_symbols and "symbol" in data:
                    raw_symbols = [data["symbol"]]
                clean_symbols = [s.strip().upper() for s in raw_symbols if s and s.strip()]
                if clean_symbols:
                    await market_ws_manager.unsubscribe(websocket, clean_symbols)
            else:
                await market_ws_manager.send_personal_message(
                    {
                        "type": "echo",
                        "received": data,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    websocket,
                )
    except WebSocketDisconnect:
        market_ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.error("WebSocket exception: %s", exc)
        market_ws_manager.disconnect(websocket)
