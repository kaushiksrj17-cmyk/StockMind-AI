"""Health check API endpoints."""

from datetime import datetime, timezone
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from backend.app.config import BASE_DIR, settings
from backend.app.database import check_db_connection

router = APIRouter(tags=["Health"])


@router.get("/health", summary="System Health Status")
async def health_check():
    """Verify application health and database connection status."""
    db_ok = await check_db_connection()

    payload = {
        "status": "healthy" if db_ok else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status_code = status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/health/detailed", summary="Comprehensive Subsystem Health & USB Audit")
async def detailed_health_check():
    """Comprehensive health check across database, market provider, AI modules, and USB storage."""
    db_ok = await check_db_connection()
    config_audit = settings.validate_configuration()

    # Check USB filesystem paths
    from pathlib import Path
    base_path = Path(BASE_DIR)
    db_file = base_path / settings.SQLITE_DB_PATH
    if not db_file.exists():
        alt_db = base_path / "database" / "stockmind.db"
        if alt_db.exists():
            db_file = alt_db
    models_dir = base_path / "models"
    logs_dir = base_path / settings.LOG_DIR

    usb_storage = {
        "root": str(base_path),
        "db_exists": db_file.exists(),
        "db_size_kb": round(db_file.stat().st_size / 1024, 2) if db_file.exists() else 0.0,
        "models_dir_exists": models_dir.exists(),
        "logs_dir_exists": logs_dir.exists(),
    }

    # Inspect AI Modules availability
    ai_status = {}
    try:
        from ai_engine.ml.model_registry import model_registry
        ai_status["ml_models_registered"] = len(model_registry.list_models())
    except Exception as exc:
        ai_status["ml_models_registered"] = f"error: {exc}"

    try:
        from ai_engine.deep_learning.predict import hybrid_ensemble
        ai_status["deep_learning"] = "ready"
    except Exception as exc:
        ai_status["deep_learning"] = f"error: {exc}"

    try:
        from ai_engine.nlp.sentiment import sentiment_analyzer
        ai_status["nlp_sentiment"] = "ready"
    except Exception as exc:
        ai_status["nlp_sentiment"] = f"error: {exc}"

    try:
        from ai_engine.signals.signal_engine import ai_signal_engine
        ai_status["signal_engine"] = "ready"
    except Exception as exc:
        ai_status["signal_engine"] = f"error: {exc}"

    try:
        from ai_engine.explainability.shap_explainer import shap_explainer
        ai_status["shap_explainer"] = "ready"
    except Exception as exc:
        ai_status["shap_explainer"] = f"error: {exc}"

    # Market service state
    from backend.app.services.market import market_service
    market_info = {
        "mode": market_service.data_mode.value,
        "is_live": market_service.data_mode.value == "LIVE",
        "provider": market_service.provider.__class__.__name__,
        "connected": market_service.is_connected,
    }

    overall_healthy = db_ok and config_audit["is_usb_confined"]

    payload = {
        "status": "healthy" if overall_healthy else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": {
            "status": "connected" if db_ok else "disconnected",
            "type": settings.DATABASE_TYPE,
            "url": settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "sqlite:local",
        },
        "market_engine": market_info,
        "ai_subsystems": ai_status,
        "usb_storage": usb_storage,
        "config_audit": config_audit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status_code = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)

