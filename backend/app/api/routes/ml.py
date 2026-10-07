"""Machine Learning REST API Endpoints.

Provides APIs for:
- Out-of-sample forward direction prediction and probability confidence
- Expected return and statistically bounded price range
- Model comparison across candidate architectures (Random Forest, Logistic Regression, SVM, Gradient Boosting)
- Model registry status and versioned tracking
- On-demand model training
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
import pandas as pd

from ai_engine.ml.predict import ml_predictor, PREDICTION_DISCLAIMER
from ai_engine.ml.model_registry import model_registry
from ai_engine.ml.train import train_full_ml_pipeline
from backend.app.schemas.ml import (
    MLPredictionResponse,
    MLModelComparisonResponse,
    MLModelComparisonItem,
    MLModelStatusResponse,
    MLTrainRequest,
    MLTrainResponse,
    DLPredictionResponse,
    ModelVote,
    EnsembleConsensusResponse,
    DLTrainRequest,
    DLTrainResponse,
)
from backend.app.services.market.market_service import market_service

router = APIRouter(prefix="/ml", tags=["Machine Learning Engine"])


async def _get_historical_df(symbol: str, timeframe: str = "1d", limit: int = 200) -> pd.DataFrame:
    """Helper to fetch historical OHLCV data from market_service and format as DataFrame."""
    bars = await market_service.get_historical_ohlc(symbol=symbol, timeframe=timeframe, limit=limit)
    if not bars or len(bars) < 30:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insufficient historical data available for '{symbol}'. Need at least 30 complete bars.",
        )
    records = [
        b.model_dump() if hasattr(b, "model_dump") else (b.dict() if hasattr(b, "dict") else b.__dict__)
        for b in bars
    ]
    return pd.DataFrame(records)


@router.get(
    "/predict/{symbol}",
    response_model=MLPredictionResponse,
    summary="Predict Next-Period Direction & Expected Movement",
)
async def predict_next_period(
    symbol: str,
    timeframe: str = Query(default="1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
    model_type: Optional[str] = Query(default=None, description="Optional preferred model: Ensemble, RandomForest, SVM, etc."),
):
    """Generate time-series safe directional prediction with confidence score and price targets.

    Clearly labeled as statistical analytical estimates and not guaranteed market outcomes.
    """
    df = await _get_historical_df(symbol=symbol, timeframe=timeframe, limit=200)
    try:
        prediction = ml_predictor.predict_next_period(
            df=df,
            symbol=symbol,
            preferred_model_type=model_type,
        )
        return prediction
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Machine learning inference error: {str(exc)}",
        )


@router.get(
    "/comparison/{symbol}",
    response_model=MLModelComparisonResponse,
    summary="Candidate Model Performance Comparison",
)
async def get_model_comparison(symbol: str):
    """Retrieve side-by-side performance metrics across all models trained for this ticker."""
    symbol = symbol.upper()
    comparison_data = model_registry.get_model_comparison(symbol=symbol)

    if not comparison_data:
        # Trigger training to generate comparison data
        df = await _get_historical_df(symbol=symbol, limit=200)
        train_full_ml_pipeline(df=df, symbol=symbol, test_size=0.20)
        comparison_data = model_registry.get_model_comparison(symbol=symbol)

    items = [
        MLModelComparisonItem(
            model_id=m["model_id"],
            model_type=m["model_type"],
            version=m["version"],
            target_type=m["target_type"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            mae=m.get("mae", 0.0),
            rmse=m.get("rmse", 0.0),
            r2=m.get("r2", 0.0),
            is_active=m.get("is_active", False),
            created_at=m.get("created_at", ""),
        )
        for m in comparison_data
    ]

    active_id = None
    for item in items:
        if item.is_active:
            active_id = item.model_id
            break

    return MLModelComparisonResponse(
        symbol=symbol,
        models=items,
        active_model_id=active_id,
        total_models=len(items),
        disclaimer=PREDICTION_DISCLAIMER,
    )


@router.get(
    "/status",
    response_model=MLModelStatusResponse,
    summary="Model Registry Health & Inventory",
)
async def get_model_status():
    """Retrieve machine learning engine and model registry status."""
    all_models = model_registry.list_models()
    active_by_sym = {}
    for m in all_models:
        sym = m.get("symbol")
        if m.get("is_active") and sym:
            active_by_sym[sym] = m.get("model_id")

    return MLModelStatusResponse(
        total_registered_models=len(all_models),
        active_models_by_symbol=active_by_sym,
        supported_model_types=[
            "RandomForest",
            "LogisticRegression",
            "SVM",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "Ensemble",
        ],
        engine_status="OPERATIONAL",
        last_updated=datetime.now().isoformat(),
    )


@router.post(
    "/train",
    response_model=MLTrainResponse,
    summary="Train Machine Learning Pipeline for Symbol",
)
async def train_models_for_symbol(request: MLTrainRequest):
    """Trigger time-series safe training pipeline across all candidate models."""
    df = await _get_historical_df(symbol=request.symbol, limit=200)
    try:
        results = train_full_ml_pipeline(
            df=df,
            symbol=request.symbol,
            test_size=request.test_size,
        )
        return MLTrainResponse(
            symbol=results["symbol"],
            active_model_id=results["active_model_id"],
            train_samples=results["train_samples"],
            test_samples=results["test_samples"],
            ensemble_metrics=results["ensemble_metrics"],
            candidate_metrics=results["candidate_metrics"],
            selected_features=results["selected_features"],
            disclaimer=PREDICTION_DISCLAIMER,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(exc)}",
        )


# =========================================================================
# Deep Learning & Consensus Ensemble Endpoints
# =========================================================================


@router.get(
    "/deep-learning/predict/{symbol}",
    response_model=DLPredictionResponse,
    summary="Deep Learning (LSTM/GRU) Sequence Direction & Price Range Prediction",
)
async def predict_deep_learning(
    symbol: str,
    timeframe: str = Query(default="1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
    model_type: str = Query(default="LSTM", description="Model architecture: LSTM or GRU"),
):
    """Generate chronological sequence prediction using recurrent deep neural network.

    Outputs direction, confidence probability, ±2σ volatility range, and MC Dropout uncertainty.
    """
    df = await _get_historical_df(symbol=symbol, timeframe=timeframe, limit=200)
    try:
        from ai_engine.deep_learning.predict import dl_predictor
        prediction = dl_predictor.predict_next_period(
            df=df,
            symbol=symbol,
            model_type=model_type.upper(),
        )
        return prediction
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deep learning inference error: {str(exc)}",
        )


@router.get(
    "/ensemble/consensus/{symbol}",
    response_model=EnsembleConsensusResponse,
    summary="Hybrid Ensemble Consensus (Classical ML + Deep Learning)",
)
async def get_ensemble_consensus(
    symbol: str,
    timeframe: str = Query(default="1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d"),
):
    """Synthesize predictions from Classical ML (RF, LR, SVM, GB) and Deep Learning (LSTM, GRU).

    Calculates consensus direction, confidence, model agreement percentage, and individual vote breakdown.
    """
    df = await _get_historical_df(symbol=symbol, timeframe=timeframe, limit=200)
    try:
        from ai_engine.deep_learning.predict import hybrid_ensemble
        consensus = hybrid_ensemble.predict_consensus(df=df, symbol=symbol)
        return consensus
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid ensemble consensus error: {str(exc)}",
        )


@router.post(
    "/deep-learning/train",
    response_model=DLTrainResponse,
    summary="Train Deep Learning Models (LSTM/GRU) with Controlled Schedule",
)
async def train_deep_learning_endpoint(request: DLTrainRequest):
    """Trigger controlled training for LSTM or GRU models on historical data.

    Enforces 60-minute cooldown schedule to prevent tick thrashing unless force_retrain=True.
    """
    df = await _get_historical_df(symbol=request.symbol, limit=200)
    try:
        from ai_engine.deep_learning.train import train_deep_learning_model
        models_to_train = ["LSTM", "GRU"] if request.model_type.upper() == "BOTH" else [request.model_type.upper()]
        results_map = {}
        paths_map = {}
        train_n, val_n, test_n = 0, 0, 0

        for m_type in models_to_train:
            res = train_deep_learning_model(
                df=df,
                symbol=request.symbol,
                model_type=m_type,
                epochs=request.epochs,
                sequence_length=request.sequence_length,
                force_retrain=request.force_retrain,
            )
            results_map[m_type] = res["test_metrics"]
            paths_map[m_type] = res["checkpoint_path"]
            train_n = res["train_samples"]
            val_n = res["val_samples"]
            test_n = res["test_samples"]

        return DLTrainResponse(
            symbol=request.symbol.upper(),
            models_trained=models_to_train,
            metrics=results_map,
            train_samples=train_n,
            val_samples=val_n,
            test_samples=test_n,
            model_paths=paths_map,
            disclaimer=PREDICTION_DISCLAIMER,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deep learning training failed: {str(exc)}",
        )


@router.get(
    "/deep-learning/comparison/{symbol}",
    summary="Compare Deep Learning vs Classical ML Models",
)
async def compare_dl_classical(symbol: str):
    """Retrieve comprehensive comparison matrix between Deep Learning and Classical ML architectures."""
    try:
        from ai_engine.deep_learning.evaluate import compare_deep_learning_with_classical
        return compare_deep_learning_with_classical(symbol=symbol.upper())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(exc)}",
        )

