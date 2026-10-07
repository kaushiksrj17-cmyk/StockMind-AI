"""Pydantic Schemas for Machine Learning Engine APIs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExpectedMovement(BaseModel):
    current_price: float
    expected_return_pct: float
    target_price: float
    expected_range_low: float
    expected_range_high: float
    range_spread_pct: float


class ModelMeta(BaseModel):
    model_id: str
    version: str
    model_type: str
    accuracy: float
    f1: float


class MLPredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_direction: str = Field(..., description="UP or DOWN")
    sentiment_bias: str = Field(..., description="BULLISH or BEARISH")
    confidence_score: float = Field(..., description="Probability of selected direction (50.0 - 100.0%)")
    probability_up: float = Field(..., description="Probability of upward price move (%)")
    probability_down: float = Field(..., description="Probability of downward price move (%)")
    expected_movement: ExpectedMovement
    model_metadata: ModelMeta
    features_used: List[str]
    feature_importance: Dict[str, float]
    disclaimer: str = Field(..., description="Mandatory statistical analytical disclaimer")


class MLModelComparisonItem(BaseModel):
    model_id: str
    model_type: str
    version: str
    target_type: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    mae: Optional[float] = 0.0
    rmse: Optional[float] = 0.0
    r2: Optional[float] = 0.0
    is_active: bool
    created_at: str


class MLModelComparisonResponse(BaseModel):
    symbol: str
    models: List[MLModelComparisonItem]
    active_model_id: Optional[str]
    total_models: int
    disclaimer: str


class MLModelStatusResponse(BaseModel):
    total_registered_models: int
    active_models_by_symbol: Dict[str, str]
    supported_model_types: List[str]
    engine_status: str
    last_updated: str


class MLTrainRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol to train models for")
    test_size: float = Field(default=0.20, ge=0.10, le=0.40)


class MLTrainResponse(BaseModel):
    symbol: str
    active_model_id: str
    train_samples: int
    test_samples: int
    ensemble_metrics: Dict[str, Any]
    candidate_metrics: Dict[str, Any]
    selected_features: List[str]
    disclaimer: str


# =========================================================================
# Deep Learning & Consensus Ensemble Schemas
# =========================================================================


class DLPredictionResponse(BaseModel):
    symbol: str
    model_type: str = Field(..., description="LSTM or GRU")
    version: str
    current_price: float
    predicted_direction: str = Field(..., description="UP or DOWN")
    confidence_score: float = Field(..., description="Probability of predicted direction (%)")
    probability_up: float
    probability_down: float
    expected_return_pct: float
    target_price: float
    expected_range_low: float
    expected_range_high: float
    uncertainty_std: float = Field(..., description="MC Dropout standard deviation")
    sequence_length: int
    features_used: List[str]
    disclaimer: str


class ModelVote(BaseModel):
    model_name: str
    category: str = Field(..., description="classical or deep_learning")
    direction: str = Field(..., description="UP or DOWN")
    confidence: float
    weight: float
    predicted_return_pct: Optional[float] = None


class EnsembleConsensusResponse(BaseModel):
    symbol: str
    current_price: float
    consensus_direction: str = Field(..., description="Consensus predicted direction (UP or DOWN)")
    consensus_confidence: float = Field(..., description="Weighted consensus confidence percentage")
    model_agreement_pct: float = Field(..., description="Percentage of models agreeing with majority direction")
    agreement_summary: str = Field(..., description="Human-readable agreement e.g. '5 of 6 models agree'")
    expected_return_pct: float
    target_price: float
    expected_range_low: float
    expected_range_high: float
    range_spread_pct: float
    classical_direction: str
    dl_direction: str
    votes: List[ModelVote]
    historical_performance: Dict[str, Any]
    model_versions: Dict[str, str]
    disclaimer: str


class DLTrainRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol to train DL models for")
    model_type: str = Field(default="LSTM", description="LSTM, GRU, or BOTH")
    epochs: int = Field(default=25, ge=5, le=100)
    sequence_length: int = Field(default=30, ge=10, le=60)
    force_retrain: bool = Field(default=False, description="Bypass cooldown schedule")


class DLTrainResponse(BaseModel):
    symbol: str
    models_trained: List[str]
    metrics: Dict[str, Any]
    train_samples: int
    val_samples: int
    test_samples: int
    model_paths: Dict[str, str]
    disclaimer: str

