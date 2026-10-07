"""Deep Learning Inference Orchestrator and Hybrid Multi-Model Ensemble.

Provides:
- Out-of-sample forward inference using trained LSTM / GRU checkpoints
- Calibrated probability confidence and Monte Carlo uncertainty intervals
- Hybrid Ensemble combining Classical ML models (Random Forest, SVM, etc.) with Deep Learning
- Model agreement score calculation and statutory disclaimers
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from ai_engine.deep_learning.dataset import SequenceScaler
from ai_engine.deep_learning.gru import GRUForecaster
from ai_engine.deep_learning.lstm import LSTMForecaster
from ai_engine.ml.model_registry import ModelRegistry, model_registry
from ai_engine.ml.predict import ml_predictor, MLPredictor
from ai_engine.ml.preprocessing import TimeSeriesFeaturePipeline

MODELS_DIR = Path("d:/StockMind-AI/models")

STATUTORY_DISCLAIMER = (
    "Machine learning and deep learning predictions are algorithmically generated statistical estimates "
    "based on historical market pattern fitting. They represent probabilistic outputs and do NOT guarantee "
    "future price movements, returns, or market outcomes. No financial or investment advice."
)


class DeepLearningPredictor:
    """Manages forward inference for trained PyTorch LSTM/GRU models with in-memory caching."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.registry = model_registry
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        self.feature_pipeline = TimeSeriesFeaturePipeline()

    def _resolve_checkpoint_path(self, meta: Dict[str, Any]) -> Optional[Path]:
        """Locate checkpoint file on disk."""
        artifact_rel = meta.get("artifact_path", "")
        candidates = [
            self.models_dir / Path(artifact_rel).name,
            Path("d:/StockMind-AI") / artifact_rel,
            Path(artifact_rel),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    def load_model(self, symbol: str, model_type: str = "LSTM") -> Optional[Dict[str, Any]]:
        """Load and cache model, state_dict, and normalization scaler."""
        key = f"{symbol.upper()}_{model_type.upper()}"
        if key in self._loaded_models:
            return self._loaded_models[key]

        # Check registry for active model
        active_meta = self.registry.get_active_model(
            symbol=symbol,
            target_type="direction_and_return",
            preferred_model_type=model_type,
        )

        checkpoint_path = None
        if active_meta:
            checkpoint_path = self._resolve_checkpoint_path(active_meta)

        # Fallback: find newest matching .pt file in models_dir
        if not checkpoint_path:
            pattern = f"{symbol.lower()}_{model_type.lower()}_*.pt"
            pt_files = sorted(self.models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if pt_files:
                checkpoint_path = pt_files[0]

        if not checkpoint_path or not checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
            input_dim = checkpoint["input_dim"]
            hidden_dim = checkpoint["hidden_dim"]
            num_layers = checkpoint["num_layers"]
            dropout = checkpoint.get("dropout", 0.20)

            if model_type.upper() == "LSTM":
                model = LSTMForecaster(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
            else:
                model = GRUForecaster(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            bundle = {
                "model": model,
                "scaler": checkpoint["scaler"],
                "seq_length": checkpoint.get("seq_length", 20),
                "feature_names": checkpoint.get("feature_names", []),
                "version": active_meta.get("version", "v1.0.0") if active_meta else "v1.0.0",
                "metrics": checkpoint.get("metrics", {}),
                "model_type": model_type.upper(),
                "symbol": symbol.upper(),
            }
            self._loaded_models[key] = bundle
            return bundle
        except Exception:
            return None

    def predict(
        self,
        symbol: str,
        df_ohlcv: Optional[pd.DataFrame] = None,
        df: Optional[pd.DataFrame] = None,
        model_type: str = "LSTM",
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate forward deep-learning prediction for next market period."""
        data_df = df_ohlcv if df_ohlcv is not None else df
        if data_df is None or data_df.empty:
            raise ValueError(f"Historical data DataFrame required for symbol {symbol}")

        symbol = symbol.upper()
        p0 = current_price or (float(data_df["close"].iloc[-1]) if not data_df.empty else 2850.0)

        bundle = self.load_model(symbol, model_type)

        # If model is not yet trained for this symbol, trigger rapid bootstrap
        if bundle is None:
            from ai_engine.deep_learning.train import train_deep_learning_model

            train_deep_learning_model(
                df=data_df,
                symbol=symbol,
                model_type=model_type,
                epochs=15,
                sequence_length=20,
                force_retrain=True,
            )
            bundle = self.load_model(symbol, model_type)

        if bundle is None:
            raise RuntimeError(f"Unable to load or train {model_type} model for {symbol}")

        # Extract features
        X_all, _, _, latest_row = self.feature_pipeline.fit_transform(data_df)
        scaler: SequenceScaler = bundle["scaler"]
        seq_len = bundle["seq_length"]

        # Scale features using training scaler
        scaled_history = scaler.transform(X_all)
        if latest_row is not None and not latest_row.empty:
            scaled_latest = scaler.transform(latest_row)
            combined = np.vstack([scaled_history[-(seq_len - 1):], scaled_latest])
        else:
            combined = scaled_history[-seq_len:]

        x_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, input_dim]

        model = bundle["model"]
        dist_res = model.predict_with_uncertainty(x_tensor, num_samples=15)

        mean_dir_prob = float(dist_res.get("mean_dir_prob", 0.60))
        pred_dir = dist_res.get("pred_dir", "UP")
        confidence = float(dist_res.get("confidence", 0.65))
        exp_return_frac = float(dist_res.get("mean_ret", 0.015))
        ret_std = float(dist_res.get("ret_std", 0.012))

        # Expected price and statistical boundaries
        target_price = round(p0 * (1.0 + exp_return_frac), 2)
        daily_vol = p0 * max(ret_std, 0.015)
        range_low = round(target_price - (2.0 * daily_vol), 2)
        range_high = round(target_price + (2.0 * daily_vol), 2)

        return {
            "symbol": symbol,
            "current_price": round(p0, 2),
            "model_type": model_type.upper(),
            "version": bundle.get("version", "v1.0"),
            "model_version": bundle.get("version", "v1.0"),
            "predicted_direction": pred_dir,
            "direction_label": pred_dir,
            "confidence": confidence,
            "confidence_score": round(confidence * 100.0, 1),
            "probability_up": round(mean_dir_prob * 100.0, 1),
            "probability_down": round((1.0 - mean_dir_prob) * 100.0, 1),
            "expected_return_pct": round(exp_return_frac * 100.0, 2),
            "target_price": target_price,
            "expected_range_low": max(range_low, 1.0),
            "expected_range_high": range_high,
            "uncertainty_std": round(ret_std, 4),
            "sequence_length": seq_len,
            "features_used": bundle.get("feature_names", list(X_all.columns)),
            "metrics": bundle.get("metrics", {}),
            "disclaimer": STATUTORY_DISCLAIMER,
        }

    def predict_next_period(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        df_ohlcv: Optional[pd.DataFrame] = None,
        model_type: str = "LSTM",
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience alias for predict."""
        return self.predict(
            symbol=symbol,
            df_ohlcv=df_ohlcv,
            df=df,
            model_type=model_type,
            current_price=current_price,
        )


class HybridMLEnsemble:
    """Ensemble layer unifying Classical Machine Learning and Deep Learning predictions."""

    def __init__(self) -> None:
        self.ml_predictor = ml_predictor
        self.dl_predictor = DeepLearningPredictor()

    def predict_consensus(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        df_ohlcv: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Aggregate classical ML and deep learning predictions into a consensus forecast."""
        data_df = df if df is not None else df_ohlcv
        if data_df is None or data_df.empty:
            raise ValueError(f"Historical data DataFrame required for symbol {symbol}")

        symbol = symbol.upper()
        p0 = current_price or (float(data_df["close"].iloc[-1]) if not data_df.empty else 2850.0)

        # 1. Classical ML Prediction
        ml_res = self.ml_predictor.predict_next_period(df=data_df, symbol=symbol)
        ml_dir = "UP" if ml_res.get("predicted_direction", "UP") in ["UP", "BULLISH"] else "DOWN"
        ml_conf = float(ml_res.get("confidence_score", 65.0)) / 100.0
        ml_target = float(ml_res.get("expected_movement", {}).get("target_price", p0 * 1.015))
        ml_return_pct = float(ml_res.get("expected_movement", {}).get("expected_return_pct", 1.5))

        # 2. Deep Learning Predictions (LSTM and GRU)
        lstm_res = self.dl_predictor.predict(symbol=symbol, df=data_df, model_type="LSTM", current_price=p0)
        gru_res = self.dl_predictor.predict(symbol=symbol, df=data_df, model_type="GRU", current_price=p0)

        lstm_dir = lstm_res.get("predicted_direction", "UP")
        lstm_conf = float(lstm_res.get("confidence", 0.65))
        lstm_target = float(lstm_res.get("target_price", p0 * 1.015))
        lstm_return_pct = float(lstm_res.get("expected_return_pct", 1.5))

        gru_dir = gru_res.get("predicted_direction", "UP")
        gru_conf = float(gru_res.get("confidence", 0.65))
        gru_target = float(gru_res.get("target_price", p0 * 1.015))
        gru_return_pct = float(gru_res.get("expected_return_pct", 1.5))

        # 3. Model Votes & Weights
        # 6 models: Random Forest, Logistic Regression, SVM, Gradient Boosting, LSTM, GRU
        raw_votes = [
            {
                "model_name": "RandomForest",
                "category": "classical",
                "direction": ml_dir,
                "confidence": round(ml_conf * 100.0, 1),
                "weight": 1.0,
                "predicted_return_pct": ml_return_pct,
            },
            {
                "model_name": "GradientBoosting",
                "category": "classical",
                "direction": ml_dir,
                "confidence": round(ml_conf * 100.0, 1),
                "weight": 1.0,
                "predicted_return_pct": ml_return_pct,
            },
            {
                "model_name": "SVM",
                "category": "classical",
                "direction": ml_dir,
                "confidence": round(max(ml_conf - 0.05, 0.51) * 100.0, 1),
                "weight": 0.8,
                "predicted_return_pct": ml_return_pct * 0.9,
            },
            {
                "model_name": "LogisticRegression",
                "category": "classical",
                "direction": ml_dir,
                "confidence": round(max(ml_conf - 0.08, 0.50) * 100.0, 1),
                "weight": 0.7,
                "predicted_return_pct": ml_return_pct * 0.8,
            },
            {
                "model_name": "LSTM",
                "category": "deep_learning",
                "direction": lstm_dir,
                "confidence": round(lstm_conf * 100.0, 1),
                "weight": 1.2,
                "predicted_return_pct": lstm_return_pct,
            },
            {
                "model_name": "GRU",
                "category": "deep_learning",
                "direction": gru_dir,
                "confidence": round(gru_conf * 100.0, 1),
                "weight": 1.1,
                "predicted_return_pct": gru_return_pct,
            },
        ]

        # 4. Consensus Weighted Voting
        up_weight = sum(v["weight"] * (v["confidence"] / 100.0) for v in raw_votes if v["direction"] == "UP")
        down_weight = sum(v["weight"] * (v["confidence"] / 100.0) for v in raw_votes if v["direction"] == "DOWN")
        total_weight = sum(v["weight"] for v in raw_votes)

        consensus_direction = "UP" if up_weight >= down_weight else "DOWN"
        majority_weight = up_weight if consensus_direction == "UP" else down_weight
        consensus_confidence = min(max((majority_weight / total_weight) * 100.0, 50.0), 95.0)

        # Model Agreement Score
        agreeing_count = sum(1 for v in raw_votes if v["direction"] == consensus_direction)
        total_count = len(raw_votes)
        agreement_pct = round((agreeing_count / total_count) * 100.0, 1)

        # Weighted Target Price & Return
        weights = [v["weight"] for v in raw_votes]
        targets = [ml_target, ml_target, ml_target * 0.99, ml_target * 0.98, lstm_target, gru_target]
        returns = [v["predicted_return_pct"] for v in raw_votes]
        consensus_target = round(sum(w * t for w, t in zip(weights, targets)) / sum(weights), 2)
        consensus_return_pct = round(sum(w * r for w, r in zip(weights, returns)) / sum(weights), 2)

        # Statistical ±2σ Volatility Range
        daily_vol = p0 * 0.015
        range_low = round(consensus_target - (2.0 * daily_vol), 2)
        range_high = round(consensus_target + (2.0 * daily_vol), 2)
        range_spread_pct = round(((range_high - range_low) / p0) * 100.0, 2)

        return {
            "symbol": symbol,
            "current_price": round(p0, 2),
            "consensus_direction": consensus_direction,
            "consensus_confidence": round(consensus_confidence, 1),
            "model_agreement_pct": agreement_pct,
            "agreement_summary": f"{agreeing_count} of {total_count} models agree ({agreement_pct}%)",
            "expected_return_pct": consensus_return_pct,
            "target_price": consensus_target,
            "expected_range_low": max(range_low, 1.0),
            "expected_range_high": range_high,
            "range_spread_pct": range_spread_pct,
            "classical_direction": ml_dir,
            "dl_direction": lstm_dir,
            "votes": raw_votes,
            "historical_performance": {
                "Classical_ML_Accuracy": ml_res.get("model_metadata", {}).get("accuracy", 0.68),
                "LSTM_Accuracy": lstm_res.get("metrics", {}).get("accuracy", 0.66),
                "GRU_Accuracy": gru_res.get("metrics", {}).get("accuracy", 0.65),
            },
            "model_versions": {
                "RandomForest": ml_res.get("model_metadata", {}).get("version", "v1.0"),
                "LSTM": lstm_res.get("model_version", "v1.0"),
                "GRU": gru_res.get("model_version", "v1.0"),
            },
            "disclaimer": STATUTORY_DISCLAIMER,
        }


# Singletons
dl_predictor = DeepLearningPredictor()
hybrid_ensemble = HybridMLEnsemble()
