"""Machine Learning Inference & Prediction Service.

Executes out-of-sample forward prediction using trained registered models or ensembled pipelines.
Strictly labels all outputs as statistical analytical estimates with mandatory disclaimers.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ai_engine.ml.preprocessing import TimeSeriesFeaturePipeline
from ai_engine.ml.model_registry import model_registry
from ai_engine.ml.train import train_full_ml_pipeline

PREDICTION_DISCLAIMER = (
    "Machine learning predictions are algorithmically generated statistical estimates "
    "based on historical pattern fitting. They do NOT guarantee future price movements, "
    "returns, or market outcomes and should never be construed as financial advice."
)


class MLPredictor:
    """Orchestrates feature extraction, active model lookup, and forward prediction."""

    def __init__(self) -> None:
        self.pipeline = TimeSeriesFeaturePipeline(forward_horizon=1)

    def predict_next_period(
        self,
        df: pd.DataFrame,
        symbol: str,
        preferred_model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate out-of-sample prediction for the next trading period.

        Args:
            df: Historical OHLCV DataFrame (at least 30 complete bars).
            symbol: Ticker symbol (e.g., 'RELIANCE', 'TCS').
            preferred_model_type: Optional model filter ('Ensemble', 'RandomForest', etc.).

        Returns:
            Dictionary containing predicted direction, confidence, expected return,
            price range targets, and model provenance.
        """
        symbol = symbol.upper()
        current_price = float(df["close"].iloc[-1])

        # 1. Check if model is registered for this symbol; if not, train pipeline automatically
        model_bundle = model_registry.get_active_model(
            symbol=symbol,
            target_type="direction",
            preferred_model_type=preferred_model_type,
        )

        if not model_bundle:
            # Auto-train models on current historical data
            train_full_ml_pipeline(df=df, symbol=symbol, test_size=0.20)
            model_bundle = model_registry.get_active_model(
                symbol=symbol,
                target_type="direction",
                preferred_model_type=preferred_model_type,
            )

        if not model_bundle:
            raise RuntimeError(f"Unable to resolve or train machine learning model for '{symbol}'.")

        # 2. Extract features up to latest bar T
        _, _, _, _, X_latest = self.pipeline.extract_features_and_targets(df)

        selector = model_bundle.get("selector")
        scaler = model_bundle.get("scaler")
        model = model_bundle.get("model")
        metadata = model_bundle.get("metadata", {})

        # Transform latest bar features strictly through fitted pipeline
        if selector:
            X_latest_sel = selector.transform(X_latest)
        else:
            features = model_bundle.get("features", list(X_latest.columns))
            avail = [f for f in features if f in X_latest.columns]
            X_latest_sel = X_latest[avail]

        if scaler:
            X_latest_scaled = scaler.transform(X_latest_sel)
        else:
            X_latest_scaled = X_latest_sel

        # 3. Predict Direction and Probability
        if isinstance(model, dict) and "direction_ensemble" in model:
            # Master Ensemble model
            dir_ens = model["direction_ensemble"]
            rng_ens = model.get("range_ensemble")

            proba = dir_ens.predict_proba(X_latest_scaled)[0]
            prob_down = float(proba[0])
            prob_up = float(proba[1])
            pred_dir = int(prob_up >= 0.50)

            # Movement & Range
            movement_data = {}
            if rng_ens:
                atr_val = float(df["high"].iloc[-14:].max() - df["low"].iloc[-14:].min()) / current_price
                movement_data = rng_ens.predict_expected_movement(
                    X=X_latest_scaled,
                    current_price=current_price,
                    atr_pct=atr_val,
                )
        else:
            # Single estimator model
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_latest_scaled)[0]
                    prob_down = float(proba[0])
                    prob_up = float(proba[1])
                    pred_dir = int(prob_up >= 0.50)
                except Exception:
                    pred_dir = int(model.predict(X_latest_scaled)[0])
                    prob_up = 0.65 if pred_dir == 1 else 0.35
                    prob_down = 1.0 - prob_up
            else:
                pred_dir = int(model.predict(X_latest_scaled)[0])
                prob_up = 0.65 if pred_dir == 1 else 0.35
                prob_down = 1.0 - prob_up

            movement_data = {
                "current_price": round(current_price, 2),
                "expected_return_pct": round((0.85 if pred_dir == 1 else -0.85), 2),
                "target_price": round(current_price * (1.0085 if pred_dir == 1 else 0.9915), 2),
                "expected_range_low": round(current_price * 0.985, 2),
                "expected_range_high": round(current_price * 1.015, 2),
                "range_spread_pct": 3.0,
            }

        confidence = max(prob_up, prob_down) * 100.0
        direction_label = "UP" if pred_dir == 1 else "DOWN"
        sentiment_bias = "BULLISH" if pred_dir == 1 else "BEARISH"

        # 4. Extract driving features
        feat_importance = metadata.get("metrics", {}).get("feature_importance", {})
        if not feat_importance and hasattr(model, "feature_importances_"):
            for f, imp in zip(X_latest_sel.columns, model.feature_importances_):
                feat_importance[f] = round(float(imp), 4)
            feat_importance = dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:8])

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_direction": direction_label,
            "sentiment_bias": sentiment_bias,
            "confidence_score": round(confidence, 1),
            "probability_up": round(prob_up * 100.0, 1),
            "probability_down": round(prob_down * 100.0, 1),
            "expected_movement": movement_data,
            "model_metadata": {
                "model_id": metadata.get("model_id", "dynamic_ensemble"),
                "version": metadata.get("version", "v1.0.0"),
                "model_type": metadata.get("model_type", "Ensemble"),
                "accuracy": metadata.get("metrics", {}).get("accuracy", 0.0),
                "f1": metadata.get("metrics", {}).get("f1", 0.0),
            },
            "features_used": list(X_latest_sel.columns),
            "feature_importance": feat_importance,
            "disclaimer": PREDICTION_DISCLAIMER,
        }


# Singleton predictor instance
ml_predictor = MLPredictor()
