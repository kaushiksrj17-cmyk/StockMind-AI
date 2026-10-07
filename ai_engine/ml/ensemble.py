
"""Ensemble Machine Learning Architecture.

Combines multiple base models (Random Forest, Logistic Regression, SVM, Gradient Boosting)
using confidence-weighted soft voting for direction classification and variance-weighted
averaging for expected movement and price range estimation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DirectionalEnsemble:
    """Soft-voting ensemble weighting models by out-of-sample directional accuracy."""

    def __init__(self, models_dict: Dict[str, Any], weights_dict: Optional[Dict[str, float]] = None) -> None:
        self.models = models_dict
        self.weights = weights_dict or {}
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        if not self.weights:
            n = len(self.models)
            self.weights = {k: 1.0 / n for k in self.models.keys()}
        else:
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}
            else:
                n = len(self.models)
                self.weights = {k: 1.0 / n for k in self.models.keys()}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted probability array [[P(down), P(up)], ...]."""
        total_proba = np.zeros((len(X), 2), dtype=float)

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if hasattr(model, "predict_proba"):
                try:
                    p = model.predict_proba(X)
                    if p.shape[1] == 2:
                        total_proba += w * p
                    else:
                        # Single-class edge case
                        total_proba[:, 1] += w * 0.5
                        total_proba[:, 0] += w * 0.5
                except Exception:
                    # Fallback to decision function or predict
                    preds = model.predict(X)
                    p_up = (preds == 1).astype(float)
                    total_proba[:, 1] += w * p_up
                    total_proba[:, 0] += w * (1.0 - p_up)
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X)
                # Sigmoid transform
                p_up = 1.0 / (1.0 + np.exp(-df))
                total_proba[:, 1] += w * p_up
                total_proba[:, 0] += w * (1.0 - p_up)
            else:
                preds = model.predict(X)
                p_up = (preds == 1).astype(float)
                total_proba[:, 1] += w * p_up
                total_proba[:, 0] += w * (1.0 - p_up)

        # Normalize rows to sum to 1.0
        row_sums = total_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return total_proba / row_sums

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Predict binary direction: 1 for UP, 0 for DOWN."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


class MovementRangeEnsemble:
    """Combines regression estimates to compute expected return and statistical confidence ranges."""

    def __init__(self, reg_models: Dict[str, Any], residual_stds: Optional[Dict[str, float]] = None) -> None:
        self.models = reg_models
        self.residual_stds = residual_stds or {}

    def predict_expected_movement(
        self,
        X: pd.DataFrame,
        current_price: float,
        atr_pct: float = 0.015,
    ) -> Dict[str, Any]:
        """Compute expected percentage return and statistically bounded price range."""
        preds = []
        for name, model in self.models.items():
            try:
                p = model.predict(X)[0]
                preds.append(float(p))
            except Exception:
                pass

        if not preds:
            expected_ret = 0.0
        else:
            expected_ret = float(np.median(preds))

        # Expected price target
        target_price = current_price * (1.0 + expected_ret)

        # Compute uncertainty range based on model residuals or ATR
        vol_margin = max(atr_pct * 1.5, 0.008)
        range_low = current_price * (1.0 + expected_ret - vol_margin)
        range_high = current_price * (1.0 + expected_ret + vol_margin)

        return {
            "current_price": round(current_price, 2),
            "expected_return_pct": round(expected_ret * 100.0, 2),
            "target_price": round(target_price, 2),
            "expected_range_low": round(range_low, 2),
            "expected_range_high": round(range_high, 2),
            "range_spread_pct": round(((range_high - range_low) / current_price) * 100.0, 2),
        }
