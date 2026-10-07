"""Feature Selection Module for Machine Learning Pipelines.

Removes collinear and noisy indicators, computes feature importance,
and selects the top predictive features to prevent overfitting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TimeSeriesFeatureSelector:
    """Selects top predictive features strictly fitted on training splits."""

    def __init__(
        self,
        max_features: int = 18,
        collinear_threshold: float = 0.90,
        variance_threshold: float = 1e-5,
        top_k: Optional[int] = None,
        collinear_thresh: Optional[float] = None,
        variance_thresh: Optional[float] = None,
    ) -> None:
        self.max_features = top_k if top_k is not None else max_features
        self.collinear_threshold = collinear_thresh if collinear_thresh is not None else collinear_threshold
        self.variance_threshold = variance_thresh if variance_thresh is not None else variance_threshold
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TimeSeriesFeatureSelector":
        """Identify optimal features strictly from training data without look-ahead bias."""
        features = list(X.columns)

        # 1. Variance threshold filter (eliminate constant or near-zero variance features)
        variances = X.var()
        valid_var_features = [f for f in features if variances.get(f, 0.0) > self.variance_threshold]
        if not valid_var_features:
            valid_var_features = features

        X_filtered = X[valid_var_features]

        # 2. Target correlation ranking (predictive power)
        target_corr = {}
        y_num = pd.to_numeric(y, errors="coerce").fillna(0.0)
        for col in valid_var_features:
            corr = X_filtered[col].corr(y_num)
            target_corr[col] = abs(corr) if not np.isnan(corr) else 0.0

        self.feature_scores_ = {k: round(v, 4) for k, v in sorted(target_corr.items(), key=lambda item: item[1], reverse=True)}

        # 3. Collinear feature filter (remove redundant indicators)
        corr_matrix = X_filtered.corr().abs()
        sorted_by_power = list(self.feature_scores_.keys())

        non_collinear: List[str] = []
        for feat in sorted_by_power:
            is_collinear = False
            for selected in non_collinear:
                if corr_matrix.loc[feat, selected] >= self.collinear_threshold:
                    is_collinear = True
                    break
            if not is_collinear:
                non_collinear.append(feat)

        # 4. Cap to max_features
        self.selected_features_ = non_collinear[: self.max_features]
        if not self.selected_features_:
            self.selected_features_ = features[: self.max_features]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame columns to selected features."""
        available = [f for f in self.selected_features_ if f in X.columns]
        if not available:
            return X
        return X[available].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit on training data and return reduced feature matrix."""
        return self.fit(X, y).transform(X)
