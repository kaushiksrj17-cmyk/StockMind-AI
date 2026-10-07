"""Machine Learning Training Pipeline Module.

Trains Random Forest, Logistic Regression, SVM, and Gradient Boosting/Ensemble models
using time-series-safe chronological validation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ai_engine.ml.preprocessing import (
    TimeSeriesFeaturePipeline,
    chronological_train_test_split,
    TimeSeriesScaler,
)
from ai_engine.ml.feature_selection import TimeSeriesFeatureSelector
from ai_engine.ml.evaluate import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    evaluate_model_performance,
)
from ai_engine.ml.ensemble import DirectionalEnsemble, MovementRangeEnsemble
from ai_engine.ml.model_registry import model_registry


def get_classifier(model_type: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """Instantiate classification model with tuned hyperparameters."""
    params = hyperparameters or {}
    mtype = model_type.lower()

    if mtype in ["randomforest", "rf", "random_forest"]:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
    elif mtype in ["logisticregression", "lr", "logistic_regression"]:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=params.get("C", 0.1),
            max_iter=params.get("max_iter", 500),
            random_state=params.get("random_state", 42),
        )
    elif mtype in ["svm", "svc", "support_vector_machine"]:
        from sklearn.svm import SVC
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            probability=True,
            random_state=params.get("random_state", 42),
        )
    elif mtype in ["xgboost", "xgb"]:
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                random_state=params.get("random_state", 42),
                eval_metric="logloss",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 80),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                random_state=params.get("random_state", 42),
            )
    elif mtype in ["lightgbm", "lgbm"]:
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                random_state=params.get("random_state", 42),
                verbosity=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 80),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                random_state=params.get("random_state", 42),
            )
    else:
        # Default gradient boosting tree
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 80),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            random_state=params.get("random_state", 42),
        )


def get_regressor(model_type: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """Instantiate regression model for price move and range estimation."""
    params = hyperparameters or {}
    mtype = model_type.lower()

    if mtype in ["randomforest", "rf", "random_forest"]:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
    elif mtype in ["svm", "svr"]:
        from sklearn.svm import SVR
        return SVR(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
        )
    elif mtype in ["ridge", "linear"]:
        from sklearn.linear_model import Ridge
        return Ridge(alpha=params.get("alpha", 1.0))
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 80),
            max_depth=params.get("max_depth", 4),
            random_state=params.get("random_state", 42),
        )


def train_direction_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "RandomForest",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train single directional model and evaluate on out-of-sample holdout."""
    clf = get_classifier(model_type, hyperparameters)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = None
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_test)
            if proba.shape[1] == 2:
                y_prob = proba[:, 1]
        except Exception:
            pass

    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

    # Feature importance if available
    feature_importance = {}
    if hasattr(clf, "feature_importances_"):
        for f, imp in zip(X_train.columns, clf.feature_importances_):
            feature_importance[f] = round(float(imp), 4)
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_
        for f, c in zip(X_train.columns, coefs):
            feature_importance[f] = round(float(abs(c)), 4)

    metrics["feature_importance"] = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    return clf, metrics


def train_return_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "RandomForest",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train regression model for next-period percentage price movement."""
    reg = get_regressor(model_type, hyperparameters)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    metrics = calculate_regression_metrics(y_test, y_pred)
    return reg, metrics


def train_full_ml_pipeline(
    df: pd.DataFrame,
    symbol: str,
    test_size: float = 0.20,
) -> Dict[str, Any]:
    """Execute complete end-to-end time-series safe training pipeline.

    - Computes features & aligned targets
    - Chronological train-test split (past vs future)
    - TimeSeriesFeatureSelector fitted strictly on train
    - TimeSeriesScaler fitted strictly on train
    - Trains Random Forest, Logistic Regression, SVM, and Gradient Boosting
    - Builds DirectionalEnsemble and MovementRangeEnsemble
    - Evaluates and registers all models in ModelRegistry
    """
    pipeline = TimeSeriesFeaturePipeline(forward_horizon=1)
    X, y_dir, y_ret, y_rng, X_latest = pipeline.extract_features_and_targets(df)

    if len(X) < 25:
        raise ValueError(f"Insufficient historical bars ({len(X)}) to train machine learning models reliably.")

    # Chronological train-test split
    X_train, X_test, y_dir_train, y_dir_test = chronological_train_test_split(X, y_dir, test_size=test_size)
    _, _, y_ret_train, y_ret_test = chronological_train_test_split(X, y_ret, test_size=test_size)

    # Feature selection fitted strictly on training data
    selector = TimeSeriesFeatureSelector(max_features=16)
    selector.fit(X_train, y_dir_train)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    X_latest_sel = selector.transform(X_latest)

    # Scaling fitted strictly on training data
    scaler = TimeSeriesScaler()
    scaler.fit(X_train_sel)
    X_train_scaled = scaler.transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    X_latest_scaled = scaler.transform(X_latest_sel)

    # Candidate models for Direction Classification
    candidate_types = ["RandomForest", "LogisticRegression", "SVM", "GradientBoosting"]
    trained_clfs: Dict[str, Any] = {}
    model_metrics: Dict[str, Any] = {}
    weights: Dict[str, float] = {}

    for m_type in candidate_types:
        clf, metrics = train_direction_model(X_train_scaled, y_dir_train, X_test_scaled, y_dir_test, model_type=m_type)
        trained_clfs[m_type] = clf
        model_metrics[m_type] = metrics
        # Weight proportional to F1 or accuracy
        weights[m_type] = max(0.1, metrics.get("f1", 50.0))

        # Register individual model in registry
        model_registry.register_model(
            model_artifact=clf,
            symbol=symbol,
            model_type=m_type,
            target_type="direction",
            metrics=metrics,
            features=list(X_train_sel.columns),
            scaler=scaler,
            selector=selector,
            is_active=False,
        )

    # Build Directional Ensemble
    ensemble = DirectionalEnsemble(models_dict=trained_clfs, weights_dict=weights)
    ens_preds = ensemble.predict(X_test_scaled)
    ens_probs = ensemble.predict_proba(X_test_scaled)[:, 1]
    ens_metrics = calculate_classification_metrics(y_dir_test, ens_preds, ens_probs)

    # Train Return Regressors
    rf_reg, rf_reg_metrics = train_return_model(X_train_scaled, y_ret_train, X_test_scaled, y_ret_test, model_type="RandomForest")
    gb_reg, gb_reg_metrics = train_return_model(X_train_scaled, y_ret_train, X_test_scaled, y_ret_test, model_type="GradientBoosting")

    range_ensemble = MovementRangeEnsemble(reg_models={"RandomForest": rf_reg, "GradientBoosting": gb_reg})

    # Register active master ensemble
    active_model_id = model_registry.register_model(
        model_artifact={
            "direction_ensemble": ensemble,
            "range_ensemble": range_ensemble,
            "individual_models": trained_clfs,
        },
        symbol=symbol,
        model_type="Ensemble",
        target_type="direction",
        metrics=ens_metrics,
        features=list(X_train_sel.columns),
        scaler=scaler,
        selector=selector,
        is_active=True,
    )

    return {
        "symbol": symbol.upper(),
        "active_model_id": active_model_id,
        "selected_features": list(X_train_sel.columns),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "ensemble_metrics": ens_metrics,
        "candidate_metrics": model_metrics,
        "regressor_metrics": {"RandomForest": rf_reg_metrics, "GradientBoosting": gb_reg_metrics},
    }
