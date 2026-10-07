"""Model Evaluation & Statistical Performance Metrics Module.

Calculates:
- Regression: MAE, RMSE, R-squared (R²), Mean Directional Accuracy
- Classification: Directional Accuracy, Precision, Recall, F1 Score, Confusion Matrix
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def calculate_regression_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
) -> Dict[str, float]:
    """Calculate regression performance metrics (MAE, RMSE, R²)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if len(yt) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mean_directional_accuracy": 0.0}

    # Mean Absolute Error
    mae = float(np.mean(np.abs(yt - yp)))

    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # R-squared (Coefficient of Determination)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot > 1e-8:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 0.0

    # Mean Directional Accuracy (Sign agreement of price moves)
    sign_true = np.sign(yt)
    sign_pred = np.sign(yp)
    mda = float(np.mean(sign_true == sign_pred)) if len(yt) > 0 else 0.5

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 4),
        "r2_score": round(r2, 4),
        "mean_directional_accuracy": round(mda * 100.0, 2),
        "directional_accuracy": round(mda * 100.0, 2),
    }


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_pred: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
) -> Dict[str, Any]:
    """Calculate directional classification metrics.

    Calculates:
    - Directional Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix (TN, FP, FN, TP)
    - Total Out-of-Sample Test Samples
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    if len(yt) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {
                "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                "true_negative": 0, "false_positive": 0, "false_negative": 0, "true_positive": 0,
            },
            "total_samples": 0,
        }

    # Confusion matrix components
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    total = len(yt)

    # Directional Accuracy
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    # Precision: TP / (TP + FP)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) > 0:
        f1 = float(2.0 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    # Brier score / log confidence check if probabilities given
    confidence_spread = 0.0
    if y_prob is not None:
        probs = np.asarray(y_prob, dtype=float)
        confidence_spread = float(np.mean(np.abs(probs - 0.5)) * 2.0)  # 0 (uncertain) to 1 (high conviction)

    return {
        "accuracy": round(accuracy * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "total_samples": total,
        "conviction_score": round(confidence_spread * 100.0, 2),
    }


def evaluate_model_performance(
    y_test_dir: pd.Series,
    y_pred_dir: np.ndarray,
    y_prob_dir: Optional[np.ndarray],
    y_test_ret: Optional[pd.Series] = None,
    y_pred_ret: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Produce comprehensive evaluation report across classification and regression."""
    clf_metrics = calculate_classification_metrics(y_test_dir, y_pred_dir, y_prob_dir)

    reg_metrics = {}
    if y_test_ret is not None and y_pred_ret is not None:
        reg_metrics = calculate_regression_metrics(y_test_ret, y_pred_ret)

    return {
        "classification": clf_metrics,
        "regression": reg_metrics,
        "is_statistically_significant": bool(clf_metrics["accuracy"] >= 52.0),
    }
