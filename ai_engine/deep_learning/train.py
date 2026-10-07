"""Deep Learning Training Loop with Early Stopping, Checkpointing, and Schedule Management.

Enforces:
- Early stopping with best checkpoint restoration
- Chronological validation and hold-out testing
- Strict training cooldown schedule (no training on live market ticks)
- Automatic registration in StockMind-AI Model Registry
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ai_engine.deep_learning.dataset import (
    SequenceScaler,
    prepare_deep_learning_dataloaders,
)
from ai_engine.deep_learning.gru import GRUForecaster
from ai_engine.deep_learning.lstm import LSTMForecaster
from ai_engine.ml.model_registry import ModelRegistry

MODELS_DIR = Path("d:/StockMind-AI/models")


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after patience epochs."""

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def __call__(
        self,
        val_metric: Optional[float] = None,
        model: Optional[nn.Module] = None,
        val_loss: Optional[float] = None,
    ) -> bool:
        loss_val = val_loss if val_loss is not None else (val_metric if val_metric is not None else 0.0)
        score = -loss_val if self.mode == "min" else loss_val

        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model is not None:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return self.early_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """Load the best model state recorded during training."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


class TrainingScheduleManager:
    """Guarantees deep learning models are not retrained on every tick.

    Enforces minimum cooldown intervals between retraining operations per symbol and model type.
    """

    def __init__(self, cooldown_minutes: int = 60) -> None:
        self.cooldown_minutes = cooldown_minutes
        self._last_trained: Dict[str, datetime] = {}

    def _get_key(self, symbol: str, model_type: str) -> str:
        return f"{symbol.upper()}_{model_type.upper()}"

    def can_retrain(self, symbol: str, model_type: str, force: bool = False) -> Tuple[bool, str]:
        """Check whether training cooldown has expired."""
        if force:
            return True, "Force retraining requested."

        key = self._get_key(symbol, model_type)
        last_time = self._last_trained.get(key)
        if last_time is None:
            return True, "No prior training record found."

        elapsed = datetime.now() - last_time
        cooldown_delta = timedelta(minutes=self.cooldown_minutes)

        if elapsed < cooldown_delta:
            remaining_mins = int((cooldown_delta - elapsed).total_seconds() / 60)
            return False, f"Cooldown active. Next scheduled training permitted in {remaining_mins} minutes."

        return True, "Cooldown window elapsed."

    def can_train(self, symbol: str, model_type: str, force: bool = False) -> bool:
        """Boolean check for training permission."""
        can, _ = self.can_retrain(symbol=symbol, model_type=model_type, force=force)
        return can

    def record_training_completed(self, symbol: str, model_type: str) -> None:
        """Mark current timestamp for training schedule."""
        key = self._get_key(symbol, model_type)
        self._last_trained[key] = datetime.now()

    def record_training(self, symbol: str, model_type: str) -> None:
        """Alias for record_training_completed."""
        self.record_training_completed(symbol, model_type)

    def remaining_cooldown(self, symbol: str, model_type: str) -> float:
        """Calculate remaining cooldown in minutes."""
        key = self._get_key(symbol, model_type)
        last_time = self._last_trained.get(key)
        if last_time is None:
            return 0.0
        elapsed = datetime.now() - last_time
        remaining = (timedelta(minutes=self.cooldown_minutes) - elapsed).total_seconds() / 60.0
        return max(remaining, 0.0)


# Global schedule manager instance
schedule_manager = TrainingScheduleManager(cooldown_minutes=60)


def train_deep_learning_model(
    symbol: str,
    features_df: Optional[pd.DataFrame] = None,
    y_direction: Optional[Union[pd.Series, np.ndarray]] = None,
    y_return: Optional[Union[pd.Series, np.ndarray]] = None,
    latest_bar_features: Optional[pd.DataFrame] = None,
    df: Optional[pd.DataFrame] = None,
    model_type: str = "LSTM",
    seq_length: Optional[int] = None,
    sequence_length: Optional[int] = None,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.20,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_epochs: Optional[int] = None,
    epochs: Optional[int] = None,
    patience: int = 8,
    force: bool = False,
    force_retrain: bool = False,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train time-series deep learning model with early stopping, checkpointing, and registry storage."""
    actual_seq_len = sequence_length if sequence_length is not None else (seq_length or 20)
    actual_epochs = epochs if epochs is not None else (max_epochs or 30)
    actual_force = force or force_retrain

    # If full dataframe was supplied instead of separate feature matrices, extract features
    if df is not None and (features_df is None or y_direction is None or y_return is None):
        from ai_engine.ml.preprocessing import TimeSeriesFeaturePipeline
        pipeline = TimeSeriesFeaturePipeline()
        features_df, y_direction, y_return, latest_bar_features = pipeline.fit_transform(df)

    seq_length = actual_seq_len
    max_epochs = actual_epochs
    force = actual_force
    """Train time-series deep learning model with early stopping, checkpointing, and registry storage.

    Args:
        symbol: Ticker symbol (e.g. RELIANCE).
        features_df: Chronologically sorted historical feature DataFrame.
        y_direction: Forward direction binary series.
        y_return: Forward return continuous series.
        latest_bar_features: Features for the latest unlabelled bar T.
        model_type: 'LSTM' or 'GRU'.
        seq_length: Number of lookback steps per sequence.
        hidden_dim: Number of hidden units.
        num_layers: Recurrent depth.
        dropout: Regularization probability.
        learning_rate: Adam optimizer learning rate.
        batch_size: Batch size.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        force: Override cooldown schedule.
        device: 'cpu' or 'cuda'.

    Returns:
        Dictionary of training results, metrics, model version, and registry artifact paths.
    """
    model_type = model_type.upper()
    if model_type not in ["LSTM", "GRU"]:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be 'LSTM' or 'GRU'.")

    # 1. Schedule check — prevent tick-by-tick thrashing
    can_train, reason = schedule_manager.can_retrain(symbol, model_type, force=force)
    if not can_train:
        return {
            "status": "skipped",
            "message": reason,
            "symbol": symbol,
            "model_type": model_type,
        }

    # 2. Select device (default CPU to avoid CUDA dependency issues)
    selected_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 3. Data preparation (strictly chronological, train scaler isolation)
    data_bundle = prepare_deep_learning_dataloaders(
        features_df=features_df,
        y_direction=y_direction,
        y_return=y_return,
        latest_bar_features=latest_bar_features,
        seq_length=seq_length,
        batch_size=batch_size,
    )

    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    scaler = data_bundle["scaler"]
    input_dim = data_bundle["input_dim"]

    # 4. Model initialization
    if model_type == "LSTM":
        model = LSTMForecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        model = GRUForecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    model.to(selected_device)

    # 5. Losses and Optimizer
    criterion_dir = nn.BCEWithLogitsLoss()
    criterion_ret = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode="min")

    history = {"train_loss": [], "val_loss": [], "val_dir_acc": []}

    # 6. Training Loop
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for x_batch, y_dir_batch, y_ret_batch in train_loader:
            x_batch = x_batch.to(selected_device)
            y_dir_batch = y_dir_batch.unsqueeze(1).to(selected_device)
            y_ret_batch = y_ret_batch.unsqueeze(1).to(selected_device)

            optimizer.zero_grad()
            logits, exp_ret, _ = model(x_batch)

            loss_dir = criterion_dir(logits, y_dir_batch)
            loss_ret = criterion_ret(exp_ret, y_ret_batch)
            total_loss = loss_dir + (10.0 * loss_ret)  # Balanced scale

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / max(batch_count, 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_dir = 0
        total_samples = 0

        with torch.no_grad():
            for x_v, y_dir_v, y_ret_v in val_loader:
                x_v = x_v.to(selected_device)
                y_dir_v = y_dir_v.unsqueeze(1).to(selected_device)
                y_ret_v = y_ret_v.unsqueeze(1).to(selected_device)

                logits_v, ret_v, _ = model(x_v)
                v_loss_dir = criterion_dir(logits_v, y_dir_v)
                v_loss_ret = criterion_ret(ret_v, y_ret_v)
                v_total = v_loss_dir + (10.0 * v_loss_ret)

                val_loss += v_total.item()
                val_batches += 1

                preds_binary = (torch.sigmoid(logits_v) >= 0.50).float()
                correct_dir += (preds_binary == y_dir_v).sum().item()
                total_samples += len(y_dir_v)

        avg_val_loss = val_loss / max(val_batches, 1)
        val_accuracy = (correct_dir / max(total_samples, 1)) * 100.0

        history["train_loss"].append(round(avg_train_loss, 5))
        history["val_loss"].append(round(avg_val_loss, 5))
        history["val_dir_acc"].append(round(val_accuracy, 2))

        scheduler.step(avg_val_loss)

        if early_stopping(avg_val_loss, model):
            break

    # Restore best checkpoint
    early_stopping.restore_best_weights(model)
    train_duration = round(time.time() - start_time, 2)

    # 7. Final Hold-Out Out-of-Sample Evaluation on Test Split
    from ai_engine.deep_learning.evaluate import evaluate_deep_learning_model
    test_metrics = evaluate_deep_learning_model(model, test_loader, selected_device)

    # 8. Checkpoint & Artifact Serialization inside D:\StockMind-AI\models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"{symbol.lower()}_{model_type.lower()}_{timestamp_str}.pt"
    artifact_path = MODELS_DIR / artifact_name

    checkpoint_payload = {
        "model_type": model_type,
        "symbol": symbol.upper(),
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_length": seq_length,
        "scaler": scaler,
        "feature_names": data_bundle["feature_names"],
        "metrics": test_metrics,
        "training_history": history,
        "created_at": datetime.now().isoformat(),
    }
    torch.save(checkpoint_payload, artifact_path)

    # 9. Register in Model Registry
    registry = ModelRegistry()
    registered_id = registry.register_model(
        model_artifact=checkpoint_payload,
        symbol=symbol,
        model_type=model_type,
        target_type="direction_and_return",
        metrics=test_metrics,
        features=data_bundle["feature_names"],
        scaler=scaler,
        hyperparameters={
            "seq_length": seq_length,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs_trained": len(history["train_loss"]),
            "train_duration_sec": train_duration,
        },
        is_active=True,
    )

    schedule_manager.record_training_completed(symbol, model_type)

    return {
        "status": "success",
        "symbol": symbol.upper(),
        "model_type": model_type,
        "model_id": registered_id,
        "artifact_path": str(artifact_path),
        "checkpoint_path": str(artifact_path),
        "epochs_trained": len(history["train_loss"]),
        "train_duration_seconds": train_duration,
        "test_metrics": test_metrics,
        "history": history,
        "latest_sequence_tensor": data_bundle["latest_sequence_tensor"],
    }
