"""Chronological Sequence Dataset and Time-Series Preprocessing for Deep Learning.

Strictly enforces:
- Look-ahead bias elimination: Target $y_t$ corresponds to $t+1$ relative to features.
- No random shuffling: Sequences and train/validation/test partitions are strictly chronological.
- Leakage-free normalization: Scalers are fitted exclusively on the training partition.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceScaler:
    """Standard scaler fitted strictly on the chronological training window."""

    def __init__(self) -> None:
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    @property
    def mean_(self) -> Optional[np.ndarray]:
        return self.means

    @property
    def scale_(self) -> Optional[np.ndarray]:
        return self.stds

    def fit(self, data: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> "SequenceScaler":
        """Compute mean and standard deviation strictly on training data."""
        if isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns)
            arr = data.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(data, dtype=np.float64)
            self.feature_names = feature_names or [f"feat_{i}" for i in range(arr.shape[1])]

        self.means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        # Avoid zero division on constant features
        self.stds = np.where(stds > 1e-7, stds, 1.0)
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Scale data using training statistics without look-ahead leakage."""
        if self.means is None or self.stds is None:
            raise ValueError("SequenceScaler must be fit on training data before transform.")

        if isinstance(data, pd.DataFrame):
            arr = data.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(data, dtype=np.float64)

        scaled = (arr - self.means) / self.stds
        clean = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return clean.astype(np.float32)

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit on data and return scaled array."""
        return self.fit(data, feature_names).transform(data)


def chronological_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split a time-series DataFrame into train, validation, and test subsets.

    Args:
        df: Chronologically sorted DataFrame.
        train_ratio: Proportion for training (default 0.70).
        val_ratio: Proportion for validation (default 0.15).
        test_ratio: Proportion for testing (default 0.15).

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    total_len = len(df)
    if total_len < 30:
        raise ValueError(f"Insufficient samples ({total_len}) for chronological 3-way split.")

    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    return df_train, df_val, df_test


class TimeSeriesSequenceDataset(Dataset):
    """PyTorch Dataset yielding sequential input windows [seq_len, num_features] and forward targets."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets_dir: np.ndarray,
        targets_ret: np.ndarray,
    ) -> None:
        """Initialize Dataset.

        Args:
            sequences: Array of shape [num_samples, seq_len, num_features].
            targets_dir: Binary classification direction targets [num_samples].
            targets_ret: Continuous fractional return targets [num_samples].
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets_dir = torch.tensor(targets_dir, dtype=torch.float32)
        self.targets_ret = torch.tensor(targets_ret, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets_dir[idx], self.targets_ret[idx]


def create_chronological_sequences(
    features: np.ndarray,
    targets_dir: np.ndarray,
    targets_ret: np.ndarray,
    seq_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct sliding sequence windows preserving time order.

    For each sequence window i:
        inputs = features[i : i + seq_length]
        target = targets[i + seq_length - 1] (forward prediction relative to window end)

    Args:
        features: Scaled feature matrix of shape [N, D].
        targets_dir: Forward direction target of shape [N].
        targets_ret: Forward return target of shape [N].
        seq_length: Number of lookback timesteps.

    Returns:
        Tuple of (sequences, targets_dir, targets_ret).
    """
    n_samples = len(features)
    if n_samples < seq_length:
        raise ValueError(f"Length of features ({n_samples}) must be at least seq_length ({seq_length}).")

    seqs, y_dir, y_ret = [], [], []
    for i in range(n_samples - seq_length + 1):
        window = features[i : i + seq_length]
        target_idx = i + seq_length - 1
        seqs.append(window)
        y_dir.append(targets_dir[target_idx])
        y_ret.append(targets_ret[target_idx])

    return np.array(seqs, dtype=np.float32), np.array(y_dir, dtype=np.float32), np.array(y_ret, dtype=np.float32)


def prepare_deep_learning_dataloaders(
    features_df: pd.DataFrame,
    y_direction: pd.Series,
    y_return: pd.Series,
    latest_bar_features: Optional[pd.DataFrame] = None,
    seq_length: int = 20,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """End-to-end data preparation pipeline for PyTorch training and inference.

    Steps:
    1. Chronological 3-way split of features and targets.
    2. Fit SequenceScaler strictly on training partition.
    3. Scale validation, test, and latest inference windows.
    4. Generate chronological sequence tensors with lookback context.
    5. Construct PyTorch DataLoaders.

    Returns:
        Dictionary containing DataLoaders, scaler, dataset shapes, and inference tensor.
    """
    total_len = len(features_df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    # Guard: ensure seq_length is feasible for training partition
    if train_end <= seq_length:
        seq_length = max(3, train_end - 1)

    # Fit scaler strictly on training split (no look-ahead leakage)
    scaler = SequenceScaler()
    scaler.fit(features_df.iloc[:train_end])
    all_features_scaled = scaler.transform(features_df)

    y_dir_arr = np.asarray(y_direction, dtype=np.float32)
    y_ret_arr = np.asarray(y_return, dtype=np.float32)

    # 1. Training sequences: strictly from [0 : train_end]
    X_train_seq, y_dir_train_seq, y_ret_train_seq = create_chronological_sequences(
        features=all_features_scaled[:train_end],
        targets_dir=y_dir_arr[:train_end],
        targets_ret=y_ret_arr[:train_end],
        seq_length=seq_length,
    )

    # 2. Validation sequences: targets evaluate [train_end : val_end], using prior lookback context
    val_start = max(0, train_end - seq_length + 1)
    X_val_seq, y_dir_val_seq, y_ret_val_seq = create_chronological_sequences(
        features=all_features_scaled[val_start:val_end],
        targets_dir=y_dir_arr[val_start:val_end],
        targets_ret=y_ret_arr[val_start:val_end],
        seq_length=seq_length,
    )

    # 3. Test sequences: targets evaluate [val_end : total_len], using prior lookback context
    test_start = max(0, val_end - seq_length + 1)
    X_test_seq, y_dir_test_seq, y_ret_test_seq = create_chronological_sequences(
        features=all_features_scaled[test_start:],
        targets_dir=y_dir_arr[test_start:],
        targets_ret=y_ret_arr[test_start:],
        seq_length=seq_length,
    )

    train_ds = TimeSeriesSequenceDataset(X_train_seq, y_dir_train_seq, y_ret_train_seq)
    val_ds = TimeSeriesSequenceDataset(X_val_seq, y_dir_val_seq, y_ret_val_seq)
    test_ds = TimeSeriesSequenceDataset(X_test_seq, y_dir_test_seq, y_ret_test_seq)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build latest sequence for forward out-of-sample inference
    if latest_bar_features is not None and not latest_bar_features.empty:
        latest_scaled = scaler.transform(latest_bar_features)
        combined_tail = np.vstack([all_features_scaled[-(seq_length - 1):], latest_scaled])
    else:
        combined_tail = all_features_scaled[-seq_length:]

    latest_tensor = torch.tensor(combined_tail, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, num_features]

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "input_dim": features_df.shape[1],
        "seq_length": seq_length,
        "feature_names": list(features_df.columns),
        "latest_sequence_tensor": latest_tensor,
        "splits_info": {
            "train_sequences": len(X_train_seq),
            "val_sequences": len(X_val_seq),
            "test_sequences": len(X_test_seq),
            "feature_dim": features_df.shape[1],
        },
    }
