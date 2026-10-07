"""LSTM Neural Network Architecture for Multi-Task Financial Time-Series Forecasting.

Provides:
- Multi-layer LSTM recurrent encoder
- Dual-head outputs: Direction Classification (BCE) & Expected Return Regression (MSE)
- Monte Carlo (MC) Dropout for epistemic uncertainty estimation & confidence intervals
- Checkpoint serialization and state restoration
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM model designed for financial sequence modeling and uncertainty estimation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
        bidirectional: bool = False,
    ) -> None:
        """Initialize LSTM Forecaster.

        Args:
            input_dim: Number of input features per timestep.
            hidden_dim: Number of recurrent hidden units.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability applied between layers and in dense heads.
            bidirectional: Whether to use bidirectional LSTM (False by default for strict causal flow).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        effective_hidden = hidden_dim * self.num_directions

        # Dense projection trunk
        self.layer_norm = nn.LayerNorm(effective_hidden)
        self.fc_shared = nn.Sequential(
            nn.Linear(effective_hidden, effective_hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        shared_dim = effective_hidden // 2

        # 1. Direction Head (Logit for binary classification: 1=Up, 0=Down)
        self.direction_head = nn.Linear(shared_dim, 1)

        # 2. Return Head (Continuous next-period fractional return)
        self.return_head = nn.Linear(shared_dim, 1)

        # 3. Uncertainty / Range Head (Log standard deviation for Bayesian Gaussian bounds)
        self.log_std_head = nn.Linear(shared_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        apply_mc_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim].
            apply_mc_dropout: If True, forces dropout on during inference for Monte Carlo sampling.

        Returns:
            Tuple of:
            - dir_logits: Logits for upward movement [batch_size, 1].
            - exp_return: Expected fractional return [batch_size, 1].
            - log_std: Log-std parameter for uncertainty intervals [batch_size, 1].
        """
        if apply_mc_dropout:
            self.train()  # Enable dropout layers for stochastic forward pass

        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]  # Take final sequential context vector

        normalized = self.layer_norm(last_timestep)
        shared_repr = self.fc_shared(normalized)

        dir_logits = self.direction_head(shared_repr)
        exp_return = self.return_head(shared_repr)
        log_std = self.log_std_head(shared_repr)

        return dir_logits, exp_return, log_std

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_mc_samples: int = 25,
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform Monte Carlo Dropout inference to estimate prediction distribution.

        Args:
            x: Single input sequence tensor [1, seq_len, input_dim].
            num_mc_samples: Number of stochastic forward passes (default 25).
            num_samples: Optional alias for num_mc_samples.

        Returns:
            Dict containing direction probability, confidence, expected return, and uncertainty range.
        """
        actual_samples = num_samples if num_samples is not None else num_mc_samples
        device = next(self.parameters()).device
        x = x.to(device)

        probs_up = []
        returns = []

        with torch.no_grad():
            for _ in range(actual_samples):
                logits, ret, _ = self.forward(x, apply_mc_dropout=True)
                prob = torch.sigmoid(logits).item()
                probs_up.append(prob)
                returns.append(ret.item())

        mean_prob_up = float(np.mean(probs_up))
        prob_std = float(np.std(probs_up))

        mean_return = float(np.mean(returns))
        return_std = float(np.std(returns))

        # Direction determination
        predicted_direction = "BULLISH" if mean_prob_up >= 0.50 else "BEARISH"
        direction_confidence = mean_prob_up if mean_prob_up >= 0.50 else (1.0 - mean_prob_up)

        return {
            "predicted_direction": predicted_direction,
            "probability_up": round(mean_prob_up, 4),
            "probability_down": round(1.0 - mean_prob_up, 4),
            "confidence": round(direction_confidence, 4),
            "probability_uncertainty": round(prob_std, 4),
            "expected_return": round(mean_return, 6),
            "expected_return_pct": round(mean_return * 100.0, 4),
            "return_std": round(return_std, 6),
        }
