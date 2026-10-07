"""GRU (Gated Recurrent Unit) Neural Network for Financial Market Sequence Modeling.

Provides:
- Compact, high-efficiency recurrent dynamics with fewer parameters than LSTM
- Multi-task output heads: Direction Classification & Expected Return Regression
- Monte Carlo (MC) Dropout uncertainty intervals
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """Gated Recurrent Unit (GRU) model for fast temporal market sequence learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
        bidirectional: bool = False,
    ) -> None:
        """Initialize GRU Forecaster.

        Args:
            input_dim: Number of input features per timestep.
            hidden_dim: Number of recurrent hidden units.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional GRU.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        effective_hidden = hidden_dim * self.num_directions

        self.layer_norm = nn.LayerNorm(effective_hidden)
        self.fc_shared = nn.Sequential(
            nn.Linear(effective_hidden, effective_hidden // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        shared_dim = effective_hidden // 2

        # 1. Direction Head (Logit for binary classification)
        self.direction_head = nn.Linear(shared_dim, 1)

        # 2. Return Head (Continuous next-period fractional return)
        self.return_head = nn.Linear(shared_dim, 1)

        # 3. Uncertainty / Range Head (Log standard deviation)
        self.log_std_head = nn.Linear(shared_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        apply_mc_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim].
            apply_mc_dropout: Enables stochastic forward pass for distribution sampling.

        Returns:
            Tuple of (dir_logits, exp_return, log_std).
        """
        if apply_mc_dropout:
            self.train()

        gru_out, _ = self.gru(x)
        last_timestep = gru_out[:, -1, :]

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
        """Perform Monte Carlo Dropout inference across stochastic samples."""
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
