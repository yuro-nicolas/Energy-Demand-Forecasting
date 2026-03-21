"""
LSTM Encoder-Decoder for energy demand forecasting.
Primary deep learning architecture.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Single-layer LSTM (baseline DL) or stacked encoder-decoder (full model).
    Set encoder_layers=1, decoder_layers=1 for the simple DL baseline.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        dropout: float = 0.2,
        horizon: int = 1,
    ):
        super().__init__()
        self.horizon = horizon

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout if decoder_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, window, input_size)
        returns : (batch,) — next-step forecast
        """
        _, (h, c) = self.encoder(x)

        # Use last observed target value as decoder seed
        dec_input = x[:, -1:, :1]          # (batch, 1, 1)
        dec_out, _ = self.decoder(dec_input, (h[-1:], c[-1:]))
        out = self.fc(self.dropout(dec_out[:, -1, :]))  # (batch, 1)
        return out.squeeze(-1)                           # (batch,)


def build_lstm_baseline(input_size: int) -> LSTMForecaster:
    """Simple single-layer LSTM — the DL baseline."""
    return LSTMForecaster(
        input_size=input_size,
        hidden_size=32,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.0,
    )


def build_lstm_full(input_size: int) -> LSTMForecaster:
    """Full stacked encoder-decoder LSTM."""
    return LSTMForecaster(
        input_size=input_size,
        hidden_size=64,
        encoder_layers=2,
        decoder_layers=1,
        dropout=0.2,
    )
