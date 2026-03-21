"""
Temporal Convolutional Network (TCN) — CNN component.
Uses dilated causal convolutions + residual blocks.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding)
        )

    def forward(self, x):
        return self.conv(x)[:, :, : x.size(2)]   # trim acausal padding


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)


class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network for time-series forecasting.
    Receptive field = 2^(num_levels) * (kernel_size - 1) + 1
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels = [num_channels] * num_levels
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = input_size if i == 0 else channels[i - 1]
            layers.append(
                ResidualBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, window, input_size)
        returns : (batch,)
        """
        x = x.permute(0, 2, 1)          # → (batch, features, time)
        out = self.network(x)            # → (batch, channels, time)
        out = self.fc(out[:, :, -1])     # last timestep → (batch, 1)
        return out.squeeze(-1)


def build_tcn(input_size: int) -> TCNForecaster:
    return TCNForecaster(
        input_size=input_size,
        num_channels=64,
        num_levels=4,
        kernel_size=3,
        dropout=0.2,
    )
