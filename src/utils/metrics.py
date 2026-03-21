"""
src/utils/metrics.py
====================
Shared evaluation metrics used across all model scripts.
Centralizes MAE, MAPE, RMSE, and RL cost-reduction calculation
so all models are measured consistently.
"""

import numpy as np
import torch


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in original units (kW)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%). Target: below 5%."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in original units (kW)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """MAPE on PyTorch tensors for use inside training loops."""
    return float(
        torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + eps))).item() * 100
    )


def cost_reduction_pct(baseline_reward: float, agent_reward: float) -> float:
    """
    Percentage improvement of RL agent over always-on baseline.
    Positive = agent is better. Target: >= 15%.
    """
    if baseline_reward == 0:
        return 0.0
    return (agent_reward - baseline_reward) / abs(baseline_reward) * 100.0


def print_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute and print all forecast metrics. Returns dict for JSON logging."""
    m = {
        f"{model_name}_mae":  mae(y_true, y_pred),
        f"{model_name}_mape": mape(y_true, y_pred),
        f"{model_name}_rmse": rmse(y_true, y_pred),
    }
    print(f"[{model_name}]  MAE={m[f'{model_name}_mae']:.4f}  "
          f"MAPE={m[f'{model_name}_mape']:.2f}%  "
          f"RMSE={m[f'{model_name}_rmse']:.4f}")
    return m
