"""
Training script for LSTM and TCN forecasting models.

Output directories:
  experiments/results/   ← model checkpoints (.pt) and metrics (.json)
  experiments/logs/      ← loss curve plots (training logs)

Run: python src/train.py --model lstm
     python src/train.py --model tcn
     python src/train.py --model lstm_baseline
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_pipeline import load_dataset, build_dataset
from src.models.lstm_model import build_lstm_baseline, build_lstm_full
from src.models.tcn_model import build_tcn

RESULTS_DIR = "experiments/results"  # checkpoints + metrics
LOGS_DIR    = "experiments/logs"     # loss curves + training logs


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(X_train, y_train, X_val, y_val, batch_size=256):
    def to_tensor(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loader = DataLoader(to_tensor(X_train, y_train),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(to_tensor(X_val,   y_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def mape(y_true, y_pred, eps=1e-8):
    return float(torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + eps))) * 100)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_mape = 0.0, 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        total_loss += criterion(preds, y_batch).item() * len(y_batch)
        total_mape += mape(y_batch, preds) * len(y_batch)
    n = len(loader.dataset)
    return total_loss / n, total_mape / n


def train_model(
    model_name: str = "lstm",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    patience: int = 5,
):
    set_seed(seed)
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    print(f"[train] Device: {device}")

    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_dataset()
    except FileNotFoundError:
        print("[train] Processed data not found — running pipeline …")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = build_dataset()

    input_size = X_train.shape[2]
    print(f"[train] input_size={input_size}, train={X_train.shape}, val={X_val.shape}")

    if model_name == "lstm_baseline":
        model = build_lstm_baseline(input_size)
    elif model_name == "lstm":
        model = build_lstm_full(input_size)
    elif model_name == "tcn":
        model = build_tcn(input_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model        = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model: {model_name} | Params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.L1Loss()

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size)

    best_val_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": [], "val_mape": []}

    # ── Checkpoint → experiments/results/ ─────────────────────────────────────
    ckpt_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mape_ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mape"].append(val_mape_)

        print(f"  Epoch {epoch:3d}/{epochs} | train_MAE={train_loss:.4f} | "
              f"val_MAE={val_loss:.4f} | val_MAPE={val_mape_:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[train] Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    # ── Test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    with torch.no_grad():
        preds    = model(torch.from_numpy(X_test).to(device))
        test_mae = criterion(preds, torch.from_numpy(y_test).to(device)).item()
        test_map = mape(torch.from_numpy(y_test).to(device), preds)

    print(f"\n[train] === Test Results ({model_name}) ===")
    print(f"  MAE  = {test_mae:.4f}")
    print(f"  MAPE = {test_map:.2f}%")

    # ── Metrics → experiments/results/ ────────────────────────────────────────
    metrics = {
        "model": model_name, "best_epoch": best_epoch,
        "val_MAE": best_val_loss, "test_MAE": test_mae,
        "test_MAPE": test_map, "params": total_params,
        "seed": seed, "lr": lr, "batch_size": batch_size,
    }
    with open(os.path.join(RESULTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Loss curves → experiments/logs/ ───────────────────────────────────────
    _save_loss_curve(history, model_name)

    return model, metrics, history


def _save_loss_curve(history, model_name):
    """Save training/validation loss curves to experiments/logs/."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"{model_name}_loss_curve.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train MAE", color="steelblue")
    ax1.plot(history["val_loss"],   label="Val MAE",   color="orange")
    ax1.set_title(f"{model_name} — Loss Curves")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MAE"); ax1.legend()

    ax2.plot(history["val_mape"], label="Val MAPE", color="darkorange")
    ax2.set_title(f"{model_name} — Val MAPE (%)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAPE (%)"); ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[train] Loss curve → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="lstm", choices=["lstm", "lstm_baseline", "tcn"])
    parser.add_argument("--epochs",   default=30,  type=int)
    parser.add_argument("--lr",       default=1e-3, type=float)
    parser.add_argument("--batch",    default=256, type=int)
    parser.add_argument("--seed",     default=42,  type=int)
    parser.add_argument("--patience", default=5,   type=int)
    args = parser.parse_args()
    train_model(model_name=args.model, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch, seed=args.seed, patience=args.patience)
