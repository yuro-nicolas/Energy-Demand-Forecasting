"""
Evaluation script for trained models.
  - Model comparison table and bar chart (all 5 models on test set)
  - Ablation study (LSTM Baseline vs LSTM Full vs TCN)
  - Error/slice analysis by hour of day
  - RL reward vs baseline summary

Saves to experiments/results/:
  model_comparison.csv
  model_comparison.png
  ablation_summary.json
  ablation_plot.png
  error_by_hour.png
  slice_by_hour.csv
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.data_pipeline import load_dataset, FEATURE_COLS
from src.models.lstm_model import build_lstm_baseline, build_lstm_full
from src.models.tcn_model import build_tcn
from src.models.tabular_baseline import flatten, load_model as load_sklearn

RESULTS_DIR = "experiments/results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_dl(name, input_size):
    path = os.path.join(RESULTS_DIR, f"{name}_best.pt")
    if not os.path.exists(path):
        return None
    builders = {
        "lstm_baseline": build_lstm_baseline,
        "lstm":          build_lstm_full,
        "tcn":           build_tcn,
    }
    m = builders[name](input_size).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def _predict(model, X):
    return model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()


def _row(y_true, y_pred, name):
    return {
        "Model":    name,
        "MAE":      round(mean_absolute_error(y_true, y_pred), 4),
        "MAPE (%)": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_evaluation():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_dataset()
    input_size = X_test.shape[2]
    rows = []

    # ── Tabular baselines ──────────────────────────────────────────────────────
    for name, fname in [("Linear Regression", "linear_regression.pkl"),
                        ("Random Forest",     "random_forest.pkl")]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            m     = load_sklearn(path)
            preds = m.predict(flatten(X_test))
            rows.append(_row(y_test, preds, name))
        else:
            print(f"[eval] Skipping {name} — not found.")

    # ── DL models ─────────────────────────────────────────────────────────────
    dl_labels = {
        "lstm_baseline": "LSTM Baseline",
        "lstm":          "LSTM Full",
        "tcn":           "TCN (CNN)",
    }
    dl_preds = {}
    for key, label in dl_labels.items():
        m = _load_dl(key, input_size)
        if m is not None:
            p = _predict(m, X_test)
            dl_preds[key] = p
            rows.append(_row(y_test, p, label))
        else:
            print(f"[eval] Skipping {label} — checkpoint not found.")

    if not rows:
        print("[eval] No trained models found. Run python run.py first.")
        return

    # ── 1. Comparison table ────────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values("MAE")
    print("\n=== MODEL COMPARISON (Test Set) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df.plot.bar(x="Model", y="MAE",      ax=axes[0],
                legend=False, color="steelblue", edgecolor="white")
    df.plot.bar(x="Model", y="MAPE (%)", ax=axes[1],
                legend=False, color="darkorange", edgecolor="white")
    axes[0].set_title("MAE by Model — Test Set")
    axes[0].set_ylabel("MAE (kW)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].set_title("MAPE by Model — Test Set")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=100)
    plt.close()
    print(f"[eval] model_comparison.png → {RESULTS_DIR}/")

    # ── 2. Ablation study ─────────────────────────────────────────────────────
    # Ablation 1: LSTM Baseline vs LSTM Full (depth ablation)
    # Ablation 2: LSTM Full vs TCN (architecture ablation)
    ablation_rows = []
    for key, label in dl_labels.items():
        if key in dl_preds:
            mae  = mean_absolute_error(y_test, dl_preds[key])
            mape = mean_absolute_percentage_error(y_test, dl_preds[key]) * 100
            ablation_rows.append({"Model": label, "MAE": round(mae,4),
                                  "MAPE (%)": round(mape,2)})

    if ablation_rows:
        df_abl = pd.DataFrame(ablation_rows)
        ablation_summary = {
            "ablation_1": {
                "name": "Depth Ablation: LSTM Baseline vs LSTM Full",
                "description": "Isolates the contribution of additional encoder layers and attention",
                "models": {r["Model"]: {"MAE": r["MAE"], "MAPE": r["MAPE (%)"]}
                           for r in ablation_rows if "LSTM" in r["Model"]},
            },
            "ablation_2": {
                "name": "Architecture Ablation: LSTM Full vs TCN",
                "description": "Compares RNN-based vs CNN-based sequence modeling",
                "models": {r["Model"]: {"MAE": r["MAE"], "MAPE": r["MAPE (%)"]}
                           for r in ablation_rows if r["Model"] in ["LSTM Full", "TCN (CNN)"]},
            },
        }
        with open(os.path.join(RESULTS_DIR, "ablation_summary.json"), "w") as f:
            json.dump(ablation_summary, f, indent=2)

        # Ablation bar chart
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = ["#5B8DB8", "#2E6DB4", "#E07B39"][:len(df_abl)]
        df_abl.plot.bar(x="Model", y="MAE",      ax=axes[0],
                        legend=False, color=colors, edgecolor="white")
        df_abl.plot.bar(x="Model", y="MAPE (%)", ax=axes[1],
                        legend=False, color=colors, edgecolor="white")
        axes[0].set_title("Ablation — MAE (Test Set)")
        axes[0].set_ylabel("MAE (kW)")
        axes[0].tick_params(axis="x", rotation=20)
        axes[1].set_title("Ablation — MAPE (Test Set)")
        axes[1].set_ylabel("MAPE (%)")
        axes[1].tick_params(axis="x", rotation=20)
        plt.suptitle("Ablation Study: LSTM Baseline → LSTM Full → TCN",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "ablation_plot.png"),
                    dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[eval] ablation_plot.png → {RESULTS_DIR}/")
        print(f"[eval] ablation_summary.json → {RESULTS_DIR}/")

        print("\n=== ABLATION 1: Depth Ablation (LSTM Baseline vs LSTM Full) ===")
        lstm_models = [r for r in ablation_rows if "LSTM" in r["Model"]]
        for r in lstm_models:
            print(f"  {r['Model']:<20}  MAE={r['MAE']:.4f}  MAPE={r['MAPE (%)']:.2f}%")

        print("\n=== ABLATION 2: Architecture Ablation (LSTM Full vs TCN) ===")
        arch_models = [r for r in ablation_rows
                       if r["Model"] in ["LSTM Full", "TCN (CNN)"]]
        for r in arch_models:
            print(f"  {r['Model']:<20}  MAE={r['MAE']:.4f}  MAPE={r['MAPE (%)']:.2f}%")

    # ── 3. Slice analysis ──────────────────────────────────────────────────────
    _slice_analysis(X_test, y_test, input_size, dl_preds)


def _slice_analysis(X_test, y_test, input_size, dl_preds):
    """Error breakdown by hour of day using best available DL model."""
    best_key  = None
    best_preds = None
    for key in ["lstm", "tcn", "lstm_baseline"]:
        if key in dl_preds:
            best_key   = key
            best_preds = dl_preds[key]
            break
    if best_preds is None:
        print("[eval] No DL model available for slice analysis.")
        return

    errors   = np.abs(best_preds - y_test)
    hour_sin = X_test[:, -1, 7]
    hour_cos = X_test[:, -1, 8]
    hours    = (np.arctan2(hour_sin, hour_cos) * 24 / (2 * np.pi)).astype(int) % 24

    df_sl      = pd.DataFrame({"hour": hours, "abs_error": errors})
    hourly_err = df_sl.groupby("hour")["abs_error"].mean()
    hourly_err.to_csv(os.path.join(RESULTS_DIR, "slice_by_hour.csv"))

    label = {"lstm_baseline": "LSTM Baseline",
             "lstm": "LSTM Full", "tcn": "TCN"}.get(best_key, best_key)

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(hourly_err.index, hourly_err.values,
                  color="steelblue", edgecolor="white")

    # Highlight peak hours (7-9am and 6-9pm)
    for i, bar in enumerate(bars):
        if i in [7, 8, 18, 19, 20, 21]:
            bar.set_color("darkorange")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Absolute Error (kW)")
    ax.set_title(f"Forecast Error by Hour of Day — {label}\n"
                 f"(orange = must-run / peak hours)")
    ax.set_xticks(range(0, 24))
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="steelblue",  label="Off-peak hours"),
        Patch(color="darkorange", label="Peak / must-run hours"),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_by_hour.png"), dpi=100)
    plt.close()
    print(f"[eval] error_by_hour.png → {RESULTS_DIR}/")
    print(f"[eval] slice_by_hour.csv → {RESULTS_DIR}/")

    peak_err    = hourly_err[[7, 8, 18, 19, 20, 21]].mean()
    offpeak_err = hourly_err.drop([7, 8, 18, 19, 20, 21]).mean()
    print(f"\n=== SLICE ANALYSIS ({label}) ===")
    print(f"  Peak hours (7-9am, 6-9pm) avg error : {peak_err:.4f} kW")
    print(f"  Off-peak hours avg error             : {offpeak_err:.4f} kW")
    print(f"  Highest error hour: {hourly_err.idxmax():02d}:00 "
          f"({hourly_err.max():.4f} kW)")
    print(f"  Lowest  error hour: {hourly_err.idxmin():02d}:00 "
          f"({hourly_err.min():.4f} kW)")


if __name__ == "__main__":
    run_evaluation()
