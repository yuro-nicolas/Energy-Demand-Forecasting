"""
src/eval.py — Week 2 version
Reads and displays individual model metrics from saved JSON files.

Week 2 scope: show each model's val MAE and MAPE individually.
Week 3 scope: full comparison table, ablation plots, slice analysis.

Run: python src/eval.py
"""

import os
import json

RESULTS_DIR = "experiments/results"


def print_week2_metrics():
    print("\n" + "="*60)
    print("  WEEK 2 — Individual Model Metrics (Validation Set)")
    print("="*60)

    # ── Tabular baselines ──────────────────────────────────────────────────────
    tabular_path = os.path.join(RESULTS_DIR, "tabular_metrics.json")
    if os.path.exists(tabular_path):
        print("\n  [Non-DL Baselines]")
        print(f"  {'Model':<25} {'Val MAE':>10} {'Val MAPE':>12}")
        print("  " + "-"*50)
        with open(tabular_path) as f:
            t = json.load(f)
        for name, m in t.items():
            mae  = f"{m['val_MAE']:.4f} kW"
            mape = f"{m['val_MAPE']:.2f}%"
            print(f"  {name:<25} {mae:>10} {mape:>12}")
    else:
        print("\n  [Non-DL Baselines] — not trained yet")

    # ── DL models ─────────────────────────────────────────────────────────────
    dl_models = {
        "lstm_baseline": "LSTM Baseline",
        "lstm":          "LSTM Full",
        "tcn":           "TCN (CNN component)",
    }
    print("\n  [Deep Learning Models]")
    print(f"  {'Model':<25} {'Val MAE':>10} {'Test MAE':>10} {'Test MAPE':>12}")
    print("  " + "-"*60)
    any_found = False
    for key, label in dl_models.items():
        path = os.path.join(RESULTS_DIR, f"{key}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            val_mae  = f"{m.get('val_MAE',  0):.4f} kW"
            test_mae = f"{m.get('test_MAE', 0):.4f} kW"
            test_map = f"{m.get('test_MAPE',0):.2f}%"
            print(f"  {label:<25} {val_mae:>10} {test_mae:>10} {test_map:>12}")
            any_found = True
        else:
            print(f"  {label:<25} {'not trained':>10}")
    if not any_found:
        print("  No DL models trained yet. Run: python run.py --skip-download")

    # ── RL agent ───────────────────────────────────────────────────────────────
    print("\n  [RL Component]")
    rl_path = os.path.join(RESULTS_DIR, "rl_agent.json")
    if os.path.exists(rl_path):
        print("  Q-Learning agent trained and saved.")
        print("  Learning curve: experiments/logs/rl_learning_curve.png")
        print("  RL reward vs baseline comparison: Week 3")
    else:
        print("  RL agent not trained yet.")

    # ── NLP component ──────────────────────────────────────────────────────────
    print("\n  [NLP Component]")
    nlp_path = os.path.join(RESULTS_DIR, "nlp_classifier.pkl")
    if os.path.exists(nlp_path):
        print("  PeriodClassifier trained and saved.")
        print("  ForecastSummarizer producing natural language summaries.")
    else:
        print("  NLP classifier not trained yet.")

    print("\n" + "="*60)
    print("  NOTE: Full comparison table, ablation analysis, and")
    print("  slice analysis will be generated in Week 3.")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_week2_metrics()
