"""
run.py — Week 2 Checkpoint Pipeline
Usage: python run.py [--skip-download] [--epochs 30] [--seed 42]

Week 2 delivers:
  - Data cleaned and split
  - All models trained with full epochs and metrics logged
  - CNN (TCN) experiment running with first results
  - NLP component trained and prototyped
  - RL agent trained with learning curves
  - Initial metrics logged per model

NOT in Week 2 (these are Week 3 only):
  - model_comparison.csv / .png  (results table)
  - error_by_hour.png            (slice analysis)
  - slice_by_hour.csv            (slice analysis)
  - ablation writeup

Output directories:
  experiments/results/  <- model checkpoints, metrics JSON, saved models
  experiments/logs/     <- loss curves, RL learning curve
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = "experiments/results"
LOGS_DIR    = "experiments/logs"


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)

    # ── 1. Download ────────────────────────────────────────────────────────────
    if not args.skip_download:
        print("\n" + "="*60)
        print("STEP 1: Downloading dataset")
        print("="*60)
        from data.get_data import download_dataset, extract_dataset
        download_dataset()
        extract_dataset()
    else:
        print("[run] Skipping download (--skip-download set)")

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Preprocessing → data/processed/")
    print("="*60)
    from src.data_pipeline import build_dataset
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = build_dataset(
        seed=args.seed
    )

    # ── 3. Tabular baselines ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Tabular Baselines (Linear Regression + Random Forest)")
    print("        → experiments/results/")
    print("="*60)
    from src.models.tabular_baseline import (
        train_linear, train_random_forest, evaluate, save_model
    )
    lr     = train_linear(X_train, y_train)
    rf     = train_random_forest(X_train, y_train, seed=args.seed)
    lr_val = evaluate(lr, X_val, y_val, "LinearRegression-val")
    rf_val = evaluate(rf, X_val, y_val, "RandomForest-val")
    save_model(lr, os.path.join(RESULTS_DIR, "linear_regression.pkl"))
    save_model(rf, os.path.join(RESULTS_DIR, "random_forest.pkl"))
    with open(os.path.join(RESULTS_DIR, "tabular_metrics.json"), "w") as f:
        json.dump({
            "LinearRegression": {
                "val_MAE":  round(lr_val["MAE"],  4),
                "val_MAPE": round(lr_val["MAPE"], 4),
            },
            "RandomForest": {
                "val_MAE":  round(rf_val["MAE"],  4),
                "val_MAPE": round(rf_val["MAPE"], 4),
            },
        }, f, indent=2)
    print(f"[run] Tabular metrics → {RESULTS_DIR}/tabular_metrics.json")

    # ── 4. LSTM Baseline ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: LSTM Baseline (Simple DL Baseline)")
    print("        checkpoint → experiments/results/")
    print("        loss curve → experiments/logs/")
    print("="*60)
    from src.train import train_model
    train_model(
        model_name="lstm_baseline",
        epochs=args.epochs,
        seed=args.seed,
        patience=args.patience,
    )

    # ── 5. Full LSTM ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Full LSTM Encoder-Decoder (Primary DL Model)")
    print("        checkpoint → experiments/results/")
    print("        loss curve → experiments/logs/")
    print("="*60)
    train_model(
        model_name="lstm",
        epochs=args.epochs,
        seed=args.seed,
        patience=args.patience,
    )

    # ── 6. TCN (CNN component) ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: TCN — CNN Component (First Results)")
    print("        checkpoint → experiments/results/")
    print("        loss curve → experiments/logs/")
    print("="*60)
    train_model(
        model_name="tcn",
        epochs=args.epochs,
        seed=args.seed,
        patience=args.patience,
    )

    # ── 7. NLP component ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7: NLP Period Classifier (Scaffolded + Prototyped)")
    print("        → experiments/results/")
    print("="*60)
    from src.models.nlp_component import PeriodClassifier, summarize_forecast
    clf = PeriodClassifier()
    clf.train()
    clf.evaluate()
    clf.save(os.path.join(RESULTS_DIR, "nlp_classifier.pkl"))
    demo = summarize_forecast(
        forecast_kwh=2.1, hour=18, rl_action=0, device="washing machine"
    )
    print(f"\n[NLP] Example forecast summary:")
    print(f"  {demo}\n")

    # ── 8. RL agent ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8: RL Load Scheduling Agent (Reward Design + Learning Curves)")
    print("        Q-table      → experiments/results/")
    print("        learning curve → experiments/logs/")
    print("="*60)
    from src.rl_agent import train_rl_agent, evaluate_agent
    hour_sin  = X_test[:, -1, 7]
    hour_cos  = X_test[:, -1, 8]
    forecasts = X_test[:, -1, 0].astype(np.float32)
    hours     = (
        np.arctan2(hour_sin, hour_cos) * 24 / (2 * np.pi)
    ).astype(int) % 24

    agent, rewards = train_rl_agent(
        forecasts,
        hours.astype(np.float32),
        n_episodes=args.rl_episodes,
        seed=args.seed,
        save_path=os.path.join(RESULTS_DIR, "rl_agent.json"),
        plot_path=os.path.join(LOGS_DIR,    "rl_learning_curve.png"),
    )
    evaluate_agent(agent, forecasts, hours.astype(np.float32))

    # ── Week 2 Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("WEEK 2 CHECKPOINT COMPLETE")
    print("="*60)
    print("\nModels saved → experiments/results/")
    print("  linear_regression.pkl")
    print("  random_forest.pkl")
    print("  tabular_metrics.json")
    print("  lstm_baseline_best.pt  +  lstm_baseline_metrics.json")
    print("  lstm_best.pt           +  lstm_metrics.json")
    print("  tcn_best.pt            +  tcn_metrics.json")
    print("  nlp_classifier.pkl")
    print("  rl_agent.json")
    print("\nTraining logs → experiments/logs/")
    print("  lstm_baseline_loss_curve.png")
    print("  lstm_loss_curve.png")
    print("  tcn_loss_curve.png")
    print("  rl_learning_curve.png")
    print("\nNOTE: Results comparison tables, ablation plots, and slice")
    print("      analysis are Week 3 deliverables. Run Week 3 run.py")
    print("      to generate model_comparison.csv/.png, error_by_hour.png,")
    print("      and slice_by_hour.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 Checkpoint Pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--epochs",      type=int, default=30,
                        help="Training epochs per model")
    parser.add_argument("--patience",    type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--rl-episodes", type=int, default=500,
                        help="RL training episodes")
    args = parser.parse_args()
    main(args)
