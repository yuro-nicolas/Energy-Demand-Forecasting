"""
src/interface.py — User-Facing Control Panel
=============================================

User Inputs (Task-Based Design)
--------------------------------
  1. Task name          — a label for what the user wants to run
  2. Power draw (kW)    — how much power the task consumes
  3. Duration needed    — how many hours the task actually needs to run
  4. Deadline           — by what hour the task must be finished
  5. Interruptible      — can the run be split across non-consecutive hours?
  6. Comfort priority   — adjusts the RL comfort penalty weight
  7. Manual override    — force-run or force-defer for the current hour
  8. NLP query          — ask about period classification in plain text
  9. Opt-out            — disengage automated scheduling entirely

System Outputs
--------------
  A. NLP plain-language summary  (ForecastSummarizer)
  B. Period classification label (PeriodClassifier)
  C. RL scheduling decision      (QLearningAgent)
  D. 24-hour forecast table      (LSTM / TCN / RF / LR, real kWh)
  E. Savings-vs-baseline meter   (real kWh × device power draw)
  F. Action log                  (timestamped history of decisions)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from typing import Optional

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  COMFORT PROFILES
# ══════════════════════════════════════════════════════════════════════════════

COMFORT_PROFILES = {
    "high_savings": 2.0,
    "balanced":     1.0,
    "high_comfort": 0.2,
}

MUST_RUN_HOURS = {7, 8, 18, 19, 20}   # mirrors LoadSchedulingEnv

# Index of Global_active_power in FEATURE_COLS (the forecast target column)
_GAP_IDX  = 0
# Indices of hour_sin and hour_cos in FEATURE_COLS
_HSIN_IDX = 7
_HCOS_IDX = 8


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL LOADERS  (unchanged — forecast/RL/NLP loading is model-side)
# ══════════════════════════════════════════════════════════════════════════════

def _load_forecast_model(model_name: str, input_size: int):
    """
    Load a trained forecast model.  Returns (predict_fn, label).

    predict_fn(x_window: ndarray [window, features]) -> float (SCALED space).
    Caller must inverse-transform for display; pass raw to RL obs.
    """
    results = "experiments/results"

    if model_name in ("lstm", "lstm_baseline", "tcn"):
        import torch
        from src.models.lstm_model import build_lstm_baseline, build_lstm_full
        from src.models.tcn_model  import build_tcn

        builders = {
            "lstm_baseline": build_lstm_baseline,
            "lstm":          build_lstm_full,
            "tcn":           build_tcn,
        }
        path = os.path.join(results, f"{model_name}_best.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Run 'python run.py' to train models first."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = builders[model_name](input_size).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()

        def predict_dl(x_window):
            import torch
            with torch.no_grad():
                t = torch.from_numpy(x_window[np.newaxis]).float().to(device)
                return float(m(t).cpu().numpy()[0])

        return predict_dl, model_name.upper()

    elif model_name in ("rf", "random_forest"):
        import pickle
        path = os.path.join(results, "random_forest.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Random Forest not found: {path}")
        with open(path, "rb") as f:
            m = pickle.load(f)
        return lambda xw: float(m.predict(xw.reshape(1, -1))[0]), "Random Forest"

    elif model_name in ("lr", "linear"):
        import pickle
        path = os.path.join(results, "linear_regression.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Linear Regression not found: {path}")
        with open(path, "rb") as f:
            m = pickle.load(f)
        return lambda xw: float(m.predict(xw.reshape(1, -1))[0]), "Linear Regression"

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose: lstm, lstm_baseline, tcn, rf, lr"
        )


def _load_rl_agent():
    """Load trained Q-Learning agent with deterministic (greedy) policy."""
    from src.rl_agent import QLearningAgent
    path = "experiments/results/rl_agent.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL agent not found: {path}\n"
            "Run 'python run.py' to train first."
        )
    agent = QLearningAgent()
    agent.load(path)
    agent.epsilon = 0.0   # deterministic inference
    return agent


def _load_nlp():
    from src.models.nlp_component import PeriodClassifier, summarize_forecast

    path = "experiments/results/nlp_classifier.pkl"

    try:
        if os.path.exists(path):
            clf = PeriodClassifier.load(path)
            print("[NLP] Loaded saved classifier.")
        else:
            raise FileNotFoundError
    except Exception:
        print("[NLP] Corrupted or incompatible model. Re-training...")
        clf = PeriodClassifier()
        clf.train()
        clf.save(path)

    return clf, summarize_forecast

def _load_data_and_scaler():
    """Return X_test, y_test, and the fitted StandardScaler."""
    from src.data_pipeline import load_dataset
    _, _, _, _, X_test, y_test, scaler = load_dataset()
    return X_test, y_test, scaler


def _inverse_scale(scaled_val: float, scaler) -> float:
    """
    Inverse-transform one scaled Global_active_power value to real kWh.
    scaler was fit on all 12 FEATURE_COLS; GAP is column index 0.
    """
    if scaler is None:
        return scaled_val
    return float(scaled_val * scaler.scale_[_GAP_IDX] + scaler.mean_[_GAP_IDX])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  24-HOUR FORECAST BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_24h_forecasts(X_test, forecast_fn, scaler, seed: int):
    """
    Randomly select a consecutive 24-sample block from X_test, run the
    forecast model on each window, and return two parallel lists:

      forecasts_real   : [(hour_of_day, real_kwh), ...]   — display & savings
      forecasts_scaled : [(hour_of_day, scaled_val), ...]  — RL observation
    """
    max_start = max(0, len(X_test) - 24)
    rng   = random.Random(seed)
    start = rng.randint(0, max_start)

    forecasts_real   = []
    forecasts_scaled = []

    for i in range(24):
        x_window = X_test[start + i]               # (window_size, n_features)

        scaled_pred = forecast_fn(x_window)
        real_kwh    = max(0.0, _inverse_scale(scaled_pred, scaler))

        h_sin = float(x_window[-1, _HSIN_IDX])
        h_cos = float(x_window[-1, _HCOS_IDX])
        hour  = int(round(np.arctan2(h_sin, h_cos) * 24 / (2 * np.pi))) % 24

        forecasts_real.append((hour, real_kwh))
        forecasts_scaled.append((hour, float(scaled_pred)))

    return forecasts_real, forecasts_scaled


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TASK DESCRIPTOR  (replaces APPLIANCES registry)
# ══════════════════════════════════════════════════════════════════════════════

class TaskDescriptor:
    """
    Plain user-defined task: a name, power draw, how long it needs to run,
    and a deadline by which it must finish.

    duration_hours  — total hours of actual operation required (e.g. 2)
    deadline_hour   — latest hour by which all runs must be completed (0-23)
    interruptible   — if False, try to keep the run block contiguous
    """

    def __init__(
        self,
        name:           str,
        power_kw:       float,
        duration_hours: int,
        deadline_hour:  int,
        interruptible:  bool = True,
    ):
        self.name           = name
        self.power_kw       = power_kw
        self.duration_hours = duration_hours
        self.deadline_hour  = deadline_hour
        self.interruptible  = interruptible

    def __str__(self):
        mode = "interruptible" if self.interruptible else "must-run-contiguously"
        return (
            f"Task '{self.name}' | {self.power_kw:.2f} kW | "
            f"{self.duration_hours} hr(s) needed | "
            f"deadline {self.deadline_hour:02d}:00 | {mode}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SCHEDULING SESSION  (task-aware)
# ══════════════════════════════════════════════════════════════════════════════

class SchedulingSession:
    """
    One scheduling session for a single user-defined task over a 24-hour horizon.

    Key differences from the appliance-based version:
      • The task runs for exactly `task.duration_hours` hours, then stops.
      • Past the deadline the task is forced on if still incomplete.
      • If not interruptible, once a run block starts it continues until
        duration_hours is exhausted (no gaps allowed mid-block).
    """

    def __init__(self, task: TaskDescriptor, comfort_penalty: float):
        self.task            = task
        self.comfort_penalty = comfort_penalty
        self.opt_out         = False
        self.action_log: list[dict] = []
        self.total_savings   = 0.0
        self._runs_so_far    = 0
        self._in_block       = False   # tracks contiguous-block state

    @property
    def _task_complete(self) -> bool:
        return self._runs_so_far >= self.task.duration_hours

    def decide(
        self,
        hour:            int,
        real_kwh:        float,
        scaled_forecast: float,
        agent,
        clf,
        summarize_fn,
        override:        Optional[int] = None,
    ) -> dict:
        """
        Produce a scheduling decision for one hour.

        Returns a result dict with all relevant fields for logging/display.
        """

        # ── period classification ─────────────────────────────────────────────
        period_text  = _hour_to_text(hour, real_kwh)
        period_label = clf.predict(period_text)

        # ── RL observation ────────────────────────────────────────────────────
        h_sin   = np.sin(2 * np.pi * hour / 24)
        h_cos   = np.cos(2 * np.pi * hour / 24)
        load_on = float(
            self.action_log[-1]["final_action"] if self.action_log else 0
        )
        rl_obs = np.array(
            [float(np.clip(scaled_forecast, -3.0, 3.0)), h_sin, h_cos, load_on],
            dtype=np.float32,
        )

        # ── base RL recommendation ────────────────────────────────────────────
        rl_action = agent.select_action(rl_obs)

        # ── task-complete: never run again once duration is met ───────────────
        if self._task_complete:
            rl_action = 0

        # ── contiguous-block enforcement ──────────────────────────────────────
        # Once we start running a non-interruptible task, keep it going until
        # duration_hours is reached (don't let the RL agent create a gap).
        if not self.task.interruptible and self._in_block and not self._task_complete:
            rl_action = 1

        # ── deadline enforcement ──────────────────────────────────────────────
        if not self._task_complete:
            hours_remaining   = self.task.deadline_hour - hour
            runs_still_needed = self.task.duration_hours - self._runs_so_far

            # No slack left before deadline — must run every remaining hour
            if 0 < hours_remaining <= runs_still_needed:
                rl_action = 1
            # Already past deadline and task not done
            elif hour >= self.task.deadline_hour:
                rl_action = 1

        # ── override / opt-out ────────────────────────────────────────────────
        if self.opt_out:
            final_action = 1 if not self._task_complete else 0
            source = "OPT-OUT (manual control)"
        elif override is not None:
            final_action = int(override)
            source = "MANUAL OVERRIDE by user"
        else:
            final_action = rl_action
            source = "RL Agent (automated)"

        # ── update contiguous-block tracker ───────────────────────────────────
        if not self.task.interruptible:
            if final_action == 1:
                self._in_block = True
            elif self._in_block and not self._task_complete:
                # Gap in a non-interruptible block — reset; must restart
                self._in_block = False

        # ── savings: real kWh × task power draw ───────────────────────────────
        # Baseline: task running every hour of the day (always-on).
        # Savings accrued only for hours where RL defers.
        cost_always_on = real_kwh * self.task.power_kw
        cost_agent     = real_kwh * self.task.power_kw * final_action
        savings_step   = cost_always_on - cost_agent

        self.total_savings += savings_step
        self._runs_so_far  += final_action

        # ── NLP summary ───────────────────────────────────────────────────────
        summary = summarize_fn(
            forecast_kwh=real_kwh,
            hour=hour,
            rl_action=final_action,
            device=self.task.name,
        )

        result = {
            "hour":               hour,
            "forecast_kwh":       round(real_kwh, 3),
            "period_label":       period_label,
            "rl_action":          rl_action,
            "final_action":       final_action,
            "source":             source,
            "summary":            summary,
            "savings_step":       round(savings_step, 4),
            "cumulative_savings": round(self.total_savings, 4),
            "runs_so_far":        self._runs_so_far,
            "task_complete":      self._task_complete,
            "timestamp":          datetime.datetime.now().isoformat(),
        }
        self.action_log.append(result)
        return result

    def save_log(self, path: str = "experiments/results/action_log.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.action_log, f, indent=2)
        print(f"\n[LOG] Action log saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _hour_to_text(hour: int, real_kwh: float) -> str:
    if 17 <= hour <= 21 or 7 <= hour <= 9:
        level = "high demand"
    elif 22 <= hour or hour <= 5:
        level = "late night minimal usage"
    else:
        level = "moderate activity"
    return f"{level} electricity forecast {real_kwh:.2f} kW hour {hour}"


def _print_header(title: str):
    w = 64
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def _print_decision(result: dict):
    if result["task_complete"] and result["final_action"] == 0:
        action_str = "✓  TASK DONE  (no more runs needed)"
    elif result["final_action"] == 1:
        action_str = "▶  RUN NOW"
    else:
        action_str = "⏸  DEFER"

    done_marker = f"  [{result['runs_so_far']} hr(s) done]"
    print(f"\n  ┌─ Hour {result['hour']:02d}:00 "
          f"──────────────────────────────────────")
    print(f"  │  Forecast    : {result['forecast_kwh']:.3f} kWh (real)")
    print(f"  │  Period      : {result['period_label'].upper()}")
    print(f"  │  Decision    : {action_str}{done_marker}")
    print(f"  │  Source      : {result['source']}")
    print(f"  │  Savings(hr) : {result['savings_step']:+.4f}  "
          f"Cumulative: {result['cumulative_savings']:+.4f}")
    print(f"  └─ {result['summary']}")


def _classify_period_simple(hour: int) -> str:
    if 17 <= hour <= 21 or 7 <= hour <= 9:
        return "peak"
    elif 22 <= hour or hour <= 5:
        return "low"
    return "off-peak"


def _print_forecast_table(forecasts_real: list, task: TaskDescriptor):
    _print_header("24-HOUR DEMAND FORECAST  (real kWh)")
    print(f"  Task   : {task.name}  |  {task.power_kw:.2f} kW  |  "
          f"{task.duration_hours} hr(s) needed  |  deadline {task.deadline_hour:02d}:00")
    print()
    print(f"  {'Hour':>4}  {'Forecast (kWh)':>15}  {'Period':>12}  "
          f"{'Load Level':>10}")
    print("  " + "-" * 52)
    for h, kwh in forecasts_real:
        period = _classify_period_simple(h)
        level  = "HIGH" if kwh > 1.5 else ("MED" if kwh > 0.8 else "LOW")
        dl_mark = " ◀ DEADLINE" if h == task.deadline_hour else ""
        print(f"  {h:>4}  {kwh:>15.3f}  {period:>12}  {level:>10}{dl_mark}")


def _print_savings_meter(session: SchedulingSession, baseline_cost: float):
    _print_header("SAVINGS VS. ALWAYS-ON BASELINE")
    savings_pct = (
        session.total_savings / baseline_cost * 100
        if baseline_cost > 0 else 0.0
    )
    bar_width = 40
    filled    = max(0, min(bar_width, int(savings_pct / 100 * bar_width)))
    bar       = "█" * filled + "░" * (bar_width - filled)
    print(f"\n  Task       : {session.task.name} ({session.task.power_kw:.2f} kW)")
    print(f"  Savings    : {session.total_savings:+.4f} kWh·units")
    print(f"  Baseline   : {baseline_cost:.4f} kWh·units")
    print(f"  [{bar}] {savings_pct:.1f}%")
    print(f"\n  Hours run  : {session._runs_so_far} / {session.task.duration_hours} required")
    status = "✓ COMPLETE" if session._task_complete else "✗ INCOMPLETE"
    print(f"  Status     : {status}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  INTERACTIVE PROMPTS  (task-based)
# ══════════════════════════════════════════════════════════════════════════════

def _prompt_task() -> TaskDescriptor:
    """Interactively gather task details from the user."""
    _print_header("TASK SETUP")
    print("  Define the task you want to schedule.")
    print("  Examples: 'Laundry', 'EV Charging', 'Dishwasher', 'Pre-cooling'\n")

    name = input("  Task name: ").strip() or "My Task"

    # Power draw
    while True:
        try:
            power_kw = float(input("  Power draw in kW (e.g. 0.5, 2.0, 7.2): ").strip())
            if power_kw > 0:
                break
            print("  Must be a positive number.")
        except ValueError:
            print("  Please enter a number.")

    # Duration
    while True:
        try:
            duration = int(input("  Hours of runtime needed (e.g. 1, 2, 3): ").strip())
            if 1 <= duration <= 23:
                break
            print("  Must be between 1 and 23.")
        except ValueError:
            print("  Please enter a whole number.")

    # Deadline
    while True:
        try:
            deadline = int(input("  Deadline — finish by hour (0-23, e.g. 7 = 7 AM): ").strip())
            if 0 <= deadline <= 23:
                break
            print("  Must be between 0 and 23.")
        except ValueError:
            print("  Please enter a whole number.")

    # Interruptible
    interruptible_in = input(
        "  Can this task be paused and resumed? [Y/n]: "
    ).strip().lower()
    interruptible = interruptible_in != "n"

    task = TaskDescriptor(
        name=name,
        power_kw=power_kw,
        duration_hours=duration,
        deadline_hour=deadline,
        interruptible=interruptible,
    )

    print(f"\n  → {task}")
    return task


def _select_comfort() -> float:
    _print_header("COMFORT / SAVINGS PRIORITY")
    print("  [1] High Savings  — aggressive deferral  (penalty = 2.0)")
    print("  [2] Balanced      — moderate tradeoff    (penalty = 1.0)")
    print("  [3] High Comfort  — rarely defer          (penalty = 0.2)")
    choice  = input("\n  Enter choice [1/2/3]: ").strip()
    mapping = {"1": 2.0, "2": 1.0, "3": 0.2}
    penalty = mapping.get(choice, 1.0)
    label   = {2.0: "High Savings", 1.0: "Balanced", 0.2: "High Comfort"}[penalty]
    print(f"  → Comfort profile: {label}  (comfort_penalty = {penalty})")
    return penalty


def _nlp_query_mode(clf, summarize_fn):
    _print_header("NLP QUERY MODE  (type 'done' to exit)")
    print("  Ask about the current period or demand level.")
    print("  Example: 'Is it a peak period right now?'")
    print("           'What should I do at midnight?'")
    while True:
        q = input("\n  Your query: ").strip()
        if q.lower() in ("done", "exit", "quit", ""):
            break
        label     = clf.predict(q)
        probs     = clf.predict_proba(q)
        best_conf = max(probs.values()) * 100
        print(f"\n  Classification   : {label.upper()}")
        print(f"  Confidence       : {best_conf:.1f}%")
        print("  All probabilities: "
              + "  ".join(f"{k}={v*100:.1f}%" for k, v in probs.items()))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN SCHEDULING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _run_full_day(
    session:          SchedulingSession,
    agent,
    clf,
    summarize_fn,
    forecasts_real:   list,
    forecasts_scaled: list,
    model_label:      str,
    interactive:      bool,
):
    task = session.task
    _print_header(
        f"24-HOUR SCHEDULING RUN  [{model_label}]  Task: {task.name}"
    )

    _print_forecast_table(forecasts_real, task)

    # Baseline: task consuming power_kw every hour across 24 hrs
    baseline_cost = sum(kwh * task.power_kw for _, kwh in forecasts_real)

    if interactive:
        print("\n  You will be asked for an override at each hour.")
        print("  Press ENTER to accept the RL decision.")
        print("  Type '1' to force RUN or '0' to force DEFER.")
        print("  Type 'optout' to disable automation for the rest of the day.")
        input("\n  [Press ENTER to begin scheduling...] ")

    _print_header("HOURLY DECISIONS")

    for (hour, real_kwh), (_, scaled_val) in zip(forecasts_real, forecasts_scaled):

        override = None
        if interactive and not session.opt_out:
            user_in = input(
                f"\n  Hour {hour:02d}:00 | Override? [ENTER/0/1/optout]: "
            ).strip().lower()
            if user_in == "optout":
                session.opt_out = True
                print("  ⚡ Opt-out activated — automated scheduling disabled.")
            elif user_in in ("0", "1"):
                override = int(user_in)

        result = session.decide(
            hour=hour,
            real_kwh=real_kwh,
            scaled_forecast=scaled_val,
            agent=agent,
            clf=clf,
            summarize_fn=summarize_fn,
            override=override,
        )
        _print_decision(result)

    _print_savings_meter(session, baseline_cost)
    session.save_log()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_interface(
    model_name:     str           = "lstm",
    task_name:      Optional[str] = None,
    power_kw:       Optional[float] = None,
    duration_hours: Optional[int] = None,
    deadline:       Optional[int] = None,
    interruptible:  bool          = True,
    comfort:        Optional[str] = None,
    interactive:    bool          = True,
    seed:           Optional[int] = None,
):
    _print_header("ENERGY DEMAND FORECASTING + RL LOAD SCHEDULING")
    print("  Academic prototype — offline simulation only.")
    print("  All decisions may be overridden by the user at any time.")

    print("\n  Loading data and models...")
    X_test, y_test, scaler = _load_data_and_scaler()
    input_size = X_test.shape[2]

    # ── Comfort ────────────────────────────────────────────────────────────────
    if comfort is not None:
        comfort_penalty = COMFORT_PROFILES.get(comfort, 1.0)
    elif interactive:
        comfort_penalty = _select_comfort()
    else:
        comfort_penalty = 1.0

    # ── Models ────────────────────────────────────────────────────────────────
    forecast_fn, model_label = _load_forecast_model(model_name, input_size)
    agent                    = _load_rl_agent()
    clf, summarize_fn        = _load_nlp()

    print(f"  ✓ Forecast model : {model_label}")
    print(f"  ✓ RL agent       : Q-Learning (deterministic, ε=0)")
    print(f"  ✓ NLP components : ForecastSummarizer + PeriodClassifier")

    # ── Task definition ────────────────────────────────────────────────────────
    all_provided = all(
        x is not None for x in [task_name, power_kw, duration_hours, deadline]
    )
    if all_provided:
        task = TaskDescriptor(
            name=task_name,
            power_kw=power_kw,
            duration_hours=duration_hours,
            deadline_hour=deadline,
            interruptible=interruptible,
        )
    elif interactive:
        task = _prompt_task()
    else:
        # Non-interactive fallback defaults
        task = TaskDescriptor(
            name="Deferrable Task",
            power_kw=1.0,
            duration_hours=2,
            deadline_hour=22,
            interruptible=True,
        )

    comfort_label = (
        "high savings" if comfort_penalty >= 2.0 else
        "high comfort" if comfort_penalty <= 0.2 else "balanced"
    )
    print(f"\n  {task}")
    print(f"  Comfort    : {comfort_penalty}  ({comfort_label})")

    # ── Optional NLP query mode ────────────────────────────────────────────────
    if interactive:
        do_nlp = input("\n  Open NLP query mode? [y/N]: ").strip().lower()
        if do_nlp == "y":
            _nlp_query_mode(clf, summarize_fn)

    # ── Build 24-hour forecasts ────────────────────────────────────────────────
    run_seed = seed if seed is not None else int(datetime.datetime.now().timestamp())
    forecasts_real, forecasts_scaled = _build_24h_forecasts(
        X_test, forecast_fn, scaler, seed=run_seed
    )

    # ── Run session ───────────────────────────────────────────────────────────
    session = SchedulingSession(task=task, comfort_penalty=comfort_penalty)

    _run_full_day(
        session=session,
        agent=agent,
        clf=clf,
        summarize_fn=summarize_fn,
        forecasts_real=forecasts_real,
        forecasts_scaled=forecasts_scaled,
        model_label=model_label,
        interactive=interactive,
    )

    _print_header("SESSION COMPLETE")
    print(f"  Task           : {session.task.name}")
    print(f"  Hours run      : {session._runs_so_far} / {session.task.duration_hours} required")
    status = "✓ COMPLETE" if session._task_complete else "✗ INCOMPLETE — check deadline settings"
    print(f"  Status         : {status}")
    print(f"  Total savings  : {session.total_savings:+.4f} kWh·units")
    print(f"  Action log     : experiments/results/action_log.json")
    if interactive:
        input("\n  [Press ENTER to exit] ")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Energy Demand Forecasting + RL Load Scheduling — User Interface"
    )
    parser.add_argument(
        "--model", default="lstm",
        choices=["lstm", "lstm_baseline", "tcn", "rf", "lr"],
        help="Forecast model (default: lstm)",
    )

    # ── Task definition ────────────────────────────────────────────────────────
    parser.add_argument(
        "--task", default=None, metavar="NAME",
        help="Task name, e.g. 'EV Charging' (default: prompt)",
    )
    parser.add_argument(
        "--power", type=float, default=None, metavar="KW",
        help="Task power draw in kW, e.g. 7.2 (default: prompt)",
    )
    parser.add_argument(
        "--duration", type=int, default=None, metavar="HOURS",
        help="Hours of runtime needed, e.g. 3 (default: prompt)",
    )
    parser.add_argument(
        "--deadline", type=int, default=None, metavar="HOUR",
        help="Finish-by hour 0-23, e.g. 22 (default: prompt)",
    )
    parser.add_argument(
        "--not-interruptible", action="store_true",
        help="Task must run as a single contiguous block (default: interruptible)",
    )

    # ── Scheduling options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--comfort", default=None,
        choices=list(COMFORT_PROFILES.keys()),
        help="Comfort/savings profile (default: prompt)",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Disable per-hour override prompts (batch/CI mode)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fix random seed for reproducible 24-h block selection",
    )
    args = parser.parse_args()

    run_interface(
        model_name     = args.model,
        task_name      = args.task,
        power_kw       = args.power,
        duration_hours = args.duration,
        deadline       = args.deadline,
        interruptible  = not args.not_interruptible,
        comfort        = args.comfort,
        interactive    = not args.no_interactive,
        seed           = args.seed,
    )