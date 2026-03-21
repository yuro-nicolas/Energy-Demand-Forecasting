"""
NLP Component — two parts:
  1. ForecastSummarizer : template-based natural language summary of forecast output.
  2. PeriodClassifier   : text classifier that maps time-period descriptions
                          to consumption categories (peak / off-peak / low).

This supports user autonomy (users read what the RL agent decided and why)
and satisfies the NLP cross-cutting requirement.
"""

import re
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle


# ── 1. Forecast Summarizer ─────────────────────────────────────────────────────

PERIOD_LABELS = {
    (0,  6):  "late night",
    (6,  9):  "morning",
    (9,  12): "mid-morning",
    (12, 14): "midday",
    (14, 17): "afternoon",
    (17, 21): "evening peak",
    (21, 24): "night",
}

ACTION_LABELS = {
    1: "RUN",
    0: "DEFER",
}


def hour_to_period(hour: int) -> str:
    for (start, end), label in PERIOD_LABELS.items():
        if start <= hour < end:
            return label
    return "night"


def summarize_forecast(
    forecast_kwh: float,
    hour: int,
    rl_action: int,
    device: str = "deferrable load",
    threshold_kwh: float = 1.5,
) -> str:
    """
    Generate a human-readable summary of the forecast + RL scheduling decision.

    Args:
        forecast_kwh : predicted demand in kWh for the next hour
        hour         : current hour (0-23)
        rl_action    : 1 = schedule now, 0 = defer
        device       : name of the load being scheduled
        threshold_kwh: cost threshold used by RL agent
    Returns:
        Natural language string the user can read and act on.
    """
    period = hour_to_period(hour)
    action_word = "run" if rl_action == 1 else "defer"
    demand_level = "high" if forecast_kwh > threshold_kwh else "low"

    lines = [
        f"Forecast ({period}, {hour:02d}:00): {forecast_kwh:.2f} kWh — demand is {demand_level}.",
    ]

    if rl_action == 1:
        lines.append(
            f"Decision: Schedule {device} NOW. "
            f"Demand is within acceptable limits, making this a cost-effective window."
        )
    else:
        lines.append(
            f"Decision: Defer {device}. "
            f"Demand is elevated during the {period} period. "
            f"Rescheduling can reduce proxy costs."
        )

    lines.append("You may override this decision at any time using the override control.")
    return " ".join(lines)


def batch_summarize(forecasts, hours, actions, device="deferrable load"):
    """Return a list of summary strings for a batch of timesteps."""
    return [
        summarize_forecast(f, h, a, device)
        for f, h, a in zip(forecasts, hours, actions)
    ]


# ── 2. Period Text Classifier ──────────────────────────────────────────────────

# Synthetic training corpus for period classification
_CORPUS = [
    # peak
    ("high demand evening rush hour lots of appliances running", "peak"),
    ("evening electricity usage spike dinner cooking", "peak"),
    ("peak hours 6pm to 9pm heavy consumption", "peak"),
    ("air conditioning heating running simultaneously afternoon", "peak"),
    ("multiple devices active high load period", "peak"),
    ("electricity bill spikes during evening peak usage", "peak"),
    ("dinner time all appliances on maximum consumption", "peak"),
    ("workday morning rush everyone getting ready high load", "peak"),
    # off-peak
    ("moderate usage mid morning light activity", "off-peak"),
    ("some appliances running but not at full load", "off-peak"),
    ("afternoon moderate electricity demand", "off-peak"),
    ("partial load a few devices running", "off-peak"),
    ("daytime moderate consumption work from home", "off-peak"),
    ("weekend afternoon some activity moderate demand", "off-peak"),
    ("midday regular consumption nothing unusual", "off-peak"),
    # low
    ("night time very low consumption everyone asleep", "low"),
    ("late night minimal electricity usage quiet household", "low"),
    ("early morning before sunrise almost no power used", "low"),
    ("very little demand standby devices only", "low"),
    ("off peak low demand period ideal for scheduling", "low"),
    ("midnight low usage opportunity to run deferrable loads", "low"),
    ("household inactive low power draw standby mode", "low"),
]


class PeriodClassifier:
    """
    TF-IDF + Logistic Regression text classifier.
    Maps a text description of a time period to: peak / off-peak / low
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
        self.clf = LogisticRegression(max_iter=500, random_state=42)
        self._trained = False

    def _get_corpus(self):
        texts  = [t for t, _ in _CORPUS]
        labels = [l for _, l in _CORPUS]
        return texts, labels

    def train(self, extra_texts=None, extra_labels=None):
        texts, labels = self._get_corpus()
        if extra_texts:
            texts  += extra_texts
            labels += extra_labels
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._trained = True
        print(f"[NLP] PeriodClassifier trained on {len(texts)} samples.")

    def predict(self, text: str) -> str:
        if not self._trained:
            self.train()
        X = self.vectorizer.transform([text])
        return self.clf.predict(X)[0]

    def predict_proba(self, text: str) -> dict:
        if not self._trained:
            self.train()
        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        return dict(zip(self.clf.classes_, probs.tolist()))

    def evaluate(self):
        texts, labels = self._get_corpus()
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)
        print("[NLP] PeriodClassifier evaluation (train set):")
        print(classification_report(labels, preds))

    def save(self, path="experiments/results/nlp_classifier.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="experiments/results/nlp_classifier.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


# ── 3. Derive period label from hour (for dataset labeling) ───────────────────

def hour_to_category(hour: int) -> str:
    if 17 <= hour <= 21 or (7 <= hour <= 9):
        return "peak"
    elif 22 <= hour or hour <= 5:
        return "low"
    else:
        return "off-peak"


def label_dataframe(df):
    """Add NLP period labels to a DataFrame that has an 'hour' column."""
    df = df.copy()
    df["period_label"] = df["hour"].apply(hour_to_category)
    return df


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test summarizer
    summary = summarize_forecast(
        forecast_kwh=2.1, hour=18, rl_action=0, device="washing machine"
    )
    print("=== Forecast Summary ===")
    print(summary)
    print()

    # Test classifier
    clf = PeriodClassifier()
    clf.train()
    clf.evaluate()

    test_inputs = [
        "late night minimal usage everyone asleep",
        "evening rush high consumption many devices",
        "midday moderate activity some appliances on",
    ]
    print("\n=== Period Classification ===")
    for t in test_inputs:
        label = clf.predict(t)
        probs = clf.predict_proba(t)
        print(f"  '{t[:45]}...' → {label}  {probs}")

    clf.save()
