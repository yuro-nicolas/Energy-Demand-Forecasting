import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ══════════════════════════════════════════════════════════════
# 1. Forecast Summarizer
# ══════════════════════════════════════════════════════════════

PERIOD_LABELS = {
    (0,  6):  "late night",
    (6,  9):  "morning",
    (9,  12): "mid-morning",
    (12, 14): "midday",
    (14, 17): "afternoon",
    (17, 21): "evening peak",
    (21, 24): "night",
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

    period = hour_to_period(hour)

    if forecast_kwh > threshold_kwh * 1.3:
        demand_level = "high"
    elif forecast_kwh > threshold_kwh:
        demand_level = "moderate"
    else:
        demand_level = "low"

    lines = [
        f"At {hour:02d}:00 ({period}), predicted demand is {forecast_kwh:.2f} kWh, which is {demand_level}."
    ]

    if rl_action == 1:
        lines.append(
            f"The system recommends running the {device} now because demand is acceptable."
        )
    else:
        lines.append(
            f"The system recommends deferring the {device} to avoid higher electricity cost."
        )

    lines.append("You can override this decision anytime.")

    return " ".join(lines)


def batch_summarize(forecasts, hours, actions, device="deferrable load"):
    return [
        summarize_forecast(f, h, a, device)
        for f, h, a in zip(forecasts, hours, actions)
    ]


# ══════════════════════════════════════════════════════════════
# 2. Improved Training Corpus
# ══════════════════════════════════════════════════════════════

_CORPUS = [
    # ── PEAK ─────────────────────────────
    ("high demand evening rush hour lots of appliances running", "peak"),
    ("evening electricity usage spike dinner cooking", "peak"),
    ("peak hours heavy consumption", "peak"),
    ("air conditioning running afternoon high load", "peak"),
    ("multiple devices active high load period", "peak"),
    ("busy time high electricity usage", "peak"),
    ("should I avoid using appliances now peak time", "peak"),
    ("is this a peak hour", "peak"),
    ("high usage time electricity expensive now", "peak"),

    # ── OFF-PEAK ─────────────────────────
    ("moderate usage mid morning light activity", "off-peak"),
    ("some appliances running not full load", "off-peak"),
    ("afternoon moderate electricity demand", "off-peak"),
    ("daytime moderate consumption", "off-peak"),
    ("normal electricity usage", "off-peak"),
    ("average demand period", "off-peak"),
    ("typical daytime electricity usage", "off-peak"),

    # ── LOW ──────────────────────────────
    ("night time very low consumption everyone asleep", "low"),
    ("late night minimal electricity usage", "low"),
    ("early morning almost no power used", "low"),
    ("standby devices only very low demand", "low"),
    ("midnight low electricity usage", "low"),
    ("should I run appliances at midnight", "low"),
    ("is midnight a good time to use electricity", "low"),
    ("overnight very low power usage", "low"),
]


# ══════════════════════════════════════════════════════════════
# 3. Period Classifier (Improved)
# ══════════════════════════════════════════════════════════════

class PeriodClassifier:

    def __init__(self):
        self.vectorizer = None
        self.clf = None
        self._trained = False

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()

    def _get_corpus(self):
        texts  = [t for t, _ in _CORPUS]
        labels = [l for _, l in _CORPUS]
        return texts, labels

    def train(self):
        texts, labels = self._get_corpus()
        texts = [self._clean_text(t) for t in texts]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2000,
            stop_words="english"
        )

        X = self.vectorizer.fit_transform(texts)

        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        self.clf.fit(X, labels)
        self._trained = True

        print(f"[NLP] Trained on {len(texts)} samples.")

    # 🔥 Hybrid Prediction (Rules + ML)
    def predict(self, text: str) -> str:
        if not self._trained:
            self.train()

        clean = self._clean_text(text)

        # ── Strong rule-based overrides ──
        if any(w in clean for w in ["midnight", "12am", "late night", "sleep", "overnight"]):
            return "low"

        if any(w in clean for w in ["evening", "rush", "peak", "dinner", "busy", "high demand"]):
            return "peak"

        if any(w in clean for w in ["morning", "afternoon", "daytime", "normal"]):
            return "off-peak"

        # ── ML fallback ──
        X = self.vectorizer.transform([clean])
        return self.clf.predict(X)[0]

    def predict_proba(self, text: str) -> dict:
        if not self._trained:
            self.train()

        clean = self._clean_text(text)
        X = self.vectorizer.transform([clean])
        probs = self.clf.predict_proba(X)[0]

        return dict(zip(self.clf.classes_, probs.tolist()))

    def evaluate(self):
        if not self._trained:
            self.train()

        texts, labels = self._get_corpus()
        texts = [self._clean_text(t) for t in texts]

        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)

        print(classification_report(labels, preds))

    # ── Safe Save ──
    def save(self, path="experiments/results/nlp_classifier.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "model": self.clf
            }, f)

    # ── Safe Load ──
    @staticmethod
    def load(path="experiments/results/nlp_classifier.pkl"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            clf = PeriodClassifier()
            clf.vectorizer = data["vectorizer"]
            clf.clf = data["model"]
            clf._trained = True
            return clf

        except Exception:
            print("[NLP] Failed to load model. Re-training...")
            clf = PeriodClassifier()
            clf.train()
            clf.save(path)
            return clf


# ══════════════════════════════════════════════════════════════
# 4. Hour → Category Mapping
# ══════════════════════════════════════════════════════════════

def hour_to_category(hour: int) -> str:
    if 17 <= hour <= 21 or (7 <= hour <= 9):
        return "peak"
    elif 22 <= hour or hour <= 5:
        return "low"
    else:
        return "off-peak"
