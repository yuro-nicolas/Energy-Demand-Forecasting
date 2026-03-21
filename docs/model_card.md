# Model Card — Energy Demand Forecasting + RL Load Scheduling

## Model Details
- **Type:** LSTM Encoder-Decoder (primary), TCN (CNN component), Q-Learning RL Agent, TF-IDF + Logistic Regression NLP Classifier
- **Version:** v0.2 (Week 2 Checkpoint)
- **Developed by:** Holy Angel University — 6INTELSY Group (AY 2025–2026)
- **Authors:** Euan Roie Nicolas, Jan Nolasco, Jeremiah Carbungco, Jared Gabriel Vitero
- **License:** Academic use only

## Intended Use
- **Primary use:** Short-term household electricity demand forecasting and deferrable load scheduling simulation
- **Intended users:** Researchers, students, academic evaluation
- **Out-of-scope:** Real-time production deployment, clinical or safety-critical decisions

## Dataset
- **Source:** UCI Individual Household Electric Power Consumption (Hébrail & Bérard, 2006)
- **Period:** December 2006 – November 2010, 1-minute intervals, resampled to hourly
- **Size:** 34,168 hourly records after cleaning (zero missing values)
- **Windowed arrays:** 27,310 train / 3,393 val / 3,393 test (24-step window, horizon = 1)
- **Splits:** Strictly temporal — 80% train / 10% val / 10% test (no leakage)
- **Features:** Global_active_power (target), Global_reactive_power, Voltage, Global_intensity, Sub_metering_1/2/3, hour_sin/cos, dow_sin/cos, is_weekend (12 features total)
- **Limitations:** Single French household; may not generalize to other regions, climates, or modern appliance profiles

## Performance (Test Set — run.py, AY 2025–2026)

### Forecasting Models
| Model | Val MAE | Test MAE | Best Epoch | Params |
|---|---|---|---|---|
| Linear Regression | 0.4022 | 0.3875 | — | — |
| Random Forest | 0.3569 | 0.3399 | — | 100 trees |
| LSTM Baseline | 0.3443 | 0.3234 | 11 / 16 | 10,401 |
| LSTM Full (primary) | 0.3414 | **0.3153** | 24 / 29 | 70,465 |
| TCN (CNN component) | 0.3406 | 0.3225 | 16 / 21 | 90,241 |

> Note: MAPE excluded — inflated by near-zero demand values in the dataset.

### RL Load Scheduling Agent
| Metric | Value |
|---|---|
| Agent avg reward (50 eval episodes) | 2.062 ± 2.885 |
| Always-on baseline reward | 0.000 ± 0.000 |
| Mean reward — first 50 training episodes | −6.415 |
| Mean reward — last 50 training episodes | +2.641 |
| Training episodes | 500 |
| Hyperparameters | α=0.10, γ=0.95, ε-decay=0.995 |

### NLP Component
| Component | Result |
|---|---|
| PeriodClassifier (TF-IDF + LogReg) | Precision / Recall / F1 = 1.00 (3 classes, 22 samples) |
| ForecastSummarizer | Template-based NLG, human-readable output confirmed |

## Evaluation Slices
- Error by hour-of-day (peak vs. off-peak) — Week 3 deliverable
- Error by weekday vs. weekend — Week 3 deliverable
- Slice analysis output: `experiments/results/slice_by_hour.csv`

## Caveats & Warnings
- Prototype system — not validated for real grid operation
- Scheduling decisions are simulations only; no real devices are controlled
- MAPE metrics are unreliable for this dataset due to near-zero demand values
- LSTM Full achieves best test MAE (0.3153) but requires more training epochs than TCN
- RL agent uses tabular Q-Learning on discretized state; performance may improve with deep RL (Week 3)
- NLP classifier trained on 22 synthetic samples; accuracy may not hold on real user descriptions

## Ethical Considerations
- Users must opt-in before automation; override and opt-out mechanisms required
- Comfort constraints built into RL reward to prevent aggressive load suppression
- Must-run hours (7h, 8h, 18–20h) enforced with penalty of −2.0 per skipped hour
- See `ethics_statement.md` for full risk register

## Changelog
| Version | Changes |
|---|---|
| v0.1 | Initial model card — all metrics TBD |
| v0.2 | Filled real test metrics from run.py; updated splits (80/10/10); added RL and NLP results; added params and best epoch columns |
