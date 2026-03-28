# Energy Demand Forecasting + RL Load Scheduling

> **6INTELSY — Final Project AY 2025–2026 | Holy Angel University**

---

## Key Features

- Hybrid LSTM + TCN demand forecasting
- Reinforcement learning-based load scheduling
- Natural language explanations of agent decisions
- Fully reproducible offline pipeline
- Built-in evaluation, ablation, and visualization tools

---

## Overview

An integrated academic prototype that forecasts short-term household electricity demand and **optimizes scheduling of deferrable electrical loads** to reduce proxy energy costs. The system combines an LSTM encoder-decoder and Temporal Convolutional Network (TCN) for demand forecasting, a Q-Learning reinforcement learning agent for load scheduling, and an NLP component that translates agent decisions into plain-language summaries with explicit user override options.

The system operates entirely as an offline simulation using the UCI Individual Household Electric Power Consumption dataset. No real devices are controlled and no real users are affected.

---

## Results Summary

### Forecasting — Test Set

| Model | Val MAE (kW) | Test MAE (kW) | Best Epoch | Parameters |
|---|---|---|---|---|
| Linear Regression | 0.4022 | 0.3875 | — | — |
| Random Forest | 0.3569 | 0.3399 | — | 100 Trees |
| LSTM Baseline | 0.3443 | 0.3234 | 11 / 16 | 10,401 |
| **LSTM Full (Primary)** | **0.3414** | **0.3153** | 24 / 29 | 70,465 |
| TCN (CNN Component) | 0.3406 | 0.3225 | 16 / 21 | 90,241 |

LSTM Full achieves the best test MAE of **0.3153 kW**, an 18.6% improvement over Linear Regression and 7.2% over Random Forest. TCN trains faster (16 epochs vs 24) with comparable accuracy, confirming its efficiency advantage.

### RL Load Scheduling

| Metric | Value |
|---|---|
| Agent avg reward (50 eval episodes) | **2.062 ± 2.885** |
| Always-on baseline reward | 0.000 ± 0.000 |
| Reward — first 50 training episodes | −6.415 |
| Reward — last 50 training episodes | +2.641 |
| Training episodes | 500 |

The agent improves from −6.415 to +2.641 mean reward across training, demonstrating a clear learning signal. The positive final reward confirms the agent successfully shifts loads to off-peak periods, reducing proxy costs beyond the always-on baseline.

### NLP Component

| Metric | Value |
|---|---|
| PeriodClassifier precision / recall / F1 | 1.00 / 1.00 / 1.00 (all classes) |
| Classes | peak, off-peak, low |
| ForecastSummarizer | Template-based, verified by manual inspection |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yuro-nicolas/Energy-Demand-Forecasting
cd Energy-Demand-Forecasting
```

### 2. Set Up Environment

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate energy-demand-rl
```

Using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

Dependencies are installed automatically at startup.

```bash
# Full pipeline including dataset download
python run.py

# Skip download if dataset already present
python run.py --skip-download

# One-command via Makefile
make repro

# One-command via shell script
bash run.sh --skip-download
```

### 4. View Results

After the pipeline completes, results are saved to:

```
experiments/
├── logs/
│   ├── lstm_baseline_loss_curve.png
│   ├── lstm_loss_curve.png
│   ├── tcn_loss_curve.png
│   └── rl_learning_curve.png
└── results/
    ├── model_comparison.csv       ← final MAE for all models
    ├── model_comparison.png       ← comparison bar chart
    ├── ablation_plot.png          ← ablation study
    ├── error_by_hour.png          ← slice analysis
    └── slice_by_hour.csv          ← error by hour of day
```

---

## Dataset

**UCI Individual Household Electric Power Consumption**
- Source: https://archive.ics.uci.edu/dataset/235
- License: Public domain (UCI ML Repository)
- Coverage: Single household, Sceaux France, Dec 2006 – Nov 2010
- Resolution: 1-minute → resampled to hourly (34,168 records)
- Splits: 80% train / 10% val / 10% test (chronological, no leakage)

Raw data is never committed to this repository. Run `python data/get_data.py` or `python run.py` to download automatically.

---

## System Components

| Requirement | Implementation |
|---|---|
| Core DL | LSTM Full Encoder-Decoder (`src/models/lstm_model.py`) |
| CNN | TCN with dilated causal convolutions (`src/models/tcn_model.py`) |
| NLP | ForecastSummarizer + PeriodClassifier (`src/models/nlp_component.py`) |
| RL | Q-Learning load scheduling agent (`src/rl_agent.py`) |
| Non-DL Baseline | Linear Regression + Random Forest (`src/models/tabular_baseline.py`) |

---

## Demo

Run a sample NLP forecast summary after training:

```
You can run directly on interface.py
"
```

Sample Expected output:
```
Forecast (evening peak, 18:00): 2.10 kWh — demand is high.
Decision: Defer washing machine. Demand is elevated during the evening peak period.
Rescheduling can reduce proxy costs.
You may override this decision at any time using the override control.
```

---

## Ethics

This system is an **academic prototype and offline simulation**. It is not intended for real-time grid control, commercial energy trading, or deployment on real devices without further validation, safety review, and regulatory compliance.

Top risks and mitigations:
- **Equity** — scheduling benefits users with smart devices; equity metrics included in evaluation
- **User autonomy** — override and opt-out mechanisms required; comfort penalty enforced in RL reward
- **Data privacy** — UCI dataset is fully anonymized; raw data never committed to repo

See `docs/ethics_statement.md` and `docs/model_card.md` for full details.

---

## Team

| Role | Member |
|---|---|
| Project Lead / Integration | [Member A] |
| Data & Ethics Lead | [Member B] |
| Modeling Lead | [Member C] |
| Evaluation & MLOps Lead | [Member D] |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
