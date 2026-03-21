# Ethics & Policy Statement
**Project:** Energy Demand Forecasting + RL Load Scheduling
**Version:** v0.2 (Week 2 Checkpoint)
**Team:** Holy Angel University — 6INTELSY Group (AY 2025–2026)

---

## Intended Use & Limitations
This system is an **academic prototype** for simulating energy demand forecasting and
deferrable load scheduling. It is not intended for deployment in real energy management
systems without further validation, safety review, and regulatory compliance.

All results reported are from an offline simulation using the UCI Individual Household
Electric Power Consumption dataset (single French household, 2006–2010). No real devices
are controlled, and no real users are affected.

---

## Risk Register

### Risk 1 — Equity and Unequal Distribution of Benefits
**Description:** Optimized scheduling benefits users with smart devices and technical
literacy, potentially shifting peak costs onto less tech-capable households.

**Status (Week 2):** EDA confirmed the dataset represents a single household in France.
Consumption patterns (morning and evening peaks, Sub_metering_3 dominance) suggest
the model captures appliance-specific behavior that may not generalize across income
levels or regions.

**Mitigation:**
- Evaluation pipeline includes an equity metric measuring cost-saving distribution
- Dataset bias limitation explicitly documented in Model Card (v0.2)
- Generalization limitations noted in all reported results
- Week 3: slice analysis by hour-of-day and weekday/weekend will surface differential error rates

---

### Risk 2 — User Autonomy
**Description:** An RL agent deferring loads (HVAC, EV, appliances) can override user
preferences without consent, especially during comfort-critical hours.

**Status (Week 2):** RL environment now fully implemented with the following autonomy
protections confirmed in code (`src/rl_agent.py`):

**Implemented mitigations:**
- **Must-run hours enforced:** Hours 7, 8, 18, 19, 20 are designated must-run. Deferring
  during these hours incurs a comfort penalty of −2.0 per step
- **Minimum run-hours constraint:** Agent must run the load for at least 4 hours per
  episode; end-of-episode penalty applied if violated
- **Override mechanism:** ForecastSummarizer (NLP component) explicitly states
  *"You may override this decision at any time"* in every generated summary
- **Opt-in design:** System documented as requiring explicit user opt-in before
  any automation activates
- **Comfort weight:** Comfort penalty is user-configurable in `LoadSchedulingEnv`
  constructor (`comfort_penalty` parameter)

**Observed results (Week 2):** RL agent mean reward rose from −6.415 (early training)
to +2.641 (late training), demonstrating the agent learned to defer loads without
excessively violating comfort constraints.

---

### Risk 3 — Data Privacy
**Description:** Granular smart meter data at 1-minute resolution can reveal household
routines, occupancy patterns, and appliance usage schedules.

**Status (Week 2):** No changes to privacy posture. UCI dataset remains the only data
source. No raw data is committed to the repository (`.gitignore` enforced).

**Implemented mitigations:**
- Dataset is anonymized and publicly released for academic research
- No PII stored or redistributed at any stage of the pipeline
- Raw data files excluded from version control via `.gitignore`
- Data used solely for model training and evaluation; never shared externally
- `data/get_data.py` downloads directly from UCI ML Repository; no intermediate servers

---

## Fairness Checks
- Forecast error will be analyzed across hour-of-day and weekday/weekend slices (Week 3)
- No demographic data present in the UCI dataset
- Single-household origin limits generalizability claims; explicitly documented in Model Card

## Misuse Considerations
- System must not be used for occupancy surveillance or behavioral profiling
- Scheduling decisions must not override safety-critical loads (medical equipment, alarms)
- Any real deployment requires informed consent from all affected occupants
- Model accuracy (Test MAE ~0.32 kW) is insufficient for financial billing applications

---

## Changelog
| Version | Changes |
|---|---|
| v0.1 | Initial ethics statement from Week 1 proposal |
| v0.2 | Updated Risk 1 with EDA findings (single-household bias confirmed); updated Risk 2 with implemented RL comfort constraints and real reward results (−6.415 → +2.641); confirmed Risk 3 mitigations unchanged; added misuse section and fairness check status |
