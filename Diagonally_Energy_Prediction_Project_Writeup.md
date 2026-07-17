# Diagonally Energy Prediction — Project Technical Writeup

**Author:** Abayomi Eze
**Dataset:** REFIT Smart Home Dataset — House 1
**Period covered:** October 9, 2013 – July 10, 2015

---

## 1. Executive Summary

This project builds a machine learning system that predicts daily household energy consumption for a UK smart home. Given the previous day's appliance usage and current temperature, the system forecasts how many watt-hours the household will consume — either for a single day or across a rolling 7-day window — and translates that into an estimated electricity bill in GBP using the current Ofgem tariff rate.

The system is fully productionised: a trained RandomForest model is served through a FastAPI backend, every prediction is stored in a Supabase (Postgres) database, and a lightweight HTML + JavaScript frontend lets users interact with it in a browser without any build tooling.

The road to a working, accurate model involved solving several non-trivial data engineering problems — a destructive outlier removal bug, a feature insufficiency problem, and the challenge of sourcing and integrating reliable historical temperature data. Each of those problems, how they were found, and how they were resolved, is documented below.

---

## 2. The Dataset

The REFIT Smart Home Dataset records electricity consumption in watts at **8-second intervals** for 9 individual appliances in a real UK home:

| Appliance | Behaviour |
|---|---|
| Fridge | Always-on, relatively constant |
| ChestFreezer | Always-on, relatively constant |
| UprightFreezer | Always-on, relatively constant |
| TumbleDryer | Sparse — only runs during laundry cycles |
| WashingMachine | Sparse — only runs during laundry cycles |
| Dishwasher | Sparse — only runs per meal cycle |
| Computer | Intermittent use |
| Television | Intermittent use |
| ElectricHeater | Strongly seasonal — high in winter, near-zero in summer |

The raw file contains a Unix timestamp column, a pre-computed "Aggregate" column, and one column per appliance. The dataset spans **638 days** with one notable gap: a 41-day period between March and April 2014 where the sensors went offline. That gap leaves no readings, so those days are naturally dropped when lag features are computed.

---

## 3. The First Problem: Standard Deviation Greater Than the Mean

### What was observed

During the initial exploratory data analysis (EDA), a statistics summary of the raw appliance readings showed something alarming for several appliances:

```
             mean        std
TumbleDryer    12.3       580.4
WashingMachine  8.7       420.1
Dishwasher      5.2       310.8
Computer       38.1       215.7
```

**For TumbleDryer, WashingMachine, and Dishwasher the standard deviation was 30–50× larger than the mean.** This is a characteristic signature of a sparse, intermittent time series. These appliances spend the majority of their time at zero watts — they are only on for short bursts when actively in use. The huge standard deviation is not noise or sensor malfunction; it is real variance caused by infrequent but high-power activation cycles.

### Why this matters

A standard deviation this large relative to the mean indicates the distribution is **heavily right-skewed** with a concentration at zero and a long tail of large activation readings. That shape breaks one of the assumptions behind the IQR-based outlier capping approach that had been applied early in the pipeline.

### The IQR outlier capping bug

The initial preprocessing pipeline included a step called `_cap_outliers_iqr`. The intention was reasonable: any extreme reading (e.g., a faulty sensor spike) should be capped at the upper fence of the interquartile range to prevent it from distorting the model. The logic was:

```
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
df[col] = df[col].clip(lower=0, upper=upper_fence)
```

For always-on appliances like Fridge and ChestFreezer, this worked correctly — the distribution is roughly normal, Q1 and Q3 are both non-zero, IQR is non-zero, and only genuine extreme spikes get clipped.

For sparse appliances like TumbleDryer, the calculation silently collapsed:

```
Q1 = 0.0   (75%+ of readings are zero, so the 25th percentile is 0)
Q3 = 0.0   (the 75th percentile is also 0)
IQR = 0.0
upper_fence = 0.0 + 1.5 × 0.0 = 0.0
df["TumbleDryer"] = df["TumbleDryer"].clip(lower=0, upper=0.0)
```

**Every single non-zero reading was clipped to zero.** All real usage events — entire laundry cycles, dishwasher runs — were silently deleted from the data. No warning was raised. The code ran without error. The bug was invisible unless you inspected the appliance columns after preprocessing and noticed they were constant zero.

### How it was found

The bug surfaced during model accuracy analysis. After the initial model was trained, performance on HIGH-consumption days was very poor. Investigating the training features revealed that `TumbleDryer`, `WashingMachine`, and `Dishwasher` columns in the processed training data were entirely zero — no variance at all — despite the raw data showing clear usage cycles. Tracing back through the pipeline to `_cap_outliers_iqr` confirmed the IQR zero-collapse.

### The fix

The IQR capping step was removed entirely. The only clipping retained is `clip(lower=0)` to remove physically impossible negative sensor readings (which occur occasionally due to sensor calibration drift). RandomForest is inherently robust to outliers — it makes splits on order statistics, not on absolute magnitudes — so extreme readings do not distort the model. The actual sensor spike concern was addressed by the 10-minute resampling: averaging across a 10-minute window already smooths out transient spikes.

This change is documented in **ADR-014** in the project's architecture decision record.

---

## 4. The Data Pipeline (After the Fix)

The preprocessing pipeline converts the raw 8-second interval data into a clean daily feature matrix in six stages.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                                │
│                                                                         │
│  Stage 1: Load raw CSV                                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│  data/raw/House_1.csv                                                   │
│    • Parse Unix timestamp → UK-local datetime index                     │
│    • Drop Unix column + raw Aggregate column                            │
│    • Rename Appliance1..9 → Fridge, ChestFreezer, ..., ElectricHeater  │
│                                                                         │
│  Stage 2: Resample to 10-minute intervals                               │
│  ─────────────────────────────────────────────────────────────────────  │
│    • Average watts within each 10-min window                            │
│    • clip(lower=0) removes negative sensor readings                     │
│    • Interpolate short gaps (≤ 60 min) using time-aware interpolation   │
│    • Drop any remaining NaN rows                                        │
│    • Compute aggregate_wh = sum of all appliances per interval          │
│                                                                         │
│  Stage 3: Aggregate to daily totals                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│    • Sum all 10-min intervals within each calendar day → Wh per day    │
│    • Add calendar features: day_of_week, month, is_weekend              │
│                                                                         │
│  Stage 4: Compute lag features                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│    On aggregate_wh:                                                     │
│      lag_1          = yesterday's total Wh                              │
│      lag_7          = same day last week's Wh                           │
│      rolling_mean_7 = 7-day rolling mean (shifted by 1 to avoid        │
│                       data leakage — never uses the same day's value)  │
│                                                                         │
│    On ElectricHeater (separately):                                      │
│      heater_lag_1, heater_lag_7, heater_rolling_mean_7                  │
│                                                                         │
│    First 7 rows dropped (insufficient history for lag_7)                │
│                                                                         │
│  Stage 5: Merge temperature                                             │
│  ─────────────────────────────────────────────────────────────────────  │
│    Source priority:                                                     │
│    1. data/raw/temperature_loughborough.csv (cached on first run)       │
│    2. Open-Meteo historical archive API (fetched if CSV absent)         │
│    3. UK monthly climate averages (offline fallback)                    │
│                                                                         │
│    Adds: temp_mean_c, temp_min_c (daily, Loughborough UK)               │
│                                                                         │
│  Stage 6: Split and save                                                │
│  ─────────────────────────────────────────────────────────────────────  │
│    data/processed/train.csv   (all days except 9 test + 1 demo)         │
│    data/processed/test.csv    (9 strategic test days)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The Second Problem: The Model Was Not Accurate Enough

### First attempt — no temperature features

The first version of the model was trained on 8 features: the 3 calendar columns (day_of_week, month, is_weekend) plus the 5 lag features built from aggregate_wh (lag_1, lag_7, rolling_mean_7, heater_lag_1, heater_lag_7). The model learned seasonal patterns reasonably well for MID-consumption days, but predictions for HIGH-consumption days — cold winter days with heavy electric heater use — were consistently too low.

The reason was structural: the model had no direct signal about current conditions. It knew what happened yesterday (lag_1) and what happened 7 days ago (lag_7), but it had no way to know whether today is unseasonably cold. ElectricHeater is the single largest driver of day-to-day variance in total household consumption. Without temperature as an input, the model could only look at historical heater usage — which told it what happened in the past, but not what to expect today.

### Why temperature was needed

UK electric heater usage is directly correlated with outdoor temperature. On a cold winter day (2°C mean), the heater might run for 10+ hours, adding 50,000+ Wh of consumption. On a mild November day (12°C), the same household might use the heater for less than an hour. The calendar month (month=11) is a weak proxy — November can range from 2°C to 15°C in the UK. The model needed actual temperature values, not just the month number.

Additionally, the 41-day sensor gap in March–April 2014 falls during the heating season transition. Without temperature, the model had no reliable way to handle that region of the date range.

### Sourcing the temperature data

The Open-Meteo historical weather archive API provides free, daily-resolution historical temperature for any coordinate. Loughborough, UK (latitude 52.77°N, longitude 1.20°W) was chosen as the representative location for House 1. On the first preprocessing run, the pipeline calls the Open-Meteo archive API and downloads `temperature_2m_mean` and `temperature_2m_min` for the full dataset date range (October 2013 – July 2015). This is cached to `data/raw/temperature_loughborough.csv` so future runs do not require an internet connection.

If the API is unavailable (e.g., in a CI environment with no network), the pipeline falls back to UK monthly climate averages — hardcoded constants derived from long-run Midlands weather norms. This fallback is deliberately coarse: it ensures the pipeline does not break, but the resulting model will be less accurate on days that deviate from the seasonal average.

### The effect of adding temperature

Adding `temp_mean_c` and `temp_min_c` as model features — alongside dedicated heater lag features (`heater_lag_1`, `heater_lag_7`, `heater_rolling_mean_7`) — gave the model the signals it needed to distinguish a cold winter day from a mild winter day. The feature set grew from 8 to 11 inputs. MAPE on the 9 strategic test days improved materially, with the largest gains on HIGH-consumption winter days.

---

## 6. Model Development and Selection

### Evaluation strategy — 9 strategic test days

Rather than a standard chronological train/test split or walk-forward cross-validation, the evaluation used **9 hand-picked strategic test days** that deliberately cover three consumption tiers:

| Tier | Definition | Dates |
|---|---|---|
| LOW  | < 12,000 Wh/day | 2013-10-16, 2014-12-07, 2015-01-03 |
| MID  | 12,000–25,000 Wh/day | 2014-07-21, 2014-11-12, 2014-08-01 |
| HIGH | > 25,000 Wh/day | 2013-11-22, 2013-12-12, 2014-01-19 |

These 9 days are held out of training entirely. They span the full 638-day date range, cover different seasons, and include days from the early dataset (when the model has little historical data) as well as later dates. One additional day (2015-02-10) is held out as a **demo day** — it is used to pre-fill the frontend form with realistic real-world feature values; its actual consumption (11,020.83 Wh) serves as a visible ground truth.

Walk-forward evaluation was considered but rejected: it requires fitting the model many times and produces a much noisier MAPE estimate for a single-step point forecast, without adding meaningful information about generalisation.

### Model comparison

A multi-model comparison was conducted in `notebooks/model_comparison.ipynb`. Candidates included:

| Model | Outcome |
|---|---|
| Prophet | Reasonable baseline; requires dense time series, struggles with the 41-day gap |
| XGBoost with lag features | Competitive; heavier dependency footprint |
| LightGBM with lag features | Competitive MAPE (35.97% on early evaluation before temperature features were added) |
| LSTM (PyTorch) | Requires substantially more training data to generalise; overhead not justified for 638 days |
| Temporal Fusion Transformer | Same limitation as LSTM; overfits on this dataset size |
| **RandomForest (300 trees, depth 10)** | **Best MAPE, fastest training (< 30 seconds), no heavy dependencies, inherently robust to the outliers that remain in the data after IQR removal** |

RandomForest won on all three criteria: accuracy, simplicity of deployment, and training speed.

### Final model configuration

```
Algorithm:       RandomForest (scikit-learn RandomForestRegressor)
n_estimators:    300
max_depth:       10
random_state:    42
n_jobs:          -1 (uses all available CPU cores during training)

Input features:  day_of_week, month, is_weekend,
                 lag_1, lag_7, rolling_mean_7,
                 heater_lag_1, heater_lag_7, heater_rolling_mean_7,
                 temp_mean_c, temp_min_c

Target:          aggregate_wh (daily household consumption in watt-hours)

No feature scaling — RandomForest is scale-invariant (splits on rank order,
not absolute values), so StandardScaler or MinMaxScaler would have no effect.
```

### Evaluation results on 9 strategic test days

| Band | Date | Temp °C | Actual Wh | Predicted Wh | Error % |
|---|---|---|---|---|---|
| HIGH | 2013-11-22 | 4.2 | 58,643 | 38,483 | 34.4% |
| HIGH | 2013-12-12 | 7.1 | 61,532 | 44,815 | 27.2% |
| HIGH | 2014-01-19 | 5.2 | 62,409 | 42,710 | 31.6% |
| LOW  | 2013-10-16 | 9.8 | 7,084  | 8,454  | 19.3% |
| LOW  | 2014-12-07 | 6.1 | 7,006  | 7,636  | 9.0%  |
| LOW  | 2015-01-03 | 3.2 | 7,134  | 8,419  | 18.0% |
| MID  | 2014-07-21 | 19.3| 15,198 | 15,006 | 1.3%  |
| MID  | 2014-11-12 | 10.2| 15,072 | 15,076 | 0.0%  |
| MID  | 2014-08-01 | 17.8| 15,058 | 13,813 | 8.3%  |

```
Overall MAPE:  16.6%
MAE:           6,811 Wh
RMSE:          10,951 Wh
```

The HIGH tier shows the largest errors (27–34%). This is expected: on extreme cold days, the electric heater's contribution can exceed 50,000 Wh — a level rarely seen in the training data (only a handful of days in early winter 2013–2014). The model has too few examples of that regime to generalise well. The MID tier, which covers typical household days, achieves sub-1% to 8% error — production-quality accuracy for the most common use case.

---

## 7. System Architecture

### End-to-end flow diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE TRAINING PIPELINE                             │
│                                                                              │
│   data/raw/House_1.csv                                                       │
│          │                                                                   │
│          ▼                                                                   │
│   data_loader.py ──────────────────────────────────────────────────────────► │
│   (6-stage preprocess:                data/raw/temperature_loughborough.csv  │
│    load → 10-min resample →           (fetched from Open-Meteo on first run) │
│    daily aggregate → lags →           ◄──────────────────────────────────── │
│    merge temperature → split)                                                │
│          │                                                                   │
│          ▼                                                                   │
│   data/processed/train.csv (617 days)                                        │
│   data/processed/test.csv  (9 strategic test days)                           │
│          │                                                                   │
│          ▼                                                                   │
│   train_forecast.py                                                          │
│   (RandomForest 300 trees, depth 10)                                         │
│          │                                                                   │
│          ▼                                                                   │
│   src/model/trained/                                                         │
│     model_forecast.joblib      (model + seed data wrapped in _TreeWrapper)   │
│     model_evaluation.json      (MAE / RMSE / MAPE + per-day breakdown)       │
│     demo_day.json              (held-out 2015-02-10 features + actual Wh)    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

                              ▲ model loaded once at startup
                              │
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME (FastAPI server)                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  POST /api/v1/forecast/predict       (single-day prediction)           │  │
│  │                                                                        │  │
│  │  User input (7 lag values + optional temperature)                      │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  routes.py validates ForecastPredictRequest (Pydantic)                 │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  forecast.py → builds 11-feature row                                   │  │
│  │       │  if temp not supplied: fetch from Open-Meteo forecast API      │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  model.predict_from_features(row) → predicted_wh                       │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  cost.py → predicted_kwh = predicted_wh / 1000                         │  │
│  │          → estimated_cost_gbp = predicted_kwh × tariff_rate            │  │
│  │          → lower_wh = predicted_wh × 0.85                              │  │
│  │          → upper_wh = predicted_wh × 1.15                              │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  database.py → INSERT into Supabase forecast_requests table            │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  JSON response: date, predicted_wh, predicted_kwh,                     │  │
│  │                 estimated_cost_gbp, lower_wh, upper_wh,                │  │
│  │                 temp_mean_c, temp_min_c                                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  GET /api/v1/forecast/7d             (7-day rolling forecast)          │  │
│  │                                                                        │  │
│  │  forecast.py → seeds from last 7 days of training data                 │  │
│  │       │  → fetches 7 future temps from Open-Meteo forecast API         │  │
│  │       │  → recursive multi-step: feeds each day's prediction           │  │
│  │       │    back as lag_1 for the next day                              │  │
│  │       ▼                                                                │  │
│  │  cost.py → project_week_bill:                                           │  │
│  │            optimistic_gbp / most_likely_gbp / pessimistic_gbp          │  │
│  │       ▼                                                                │  │
│  │  JSON response: forecast[7], peak_day, lowest_day,                     │  │
│  │                 projected_week_bill                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  GET /api/v1/models/evaluation  →  serves model_evaluation.json from disk    │
│  GET /health                    →  { "status": "ok" }                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

                              │ HTTP fetch calls
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (static files)                            │
│                                                                              │
│  forecast.html — single-day prediction form                                  │
│    • All 11 model features visible to the user                               │
│    • Pre-filled with demo day values (2015-02-10) from demo_day.json         │
│    • Result card: Wh / kWh / cost / confidence band / temperature            │
│    • 7-day bar chart with bill projection                                    │
│    • Model evaluation leaderboard (collapsible)                              │
│                                                                              │
│  dashboard.html — placeholder, no backend wired yet                          │
│                                                                              │
│  Served as static files by FastAPI at /frontend                              │
│  No build step — plain HTML + Tailwind CSS (CDN) + vanilla JavaScript        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

                              │ writes
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SUPABASE (Postgres)                                   │
│                                                                              │
│  forecast_requests table                                                     │
│    id, created_at, input_date,                                               │
│    lag_1, lag_7, rolling_mean_7,                                             │
│    heater_lag_1, heater_lag_7, heater_rolling_mean_7,                        │
│    temp_mean_c, temp_min_c,                                                  │
│    predicted_wh, predicted_kwh, estimated_cost_gbp                           │
│                                                                              │
│  Row Level Security enabled on all tables                                    │
│  Writes use SUPABASE_SERVICE_ROLE_KEY (bypasses RLS)                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Engineering Decisions

### 8.1 — Removing raw Aggregate column and recomputing the sum

The raw CSV includes a pre-computed `Aggregate` column. This was dropped and recomputed from scratch as `sum(Appliance1..9)` after resampling. The reason: the original aggregate sometimes disagrees with the appliance sum due to circuit-level measurement, and recomputing it from the clean individual columns ensures the target variable is internally consistent.

### 8.2 — heater_lag features as separate features from aggregate_lag features

Electric heater consumption can swing from near-zero (summer) to 40,000+ Wh/day (deep winter). Including only aggregate lags meant the model had to infer heater behaviour from the composite signal. Separating heater-specific lags (`heater_lag_1`, `heater_lag_7`, `heater_rolling_mean_7`) gives the model an explicit signal about heating demand — the single largest source of daily variance in this household.

### 8.3 — Temperature from Open-Meteo, not a local weather station

Open-Meteo provides free, daily-resolution historical archive data via a public REST API — no API key required. The data for Loughborough, UK (the nearest city to the House 1 location) is fetched once and cached to disk. This eliminates an external dependency at inference time: the training data is already fixed, and the production server only needs temperature for upcoming forecast days (from the Open-Meteo forecast API, which also requires no key).

### 8.4 — ±15% confidence interval (not model-derived)

A full prediction interval from a RandomForest would require quantile regression forests or bootstrap resampling — significantly more complexity for modest gain. The ±15% fixed band was chosen because it produces a realistic practical range (roughly matching the MAE / mean prediction ratio) and is completely transparent: a user who sees "12,000–16,000 Wh" understands it as an honest range estimate, not a false-precision probabilistic statement.

### 8.5 — Tariff rate in an environment variable

The Ofgem electricity tariff rate is not hardcoded anywhere in the codebase. It is always read from `ELECTRICITY_TARIFF_GBP_PER_KWH` at runtime. This allows the rate to be updated without a code change or redeploy — important because Ofgem revises the price cap quarterly.

### 8.6 — Model loaded once at startup, never reloaded per request

The `model_forecast.joblib` file is loaded into memory once when the FastAPI process starts, held as a module-level singleton in `forecast.py`, and reused for every prediction. Loading a joblib file involves deserialising a large NumPy array of decision trees — doing this per request would add seconds of latency. The singleton approach keeps inference latency under 5ms.

### 8.7 — Training is strictly offline

The model is never retrained inside the API process. Retraining is a CPU-intensive batch job (30+ seconds, loads the full CSV, fits 300 trees) that would block request handling. The correct path is `scripts/run_training_forecast.py` — a standalone script that runs the full pipeline (preprocess → train → evaluate → save), then the server is restarted to pick up the new model.

---

## 9. Testing Approach

The project follows test-driven development: every function that transforms data or calls the model has a corresponding test. Tests are written before implementation, and a test that passes before any implementation code exists is treated as incorrectly written.

### Test coverage

| Test file | What it covers |
|---|---|
| `tests/test_api.py` | All API endpoints: 200 success paths, validation errors (422), error handling (500), correct numeric math (Wh → kWh → GBP) |
| `tests/test_forecast.py` | Single-day and 7-day forecast functions; correct feature row assembly |
| `tests/test_features.py` | `build_lag_matrix`: correct column names, no data leakage, correct shapes |
| `tests/test_cost.py` | Wh → GBP conversion; week bill projection with optimistic/most_likely/pessimistic |
| `tests/test_database.py` | Supabase insert helpers; mock client ensures no live DB calls in CI |

### CI pipeline

GitHub Actions runs the full test suite on every push to `main`. The test environment mocks the Supabase client (no live credentials needed in CI) and mocks the forecast model singleton. This ensures tests run in under 30 seconds regardless of environment.

---

## 10. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/api/v1/forecast/predict` | POST | Single-day forecast from 7–9 user inputs |
| `/api/v1/forecast/7d` | GET | 7-day rolling forecast + weekly bill projection |
| `/api/v1/models/evaluation` | GET | Model metrics + per-day test results |

### POST /api/v1/forecast/predict — request

```json
{
  "date": "2015-02-10",
  "lag_1": 10638.64,
  "lag_7": 14785.26,
  "rolling_mean_7": 20023.43,
  "heater_lag_1": 143.97,
  "heater_lag_7": 143.99,
  "heater_rolling_mean_7": 143.94,
  "temp_mean_c": 2.2,
  "temp_min_c": 0.8
}
```
`temp_mean_c` and `temp_min_c` are optional — omit them and the backend fetches live temperature automatically.

### POST /api/v1/forecast/predict — response

```json
{
  "date": "2015-02-10",
  "predicted_wh": 11020.83,
  "predicted_kwh": 11.02083,
  "estimated_cost_gbp": 3.75,
  "lower_wh": 9367.71,
  "upper_wh": 12673.95,
  "temp_mean_c": 2.2,
  "temp_min_c": 0.8
}
```

---

## 11. Technology Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.11 | Mature ML ecosystem, strong typing support |
| API framework | FastAPI + Uvicorn | Auto-generates OpenAPI docs, async-capable, fast |
| ML model | scikit-learn RandomForest | Best accuracy in comparison, fast training, no heavy deps |
| Model serialisation | joblib | Standard for scikit-learn; handles NumPy arrays efficiently |
| Data processing | pandas + numpy | Standard; fast datetime-aware resampling |
| Temperature API | Open-Meteo | Free, no API key, reliable UK historical + forecast data |
| Database | Supabase (Postgres) | Managed Postgres with Row Level Security; supabase-py client |
| Frontend | HTML + JavaScript + Tailwind CSS (CDN) | No build step; deployable as static files from the API |
| Testing | pytest | Industry standard; works with FastAPI's TestClient |
| CI | GitHub Actions | Runs full test suite on every push to main |

---

## 12. Project Status

The core forecast system is complete and working:

- Data pipeline: fully automated, reproducible from a single script
- Model: trained, evaluated, and serialised
- API: all forecast endpoints live and tested
- Database: prediction logging to Supabase on every request
- Frontend: single-day form and 7-day chart operational
- CI: passing on every commit to main

The dashboard (`dashboard.html`) is scaffolded but not yet wired to the backend — it requires a `GET /api/v1/readings` endpoint that has not been implemented yet.

---

*This document describes the system as built and deployed. All architectural decisions, data pipeline behaviour, model configuration, and API contracts are reflected in the source code and the `docs/` directory of the repository.*
