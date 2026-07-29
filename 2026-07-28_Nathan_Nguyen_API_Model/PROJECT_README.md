# Dengue outbreak forecasting model

First iteration of a leakage-safe dengue forecasting pipeline, plus a separate
Puerto Rico-wide operational case forecaster and outbreak detector that refresh
from public APIs.

The project compares two related goals:

- Weekly case-count forecasting, where the main score is mean absolute error (MAE).
- Outbreak alerting, where recall and precision matter more than raw case-count MAE.

## Public dashboard

The bilingual **Dengue Signal Puerto Rico** app is in `docs/app/`. It presents
the newest island-wide case estimate, clearly labels data delays, and provides
an interactive timeline with cases, hospitalizations, exact week selection,
and adjustable time windows.

Generate its public JSON snapshot from the operational ledger:

```bash
python src/generate_public_dashboard.py
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000/app/`. The GitHub Actions workflow regenerates
the snapshot after a successful weekly update or guarded retraining check.
Email signup is not shown because the app does not yet have a secure
subscription and delivery backend.

For a server-free local preview, open `docs/app/index.html` directly. The
generated `dashboard-data.js` snapshot allows the app to work from a local file
even when no preview server is running.

## Current results

These results come from expanding-window validation by city. Each validation year is predicted recursively: after the first validation week, earlier model predictions are fed back into the case-lag features instead of using hidden future case counts.

| Output | Overall MAE | Normal-week MAE | Outbreak-week MAE | Best use |
| --- | ---: | ---: | ---: | --- |
| Single-stage baseline | 11.68 | 5.92 | 31.69 | Simple reference forecast |
| Two-stage MAE policy | **11.09** | **4.51** | 33.93 | Best weekly case-count forecast |
| Two-stage recall policy | 11.68 | 8.68 | **22.11** | More outbreak-sensitive alerting |

Alert metrics for the recall-focused policy:

- Outbreak recall: **51.6%**
- Alert precision: **31.6%**

That means the alert policy catches about 52 of every 100 actual outbreak weeks, and about 32 of every 100 alerts correspond to actual outbreak weeks.

### Puerto Rico-wide operational baseline

The current Puerto Rico-wide champion was trained through **May 18, 2026** on
**385 complete official weeks**. Its expanding-time validation results are:

| Model available at prediction time | Precision | Recall | F1 | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weather only | 77.6% | **100.0%** | 0.874 | 0.792 | 0.174 |
| Weather + delayed case history (8-week validation delay) | 80.0% | 90.7% | 0.850 | 0.788 | 0.172 |
| Weather + eight exact recent case weeks | **83.5%** | 99.0% | **0.906** | **0.873** | **0.163** |

The same held-out weeks are also used to evaluate weekly case-count forecasts.
P50 is the median prediction and is the point forecast scored with MAE.

| Case forecast available at prediction time | Overall MAE | Normal-week MAE | Outbreak-week MAE | Persistence reference* |
| --- | ---: | ---: | ---: | ---: |
| Weather only | 56.94 | 16.66 | 68.57 | Not available live |
| Weather + delayed case history (8-week validation delay) | 23.99 | 12.96 | 27.18 | 23.99 |
| Weather + eight exact recent case weeks | **10.69** | **6.61** | **11.86** | 10.90 |

The recent-case model predicts a time-calibrated change from last week's count.
The delayed model is trained across simulated 4–12 week reporting delays and
validated at eight weeks. Its learned P50 corrections did not beat carrying the
latest delayed report forward, so the guarded policy uses that report as P50;
the classifier and P80/P90 range still use weather, case trends, report age, and
hospitalizations. This is much more useful than the weather-only 56.94 MAE, but
it remains weaker than having exact recent reports.

\*Persistence repeats the latest report available to that variant: t-1 for the
exact model and t-8 for the delayed validation scenario.

These are held-out historical results from only 125 validation rows. The folds
are time-ordered, but their target-week weather comes from the finalized
historical record, so they are explicitly treated as **historical proxy**
results. They are useful for comparing model versions, but they are not a
clinical validation or an official outbreak declaration.

The stricter `audit` command replaces target-week validation weather with
archived ECMWF forecasts issued before each covered week. That result is kept
separate because ECMWF is a sensitivity test, not the exact live Open-Meteo
“best match” provider. Archive metrics also use only fully covered weeks, while
the headline proxy uses the full held-out sample, so their difference is not a
paired estimate of weather-provider impact. The strongest deployment score will accumulate
prospectively in `data/operational/monitoring/latest_metrics.json`, using only
predictions whose exact inputs were frozen before the outcome was known.

## Project structure

```text
dengue-forecasting-model/
├── docs/
│   └── model-explainer.html
├── config/
│   └── operations.json
├── data/operational/
│   ├── cases/
│   ├── monitoring/
│   ├── vintages/
│   └── predictions.csv
├── models/operational/pr/
│   ├── champion.joblib
│   ├── champion.json
│   ├── history.jsonl
│   └── validation_predictions.csv
├── notebooks/
│   └── dengue_forecasting_model.ipynb
├── src/
│   ├── continuous_pipeline.py
│   ├── dengue_forecast_model.py
│   ├── near_realtime_outbreak_detection.py
│   ├── official_case_data.py
│   └── puerto_rico_operational.py
├── tests/
├── automation/
│   └── puerto-rico-dengue-model.yml
├── requirements-automation.lock
├── requirements.txt
└── README.md
```

## Model architecture

The pipeline has two model families:

1. **Single-stage model**
   - Trains one regressor to directly predict weekly dengue cases.
   - Uses weather, seasonality, and previous case-count features.
   - Serves as the baseline.

2. **Two-stage model**
   - First predicts whether a week looks outbreak-like.
   - Then chooses between:
     - a normal-week case-count model, and
     - an outbreak/near-outbreak case-count model.
   - Uses two decision policies:
     - an MAE-focused gate for best overall case-count accuracy;
     - a recall-focused gate for catching more outbreak weeks.

## What was fixed from the earlier notebook

- Weather lags are separated from case lags.
- `cases_lag_1` remains the previous week’s case count instead of being accidentally shifted by a global weather delay.
- Missing values are handled inside each temporal fold to reduce preprocessing leakage.
- Validation is recursive, matching the real forecasting scenario more closely.
- Expanding-window validation replaces a single 80/20 split.
- Outbreak thresholds are seasonal, not one city-wide cutoff.
- Classifier probabilities are calibrated with time-based out-of-fold predictions.
- The two-stage model uses hard policy gates instead of averaging normal and outbreak predictions.

## Data source

The code reads public DrivenData DengAI training files directly from:

```text
https://s3.amazonaws.com/drivendata/data/44/public
```

No Kaggle API credentials are required.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/dengue_forecast_model.py
```

The script prints fold-level, city-level, and overall model metrics.

## Near-real-time outbreak detection

The separate live detector keeps the historical dengue case labels and model
architecture, but rebuilds the weather history with Open-Meteo before training.
This avoids asking a model trained on the competition's weather definitions to
interpret values produced by a different API.

For each city, the live program:

1. Loads the historical DengAI case dates and counts.
2. Downloads matching historical Open-Meteo weather and caches it locally.
3. Defines outbreak labels using seasonal thresholds calculated from earlier
   years only.
4. Trains and time-calibrates a weather-only outbreak classifier.
5. Trains a second classifier that also uses recent case lags.
6. Downloads recent conditions and a 16-day Open-Meteo forecast.
7. Scores the current city-specific week and the following week.

Run both cities:

```bash
python src/near_realtime_outbreak_detection.py --city all
```

The first run downloads the historical data, performs time-based calibration,
and saves versioned fitted detectors, so it can take several minutes. Later
runs load compatible models from `.cache/near_realtime/`; live forecast
responses are still refreshed after six hours. Detailed alert rows are saved to
`outputs/near_realtime_outbreak_alerts.csv`.

### Optional recent cases

Add a CSV when recent official case reports are available:

```bash
python src/near_realtime_outbreak_detection.py \
  --city sj \
  --recent-cases data/recent_cases.csv
```

The CSV format is:

```csv
city,week_start_date,total_cases
sj,2026-05-11,18
sj,2026-05-18,21
```

The example is abbreviated; supply at least eight consecutive weekly rows for
the city if you want the current-week prediction to use case-lag features.

The case-aware classifier is used only when the file contains all eight exact
weekly counts immediately preceding a scored week. If even one required week is
missing, that row is explicitly labeled `weather_only`. This prevents delayed
case reporting from becoming a hidden or fabricated input.

Useful options:

```text
--weeks-ahead 0       Score only the current week
--alert-gate 0.35     Override the time-selected probability cutoff
--target-recall 0.80  Set the recall goal used to select the cutoff
--refresh             Redownload historical and live inputs
--output PATH         Choose the output CSV location
```

The live output includes the probability, alert gate, model variant, held-out
precision/recall/F1, weather cache state, forecast-day count, recent-case age,
and the historical seasonal case threshold. A stale API cache is labeled
`stale_fallback` rather than being silently presented as fresh data.

## Puerto Rico-wide continuous pipeline

The operational geography is all of Puerto Rico, not only San Juan. The old
San Juan and Iquitos model remains available as a historical benchmark, but it
is not reused for Puerto Rico-wide predictions because it learned a different
target population.

### Inputs

- Official case labels come from the Puerto Rico Department of Health's
  [`arbovirus_cases_summary` catalog](https://biostatistics.salud.pr.gov/swagger/index.html).
  Daily PCR and IgM counts are added, hospitalizations are retained, and values
  are aggregated into complete Monday–Sunday weeks. Missing dates make a week
  incomplete rather than silently becoming zero cases.
- Weather comes from [Open-Meteo](https://open-meteo.com/) at San Juan, Arecibo,
  Mayagüez, Ponce, Caguas, and Fajardo. Inputs include temperature, total rain,
  rainy hours, wet-day count, heavy-rain-day count, maximum daily rain, longest
  dry spell, humidity, dew point, shallow soil moisture, and
  evapotranspiration. Heavy-rain days use at least 10 mm of rain; dry days use
  less than 0.1 mm. The code spatially averages most fields and preserves island
  sampled temperature extremes. Every point must be complete for a daily island
  row to be used.
- The Peru open-data catalog is checked for the legacy Iquitos experiment. Its
  large official CSV currently requires a manual download, which can be
  normalized with `import-peru`; it does not affect the Puerto Rico model.

### Outbreak definition

An actual outbreak week is a week whose official total cases are at or above
the historical 75th percentile for that part of the year. The threshold uses a
five-week seasonal neighborhood: the target week, two nearby weeks before, and
two after. During validation, thresholds are calculated only from earlier
years. This is a statistical research label, not the Department of Health's
official declaration process.

### Training and prediction policy

The pipeline trains two model families for each data-availability scenario:

- A calibrated classifier estimates outbreak probability and applies an alert
  gate chosen on time-held-out data.
- Three quantile regressors estimate P50, P80, and P90 weekly cases. P50 is the
  main MAE forecast; P80 and P90 show progressively more cautious upper values.

The feature lists are intentionally separate. Time-held-out ablations found
that heavy-rain frequency and the longest dry spell improved case-count MAE,
but weakened weather-only PR-AUC when added to the alert classifiers. Those two
variables therefore feed only the MAE-focused P50/P80/P90 regressors; the
recall-focused classifiers keep their previously validated weather inputs.

There are three variants of both model families:

1. `weather_only`, which is always usable.
2. `weather_plus_delayed_cases`, used when an older report and eight contiguous
   weeks ending at that report are available. Training simulates report ages
   from 4–12 weeks; headline validation uses an eight-week delay.
3. `weather_plus_recent_cases`, used only when all eight exact prior official
   weeks are present. Live routing prefers exact cases, then delayed cases, then
   weather only.

Case-aware regressors predict a change from the latest usable report. A nested
time split shrinks uncertain changes, and P50 falls back to persistence when
held-out corrections are worse. Time-calibrated upper cushions preserve useful
P80/P90 ranges. On held-out data, delayed P80/P90 covered 76.0%/80.8% and exact
recent P80/P90 covered 76.8%/84.0%; these remain research ranges rather than
guaranteed statistical intervals.

The latest four weeks are considered provisional and cannot become retraining
labels. Monthly retraining also waits for at least 13 newly finalized weeks.
The candidate replaces the champion only if every guardrail passes:

- precision falls by no more than 0.03;
- recall falls by no more than 0.03;
- PR-AUC falls by no more than 0.02; and
- Brier score rises by no more than 0.02;
- case MAE rises by no more than 10% for each data-availability variant;
- outbreak-week MAE rises by no more than 15% for each variant; and
- each case-aware P50 stays within 5% of its matching persistence baseline.

This is scheduled retraining, not an uncontrolled model that continuously
changes itself after every prediction.

### Information-available validation

Every official-data refresh writes a content-addressed, immutable normalized
case snapshot. Every live forecast also writes the exact processed feature row
used by the champion. Repeated identical inputs reuse the same identifier;
revised inputs create a new one. These files live under
`data/operational/vintages/` and deliberately exclude temporary signed download
URLs or credentials.

Validation results are labeled in three groups:

1. **Historical proxy:** expanding-year validation with finalized historical
   target-week weather. This remains the promotion baseline until the newer
   evidence is sufficiently mature and comparable.
2. **Archived-weather audit:** target-week weather rebuilt from an ECMWF model
   run issued one day before the week. Available from March 2024 onward and
   saved to
   `reports/operational/pr_information_available_validation_latest.json`.
3. **Prospective deployment score:** only live predictions with persisted
   issue-time inputs, evaluated after official outcomes pass the four-week
   stabilization window. This is the most realistic score, but it starts small
   and grows one week at a time.

### Run the operational flow locally

```bash
# Refresh official cases and source status only
python src/continuous_pipeline.py ingest

# Weekly official-data refresh, two predictions, and outcome monitoring
python src/continuous_pipeline.py weekly

# Candidate retraining plus guarded promotion, prediction, and monitoring
python src/continuous_pipeline.py monthly

# Recalculate monitoring metrics without downloading weather
python src/continuous_pipeline.py monitor

# Run the stricter archived issue-time weather validation without promotion
python src/continuous_pipeline.py audit

# Optional legacy Iquitos official-data import
python src/continuous_pipeline.py import-peru --csv /path/to/peru_dengue.csv
```

Useful reproducibility options are `--as-of YYYY-MM-DD` and `--refresh`.

### GitHub Actions and bot commits

The repository-level `.github/workflows/puerto-rico-dengue-model.yml` runs in
weekly mode every Wednesday and monthly mode on the first day of each month.
The dated folder keeps an identical reference copy at
`automation/puerto-rico-dengue-model.yml`. Each run:

1. installs the pinned automation dependencies;
2. runs all offline tests;
3. executes the appropriate pipeline command;
4. rebuilds the dashboard and both explainers;
5. reruns the complete test suite; and
6. commits only versioned operational data, reports, champion artifacts, and
   generated public pages as `github-actions[bot]`.

The monthly run checks whether at least 13 new finalized case weeks exist.
Until then, it records the decision and keeps the current model. When
retraining is due, a candidate replaces the champion only if the configured
time-based case-error and outbreak-detection limits remain acceptable. If an
API, generator, or test fails, the run stops before any bot commit.

The repository must allow GitHub Actions **read and write** workflow permission
for bot commits. Branch rules must also allow this workflow to update the
default branch; otherwise the safe alternative is to change the final workflow
step to open a pull request.

Run the offline tests with:

```bash
python -m unittest discover -s tests -v
```

## Model explainer

- Open [the focused Puerto Rico API model explainer](docs/puerto-rico-api-model-focused.html)
  for a concise walkthrough of the operational model, its three data-availability
  routes, MAE versus recall, the archived issue-time audit, current results,
  prediction-versus-actual chart, and code map.
- Open [the Puerto Rico API model explainer](docs/puerto-rico-api-model-explainer.html)
  for the current operational architecture, prediction-versus-actual graph,
  MAE and alert charts, live example, limitations, and links to relevant files.
- Open [the historical city-model explainer](docs/model-explainer.html) for the
  earlier San Juan/Iquitos experiment.

Both API explainers are self-contained and are regenerated from the committed
champion, validation, audit, and live-prediction artifacts with:

```bash
python src/generate_api_model_explainer.py
```

## Caveats

This is a research/learning model, not a production public-health alert system. The outbreak label is a statistical definition based on historical seasonal case counts, not an official epidemiological declaration.
