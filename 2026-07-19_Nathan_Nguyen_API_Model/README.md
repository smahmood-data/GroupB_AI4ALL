# 2026-07-19 — Nathan Nguyen API Model

## Puerto Rico Dengue Forecasting

This folder contains a Puerto Rico-wide dengue forecasting system built from
official case reports and near-real-time weather data. It produces two separate
outputs:

- a weekly case-count forecast optimized for mean absolute error (MAE); and
- an outbreak alert optimized to catch a high proportion of outbreak weeks.

The system is intended for research and educational use. Its outbreak label is
a statistical threshold, not an official public-health declaration.

## Model variants

The pipeline chooses the strongest input history available for each forecast:

1. **Weather only** uses weather and seasonality when case history is missing.
2. **Weather + delayed cases** uses weather and the newest case report even when
   that report is several weeks old. Report age is included as a feature.
3. **Weather + exact recent cases** uses weather plus all eight case-report weeks
   immediately preceding the prediction week.

Each variant contains an MAE-focused case model and a separate recall-focused
outbreak classifier. Heavy-rain frequency and longest dry spell are used only
by the case-count models because time-held-out testing showed they improved MAE
but weakened alert classification.

## Current held-out results

The saved champion was trained on 385 complete Puerto Rico weeks through
May 18, 2026. Results below use 125 later, time-held-out observations.

| Available inputs | Case MAE | Classifier precision | Classifier recall | Classifier F1 |
| --- | ---: | ---: | ---: | ---: |
| Weather only | 56.94 | 77.6% | 100.0% | 0.874 |
| Weather + delayed cases | 23.99 | 80.0% | 90.7% | 0.850 |
| Weather + exact recent cases | **10.69** | **83.5%** | 99.0% | **0.906** |

## Core code

- `src/puerto_rico_operational.py` trains the three model variants, forecasts
  P50/P80/P90 case counts, scores outbreak alerts, and applies promotion rules.
- `src/near_realtime_outbreak_detection.py` downloads weather, creates
  independent lags and rolling features, calibrates probabilities, and selects
  the recall-focused alert gate.
- `src/official_case_data.py` downloads, cleans, aggregates, and audits official
  Puerto Rico case and hospitalization reports.
- `src/continuous_pipeline.py` orchestrates ingestion, guarded retraining,
  champion promotion, prediction, and monitoring.

Supporting files include:

- `src/dengue_forecast_model.py` for shared model factories and seasonal
  threshold calculations;
- `config/operations.json` for weather locations and operational policies;
- `models/operational/pr/` for the saved champion and validation predictions;
- `docs/puerto-rico-api-model-explainer.html` for the visual model explainer;
- `data/operational/` for the committed case and prediction snapshots; and
- `tests/` for leakage, feature-schema, ingestion, promotion, and explainer
  checks.

## Setup

Run these commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Operational commands

Refresh official data and create current-week and next-week predictions:

```bash
python src/continuous_pipeline.py weekly
```

Train a candidate, apply promotion guardrails, predict, and monitor:

```bash
python src/continuous_pipeline.py monthly
```

Regenerate the explainer after model artifacts change:

```bash
python src/generate_api_model_explainer.py
```

The monthly command is scheduled retraining, not uncontrolled continuous
learning. A candidate replaces the saved champion only when its time-held-out
metrics satisfy the configured guardrails.

## Data sources

- Puerto Rico Department of Health arbovirus case summaries provide case and
  hospitalization counts.
- Open-Meteo provides historical, recent, and forecast weather across six
  Puerto Rico locations.

Missing dates do not silently become zero cases. Incomplete weeks are excluded,
recent provisional labels are held back from training, and validation follows
time order to prevent future information from entering earlier predictions.
