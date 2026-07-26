# 2026-07-25 — Nathan Nguyen API Model

## Puerto Rico Dengue Forecasting App

This folder is a self-contained snapshot of the Puerto Rico-wide dengue
forecasting project as of July 25, 2026. It includes the public bilingual app,
the focused model explainer, the operational API model, saved model artifacts,
data snapshots, validation results, reports, and tests.

The project is for research and educational use. An outbreak in this model
means that weekly reported cases exceed the seasonal cutoff: the 75th
percentile, or top 25%, of comparable weeks in earlier training years. This is
not an official government outbreak declaration.

## Open the app

The folder-level `index.html` sends visitors directly to the public dashboard.
After a repository administrator enables GitHub Pages from the `main` branch
and repository root, the shareable address will be:

<https://smahmood-data.github.io/GroupB_AI4ALL/2026-07-25_Nathan_Nguyen_API_Model/>

The focused technical explainer will be available at:

<https://smahmood-data.github.io/GroupB_AI4ALL/2026-07-25_Nathan_Nguyen_API_Model/docs/puerto-rico-api-model-focused.html>

## What is included

- `index.html` opens the public dashboard from the folder URL.
- `docs/app/` contains the bilingual English/Spanish dashboard and generated
  public data snapshot.
- `docs/puerto-rico-api-model-focused.html` explains the model architecture,
  variables, lags, testing process, results, limitations, and terminology.
- `docs/puerto-rico-api-model-explainer.html` contains the longer API model
  explainer.
- `src/` contains API ingestion, data cleaning, feature engineering, model
  training, forecasting, monitoring, and page-generation code.
- `config/operations.json` contains the Puerto Rico locations and operating
  rules used by the pipeline.
- `models/operational/pr/` contains the saved model and its held-out validation
  predictions.
- `data/operational/` contains committed case, prediction, monitoring, and
  input-vintage snapshots.
- `reports/operational/` contains retraining and information-available
  validation reports.
- `tests/` contains the automated model, leakage, operational-pipeline,
  explainer, and public-dashboard tests.
- `automation/weekly-operational-update.yml` is a reference copy of the weekly
  GitHub Actions workflow.
- `PROJECT_README.md` contains the full development documentation.

Everything for this iteration is contained inside this dated subfolder.

## Model outputs

The system produces two related outputs:

1. A case-count forecast with a starting estimate and two higher estimates.
2. The estimated chance that cases will exceed that week’s seasonal outbreak
   cutoff.

It chooses among three data routes according to what was available before the
forecast:

1. Weather only.
2. Weather plus delayed case reports.
3. Weather plus exact recent case reports.

The current delayed-case route uses the latest official count as its starting
estimate because that simple fallback was more accurate than the learned
adjustment in time-based testing. Weather and delayed health information still
affect the higher estimates and outbreak probability.

## Current held-out results

These results use 125 later weeks that were excluded from their corresponding
training periods.

| Available inputs | Case MAE | Alerts correct | Outbreaks caught | F1 |
| --- | ---: | ---: | ---: | ---: |
| Weather only | 56.94 | 77.6% | 100.0% | 87.4% |
| Weather + delayed cases | 23.99 | 80.0% | 90.7% | 85.0% |
| Weather + exact recent cases | **10.69** | **83.5%** | 99.0% | **90.6%** |

MAE is the average number of cases by which the case-count prediction missed.
“Outbreaks caught” is sensitivity, also called recall. “Alerts correct” is
precision. F1 balances precision and recall.

## Run locally

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m http.server 8000
```

Open:

- App: <http://localhost:8000/>
- Focused explainer:
  <http://localhost:8000/docs/puerto-rico-api-model-focused.html>

Run all tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Refresh the public app snapshot after producing new operational predictions:

```bash
python src/generate_public_dashboard.py
```

Regenerate both model explainers:

```bash
python src/generate_api_model_explainer.py
```

## Scheduled updates

The included workflow file is kept inside this subfolder so the entire
iteration remains self-contained. GitHub only runs workflows placed in the
repository-level `.github/workflows/` directory. A repository administrator
must copy or adapt `automation/weekly-operational-update.yml` there if the
GroupB repository should run this model automatically.

The model does not silently retrain itself after each prediction. Scheduled
retraining creates a candidate model, and the current model is replaced only
when the configured time-based safety checks remain acceptable.

