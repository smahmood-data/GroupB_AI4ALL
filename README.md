# Predicting Dengue Fever Outbreaks Using Climate Variables

Built a two-stage machine learning system that forecasts weekly dengue case counts and estimates outbreak probability from climate and public health data, applying gradient-boosted trees, probability calibration, and time-aware validation — developed through **AI4ALL 2026 (Group B, Healthcare & Life Sciences)**. The project trains and compares models on a historical 1990–2010 research dataset (San Juan, PR and Iquitos, Peru) and on a near-real-time operational pipeline built on live APIs (Puerto Rico), deploying the historical model in an interactive web app and the operational model in a public bilingual dashboard.

> ⚠️ This is a research/screening tool for an AI4ALL project — **not an official outbreak declaration or medical guidance.**

🔗 **Live demo:** Dengue Signal — Puerto Rico (public bilingual dashboard, `docs/app/` in [`2026-07-28_Nathan_Nguyen_API_Model/`](./2026-07-28_Nathan_Nguyen_API_Model))
📁 **Source code:** [github.com/smahmood-data/GroupB_AI4ALL](https://github.com/smahmood-data/GroupB_AI4ALL) *(verify link before publishing)*

## Problem Statement

Half the world's population is at risk of dengue fever, with an estimated 100–400 million infections occurring annually (WHO). Outbreaks are difficult to predict in densely populated urban and semi-urban areas, where many climate variables intersect at once, and public health responses remain largely reactive rather than proactive. Under-resourced and over-subscribed healthcare systems lack an outbreak alert system to help them prepare. Combining climate variables with dengue case history into a calibrated, responsible AI model could give public health systems earlier, more actionable warning of rising case counts.

## Key Results

1. **Built a two-stage historical model (San Juan, PR & Iquitos, Peru, 1990–2010)** that routes each week through a calibrated outbreak classifier and one of two specialist case-count regressors, achieving **11.09 MAE cases/week** with **51.6% outbreak recall (sensitivity)**, **31.6% alert precision**, and an **F1 of 39.2%** (two-stage, balanced policy).
2. **Built a near-real-time operational model for Puerto Rico** on live Department of Health and Open-Meteo weather APIs, improving to **23.99 MAE cases/week** with **90.7% outbreak recall**, **80.0% alert precision**, and an **F1 of 85.0%** (weather + delayed case history route).
3. **Identified minimum temperature and dew point as the most persistent climate predictors** of case counts across lags of 0–20 weeks, informing feature selection for the operational model.
4. **Deployed the historical model in a Streamlit web app** and the operational model in a live, automatically retraining, bilingual (English/Spanish) public dashboard.
5. **Built a guardrailed, self-retraining pipeline** that retrains once 13+ new finalized weekly case reports arrive (roughly quarterly) and only promotes a candidate model when it beats the live model on time-held-out accuracy and outbreak-detection guardrails.

## Methodologies

We approached dengue outbreak prediction in two phases, moving from a fixed historical research dataset to a live operational pipeline:

- **Phase 1 — Historical model (DrivenData, 1990–2010).** Engineered weather lags, sine/cosine seasonality encoding, and case lags/changes from NDVI, reanalysis, and station climate data, and computed a seasonal 75th-percentile outbreak threshold per city. Trained a **Histogram Gradient Boosted Classifier** to estimate outbreak probability, calibrated its output with **Logistic Regression**, then routed each week to one of two specialist **Histogram Gradient Boosted Regressors** (one trained on non-outbreak weeks, one on outbreak weeks) to forecast case counts.
- **Phase 2 — Operational API model (Puerto Rico).** Rebuilt the same core two-stage idea as a live pipeline: pulled weekly cases/hospitalizations from the PR Department of Health API and island-wide weather from Open-Meteo, engineered weather lags (0/2/4/8 weeks) and rolling windows, and trained a **quantile HistGB regressor bundle** (P50/P80/P90 case forecasts) alongside a calibrated **HistGB outbreak classifier**. The pipeline chooses among three data-availability routes (weather only; weather + delayed cases; weather + exact recent cases) depending on how current the case data is at prediction time.
- **Validation.** All models were validated on time-held-out weeks (never randomly shuffled) to prevent future information from leaking into earlier predictions; the operational pipeline additionally enforces guardrails before promoting any retrained candidate.

## Data Sources

- **[DrivenData "DengAI" dataset](https://www.drivendata.org/)** — weekly satellite (NDVI), reanalysis (temperature, humidity, precipitation), and ground-station climate data paired with historical dengue case counts for San Juan, Puerto Rico and Iquitos, Peru (1990–2010).
- **Puerto Rico Department of Health arbovirus case API** — near-real-time weekly dengue case and hospitalization reports for Puerto Rico.
- **[Open-Meteo](https://open-meteo.com/)** — historical, recent, and forecast weather variables across six Puerto Rico locations.

## Technologies Used

- **Python**
- **scikit-learn** — Histogram Gradient Boosted Regressor/Classifier, Logistic Regression, pipelines, calibration, cross-validation
- **pandas / numpy** — data cleaning and feature engineering
- **matplotlib / seaborn** — exploratory data analysis and visualization
- **Streamlit** — deployed demo app for the historical model
- **GitHub Actions** — scheduled weekly data refresh and guardrailed monthly retraining
- **Jupyter** — exploratory notebooks

## Authors

This project was completed as part of AI4ALL 2026 (Group B, Healthcare & Life Sciences) by:

- Nathan Nguyen
- Syed Mahmood
- Sahar Abid
- Zita Addy
- Ella Jeon

## Citations

- World Health Organization. (2026, March 18). *Dengue and severe dengue.* <https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue>
- Naik, K. (2020, April 20). *What is AdaBoost* [Video]. YouTube. <https://www.youtube.com/watch?v=NLRO1-jp5F8>
- *Multiyear climate variability and dengue.* (n.d.). <https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0000670>
