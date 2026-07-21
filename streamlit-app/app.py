"""Streamlit app for the DengAI case-count model.

Loads the model trained by train_final_model.py and lets a user enter
current weather conditions and recent case history to get a predicted
case count for the upcoming week.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dengue_forecast_model import WEATHER_COLUMNS  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"

# Friendlier labels for the raw weather column names.
WEATHER_LABELS = {
    "ndvi_ne": "Vegetation index (NE)",
    "ndvi_nw": "Vegetation index (NW)",
    "ndvi_se": "Vegetation index (SE)",
    "ndvi_sw": "Vegetation index (SW)",
    "precipitation_amt_mm": "Precipitation (mm)",
    "reanalysis_air_temp_k": "Air temp, reanalysis (K)",
    "reanalysis_avg_temp_k": "Avg temp, reanalysis (K)",
    "reanalysis_dew_point_temp_k": "Dew point temp (K)",
    "reanalysis_max_air_temp_k": "Max air temp, reanalysis (K)",
    "reanalysis_min_air_temp_k": "Min air temp, reanalysis (K)",
    "reanalysis_precip_amt_kg_per_m2": "Precip, reanalysis (kg/m^2)",
    "reanalysis_relative_humidity_percent": "Relative humidity (%)",
    "reanalysis_sat_precip_amt_mm": "Satellite precip (mm)",
    "reanalysis_specific_humidity_g_per_kg": "Specific humidity (g/kg)",
    "reanalysis_tdtr_k": "Diurnal temp range, reanalysis (K)",
    "station_avg_temp_c": "Station avg temp (C)",
    "station_diur_temp_rng_c": "Station diurnal temp range (C)",
    "station_max_temp_c": "Station max temp (C)",
    "station_min_temp_c": "Station min temp (C)",
    "station_precip_mm": "Station precipitation (mm)",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def build_feature_row(city_bundle: dict, weekofyear: int, weather_values: dict, last_8_cases: list, cases_last_year: float) -> pd.DataFrame:
    """Assemble a single-row DataFrame matching the model's training columns."""

    training_columns = city_bundle["training_columns"]
    medians = city_bundle["medians"]

    row = dict(medians)  # start from training medians as a safe fallback

    # Seasonal (week-of-year) features
    for harmonic in [1, 2]:
        row[f"week_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * weekofyear / 52.0)
        row[f"week_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * weekofyear / 52.0)

    # Weather features: user enters ONE current-conditions value per variable;
    # we use it for both the lag and rolling-mean columns, and mark missing=0.
    lags = city_bundle["selected_lags"]
    for column in WEATHER_COLUMNS:
        value = weather_values[column]
        lag = lags[column]
        row[f"{column}_lag_{lag}"] = value
        row[f"{column}_mean_4"] = value
        row[f"{column}_missing"] = 0

    # Case-history features
    recent = last_8_cases  # most recent last, e.g. index -1 = last week
    row["cases_lag_1"] = recent[-1]
    row["cases_lag_2"] = recent[-2]
    row["cases_lag_4"] = recent[-4]
    row["cases_lag_52"] = cases_last_year
    row["cases_mean_4"] = float(np.mean(recent[-4:]))
    row["cases_mean_8"] = float(np.mean(recent))
    row["cases_change_1"] = recent[-1] - recent[-2]

    ordered = {col: row.get(col, 0.0) for col in training_columns}
    return pd.DataFrame([ordered])[training_columns]


st.set_page_config(page_title="DengAI Case Predictor", page_icon="🦟")
st.title("Dengue Case Count Predictor")
st.caption("Single-stage model | San Juan & Iquitos | DengAI dataset")

city_models = load_model()

if city_models is None:
    st.error(
        "No trained model found at `model/model.pkl`. "
        "Run `python train_final_model.py` first to generate it."
    )
    st.stop()

city_display = {"sj": "San Juan", "iq": "Iquitos"}
city = st.selectbox("City", options=list(city_display.keys()), format_func=lambda c: city_display[c])
city_bundle = city_models[city]

st.subheader("Timing")
weekofyear = st.slider("Week of year", min_value=1, max_value=53, value=25)

st.subheader("Recent case counts")
st.caption("Enter reported dengue case counts for the last 8 weeks (oldest to most recent).")
default_recent = [10] * 8
last_8_cases = []
cols = st.columns(8)
for i, col in enumerate(cols):
    with col:
        val = st.number_input(f"Wk -{8 - i}", min_value=0, value=default_recent[i], key=f"wk_{i}")
        last_8_cases.append(val)

cases_last_year = st.number_input("Cases in this same week, one year ago", min_value=0, value=10)

st.subheader("Current weather conditions")
weather_values = {}
with st.expander("Enter current weather readings (defaults are historical averages)"):
    for column in WEATHER_COLUMNS:
        default = float(city_bundle["medians"].get(f"{column}_lag_{city_bundle['selected_lags'][column]}", 0.0))
        weather_values[column] = st.number_input(
            WEATHER_LABELS.get(column, column), value=round(default, 2), key=column
        )

if st.button("Predict case count", type="primary"):
    X = build_feature_row(city_bundle, weekofyear, weather_values, last_8_cases, cases_last_year)
    prediction = max(0.0, float(city_bundle["model"].predict(X)[0]))
    st.success(f"Predicted cases for {city_display[city]} next week: **{prediction:.0f}**")
    st.caption(
        "This is a point estimate from a single-stage gradient boosting regressor. "
        "It does not include the outbreak-classifier logic from the research script."
    )  