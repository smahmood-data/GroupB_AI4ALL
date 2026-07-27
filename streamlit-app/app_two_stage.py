"""Streamlit app for the DengAI TWO-STAGE case-count model.

Loads the model trained by train_final_model_two_stage.py. This reflects
the full two-stage system from dengue_forecast_model.py: an outbreak
probability classifier gates between a "normal week" specialist regressor
and an "outbreak week" specialist regressor.

Run:
    streamlit run app_two_stage.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dengue_forecast_model import WEATHER_COLUMNS  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model_two_stage.pkl"

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

    row = dict(medians)

    for harmonic in [1, 2]:
        row[f"week_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * weekofyear / 52.0)
        row[f"week_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * weekofyear / 52.0)

    lags = city_bundle["selected_lags"]
    for column in WEATHER_COLUMNS:
        value = weather_values[column]
        lag = lags[column]
        row[f"{column}_lag_{lag}"] = value
        row[f"{column}_mean_4"] = value
        row[f"{column}_missing"] = 0

    recent = last_8_cases
    row["cases_lag_1"] = recent[-1]
    row["cases_lag_2"] = recent[-2]
    row["cases_lag_4"] = recent[-4]
    row["cases_lag_52"] = cases_last_year
    row["cases_mean_4"] = float(np.mean(recent[-4:]))
    row["cases_mean_8"] = float(np.mean(recent))
    row["cases_change_1"] = recent[-1] - recent[-2]

    ordered = {col: row.get(col, 0.0) for col in training_columns}
    return pd.DataFrame([ordered])[training_columns]


st.set_page_config(page_title="DengAI Two-Stage Predictor", page_icon="🦟")
st.title("Dengue Case Count Predictor — Two-Stage Model")
st.caption("Outbreak classifier + specialist regressors | San Juan & Iquitos | DengAI dataset")

st.markdown(
    """
This tool predicts the number of dengue cases expected next week in San Juan
or Iquitos, using historical weather and case data. It first estimates how
"outbreak-like" the week looks, then hands the prediction off to whichever of
two specialist models is better suited: one trained on ordinary weeks, one
trained on high-case weeks. This two-step approach outperforms a single
general-purpose model, especially during outbreaks.
"""
)

with st.expander("Model performance (from validated test years)"):
    st.markdown(
        """
| | Overall MAE | Outbreak-week MAE |
|---|---|---|
| Single-stage baseline | 11.68 | 31.69 |
| Two-stage, balanced policy | **11.09** | 33.93 |
| Two-stage, sensitive policy | 11.68 | **22.11** |

*MAE = mean absolute error in predicted case count; lower is better.*

**Outbreak alerts:** catch about 52% of true outbreak weeks (recall), and
about 32% of alerts correspond to a real outbreak (precision). These alerts
are a useful research signal, but they miss nearly half of real outbreaks
and produce many false alarms — **they are not accurate enough to drive
public-health action on their own.**
"""
    )

city_models = load_model()

if city_models is None:
    st.error(
        "No trained model found at `model/model_two_stage.pkl`. "
        "Run `python train_final_model_two_stage.py` first to generate it."
    )
    st.stop()

city_display = {"sj": "San Juan", "iq": "Iquitos"}
city = st.selectbox("City", options=list(city_display.keys()), format_func=lambda c: city_display[c])
city_bundle = city_models[city]
gates = city_bundle["gates"]

st.subheader("Timing")
st.caption("Dengue is seasonal, so the model factors in which week of the year you're predicting for.")
weekofyear = st.slider("Week of year", min_value=1, max_value=53, value=25)

st.subheader("Recent case counts")
st.caption(
    "Case counts don't change randomly week to week — recent trends are one of the "
    "strongest predictors of what happens next. Enter reported dengue case counts for "
    "the last 8 weeks (oldest to most recent)."
)
default_recent = [10] * 8
last_8_cases = []
cols = st.columns(8)
for i, col in enumerate(cols):
    with col:
        val = st.number_input(f"Wk -{8 - i}", min_value=0, value=default_recent[i], key=f"ts_wk_{i}")
        last_8_cases.append(val)

st.caption("Dengue outbreaks often recur around the same time each year, so this gives the model a year-over-year comparison point.")
cases_last_year = st.number_input("Cases in this same week, one year ago", min_value=0, value=10, key="ts_last_year")

st.subheader("Current weather conditions")
st.caption(
    "Mosquito breeding and survival are strongly tied to temperature, humidity, and "
    "rainfall, so these readings feed directly into the prediction."
)
weather_values = {}
with st.expander("Enter current weather readings (defaults are historical averages)"):
    for column in WEATHER_COLUMNS:
        default = float(city_bundle["medians"].get(f"{column}_lag_{city_bundle['selected_lags'][column]}", 0.0))
        weather_values[column] = st.number_input(
            WEATHER_LABELS.get(column, column), value=round(default, 2), key=f"ts_{column}"
        )

st.subheader("Alert policy")
st.caption(
    "The outbreak classifier's probability is compared against a threshold (\"gate\") "
    "to decide which specialist regressor makes the final prediction."
)
policy = st.radio(
    "Choose which gate to use",
    options=["Balanced (minimizes overall prediction error)", "Sensitive (catches more true outbreaks, more false alarms)"],
)
gate_value = gates.mae_gate if policy.startswith("Balanced") else gates.recall_gate

if st.button("Predict case count", type="primary"):
    X = build_feature_row(city_bundle, weekofyear, weather_values, last_8_cases, cases_last_year)

    raw_probability = city_bundle["classifier"].predict_proba(X)[:, 1]
    if city_bundle["calibrator"] is not None:
        probability = city_bundle["calibrator"].predict_proba(raw_probability.reshape(-1, 1))[:, 1][0]
    else:
        probability = raw_probability[0]

    is_alert = probability >= gate_value
    if is_alert:
        prediction = max(0.0, float(city_bundle["outbreak_model"].predict(X)[0]))
        specialist_used = "outbreak specialist"
    else:
        prediction = max(0.0, float(city_bundle["normal_model"].predict(X)[0]))
        specialist_used = "normal-week specialist"

    st.success(f"Predicted cases for {city_display[city]} next week: **{prediction:.0f}**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Outbreak probability", f"{probability:.0%}")
    with col2:
        st.metric("Alert triggered?", "Yes" if is_alert else "No")

    st.caption(
        f"Prediction generated by the {specialist_used}, since the calibrated outbreak "
        f"probability was {'above' if is_alert else 'below'} the {policy.split(' ')[0].lower()} "
        f"gate ({gate_value:.2f})."
    )

    if is_alert:
        st.warning(
            "This alert is based on a model that only catches about half of real "
            "outbreak weeks and is correct about a third of the time it fires. "
            "Treat it as a signal worth investigating further, not a confirmed outbreak."
        )