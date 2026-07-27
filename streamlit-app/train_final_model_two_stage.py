"""Train and save the final TWO-STAGE dengue case model, per city.

This mirrors the two-stage system in dengue_forecast_model.py (outbreak
classifier + calibrator + normal/outbreak specialist regressors + tuned
policy gates), but trains it on the FULL historical dataset for each city
instead of the expanding-window validation splits used for evaluation.

Output: model/model_two_stage.pkl
    A dict keyed by city ("sj", "iq"), each containing:
        - "classifier": outbreak probability classifier
        - "calibrator": probability calibrator (or None)
        - "normal_model": regressor specialized on normal weeks
        - "outbreak_model": regressor specialized on outbreak/near-outbreak weeks
        - "gates": PolicyGates(mae_gate, recall_gate)
        - "thresholds": seasonal outbreak/near-outbreak threshold table
        - "selected_lags": dict of weather column -> chosen lag
        - "training_columns": exact column order the models expect
        - "medians": training medians, used as safe defaults in the UI

Run from the streamlit-app/ folder:
    python train_final_model_two_stage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dengue_forecast_model import (  # noqa: E402
    WEATHER_COLUMNS,
    add_weather_and_calendar_features,
    attach_thresholds,
    build_training_matrix,
    fit_case_regressor,
    fit_outbreak_classifier,
    fit_probability_calibrator,
    load_drivendata_dengai,
    prepare_raw_data,
    seasonal_threshold_table,
    select_weather_lags,
    tune_policy_gates,
)


def train_city_two_stage(city_data: pd.DataFrame) -> dict:
    """Train the full two-stage system for one city's complete history."""

    city_data = city_data.sort_values("week_start_date").reset_index(drop=True).copy()

    for column in WEATHER_COLUMNS:
        city_data[f"{column}_missing"] = city_data[column].isna().astype(int)
    medians = city_data[WEATHER_COLUMNS].median(numeric_only=True)
    city_data[WEATHER_COLUMNS] = city_data[WEATHER_COLUMNS].ffill().fillna(medians)

    selected_lags = select_weather_lags(city_data)

    empty_validation = city_data.iloc[0:0]
    train_features, _, base_columns = add_weather_and_calendar_features(
        city_data, empty_validation, selected_lags
    )

    thresholds = seasonal_threshold_table(train_features)
    train_features = attach_thresholds(train_features, thresholds)

    X_train, y_train = build_training_matrix(train_features, base_columns)
    training_columns = X_train.columns.tolist()

    train_outbreak_label = (
        train_features["total_cases"] >= train_features["outbreak_threshold"]
    ).astype(int).to_numpy()
    train_near_outbreak = (
        train_features["total_cases"] >= train_features["near_outbreak_threshold"]
    ).to_numpy()

    # Outbreak classifier + calibrator
    classifier = fit_outbreak_classifier()
    classifier.fit(X_train, train_outbreak_label)
    raw_train_probability = classifier.predict_proba(X_train)[:, 1]
    calibrator = fit_probability_calibrator(raw_train_probability, train_outbreak_label)

    # Specialist regressors
    normal_mask = train_outbreak_label == 0
    normal_model = fit_case_regressor()
    normal_model.fit(X_train.loc[normal_mask], y_train.loc[normal_mask])

    outbreak_model = fit_case_regressor()
    outbreak_weights = np.where(train_outbreak_label[train_near_outbreak] == 1, 2.0, 1.0)
    outbreak_model.fit(
        X_train.loc[train_near_outbreak],
        y_train.loc[train_near_outbreak],
        sample_weight=outbreak_weights,
    )

    # Policy gates (tuned on training data, matching the original script's approach)
    calibrated_train_probability = raw_train_probability
    if calibrator is not None:
        calibrated_train_probability = calibrator.predict_proba(
            raw_train_probability.reshape(-1, 1)
        )[:, 1]

    gates = tune_policy_gates(
        calibrated_train_probability,
        train_outbreak_label,
        normal_model.predict(X_train),
        outbreak_model.predict(X_train),
        y_train.to_numpy(),
    )

    return {
        "classifier": classifier,
        "calibrator": calibrator,
        "normal_model": normal_model,
        "outbreak_model": outbreak_model,
        "gates": gates,
        "thresholds": thresholds,
        "selected_lags": selected_lags,
        "training_columns": training_columns,
        "medians": X_train.median(numeric_only=True).to_dict(),
    }


def main() -> None:
    print("Loading data...")
    raw_features, raw_labels = load_drivendata_dengai()
    data = prepare_raw_data(raw_features, raw_labels)

    city_models = {}
    for city, city_data in data.groupby("city"):
        print(f"Training two-stage model for {city.upper()}...")
        city_models[city] = train_city_two_stage(city_data)
        gates = city_models[city]["gates"]
        print(f"  {city.upper()} gates -> mae_gate={gates.mae_gate}, recall_gate={gates.recall_gate}")

    output_dir = Path(__file__).resolve().parent / "model"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "model_two_stage.pkl"

    joblib.dump(city_models, output_path)
    print(f"Saved two-stage model to {output_path}")


if __name__ == "__main__":
    main()