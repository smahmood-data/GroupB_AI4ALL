"""Train the final single-stage dengue case model and save it for deployment.

This script reuses the leakage-safe feature engineering from
`dengue_forecast_model.py` but trains ONE final model per city on ALL
available historical data (no train/validation split), since the goal here
is a deployable model rather than an evaluation report.

Output: model/model.pkl
    A dict keyed by city ("sj", "iq"), each containing:
        - "model": the trained HistGradientBoostingRegressor
        - "selected_lags": dict of weather column -> chosen lag (for reference)
        - "training_columns": exact column order the model expects
        - "medians": training medians, used to fill any missing inputs

Run from the streamlit-app/ folder:
    python train_final_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import the team's existing pipeline functions instead of duplicating them.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dengue_forecast_model import (  # noqa: E402
    WEATHER_COLUMNS,
    add_weather_and_calendar_features,
    build_training_matrix,
    fit_case_regressor,
    load_drivendata_dengai,
    prepare_raw_data,
    select_weather_lags,
)


def train_city_model(city_data: pd.DataFrame) -> dict:
    """Train one final single-stage model for a single city's full history."""

    city_data = city_data.sort_values("week_start_date").reset_index(drop=True)

    # Fill missing weather values using only past data (same approach as
    # preprocess_fold, but with no validation split since we're training final).
    city_data = city_data.copy()
    for column in WEATHER_COLUMNS:
        city_data[f"{column}_missing"] = city_data[column].isna().astype(int)
    medians = city_data[WEATHER_COLUMNS].median(numeric_only=True)
    city_data[WEATHER_COLUMNS] = city_data[WEATHER_COLUMNS].ffill().fillna(medians)

    selected_lags = select_weather_lags(city_data)

    # add_weather_and_calendar_features expects a train/validation pair; give it
    # an empty validation frame since we only need the "train" side back.
    empty_validation = city_data.iloc[0:0]
    train_features, _, base_columns = add_weather_and_calendar_features(
        city_data, empty_validation, selected_lags
    )

    X, y = build_training_matrix(train_features, base_columns)
    training_columns = X.columns.tolist()

    model = fit_case_regressor()
    model.fit(X, y)

    return {
        "model": model,
        "selected_lags": selected_lags,
        "training_columns": training_columns,
        "medians": X.median(numeric_only=True).to_dict(),
    }


def main() -> None:
    print("Loading data...")
    raw_features, raw_labels = load_drivendata_dengai()
    data = prepare_raw_data(raw_features, raw_labels)

    city_models = {}
    for city, city_data in data.groupby("city"):
        print(f"Training final model for {city.upper()}...")
        city_models[city] = train_city_model(city_data)

    output_dir = Path(__file__).resolve().parent / "model"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "model.pkl"

    joblib.dump(city_models, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()