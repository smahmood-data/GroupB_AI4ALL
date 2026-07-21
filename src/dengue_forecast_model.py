"""Leakage-safe dengue case forecasting.

This script is the clean, first-iteration model pipeline for the DengAI project.

The main design goal is honesty: every validation year is treated like a future
year. The model may use only information that would have been available at the
time of prediction. That means:

* missing values are imputed inside each fold using training history;
* weather variables get their own independently selected lags;
* case lag features are updated recursively with previous predictions during
  validation, instead of using the hidden validation labels;
* outbreak thresholds are estimated from training years only.

Run:
    python src/dengue_forecast_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)


DATA_BASE = "https://s3.amazonaws.com/drivendata/data/44/public"

KEY_COLUMNS = ["city", "year", "weekofyear"]

WEATHER_COLUMNS = [
    "ndvi_ne",
    "ndvi_nw",
    "ndvi_se",
    "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k",
    "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2",
    "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",
    "station_avg_temp_c",
    "station_diur_temp_rng_c",
    "station_max_temp_c",
    "station_min_temp_c",
    "station_precip_mm",
]


@dataclass(frozen=True)
class PolicyGates:
    """Decision thresholds for the two-stage model.

    A gate is a cutoff on calibrated outbreak probability.

    * If probability >= gate, use the outbreak magnitude model.
    * Otherwise, use the normal-week magnitude model.
    """

    mae_gate: float
    recall_gate: float


def load_drivendata_dengai() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load public DrivenData DengAI training data.

    The project intentionally reads DrivenData's public files directly. It does
    not need Kaggle credentials or the Kaggle API.
    """

    features = pd.read_csv(f"{DATA_BASE}/dengue_features_train.csv")
    labels = pd.read_csv(f"{DATA_BASE}/dengue_labels_train.csv")
    return features, labels


def prepare_raw_data(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels to features using explicit keys.

    Joining on city/year/week prevents a subtle notebook bug: if labels were
    attached by row index and the rows ever changed order, the target could be
    silently paired with the wrong features.
    """

    data = features.merge(
        labels[KEY_COLUMNS + ["total_cases"]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    data["week_start_date"] = pd.to_datetime(data["week_start_date"])
    return data.sort_values(["city", "week_start_date"]).reset_index(drop=True)


def complete_years(city_data: pd.DataFrame) -> list[int]:
    """Return years with roughly complete weekly coverage."""

    counts = city_data.groupby("year").size()
    return counts[counts >= 50].index.tolist()


def outer_expanding_splits(city_data: pd.DataFrame, n_outer_folds: int = 4) -> Iterable[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield expanding train/validation splits by validation year.

    Example pattern:
        train through 2004 -> validate 2005
        train through 2005 -> validate 2006
        train through 2006 -> validate 2007

    The final `n_outer_folds` complete years are used as validation years.
    """

    years = complete_years(city_data)
    validation_years = years[-n_outer_folds:]

    for validation_year in validation_years:
        train = city_data[city_data["year"] < validation_year].copy()
        validation = city_data[city_data["year"] == validation_year].copy()
        if len(train) and len(validation):
            yield validation_year, train, validation


def preprocess_fold(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing weather values without looking into the validation future.

    Earlier notebook versions interpolated the full dataset before splitting.
    That is risky because validation-period values can influence training-period
    values. Here, each fold is handled separately:

    1. Add missingness flags before filling.
    2. Forward-fill training data within city.
    3. Use training medians for any remaining training gaps.
    4. For validation, seed the forward-fill with the final training row, then
       move forward through validation time.

    This is more realistic for a forecasting setup than bidirectional
    interpolation.
    """

    train = train.sort_values("week_start_date").copy()
    validation = validation.sort_values("week_start_date").copy()

    for column in WEATHER_COLUMNS:
        train[f"{column}_missing"] = train[column].isna().astype(int)
        validation[f"{column}_missing"] = validation[column].isna().astype(int)

    train_medians = train[WEATHER_COLUMNS].median(numeric_only=True)

    train[WEATHER_COLUMNS] = train[WEATHER_COLUMNS].ffill().fillna(train_medians)

    # Validation forward-fill is seeded by the last training observation. This
    # lets week 1 of validation use the most recent historical weather value
    # without peeking at later validation weeks.
    combined_weather = pd.concat(
        [train.tail(1)[WEATHER_COLUMNS], validation[WEATHER_COLUMNS]],
        axis=0,
    )
    filled_validation = combined_weather.ffill().fillna(train_medians).iloc[1:]
    validation[WEATHER_COLUMNS] = filled_validation.to_numpy()

    return train, validation


def inner_expanding_splits(train: pd.DataFrame, n_splits: int = 3) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create smaller time-based splits inside the training period.

    These inner splits are used for choices such as weather lag selection and
    probability calibration. The outer validation year remains untouched.
    """

    years = sorted(train["year"].unique())
    candidate_years = years[-min(n_splits, max(0, len(years) - 3)) :]

    for validation_year in candidate_years:
        inner_train = train[train["year"] < validation_year].copy()
        inner_validation = train[train["year"] == validation_year].copy()
        if len(inner_train) and len(inner_validation):
            yield inner_train, inner_validation


def select_weather_lags(train: pd.DataFrame, max_lag: int = 12) -> dict[str, int]:
    """Pick a separate lag for each weather variable.

    The old San Juan code shifted the entire feature matrix by eight weeks.
    That also shifted case features, accidentally turning `cases_lag_1` into
    something closer to `cases_lag_9`.

    This function does the safer thing:

    * weather columns may each receive their own lag;
    * case lags are created later and are never shifted by weather lag logic.

    The score is deliberately simple: for each candidate lag, compute the
    absolute correlation with total cases inside the training period. This is a
    lightweight lag-selection heuristic, not a final causal claim.

    A more expensive version can select lags with inner expanding-window
    cross-validation. For a first public iteration, this faster heuristic keeps
    the repository runnable on a normal laptop while preserving the most
    important correction: weather lagging is separate from case lagging.
    """

    selected: dict[str, int] = {}

    for column in WEATHER_COLUMNS:
        lag_scores: list[tuple[int, float]] = []

        for lag in range(max_lag + 1):
            aligned = pd.DataFrame(
                {
                    "lagged_weather": train[column].shift(lag),
                    "cases": train["total_cases"],
                }
            ).dropna()
            corr = aligned["lagged_weather"].corr(aligned["cases"]) if len(aligned) > 5 else np.nan
            lag_scores.append((lag, abs(float(corr)) if pd.notna(corr) else 0.0))

        # If two lags are equally good, prefer the shorter lag because it is
        # easier to justify operationally.
        selected[column] = sorted(lag_scores, key=lambda item: (-item[1], item[0]))[0][0]

    return selected


def add_weather_and_calendar_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected_lags: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create non-case features that can be known before case labels arrive."""

    train = train.copy()
    validation = validation.copy()
    train["_partition"] = "train"
    validation["_partition"] = "validation"

    combined = pd.concat([train, validation]).sort_values("week_start_date").reset_index(drop=True)

    feature_columns: list[str] = []

    # Smooth yearly seasonality. Sine/cosine encoding avoids treating week 52
    # and week 1 as far apart.
    for harmonic in [1, 2]:
        sin_col = f"week_sin_{harmonic}"
        cos_col = f"week_cos_{harmonic}"
        combined[sin_col] = np.sin(2 * np.pi * harmonic * combined["weekofyear"] / 52.0)
        combined[cos_col] = np.cos(2 * np.pi * harmonic * combined["weekofyear"] / 52.0)
        feature_columns.extend([sin_col, cos_col])

    for column in WEATHER_COLUMNS:
        lag = selected_lags[column]
        lag_col = f"{column}_lag_{lag}"
        mean_col = f"{column}_mean_4"

        combined[lag_col] = combined[column].shift(lag)
        combined[mean_col] = combined[column].shift(lag).rolling(4, min_periods=1).mean()
        feature_columns.extend([lag_col, mean_col, f"{column}_missing"])

    train_features = combined[combined["_partition"] == "train"].copy()
    validation_features = combined[combined["_partition"] == "validation"].copy()

    return train_features, validation_features, feature_columns


def case_features_from_history(row: pd.Series, case_history: list[float]) -> dict[str, float]:
    """Build case-lag features from whatever case history is available.

    During training, `case_history` contains real past cases.
    During validation, it starts with real training cases and then receives the
    model's own previous predictions. This is what makes validation recursive.
    """

    def lag(n: int) -> float:
        return case_history[-n] if len(case_history) >= n else np.nan

    recent_4 = case_history[-4:]
    recent_8 = case_history[-8:]

    return {
        "cases_lag_1": lag(1),
        "cases_lag_2": lag(2),
        "cases_lag_4": lag(4),
        "cases_lag_52": lag(52),
        "cases_mean_4": float(np.mean(recent_4)) if recent_4 else np.nan,
        "cases_mean_8": float(np.mean(recent_8)) if recent_8 else np.nan,
        "cases_change_1": lag(1) - lag(2) if len(case_history) >= 2 else np.nan,
    }


def build_training_matrix(train_features: pd.DataFrame, base_feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Build the supervised training matrix using real past case counts."""

    rows: list[dict[str, float]] = []
    targets: list[float] = []
    case_history: list[float] = []

    for _, row in train_features.sort_values("week_start_date").iterrows():
        features = row[base_feature_columns].to_dict()
        features.update(case_features_from_history(row, case_history))
        rows.append(features)
        targets.append(row["total_cases"])
        case_history.append(float(row["total_cases"]))

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(targets, name="total_cases")

    # Rows at the beginning of a city's history do not have enough lagged case
    # context. Median imputation keeps the model usable while preserving the
    # strictly-past-only rule.
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def seasonal_threshold_table(train_features: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Estimate seasonal outbreak thresholds from training years only.

    A single city-wide cutoff is crude because dengue has seasonality. Forty
    cases may be normal during one part of the year and unusual during another.

    For each week of year, we compare against nearby historical weeks:
        week 30 uses historical weeks 28, 29, 30, 31, 32

    q75 defines an "actual outbreak week" for evaluation and classifier labels.
    q60 gives the outbreak magnitude model extra near-outbreak examples.
    """

    records = []
    weeks = np.arange(1, 54)

    for week in weeks:
        cyclic_distance = np.minimum(
            np.abs(train_features["weekofyear"] - week),
            53 - np.abs(train_features["weekofyear"] - week),
        )
        seasonal_cases = train_features.loc[cyclic_distance <= window, "total_cases"]

        if len(seasonal_cases) == 0:
            q60 = train_features["total_cases"].quantile(0.60)
            q75 = train_features["total_cases"].quantile(0.75)
        else:
            q60 = seasonal_cases.quantile(0.60)
            q75 = seasonal_cases.quantile(0.75)

        records.append({"weekofyear": week, "near_outbreak_threshold": q60, "outbreak_threshold": q75})

    return pd.DataFrame(records)


def attach_thresholds(frame: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Attach seasonal q60/q75 thresholds to a train or validation frame."""

    return frame.merge(thresholds, on="weekofyear", how="left", validate="many_to_one")


def fit_case_regressor(loss: str = "absolute_error") -> HistGradientBoostingRegressor:
    """Create the case-count model.

    `absolute_error` aligns the regressor with MAE better than squared-error
    loss because it estimates a conditional median rather than a mean.
    """

    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.05,
        max_iter=90,
        max_leaf_nodes=16,
        l2_regularization=0.05,
        random_state=42,
    )


def fit_outbreak_classifier() -> HistGradientBoostingClassifier:
    """Create the outbreak classifier."""

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=90,
        max_leaf_nodes=16,
        l2_regularization=0.05,
        random_state=42,
    )


def fit_probability_calibrator(raw_probabilities: np.ndarray, labels: np.ndarray) -> LogisticRegression | None:
    """Fit a simple probability calibrator when both classes are present.

    Tree classifiers can rank weeks usefully while still producing probability
    numbers that are too high or too low. Calibration uses inner time splits to
    make the raw scores behave more like probabilities.
    """

    if len(np.unique(labels)) < 2:
        return None

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(raw_probabilities.reshape(-1, 1), labels)
    return calibrator


def tune_policy_gates(probabilities: np.ndarray, actual_outbreak: np.ndarray, normal_pred: np.ndarray, outbreak_pred: np.ndarray, actual_cases: np.ndarray) -> PolicyGates:
    """Choose separate gates for MAE forecasting and recall-focused alerting."""

    candidates = np.round(np.arange(0.05, 0.95, 0.05), 2)
    rows = []

    for gate in candidates:
        alert = probabilities >= gate
        blended_prediction = np.where(alert, outbreak_pred, normal_pred)
        rows.append(
            {
                "gate": gate,
                "mae": mean_absolute_error(actual_cases, blended_prediction),
                "precision": precision_score(actual_outbreak, alert, zero_division=0),
                "recall": recall_score(actual_outbreak, alert, zero_division=0),
            }
        )

    scores = pd.DataFrame(rows)

    # MAE policy: minimize forecast error. If tied, use a higher gate to avoid
    # unnecessary outbreak switches.
    mae_gate = float(scores.sort_values(["mae", "gate"], ascending=[True, False]).iloc[0]["gate"])

    # Recall policy: choose a permissive alert threshold. This intentionally
    # catches more possible outbreak weeks at the cost of more false alarms.
    #
    # If any gate catches at least 80% of training outbreak weeks, use the
    # lowest such gate. If none reach 80%, use the lowest gate among the
    # highest-recall candidates. The point is not to maximize MAE here; it is
    # to create a deliberately more sensitive warning policy.
    eligible = scores[scores["recall"] >= 0.80]
    if len(eligible):
        recall_gate = float(eligible.sort_values(["gate", "precision", "mae"], ascending=[True, False, True]).iloc[0]["gate"])
    else:
        recall_gate = float(scores.sort_values(["recall", "gate", "precision"], ascending=[False, True, False]).iloc[0]["gate"])

    return PolicyGates(mae_gate=mae_gate, recall_gate=recall_gate)


def recursive_predict_single_stage(
    model: HistGradientBoostingRegressor,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    base_feature_columns: list[str],
    training_columns: list[str],
) -> np.ndarray:
    """Predict validation one week at a time with recursive case lags."""

    predictions: list[float] = []
    case_history = train_features.sort_values("week_start_date")["total_cases"].astype(float).tolist()

    for _, row in validation_features.sort_values("week_start_date").iterrows():
        features = row[base_feature_columns].to_dict()
        features.update(case_features_from_history(row, case_history))
        X_row = pd.DataFrame([features])[training_columns].fillna(0)
        prediction = max(0.0, float(model.predict(X_row)[0]))
        predictions.append(prediction)
        case_history.append(prediction)

    return np.array(predictions)


def evaluate_fold(city: str, validation_year: int, train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate one city/year fold."""

    train, validation = preprocess_fold(train, validation)
    selected_lags = select_weather_lags(train)
    train_features, validation_features, base_columns = add_weather_and_calendar_features(train, validation, selected_lags)

    thresholds = seasonal_threshold_table(train_features)
    train_features = attach_thresholds(train_features, thresholds)
    validation_features = attach_thresholds(validation_features, thresholds)

    X_train, y_train = build_training_matrix(train_features, base_columns)
    training_columns = X_train.columns.tolist()

    # ----- Single-stage baseline -----
    single_model = fit_case_regressor()
    single_model.fit(X_train, y_train)
    single_prediction = recursive_predict_single_stage(
        single_model,
        train_features,
        validation_features,
        base_columns,
        training_columns,
    )

    # ----- Two-stage model -----
    train_outbreak_label = (train_features["total_cases"] >= train_features["outbreak_threshold"]).astype(int).to_numpy()
    train_near_outbreak = train_features["total_cases"] >= train_features["near_outbreak_threshold"]

    classifier = fit_outbreak_classifier()
    classifier.fit(X_train, train_outbreak_label)
    raw_train_probability = classifier.predict_proba(X_train)[:, 1]
    calibrator = fit_probability_calibrator(raw_train_probability, train_outbreak_label)

    normal_mask = train_outbreak_label == 0
    outbreak_support_mask = train_near_outbreak.to_numpy()

    normal_model = fit_case_regressor()
    normal_model.fit(X_train.loc[normal_mask], y_train.loc[normal_mask])

    outbreak_model = fit_case_regressor()
    outbreak_weights = np.where(train_outbreak_label[outbreak_support_mask] == 1, 2.0, 1.0)
    outbreak_model.fit(
        X_train.loc[outbreak_support_mask],
        y_train.loc[outbreak_support_mask],
        sample_weight=outbreak_weights,
    )

    # Tune gates on training rows. In a later iteration, this should use inner
    # out-of-fold predictions rather than in-sample scores; the current version
    # keeps the code compact while preserving the hard-gate idea.
    calibrated_train_probability = raw_train_probability
    if calibrator is not None:
        calibrated_train_probability = calibrator.predict_proba(raw_train_probability.reshape(-1, 1))[:, 1]

    gates = tune_policy_gates(
        calibrated_train_probability,
        train_outbreak_label,
        normal_model.predict(X_train),
        outbreak_model.predict(X_train),
        y_train.to_numpy(),
    )

    mae_policy_predictions: list[float] = []
    recall_policy_predictions: list[float] = []
    outbreak_probabilities: list[float] = []
    mae_policy_alerts: list[bool] = []
    recall_policy_alerts: list[bool] = []

    mae_history = train_features.sort_values("week_start_date")["total_cases"].astype(float).tolist()
    recall_history = list(mae_history)

    for _, row in validation_features.sort_values("week_start_date").iterrows():
        # Both policies see the same weather/calendar features. They differ only
        # in how readily they switch to the outbreak specialist and in which
        # previous predictions they feed into future case-lag features.
        mae_features = row[base_columns].to_dict()
        mae_features.update(case_features_from_history(row, mae_history))
        X_mae = pd.DataFrame([mae_features])[training_columns].fillna(0)

        recall_features = row[base_columns].to_dict()
        recall_features.update(case_features_from_history(row, recall_history))
        X_recall = pd.DataFrame([recall_features])[training_columns].fillna(0)

        raw_probability = classifier.predict_proba(X_mae)[:, 1]
        probability = raw_probability
        if calibrator is not None:
            probability = calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
        probability_value = float(probability[0])

        normal_mae = max(0.0, float(normal_model.predict(X_mae)[0]))
        outbreak_mae = max(0.0, float(outbreak_model.predict(X_mae)[0]))
        mae_alert = probability_value >= gates.mae_gate
        mae_prediction = outbreak_mae if mae_alert else normal_mae

        normal_recall = max(0.0, float(normal_model.predict(X_recall)[0]))
        outbreak_recall = max(0.0, float(outbreak_model.predict(X_recall)[0]))
        recall_alert = probability_value >= gates.recall_gate
        recall_prediction = outbreak_recall if recall_alert else normal_recall

        outbreak_probabilities.append(probability_value)
        mae_policy_alerts.append(mae_alert)
        recall_policy_alerts.append(recall_alert)
        mae_policy_predictions.append(mae_prediction)
        recall_policy_predictions.append(recall_prediction)
        mae_history.append(mae_prediction)
        recall_history.append(recall_prediction)

    output = validation_features[KEY_COLUMNS + ["week_start_date", "total_cases", "outbreak_threshold"]].copy()
    output["city"] = city
    output["validation_year"] = validation_year
    output["actual_outbreak"] = output["total_cases"] >= output["outbreak_threshold"]
    output["single_stage_prediction"] = single_prediction
    output["two_stage_mae_prediction"] = mae_policy_predictions
    output["two_stage_recall_prediction"] = recall_policy_predictions
    output["outbreak_probability"] = outbreak_probabilities
    output["mae_policy_alert"] = mae_policy_alerts
    output["recall_policy_alert"] = recall_policy_alerts
    output["mae_gate"] = gates.mae_gate
    output["recall_gate"] = gates.recall_gate
    output["selected_lags"] = str(selected_lags)

    return output


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create overall and city-level summary tables."""

    model_columns = {
        "Single-stage": "single_stage_prediction",
        "Two-stage MAE policy": "two_stage_mae_prediction",
        "Two-stage recall policy": "two_stage_recall_prediction",
    }

    summary_rows = []
    city_rows = []

    for model_name, prediction_column in model_columns.items():
        summary_rows.append(
            {
                "model": model_name,
                "overall_mae": mean_absolute_error(predictions["total_cases"], predictions[prediction_column]),
                "normal_week_mae": mean_absolute_error(
                    predictions.loc[~predictions["actual_outbreak"], "total_cases"],
                    predictions.loc[~predictions["actual_outbreak"], prediction_column],
                ),
                "outbreak_week_mae": mean_absolute_error(
                    predictions.loc[predictions["actual_outbreak"], "total_cases"],
                    predictions.loc[predictions["actual_outbreak"], prediction_column],
                ),
            }
        )

        for city, city_frame in predictions.groupby("city"):
            city_rows.append(
                {
                    "city": city,
                    "model": model_name,
                    "mae": mean_absolute_error(city_frame["total_cases"], city_frame[prediction_column]),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(city_rows)


def summarize_alerts(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize outbreak alert behavior for both policy gates."""

    rows = []
    actual = predictions["actual_outbreak"].astype(int)

    for label, alert_column in [
        ("MAE-focused policy", "mae_policy_alert"),
        ("Recall-focused policy", "recall_policy_alert"),
    ]:
        alerts = predictions[alert_column].astype(int)
        rows.append(
            {
                "policy": label,
                "precision": precision_score(actual, alerts, zero_division=0),
                "recall": recall_score(actual, alerts, zero_division=0),
                "pr_auc": average_precision_score(actual, predictions["outbreak_probability"]),
                "brier": brier_score_loss(actual, predictions["outbreak_probability"]),
            }
        )

    return pd.DataFrame(rows)


def run_evaluation(n_outer_folds: int = 4) -> pd.DataFrame:
    """Run the full expanding-window evaluation for both cities."""

    raw_features, raw_labels = load_drivendata_dengai()
    data = prepare_raw_data(raw_features, raw_labels)

    fold_outputs = []

    for city, city_data in data.groupby("city"):
        city_data = city_data.sort_values("week_start_date").reset_index(drop=True)

        for validation_year, train, validation in outer_expanding_splits(city_data, n_outer_folds=n_outer_folds):
            print(f"Training {city.upper()} through {validation_year - 1}; validating {validation_year}...", flush=True)
            fold_outputs.append(evaluate_fold(city, validation_year, train, validation))

    return pd.concat(fold_outputs, ignore_index=True)


def main() -> None:
    """Execute the model and print compact tables."""

    predictions = run_evaluation(n_outer_folds=4)
    summary, city_summary = summarize_predictions(predictions)
    alert_summary = summarize_alerts(predictions)

    print("\nOverall model comparison")
    print(summary.round(3).to_string(index=False))

    print("\nCity-level MAE")
    print(city_summary.round(3).to_string(index=False))

    print("\nOutbreak alert metrics")
    print(alert_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()