"""Puerto Rico-wide dengue outbreak training, promotion, and scoring.

The historical DengAI/San Juan model remains useful as a benchmark, but its
target is not interchangeable with Puerto Rico-wide surveillance counts.  This
module trains a separate operational classifier using official island-wide
case labels and weather sampled across six parts of Puerto Rico.

The model registry follows a challenger/champion pattern:

* a newly trained candidate is evaluated with expanding, time-held-out years;
* the candidate is promoted automatically only after enough new finalized
  labels exist and its alert metrics stay inside configured guardrails; and
* weekly prediction jobs load only the current champion.

Recent surveillance values are allowed as prediction features, but the newest
four weeks are never used as training labels.  This separates fast provisional
signals from the slower, more stable truth needed for retraining.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    median_absolute_error,
    precision_score,
    recall_score,
)

from dengue_forecast_model import seasonal_threshold_table
from near_realtime_outbreak_detection import (
    CASE_FEATURE_COLUMNS,
    DAILY_WEATHER_COLUMNS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    WEATHER_LAGS,
    CityConfig,
    TrainedOutbreakDetector,
    _detector_from_payload,
    _detector_to_payload,
    add_expanding_outbreak_labels,
    add_historical_case_features,
    add_weather_features,
    aggregate_daily_to_weeks,
    case_forecast_weather_feature_columns,
    combine_daily_weather,
    fetch_json_with_cache,
    fit_time_aware_detector,
    live_case_features,
    parse_open_meteo_daily,
    weather_feature_columns,
)


MODEL_SCHEMA_VERSION = 5
CASE_FORECAST_SCHEMA_VERSION = 6
CASE_QUANTILES = (0.50, 0.80, 0.90)

# Exact-case models can use only values reported before the target week.  The
# hospitalization rate is calculated over four complete lagged weeks because a
# single-day or single-week rate would be unnecessarily noisy.
HOSPITAL_FEATURE_COLUMNS = [
    "hospitalized_lag_1",
    "hospitalized_lag_2",
    "hospitalized_mean_4",
    "hospitalization_rate_mean_4",
]
EXACT_HEALTH_FEATURE_COLUMNS = CASE_FEATURE_COLUMNS + HOSPITAL_FEATURE_COLUMNS

# The delayed model anchors every feature to the latest report that genuinely
# existed at forecast time.  ``report_age_weeks`` tells the model how far that
# anchor is from the week being predicted.
DELAYED_HEALTH_FEATURE_COLUMNS = [
    "delayed_cases_latest",
    "delayed_cases_lag_1",
    "delayed_cases_lag_2",
    "delayed_cases_lag_4",
    "delayed_cases_mean_4",
    "delayed_cases_mean_8",
    "delayed_cases_change_1",
    "delayed_hospitalized_latest",
    "delayed_hospitalized_mean_4",
    "delayed_hospitalization_rate_mean_4",
    "report_age_weeks",
]
PREDICTION_COLUMNS = [
    "geography",
    "geography_name",
    "generated_at_utc",
    "as_of_date",
    "week_start_date",
    "lead_weeks",
    "time_scope",
    "model_version",
    "model_variant",
    "predicted_cases_p50",
    "predicted_cases_p80",
    "predicted_cases_p90",
    "case_forecast_reliability",
    "case_risk_level",
    "p50_above_threshold",
    "p80_above_threshold",
    "p90_above_threshold",
    "held_out_case_mae",
    "held_out_normal_week_mae",
    "held_out_outbreak_week_mae",
    "outbreak_probability",
    "alert_gate",
    "outbreak_alert",
    "historical_outbreak_threshold_cases",
    "outbreak_definition",
    "held_out_precision",
    "held_out_recall",
    "held_out_f1",
    "held_out_pr_auc",
    "held_out_brier",
    "training_data_cutoff",
    "case_source_publication_date",
    "latest_case_week",
    "case_data_age_weeks",
    "case_report_anchor_week",
    "case_report_age_weeks",
    "weather_days_available",
    "forecast_input_days",
    "weather_sources",
    "actual_cases",
    "absolute_case_error",
    "actual_outbreak",
    "evaluated_at_utc",
]


@dataclass(frozen=True)
class PromotionDecision:
    """A reviewable guarded-promotion result."""

    promote: bool
    reasons: list[str]


@dataclass
class CaseForecastBundle:
    """Three fitted case-count quantile models with honest validation results.

    P50 is the median forecast used for MAE. P80 and P90 describe progressively
    more cautious upper outcomes. Keeping all three models in one bundle makes
    it impossible for live scoring to accidentally mix feature schemas.
    """

    models: dict[float, HistGradientBoostingRegressor]
    feature_columns: list[str]
    validation_metrics: dict[str, float | None]
    validation_predictions: list[dict[str, Any]]
    baseline_column: str | None = None
    residual_scales: dict[float, float] | None = None
    upper_offsets: dict[float, float] | None = None

    def predict(self, feature_row: pd.Series | dict[str, Any]) -> dict[str, float]:
        """Return nonnegative, monotonically ordered P50/P80/P90 case forecasts."""

        matrix = pd.DataFrame([dict(feature_row)]).reindex(columns=self.feature_columns)
        baseline = (
            float(matrix.iloc[0][self.baseline_column])
            if self.baseline_column is not None
            else 0.0
        )
        scales = self.residual_scales or {q: 1.0 for q in CASE_QUANTILES}
        raw = [
            max(
                0.0,
                baseline
                + float(scales.get(q, 1.0))
                * float(self.models[q].predict(matrix)[0]),
            )
            for q in CASE_QUANTILES
        ]
        offsets = self.upper_offsets or {}
        raw[1] = max(raw[1], raw[0] + float(offsets.get(0.80, 0.0)))
        raw[2] = max(raw[2], raw[0] + float(offsets.get(0.90, 0.0)))
        ordered = np.maximum.accumulate(raw)
        return {
            "p50": float(ordered[0]),
            "p80": float(ordered[1]),
            "p90": float(ordered[2]),
        }


def load_operations_config(path: Path) -> dict[str, Any]:
    """Read the versioned operational policy shared by local and CI runs."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_official_cases(path: Path, geography: str = "pr") -> pd.DataFrame:
    """Load one normalized geography and add model calendar fields."""

    if not path.exists():
        raise FileNotFoundError(
            f"Official case table does not exist: {path}. Run the ingest command first."
        )
    frame = pd.read_csv(path, parse_dates=["week_start_date"])
    frame = frame[frame["geography"] == geography].copy()
    if frame.empty:
        raise ValueError(f"Official case table has no rows for geography={geography}")
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"]).dt.normalize()
    frame["year"] = frame["week_start_date"].dt.isocalendar().year.astype(int)
    frame["weekofyear"] = frame["week_start_date"].dt.isocalendar().week.astype(int)
    frame["city"] = geography
    frame["total_cases"] = pd.to_numeric(frame["total_cases"], errors="raise")
    if "hospitalized_cases" not in frame:
        raise ValueError(
            "Official case table predates the hospitalization schema; run ingest again"
        )
    frame["hospitalized_cases"] = pd.to_numeric(
        frame["hospitalized_cases"], errors="raise"
    )
    return frame.sort_values("week_start_date").reset_index(drop=True)


def finalized_cases(cases: pd.DataFrame, as_of: date, stabilization_weeks: int) -> pd.DataFrame:
    """Return labels whose full week ended before the stabilization buffer."""

    cutoff_date = pd.Timestamp(as_of) - pd.Timedelta(weeks=stabilization_weeks)
    finalized = cases[cases["week_start_date"] + pd.Timedelta(days=6) <= cutoff_date].copy()
    if finalized.empty:
        raise ValueError("No official case weeks are old enough to be finalized")
    return finalized


def add_historical_health_features(cases: pd.DataFrame) -> pd.DataFrame:
    """Add exact past-case and hospitalization features for model training.

    Every feature is shifted by at least one week.  The target week's cases or
    hospitalizations therefore never leak into the row being predicted.
    """

    frame = add_historical_case_features(cases)
    hospitalized = frame["hospitalized_cases"].astype(float)
    cases_shifted = frame["total_cases"].astype(float).shift(1)
    hospitalized_shifted = hospitalized.shift(1)

    frame["hospitalized_lag_1"] = hospitalized.shift(1)
    frame["hospitalized_lag_2"] = hospitalized.shift(2)
    frame["hospitalized_mean_4"] = hospitalized_shifted.rolling(
        4, min_periods=4
    ).mean()
    case_sum_4 = cases_shifted.rolling(4, min_periods=4).sum()
    hospital_sum_4 = hospitalized_shifted.rolling(4, min_periods=4).sum()
    frame["hospitalization_rate_mean_4"] = hospital_sum_4 / case_sum_4.replace(
        0, np.nan
    )
    # Zero cases and zero hospitalizations describe a real zero-burden window.
    frame.loc[
        (case_sum_4 == 0) & (hospital_sum_4 == 0),
        "hospitalization_rate_mean_4",
    ] = 0.0
    return frame


def add_delayed_health_features(
    cases: pd.DataFrame,
    report_delays: range | list[int] | tuple[int, ...],
) -> pd.DataFrame:
    """Simulate what historical forecasts knew under several report delays.

    A target row is copied once for each configured delay.  For an eight-week
    delay, for example, ``delayed_cases_latest`` is the true count from t-8,
    never t-1.  Training across a small delay range makes one model usable as
    the live data age changes; headline validation later selects only the
    configured eight-week scenario so each target week is counted once.
    """

    base = cases.sort_values("week_start_date").copy()
    parts: list[pd.DataFrame] = []
    for raw_delay in report_delays:
        delay = int(raw_delay)
        if delay < 1:
            raise ValueError("Case reporting delay must be at least one week")

        frame = base.copy()
        latest_cases = base["total_cases"].astype(float).shift(delay)
        latest_hospitalized = base["hospitalized_cases"].astype(float).shift(delay)
        frame["delayed_cases_latest"] = latest_cases
        frame["delayed_cases_lag_1"] = base["total_cases"].astype(float).shift(
            delay + 1
        )
        frame["delayed_cases_lag_2"] = base["total_cases"].astype(float).shift(
            delay + 2
        )
        frame["delayed_cases_lag_4"] = base["total_cases"].astype(float).shift(
            delay + 4
        )
        frame["delayed_cases_mean_4"] = latest_cases.rolling(
            4, min_periods=4
        ).mean()
        frame["delayed_cases_mean_8"] = latest_cases.rolling(
            8, min_periods=8
        ).mean()
        frame["delayed_cases_change_1"] = (
            frame["delayed_cases_latest"] - frame["delayed_cases_lag_1"]
        )
        frame["delayed_hospitalized_latest"] = latest_hospitalized
        frame["delayed_hospitalized_mean_4"] = latest_hospitalized.rolling(
            4, min_periods=4
        ).mean()
        delayed_cases_sum = latest_cases.rolling(4, min_periods=4).sum()
        delayed_hospital_sum = latest_hospitalized.rolling(4, min_periods=4).sum()
        frame["delayed_hospitalization_rate_mean_4"] = (
            delayed_hospital_sum / delayed_cases_sum.replace(0, np.nan)
        )
        frame.loc[
            (delayed_cases_sum == 0) & (delayed_hospital_sum == 0),
            "delayed_hospitalization_rate_mean_4",
        ] = 0.0
        frame["report_age_weeks"] = float(delay)
        parts.append(frame)

    return pd.concat(parts, ignore_index=True).sort_values(
        ["week_start_date", "report_age_weeks"]
    )


def live_exact_health_features(
    recent: pd.DataFrame, target_week_start: pd.Timestamp
) -> dict[str, float] | None:
    """Return exact case plus hospital lags only when all source weeks exist."""

    case_features = live_case_features(recent, target_week_start)
    if case_features is None or "hospitalized_cases" not in recent:
        return None

    indexed = recent.set_index("week_start_date")
    required = [
        target_week_start - pd.Timedelta(weeks=lag) for lag in range(1, 5)
    ]
    if any(value not in indexed.index for value in required):
        return None
    hospitals = indexed.loc[required, "hospitalized_cases"].astype(float)
    cases = indexed.loc[required, "total_cases"].astype(float)
    if hospitals.isna().any() or cases.isna().any():
        return None
    case_total = float(cases.sum())
    hospital_total = float(hospitals.sum())
    hospitalization_rate = hospital_total / case_total if case_total else 0.0
    return {
        **case_features,
        "hospitalized_lag_1": float(hospitals.iloc[0]),
        "hospitalized_lag_2": float(hospitals.iloc[1]),
        "hospitalized_mean_4": float(hospitals.mean()),
        "hospitalization_rate_mean_4": hospitalization_rate,
    }


def live_delayed_health_features(
    recent: pd.DataFrame,
    target_week_start: pd.Timestamp,
    minimum_age_weeks: int,
    maximum_age_weeks: int,
) -> tuple[dict[str, float], pd.Timestamp] | None:
    """Anchor delayed features to the newest genuinely available report.

    The function requires eight contiguous weeks ending at the report anchor.
    It does not fill missing surveillance weeks or pretend that an old value is
    a t-1 lag.  The returned anchor date is written beside every live forecast.
    """

    if recent.empty or "hospitalized_cases" not in recent:
        return None
    available = recent[
        pd.to_datetime(recent["week_start_date"]) < target_week_start
    ].copy()
    if available.empty:
        return None
    anchor = pd.Timestamp(available["week_start_date"].max()).normalize()
    age = (target_week_start - anchor).days / 7
    if age != int(age) or not minimum_age_weeks <= age <= maximum_age_weeks:
        return None

    indexed = available.set_index("week_start_date")
    required = [anchor - pd.Timedelta(weeks=lag) for lag in range(0, 8)]
    if any(value not in indexed.index for value in required):
        return None
    history = indexed.loc[required, ["total_cases", "hospitalized_cases"]].astype(
        float
    )
    if history.isna().any().any():
        return None

    latest_cases = float(history.iloc[0]["total_cases"])
    latest_hospitalized = float(history.iloc[0]["hospitalized_cases"])
    case_total_4 = float(history.iloc[:4]["total_cases"].sum())
    hospital_total_4 = float(history.iloc[:4]["hospitalized_cases"].sum())
    features = {
        "delayed_cases_latest": latest_cases,
        "delayed_cases_lag_1": float(history.iloc[1]["total_cases"]),
        "delayed_cases_lag_2": float(history.iloc[2]["total_cases"]),
        "delayed_cases_lag_4": float(history.iloc[4]["total_cases"]),
        "delayed_cases_mean_4": float(history.iloc[:4]["total_cases"].mean()),
        "delayed_cases_mean_8": float(history["total_cases"].mean()),
        "delayed_cases_change_1": latest_cases
        - float(history.iloc[1]["total_cases"]),
        "delayed_hospitalized_latest": latest_hospitalized,
        "delayed_hospitalized_mean_4": float(
            history.iloc[:4]["hospitalized_cases"].mean()
        ),
        "delayed_hospitalization_rate_mean_4": (
            hospital_total_4 / case_total_4 if case_total_4 else 0.0
        ),
        "report_age_weeks": float(age),
    }
    return features, anchor


def _point_config(point: dict[str, Any], timezone_name: str) -> CityConfig:
    """Convert a JSON weather point into the existing typed API configuration."""

    return CityConfig(
        code=str(point["code"]),
        name=str(point["name"]),
        latitude=float(point["latitude"]),
        longitude=float(point["longitude"]),
        timezone_name=timezone_name,
    )


def _weather_parameters(
    point: CityConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    """Build source-consistent Open-Meteo archive or forecast parameters."""

    parameters: dict[str, Any] = {
        "latitude": point.latitude,
        "longitude": point.longitude,
        "daily": ",".join(DAILY_WEATHER_COLUMNS),
        "timezone": point.timezone_name,
    }
    if start_date is None or end_date is None:
        parameters.update({"past_days": 92, "forecast_days": 16})
    else:
        parameters.update(
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
        )
    return parameters


def spatially_aggregate_daily_weather(
    point_frames: list[pd.DataFrame], expected_points: int
) -> pd.DataFrame:
    """Create one island-wide daily weather row from all configured points.

    Island mean temperature, humidity, rainfall, and evapotranspiration are
    spatial averages. Maximum and minimum temperature preserve the hottest and
    coolest sampled values. A date is usable only when every configured point
    supplies every weather field, preventing silent geography changes.
    """

    if not point_frames:
        raise ValueError("At least one Puerto Rico weather point is required")
    combined = pd.concat(point_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    if "weather_point" not in combined:
        raise ValueError("Island weather rows must identify their weather_point")
    complete = combined[DAILY_WEATHER_COLUMNS].notna().all(axis=1)
    available = combined.loc[complete].groupby("date")["weather_point"].nunique()

    grouped = combined.groupby("date", as_index=False).agg(
        temperature_2m_mean=("temperature_2m_mean", "mean"),
        temperature_2m_max=("temperature_2m_max", "max"),
        temperature_2m_min=("temperature_2m_min", "min"),
        precipitation_sum=("precipitation_sum", "mean"),
        rain_sum=("rain_sum", "mean"),
        precipitation_hours=("precipitation_hours", "mean"),
        relative_humidity_2m_mean=("relative_humidity_2m_mean", "mean"),
        dew_point_2m_mean=("dew_point_2m_mean", "mean"),
        soil_moisture_0_to_7cm_mean=("soil_moisture_0_to_7cm_mean", "mean"),
        et0_fao_evapotranspiration=("et0_fao_evapotranspiration", "mean"),
    )
    grouped["point_count"] = grouped["date"].map(available).fillna(0).astype(int)
    incomplete = grouped["point_count"] != expected_points
    grouped.loc[incomplete, DAILY_WEATHER_COLUMNS] = np.nan
    sources = sorted(combined["weather_source"].dropna().unique())
    grouped["weather_source"] = "+".join(sources) if sources else "missing"
    return grouped.sort_values("date").reset_index(drop=True)


def fetch_island_training_weather(
    cases: pd.DataFrame,
    config: dict[str, Any],
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, str]:
    """Fetch historical weather at every island point and aggregate by week."""

    settings = config["puerto_rico"]
    start = pd.Timestamp(cases["week_start_date"].min()).date()
    end = (pd.Timestamp(cases["week_start_date"].max()) + pd.Timedelta(days=6)).date()
    frames: list[pd.DataFrame] = []
    states: set[str] = set()
    for raw_point in settings["weather_points"]:
        point = _point_config(raw_point, settings["timezone"])
        response = fetch_json_with_cache(
            OPEN_METEO_ARCHIVE_URL,
            _weather_parameters(point, start, end),
            cache_dir,
            cache_prefix=f"pr_{point.code}_training_weather",
            max_cache_age_hours=None,
            refresh=refresh,
        )
        frame = parse_open_meteo_daily(
            response.payload, source=f"open_meteo_archive:{point.code}"
        )
        frame["weather_point"] = point.code
        frames.append(frame)
        states.add(response.cache_state)

    daily = spatially_aggregate_daily_weather(frames, len(settings["weather_points"]))
    weekly = aggregate_daily_to_weeks(daily, cases["week_start_date"].tolist())
    return weekly, "+".join(sorted(states))


def fetch_island_live_weather(
    target_starts: list[pd.Timestamp],
    as_of: date,
    config: dict[str, Any],
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, str]:
    """Build recent plus forecast island-wide weather for live scoring."""

    settings = config["puerto_rico"]
    earliest = min(target_starts) - pd.Timedelta(weeks=11)
    # Archive data can lag several days.  The forecast endpoint's 92-day past
    # window fills that recent overlap and is preferred when both sources exist.
    archive_end = min(as_of - timedelta(days=5), max(target_starts).date() + timedelta(days=6))
    point_frames: list[pd.DataFrame] = []
    states: set[str] = set()

    for raw_point in settings["weather_points"]:
        point = _point_config(raw_point, settings["timezone"])
        archive_response = fetch_json_with_cache(
            OPEN_METEO_ARCHIVE_URL,
            _weather_parameters(point, earliest.date(), archive_end),
            cache_dir,
            cache_prefix=f"pr_{point.code}_recent_archive",
            max_cache_age_hours=24,
            refresh=refresh,
        )
        live_response = fetch_json_with_cache(
            OPEN_METEO_FORECAST_URL,
            _weather_parameters(point),
            cache_dir,
            cache_prefix=f"pr_{point.code}_live_forecast",
            max_cache_age_hours=6,
            refresh=refresh,
        )
        archive = parse_open_meteo_daily(
            archive_response.payload, source=f"open_meteo_archive:{point.code}"
        )
        live = parse_open_meteo_daily(
            live_response.payload, source=f"open_meteo_live:{point.code}"
        )
        archive["weather_point"] = point.code
        live["weather_point"] = point.code
        combined = combine_daily_weather(archive, live)
        combined["weather_source"] = combined["weather_source"].astype(str)
        point_frames.append(combined)
        states.update({archive_response.cache_state, live_response.cache_state})

    daily = spatially_aggregate_daily_weather(
        point_frames, len(settings["weather_points"])
    )
    all_starts = pd.date_range(earliest, max(target_starts), freq="7D").tolist()
    weekly = aggregate_daily_to_weeks(daily, all_starts, as_of_date=as_of)
    return weekly, "+".join(sorted(states))


def _case_forecaster_to_payload(bundle: CaseForecastBundle) -> dict[str, Any]:
    """Serialize a trusted case-forecast bundle into stable built-in containers."""

    return {
        "models": bundle.models,
        "feature_columns": bundle.feature_columns,
        "validation_metrics": bundle.validation_metrics,
        "validation_predictions": bundle.validation_predictions,
        "baseline_column": bundle.baseline_column,
        "residual_scales": bundle.residual_scales,
        "upper_offsets": bundle.upper_offsets,
    }


def _case_forecaster_from_payload(payload: dict[str, Any]) -> CaseForecastBundle:
    """Reconstruct a case forecaster from the repository-controlled champion."""

    return CaseForecastBundle(
        models={float(key): value for key, value in payload["models"].items()},
        feature_columns=list(payload["feature_columns"]),
        validation_metrics=dict(payload["validation_metrics"]),
        validation_predictions=list(payload.get("validation_predictions", [])),
        baseline_column=payload.get("baseline_column"),
        residual_scales={
            float(key): float(value)
            for key, value in (payload.get("residual_scales") or {}).items()
        }
        or None,
        upper_offsets={
            float(key): float(value)
            for key, value in (payload.get("upper_offsets") or {}).items()
        }
        or None,
    )


def _fit_quantile_regressor(quantile: float) -> HistGradientBoostingRegressor:
    """Create a conservative boosted-tree case model for one forecast quantile."""

    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=90,
        max_leaf_nodes=16,
        l2_regularization=0.05,
        random_state=42,
    )


def _case_forecast_metrics(
    predictions: pd.DataFrame, persistence_column: str = "cases_lag_1"
) -> dict[str, float | None]:
    """Summarize point error, upper-quantile coverage, and threshold usefulness."""

    actual = predictions["actual_cases"].astype(float)
    p50 = predictions["predicted_cases_p50"].astype(float)
    p80 = predictions["predicted_cases_p80"].astype(float)
    p90 = predictions["predicted_cases_p90"].astype(float)
    outbreak = predictions["actual_outbreak"].astype(int)
    threshold_signal = p50 >= predictions["outbreak_threshold"].astype(float)
    normal = outbreak == 0
    elevated = outbreak == 1

    return {
        "held_out_rows": float(len(predictions)),
        "mae": float(mean_absolute_error(actual, p50)),
        "median_absolute_error": float(median_absolute_error(actual, p50)),
        "normal_week_mae": (
            float(mean_absolute_error(actual[normal], p50[normal])) if normal.any() else None
        ),
        "outbreak_week_mae": (
            float(mean_absolute_error(actual[elevated], p50[elevated]))
            if elevated.any()
            else None
        ),
        "p80_pinball_loss": float(mean_pinball_loss(actual, p80, alpha=0.80)),
        "p90_pinball_loss": float(mean_pinball_loss(actual, p90, alpha=0.90)),
        "p80_coverage": float((actual <= p80).mean()),
        "p90_coverage": float((actual <= p90).mean()),
        "p50_threshold_precision": float(
            precision_score(outbreak, threshold_signal, zero_division=0)
        ),
        "p50_threshold_recall": float(
            recall_score(outbreak, threshold_signal, zero_division=0)
        ),
        "persistence_mae": float(
            mean_absolute_error(actual, predictions[persistence_column].astype(float))
        ),
    }


def _calibrate_residual_scales(
    available_training: pd.DataFrame,
    feature_columns: list[str],
    baseline_column: str,
    validation_filter: tuple[str, float] | None,
) -> dict[float, float]:
    """Choose conservative residual sizes on a later, internal time split.

    A residual model predicts how far the target may move from the latest known
    case count.  Older reports invite larger but less certain corrections.  To
    prevent those corrections from making MAE worse than simply carrying the
    report forward, this helper fits on earlier years and chooses one multiplier
    on the newest available training year.  The outer validation year remains
    untouched, so the reported score stays honest.
    """

    years = sorted(available_training["year"].astype(int).unique())
    if len(years) < 3:
        return {quantile: 1.0 for quantile in CASE_QUANTILES}
    calibration_year = years[-1]
    inner_train = available_training[
        available_training["year"] < calibration_year
    ].copy()
    calibration = available_training[
        available_training["year"] == calibration_year
    ].copy()
    if validation_filter is not None:
        column, value = validation_filter
        calibration = calibration[calibration[column] == value].copy()
    if len(inner_train) < 100 or calibration.empty:
        return {quantile: 1.0 for quantile in CASE_QUANTILES}

    target = inner_train["total_cases"].astype(float) - inner_train[
        baseline_column
    ].astype(float)
    calibration_actual = calibration["total_cases"].astype(float).to_numpy()
    calibration_baseline = calibration[baseline_column].astype(float).to_numpy()
    candidates = np.linspace(0.0, 1.5, 31)
    scales: dict[float, float] = {}

    for quantile in CASE_QUANTILES:
        model = _fit_quantile_regressor(quantile)
        model.fit(inner_train[feature_columns], target)
        residual = model.predict(calibration[feature_columns])
        scores: list[tuple[float, float]] = []
        for scale in candidates:
            prediction = np.maximum(0.0, calibration_baseline + scale * residual)
            if quantile == 0.50:
                loss = mean_absolute_error(calibration_actual, prediction)
            else:
                loss = mean_pinball_loss(
                    calibration_actual, prediction, alpha=quantile
                )
            scores.append((float(loss), float(scale)))
        # Prefer the smaller correction if two scales have indistinguishable
        # loss; it is safer when the next outbreak differs from the last one.
        scales[quantile] = min(scores, key=lambda item: (item[0], item[1]))[1]
    return scales


def _calibrate_upper_offsets(
    available_training: pd.DataFrame,
    feature_columns: list[str],
    baseline_column: str | None,
    validation_filter: tuple[str, float] | None,
    residual_scales: dict[float, float],
) -> dict[float, float]:
    """Estimate time-safe P80/P90 cushions above the median forecast.

    Independent quantile trees can collapse onto P50 when a recent calibration
    year trends downward. The cushion pools expanding predictions from up to
    three later years *inside* the available training period, never the outer
    validation year, and preserves a useful range without leaking outcomes.
    """

    years = sorted(available_training["year"].astype(int).unique())
    if len(years) < 3:
        return {0.80: 0.0, 0.90: 0.0}
    calibration_years = years[-min(3, len(years) - 2) :]
    errors: list[np.ndarray] = []
    for calibration_year in calibration_years:
        inner_train = available_training[
            available_training["year"] < calibration_year
        ].copy()
        calibration = available_training[
            available_training["year"] == calibration_year
        ].copy()
        if validation_filter is not None:
            column, value = validation_filter
            calibration = calibration[calibration[column] == value].copy()
        if len(inner_train) < 100 or calibration.empty:
            continue

        train_target = inner_train["total_cases"].astype(float)
        if baseline_column is not None:
            train_target = train_target - inner_train[baseline_column].astype(float)
            calibration_baseline = calibration[baseline_column].astype(float).to_numpy()
        else:
            calibration_baseline = np.zeros(len(calibration), dtype=float)
        median_model = _fit_quantile_regressor(0.50)
        median_model.fit(inner_train[feature_columns], train_target)
        median_prediction = np.maximum(
            0.0,
            calibration_baseline
            + residual_scales.get(0.50, 1.0)
            * median_model.predict(calibration[feature_columns]),
        )
        errors.append(
            calibration["total_cases"].astype(float).to_numpy() - median_prediction
        )
    if not errors:
        return {0.80: 0.0, 0.90: 0.0}
    upper_error = np.concatenate(errors)
    return {
        quantile: max(0.0, float(np.quantile(upper_error, quantile)))
        for quantile in (0.80, 0.90)
    }


def fit_time_aware_case_forecaster(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    variant: str,
    baseline_column: str | None = None,
    n_validation_years: int = 4,
    validation_filter: tuple[str, float] | None = None,
    fallback_to_persistence: bool = False,
) -> CaseForecastBundle:
    """Fit P50/P80/P90 case models with expanding, later-year validation.

    The validation years match the outbreak classifier's eligible labeled
    years. The case-aware variant is a conditional evaluation: it uses actual
    earlier case weeks only because live routing selects this variant solely
    when those exact reports exist. When they do not exist, live scoring uses
    the independently validated delayed-case or weather-only forecaster.
    """

    frame = training_frame.sort_values("week_start_date").copy()
    frame = frame.dropna(subset=feature_columns + ["total_cases"])
    labeled_years = sorted(
        frame.loc[frame["outbreak_label"].notna(), "year"].astype(int).unique()
    )
    if len(labeled_years) < 4:
        raise ValueError("At least four labeled years are required for case validation")
    validation_years = labeled_years[-min(n_validation_years, len(labeled_years) - 2) :]
    held_out_parts: list[pd.DataFrame] = []

    for validation_year in validation_years:
        train = frame[frame["year"] < validation_year]
        validation = frame[
            (frame["year"] == validation_year) & frame["outbreak_label"].notna()
        ].copy()
        if validation_filter is not None:
            column, value = validation_filter
            validation = validation[validation[column] == value].copy()
        if len(train) < 100 or validation.empty:
            continue

        fold_predictions: list[np.ndarray] = []
        train_target = train["total_cases"].astype(float)
        fold_scales = {quantile: 1.0 for quantile in CASE_QUANTILES}
        if baseline_column is not None:
            train_target = train_target - train[baseline_column].astype(float)
            fold_scales = _calibrate_residual_scales(
                train,
                feature_columns,
                baseline_column,
                validation_filter,
            )
        fold_upper_offsets = _calibrate_upper_offsets(
            train,
            feature_columns,
            baseline_column,
            validation_filter,
            fold_scales,
        )

        for quantile in CASE_QUANTILES:
            model = _fit_quantile_regressor(quantile)
            model.fit(train[feature_columns], train_target)
            validation_baseline = (
                validation[baseline_column].astype(float).to_numpy()
                if baseline_column is not None
                else 0.0
            )
            fold_predictions.append(
                np.maximum(
                    0.0,
                    validation_baseline
                    + fold_scales[quantile]
                    * model.predict(validation[feature_columns]),
                )
            )

        # Quantile models are fitted independently and can occasionally cross.
        # Sorting each row preserves the intended P50 <= P80 <= P90 contract.
        ordered = np.maximum.accumulate(np.column_stack(fold_predictions), axis=1)
        ordered[:, 1] = np.maximum(
            ordered[:, 1], ordered[:, 0] + fold_upper_offsets[0.80]
        )
        ordered[:, 2] = np.maximum(
            ordered[:, 2], ordered[:, 0] + fold_upper_offsets[0.90]
        )
        validation["predicted_cases_p50"] = ordered[:, 0]
        validation["predicted_cases_p80"] = ordered[:, 1]
        validation["predicted_cases_p90"] = ordered[:, 2]
        validation["actual_cases"] = validation["total_cases"].astype(float)
        validation["actual_outbreak"] = validation["outbreak_label"].astype(int)
        validation["outbreak_threshold"] = validation["outbreak_threshold"].astype(float)
        validation["forecast_variant"] = variant
        validation["_upper_offset_p80"] = fold_upper_offsets[0.80]
        validation["_upper_offset_p90"] = fold_upper_offsets[0.90]
        held_out_parts.append(validation)

    if not held_out_parts:
        raise ValueError("Unable to create time-held-out case-count predictions")
    held_out = pd.concat(held_out_parts, ignore_index=True).sort_values("week_start_date")
    persistence_column = baseline_column or "cases_lag_1"
    used_persistence_fallback = False
    if fallback_to_persistence and baseline_column is not None:
        model_mae = mean_absolute_error(
            held_out["actual_cases"], held_out["predicted_cases_p50"]
        )
        persistence_mae = mean_absolute_error(
            held_out["actual_cases"], held_out[persistence_column]
        )
        if model_mae > persistence_mae:
            # P50 is the MAE-focused decision.  If the time-held-out residual
            # correction cannot beat the available report itself, keep that
            # report as P50 instead of shipping a demonstrably harmful change.
            held_out["predicted_cases_p50"] = held_out[persistence_column]
            held_out["predicted_cases_p80"] = np.maximum(
                held_out["predicted_cases_p80"],
                held_out["predicted_cases_p50"] + held_out["_upper_offset_p80"],
            )
            held_out["predicted_cases_p90"] = np.maximum(
                held_out["predicted_cases_p90"],
                held_out["predicted_cases_p50"] + held_out["_upper_offset_p90"],
            )
            used_persistence_fallback = True
    metrics = _case_forecast_metrics(held_out, persistence_column)
    metrics["p50_used_persistence_fallback"] = used_persistence_fallback

    final_models: dict[float, HistGradientBoostingRegressor] = {}
    final_scales = {quantile: 1.0 for quantile in CASE_QUANTILES}
    if baseline_column is not None:
        final_scales = _calibrate_residual_scales(
            frame,
            feature_columns,
            baseline_column,
            validation_filter,
        )
        if used_persistence_fallback:
            final_scales[0.50] = 0.0
    final_upper_offsets = _calibrate_upper_offsets(
        frame,
        feature_columns,
        baseline_column,
        validation_filter,
        final_scales,
    )
    for quantile in CASE_QUANTILES:
        model = _fit_quantile_regressor(quantile)
        final_target = frame["total_cases"].astype(float)
        if baseline_column is not None:
            final_target = final_target - frame[baseline_column].astype(float)
        model.fit(frame[feature_columns], final_target)
        final_models[quantile] = model

    prediction_columns = [
        "week_start_date",
        "year",
        "weekofyear",
        "forecast_variant",
        "actual_cases",
        "outbreak_threshold",
        "actual_outbreak",
        "predicted_cases_p50",
        "predicted_cases_p80",
        "predicted_cases_p90",
    ]
    validation_records = held_out[prediction_columns].copy()
    validation_records["week_start_date"] = (
        validation_records["week_start_date"].dt.date.astype(str)
    )
    return CaseForecastBundle(
        models=final_models,
        feature_columns=feature_columns,
        validation_metrics=metrics,
        validation_predictions=validation_records.to_dict(orient="records"),
        baseline_column=baseline_column,
        residual_scales=final_scales,
        upper_offsets=final_upper_offsets,
    )


def train_candidate(
    finalized: pd.DataFrame,
    config: dict[str, Any],
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train outbreak classifiers and case-forecast variants together."""

    weekly_weather, weather_cache_state = fetch_island_training_weather(
        finalized, config, cache_dir, refresh
    )
    weekly_weather["weekofyear"] = (
        weekly_weather["week_start_date"].dt.isocalendar().week.astype(int)
    )
    weather_features = add_weather_features(weekly_weather)
    labeled = add_expanding_outbreak_labels(finalized)
    health_with_lags = add_historical_health_features(finalized)
    delay_policy = config["delayed_case_model"]
    minimum_delay = int(delay_policy["minimum_report_age_weeks"])
    maximum_delay = int(delay_policy["maximum_report_age_weeks"])
    validation_delay = int(delay_policy["validation_report_age_weeks"])
    if not minimum_delay <= validation_delay <= maximum_delay:
        raise ValueError("Delayed-case validation age must be inside the training range")
    delayed_health = add_delayed_health_features(
        finalized, range(minimum_delay, maximum_delay + 1)
    )

    classifier_training = labeled.merge(
        weather_features[["week_start_date"] + weather_feature_columns()],
        on="week_start_date",
        how="inner",
        validate="one_to_one",
    ).merge(
        health_with_lags[["week_start_date"] + EXACT_HEALTH_FEATURE_COLUMNS],
        on="week_start_date",
        how="left",
        validate="one_to_one",
    )

    weather_columns = weather_feature_columns()
    case_forecast_weather_columns = case_forecast_weather_feature_columns()
    classifier_training = classifier_training.dropna(
        subset=weather_columns + ["outbreak_label"]
    )
    weather_detector = fit_time_aware_detector(
        classifier_training,
        weather_columns,
        target_recall=float(config["target_recall"]),
    )
    case_training = classifier_training.dropna(subset=EXACT_HEALTH_FEATURE_COLUMNS)
    case_detector = fit_time_aware_detector(
        case_training,
        weather_columns + EXACT_HEALTH_FEATURE_COLUMNS,
        target_recall=float(config["target_recall"]),
    )
    delayed_classifier_training = labeled.merge(
        weather_features[["week_start_date"] + weather_columns],
        on="week_start_date",
        how="inner",
        validate="one_to_one",
    ).merge(
        delayed_health[["week_start_date"] + DELAYED_HEALTH_FEATURE_COLUMNS],
        on="week_start_date",
        how="left",
        validate="one_to_many",
    ).dropna(subset=weather_columns + DELAYED_HEALTH_FEATURE_COLUMNS + ["outbreak_label"])
    delayed_detector = fit_time_aware_detector(
        delayed_classifier_training,
        weather_columns + DELAYED_HEALTH_FEATURE_COLUMNS,
        target_recall=float(config["target_recall"]),
        validation_filter=("report_age_weeks", float(validation_delay)),
    )

    # Case-count forecasting can learn from the early baseline years even
    # though those years cannot yet receive outbreak labels. Outbreak labels
    # and thresholds are attached only where enough prior seasonal history
    # exists, and they are used solely to break validation MAE into normal and
    # outbreak weeks.
    regression_base = finalized.merge(
        weather_features[["week_start_date"] + case_forecast_weather_columns],
        on="week_start_date",
        how="inner",
        validate="one_to_one",
    ).merge(
        labeled[
            ["week_start_date", "outbreak_threshold", "outbreak_label"]
        ],
        on="week_start_date",
        how="left",
        validate="one_to_one",
    )
    regression_training = regression_base.merge(
        health_with_lags[["week_start_date"] + EXACT_HEALTH_FEATURE_COLUMNS],
        on="week_start_date",
        how="left",
        validate="one_to_one",
    )
    delayed_regression_training = regression_base.merge(
        delayed_health[["week_start_date"] + DELAYED_HEALTH_FEATURE_COLUMNS],
        on="week_start_date",
        how="left",
        validate="one_to_many",
    )
    weather_case_forecaster = fit_time_aware_case_forecaster(
        regression_training,
        case_forecast_weather_columns,
        variant="weather_only",
    )
    recent_case_forecaster = fit_time_aware_case_forecaster(
        regression_training.dropna(subset=EXACT_HEALTH_FEATURE_COLUMNS),
        case_forecast_weather_columns + EXACT_HEALTH_FEATURE_COLUMNS,
        variant="weather_plus_recent_cases",
        baseline_column="cases_lag_1",
        fallback_to_persistence=True,
    )
    delayed_case_forecaster = fit_time_aware_case_forecaster(
        delayed_regression_training.dropna(subset=DELAYED_HEALTH_FEATURE_COLUMNS),
        case_forecast_weather_columns + DELAYED_HEALTH_FEATURE_COLUMNS,
        variant="weather_plus_delayed_cases",
        baseline_column="delayed_cases_latest",
        validation_filter=("report_age_weeks", float(validation_delay)),
        fallback_to_persistence=True,
    )

    cutoff = pd.Timestamp(finalized["week_start_date"].max()).date().isoformat()
    fingerprint = {
        "schema": MODEL_SCHEMA_VERSION,
        "case_forecast_schema": CASE_FORECAST_SCHEMA_VERSION,
        "cutoff": cutoff,
        "rows": len(finalized),
        "case_total": int(finalized["total_cases"].sum()),
        "weather_points": config["puerto_rico"]["weather_points"],
        "target_recall": config["target_recall"],
        "delayed_case_model": delay_policy,
        "classifier_weather_features": weather_columns,
        "case_forecast_weather_features": case_forecast_weather_columns,
    }
    version = "pr-" + hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    source_publication = str(finalized["source_publication_date"].max())
    metadata = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "case_forecast_schema_version": CASE_FORECAST_SCHEMA_VERSION,
        "model_version": version,
        "geography": "pr",
        "geography_name": "Puerto Rico",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_cutoff": cutoff,
        "training_rows": int(len(finalized)),
        "case_source_publication_date": source_publication,
        "training_weather_cache_state": weather_cache_state,
        "outbreak_definition": "seasonal_prior_years_q75",
        "weather_only_metrics": weather_detector.validation_metrics,
        "weather_plus_recent_cases_metrics": case_detector.validation_metrics,
        "weather_plus_delayed_cases_metrics": delayed_detector.validation_metrics,
        "weather_only_case_forecast_metrics": (
            weather_case_forecaster.validation_metrics
        ),
        "weather_plus_recent_cases_case_forecast_metrics": (
            recent_case_forecaster.validation_metrics
        ),
        "weather_plus_delayed_cases_case_forecast_metrics": (
            delayed_case_forecaster.validation_metrics
        ),
        "weather_only_alert_gate": weather_detector.alert_gate,
        "weather_plus_recent_cases_alert_gate": case_detector.alert_gate,
        "weather_plus_delayed_cases_alert_gate": delayed_detector.alert_gate,
        "delayed_case_training_age_weeks": [minimum_delay, maximum_delay],
        "delayed_case_validation_age_weeks": validation_delay,
        "feature_manifest": {
            "weather": weather_columns,
            "weather_case_forecast": case_forecast_weather_columns,
            "exact_health": EXACT_HEALTH_FEATURE_COLUMNS,
            "delayed_health": DELAYED_HEALTH_FEATURE_COLUMNS,
        },
        "case_forecast_residual_scales": {
            "weather_plus_recent_cases": recent_case_forecaster.residual_scales,
            "weather_plus_delayed_cases": delayed_case_forecaster.residual_scales,
        },
        "case_forecast_upper_offsets": {
            "weather_only": weather_case_forecaster.upper_offsets,
            "weather_plus_recent_cases": recent_case_forecaster.upper_offsets,
            "weather_plus_delayed_cases": delayed_case_forecaster.upper_offsets,
        },
    }
    thresholds = seasonal_threshold_table(finalized)[
        ["weekofyear", "outbreak_threshold"]
    ].to_dict(orient="records")
    artifact = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "case_forecast_schema_version": CASE_FORECAST_SCHEMA_VERSION,
        "metadata": metadata,
        "weather_detector": _detector_to_payload(weather_detector),
        "case_detector": _detector_to_payload(case_detector),
        "delayed_case_detector": _detector_to_payload(delayed_detector),
        "weather_case_forecaster": _case_forecaster_to_payload(
            weather_case_forecaster
        ),
        "recent_case_forecaster": _case_forecaster_to_payload(
            recent_case_forecaster
        ),
        "delayed_case_forecaster": _case_forecaster_to_payload(
            delayed_case_forecaster
        ),
        "seasonal_thresholds": thresholds,
    }
    return artifact, metadata


def evaluate_promotion(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    config: dict[str, Any],
) -> PromotionDecision:
    """Apply minimum-label and non-regression guardrails to a candidate."""

    if champion is None:
        return PromotionDecision(True, ["bootstrap: no champion exists"])

    candidate_cutoff = pd.Timestamp(candidate["training_data_cutoff"])
    champion_cutoff = pd.Timestamp(champion["training_data_cutoff"])
    new_weeks = int((candidate_cutoff - champion_cutoff).days // 7)
    minimum = int(config["minimum_new_finalized_weeks"])
    schema_upgrade = (
        candidate.get("model_schema_version") != champion.get("model_schema_version")
        or candidate.get("case_forecast_schema_version")
        != champion.get("case_forecast_schema_version")
    )
    reasons = [
        f"new finalized weeks: {new_weeks} (minimum {minimum})",
        (
            "schema upgrade: label-count wait bypassed for one architecture migration"
            if schema_upgrade
            else "schema unchanged"
        ),
    ]
    passed = schema_upgrade or new_weeks >= minimum

    tolerances = config["promotion_guardrails"]
    candidate_metrics = candidate["weather_only_metrics"]
    champion_metrics = champion["weather_only_metrics"]
    rules = {
        "precision": candidate_metrics["precision"]
        >= champion_metrics["precision"] - float(tolerances["precision_tolerance"]),
        "recall": candidate_metrics["recall"]
        >= champion_metrics["recall"] - float(tolerances["recall_tolerance"]),
        "pr_auc": candidate_metrics["pr_auc"]
        >= champion_metrics["pr_auc"] - float(tolerances["pr_auc_tolerance"]),
        "brier": candidate_metrics["brier"]
        <= champion_metrics["brier"] + float(tolerances["brier_tolerance"]),
    }
    for metric, rule_passed in rules.items():
        reasons.append(
            f"{metric} guardrail: {'pass' if rule_passed else 'fail'} "
            f"(candidate={candidate_metrics[metric]:.4f}, "
            f"champion={champion_metrics[metric]:.4f})"
        )
        passed = passed and rule_passed

    # Case-aware alert routes are also guarded once both the candidate and the
    # champion contain their metrics.  A schema migration establishes the
    # first baseline for a newly introduced route.
    for variant in ("weather_plus_recent_cases", "weather_plus_delayed_cases"):
        candidate_variant = candidate.get(f"{variant}_metrics")
        champion_variant = champion.get(f"{variant}_metrics")
        if candidate_variant is None and champion_variant is None:
            continue
        if candidate_variant is None:
            reasons.append(f"{variant} classifier guardrails: fail (metrics missing)")
            passed = False
            continue
        if champion_variant is None and schema_upgrade:
            reasons.append(f"{variant} classifier guardrails: baseline established")
            continue
        if champion_variant is None:
            reasons.append(f"{variant} classifier guardrails: fail (champion missing)")
            passed = False
            continue
        variant_rules = {
            "precision": candidate_variant["precision"]
            >= champion_variant["precision"] - float(tolerances["precision_tolerance"]),
            "recall": candidate_variant["recall"]
            >= champion_variant["recall"] - float(tolerances["recall_tolerance"]),
            "pr_auc": candidate_variant["pr_auc"]
            >= champion_variant["pr_auc"] - float(tolerances["pr_auc_tolerance"]),
            "brier": candidate_variant["brier"]
            <= champion_variant["brier"] + float(tolerances["brier_tolerance"]),
        }
        for metric, rule_passed in variant_rules.items():
            reasons.append(
                f"{variant} {metric} guardrail: "
                f"{'pass' if rule_passed else 'fail'} "
                f"(candidate={candidate_variant[metric]:.4f}, "
                f"champion={champion_variant[metric]:.4f})"
            )
            passed = passed and rule_passed

    # The first schema containing case forecasts establishes the MAE baseline.
    # Later candidates must preserve the always-available weather model, the
    # delayed-report bridge, and the strongest exact-recent-report model.
    case_metric_names = (
        "weather_only_case_forecast_metrics",
        "weather_plus_recent_cases_case_forecast_metrics",
        "weather_plus_delayed_cases_case_forecast_metrics",
    )
    candidate_has_case_forecasts = candidate.get(case_metric_names[0]) is not None
    champion_has_case_forecasts = champion.get(case_metric_names[0]) is not None
    persistence_tolerance = float(tolerances.get("persistence_relative_tolerance", 0.05))
    for variant in ("weather_plus_recent_cases", "weather_plus_delayed_cases"):
        candidate_case = candidate.get(f"{variant}_case_forecast_metrics")
        if candidate_case is None:
            continue
        if candidate_case.get("persistence_mae") is None:
            reasons.append(f"{variant} persistence guardrail: fail (metric missing)")
            passed = False
            continue
        persistence_passed = candidate_case["mae"] <= candidate_case[
            "persistence_mae"
        ] * (1 + persistence_tolerance)
        reasons.append(
            f"{variant} persistence guardrail: "
            f"{'pass' if persistence_passed else 'fail'} "
            f"(model_mae={candidate_case['mae']:.4f}, "
            f"persistence_mae={candidate_case['persistence_mae']:.4f})"
        )
        passed = passed and persistence_passed
    if candidate_has_case_forecasts and not champion_has_case_forecasts:
        reasons.append("case MAE guardrails: baselines established by this schema upgrade")
    elif champion_has_case_forecasts:
        mae_tolerance = float(tolerances["case_mae_relative_tolerance"])
        outbreak_tolerance = float(tolerances["outbreak_mae_relative_tolerance"])
        for metadata_key in case_metric_names:
            variant = metadata_key.removesuffix("_case_forecast_metrics")
            candidate_case = candidate.get(metadata_key)
            champion_case = champion.get(metadata_key)
            if candidate_case is None and champion_case is None:
                continue
            if candidate_case is not None and champion_case is None and schema_upgrade:
                reasons.append(f"{variant} case MAE guardrails: baseline established")
                continue
            if candidate_case is None or champion_case is None:
                reasons.append(f"{variant} case MAE guardrails: fail (metrics missing)")
                passed = False
                continue
            case_rules = {
                "case_mae": candidate_case["mae"]
                <= champion_case["mae"] * (1 + mae_tolerance),
                "outbreak_week_mae": candidate_case["outbreak_week_mae"]
                <= champion_case["outbreak_week_mae"] * (1 + outbreak_tolerance),
            }
            for metric, rule_passed in case_rules.items():
                candidate_name = "mae" if metric == "case_mae" else metric
                reasons.append(
                    f"{variant} {metric} guardrail: "
                    f"{'pass' if rule_passed else 'fail'} "
                    f"(candidate={candidate_case[candidate_name]:.4f}, "
                    f"champion={champion_case[candidate_name]:.4f})"
                )
                passed = passed and rule_passed
    return PromotionDecision(passed, reasons)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a small JSON registry or report file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def promote_candidate(
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    registry_dir: Path,
    report_dir: Path,
    config: dict[str, Any],
) -> PromotionDecision:
    """Evaluate a candidate, save the decision report, and promote if allowed."""

    champion_json = registry_dir / "champion.json"
    champion = (
        json.loads(champion_json.read_text(encoding="utf-8"))
        if champion_json.exists()
        else None
    )
    decision = evaluate_promotion(metadata, champion, config)
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": metadata,
        "previous_champion": champion,
        "promoted": decision.promote,
        "reasons": decision.reasons,
    }
    report_name = f"pr_retraining_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    _write_json(report_dir / report_name, report)

    if decision.promote:
        registry_dir.mkdir(parents=True, exist_ok=True)
        temporary = registry_dir / "champion.joblib.tmp"
        joblib.dump(artifact, temporary)
        temporary.replace(registry_dir / "champion.joblib")
        _write_json(champion_json, metadata)
        validation_rows = [
            *artifact["weather_case_forecaster"].get(
                "validation_predictions", []
            ),
            *artifact["recent_case_forecaster"].get(
                "validation_predictions", []
            ),
            *artifact["delayed_case_forecaster"].get(
                "validation_predictions", []
            ),
        ]
        if validation_rows:
            pd.DataFrame(validation_rows).sort_values(
                ["forecast_variant", "week_start_date"]
            ).to_csv(registry_dir / "validation_predictions.csv", index=False)
        history_path = registry_dir / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, sort_keys=True) + "\n")
    return decision


def load_champion(registry_dir: Path) -> dict[str, Any]:
    """Load only the repository-controlled champion artifact."""

    path = registry_dir / "champion.joblib"
    if not path.exists():
        raise FileNotFoundError("No Puerto Rico champion exists; run guarded training first")
    payload = joblib.load(path)
    if payload.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("Champion model schema is incompatible with this code")
    if payload.get("case_forecast_schema_version") != CASE_FORECAST_SCHEMA_VERSION:
        raise ValueError("Champion case-forecast schema is incompatible with this code")
    return payload


def score_champion(
    artifact: dict[str, Any],
    all_cases: pd.DataFrame,
    config: dict[str, Any],
    cache_dir: Path,
    as_of: date | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Generate current-week and next-week Puerto Rico alert rows."""

    timezone_name = config["puerto_rico"]["timezone"]
    as_of = as_of or datetime.now(ZoneInfo(timezone_name)).date()
    current_start = pd.Timestamp(as_of) - pd.Timedelta(days=pd.Timestamp(as_of).weekday())
    target_starts = [
        current_start + pd.Timedelta(weeks=lead)
        for lead in range(int(config["prediction_weeks_ahead"]) + 1)
    ]
    weekly, _ = fetch_island_live_weather(
        target_starts, as_of, config, cache_dir, refresh
    )
    weekly["weekofyear"] = weekly["week_start_date"].dt.isocalendar().week.astype(int)
    features = add_weather_features(weekly).set_index("week_start_date")
    weather_detector = _detector_from_payload(artifact["weather_detector"])
    case_detector = _detector_from_payload(artifact["case_detector"])
    delayed_case_detector = _detector_from_payload(
        artifact["delayed_case_detector"]
    )
    weather_case_forecaster = _case_forecaster_from_payload(
        artifact["weather_case_forecaster"]
    )
    recent_case_forecaster = _case_forecaster_from_payload(
        artifact["recent_case_forecaster"]
    )
    delayed_case_forecaster = _case_forecaster_from_payload(
        artifact["delayed_case_forecaster"]
    )
    thresholds = {
        int(row["weekofyear"]): float(row["outbreak_threshold"])
        for row in artifact["seasonal_thresholds"]
    }
    metadata = artifact["metadata"]
    latest_case = pd.Timestamp(all_cases["week_start_date"].max())
    source_publication = str(all_cases["source_publication_date"].max())
    recent = all_cases[
        ["week_start_date", "total_cases", "hospitalized_cases"]
    ].copy()
    delay_policy = config["delayed_case_model"]

    rows: list[dict[str, Any]] = []
    generated = datetime.now(timezone.utc).isoformat()
    for lead, target in enumerate(target_starts):
        row = features.loc[target].copy()
        if int(row["weather_days_available"]) < 7:
            raise RuntimeError(
                f"Puerto Rico week {target.date()} has only "
                f"{int(row['weather_days_available'])}/7 complete island weather days"
            )
        exact_cases = live_exact_health_features(recent, target)
        delayed_cases = live_delayed_health_features(
            recent,
            target,
            minimum_age_weeks=int(delay_policy["minimum_report_age_weeks"]),
            maximum_age_weeks=int(delay_policy["maximum_report_age_weeks"]),
        )
        detector: TrainedOutbreakDetector
        report_anchor: pd.Timestamp | None = None
        if exact_cases is None:
            if delayed_cases is None:
                detector = weather_detector
                case_forecaster = weather_case_forecaster
                variant = "weather_only"
                case_forecast_reliability = "limited_no_usable_case_history"
            else:
                delayed_values, report_anchor = delayed_cases
                for name, value in delayed_values.items():
                    row[name] = value
                detector = delayed_case_detector
                case_forecaster = delayed_case_forecaster
                variant = "weather_plus_delayed_cases"
                case_forecast_reliability = "moderate_delayed_case_history"
        else:
            for name, value in exact_cases.items():
                row[name] = value
            detector = case_detector
            case_forecaster = recent_case_forecaster
            variant = "weather_plus_recent_cases"
            case_forecast_reliability = "stronger_exact_recent_cases"
        probability = detector.predict_probability(row)
        week = int(row["weekofyear"])
        threshold = thresholds[week]
        case_forecast = case_forecaster.predict(row)
        if case_forecast["p50"] >= threshold:
            case_risk_level = "expected_above_threshold"
        elif case_forecast["p80"] >= threshold:
            case_risk_level = "elevated_upper_range"
        elif case_forecast["p90"] >= threshold:
            case_risk_level = "possible_upper_tail"
        else:
            case_risk_level = "below_threshold_range"
        case_metrics = case_forecaster.validation_metrics
        rows.append(
            {
                "geography": "pr",
                "geography_name": "Puerto Rico",
                "generated_at_utc": generated,
                "as_of_date": as_of.isoformat(),
                "week_start_date": target.date().isoformat(),
                "lead_weeks": lead,
                "time_scope": "current_week" if lead == 0 else "forecast_week",
                "model_version": metadata["model_version"],
                "model_variant": variant,
                "predicted_cases_p50": case_forecast["p50"],
                "predicted_cases_p80": case_forecast["p80"],
                "predicted_cases_p90": case_forecast["p90"],
                # Keep the selected route's historical reliability visible in
                # every row; delayed reports no longer force weather-only use.
                "case_forecast_reliability": case_forecast_reliability,
                "case_risk_level": case_risk_level,
                "p50_above_threshold": bool(case_forecast["p50"] >= threshold),
                "p80_above_threshold": bool(case_forecast["p80"] >= threshold),
                "p90_above_threshold": bool(case_forecast["p90"] >= threshold),
                "held_out_case_mae": case_metrics["mae"],
                "held_out_normal_week_mae": case_metrics["normal_week_mae"],
                "held_out_outbreak_week_mae": case_metrics["outbreak_week_mae"],
                "outbreak_probability": probability,
                "alert_gate": detector.alert_gate,
                "outbreak_alert": bool(probability >= detector.alert_gate),
                "historical_outbreak_threshold_cases": threshold,
                "outbreak_definition": "seasonal_training_q75",
                "held_out_precision": detector.validation_metrics["precision"],
                "held_out_recall": detector.validation_metrics["recall"],
                "held_out_f1": detector.validation_metrics["f1"],
                "held_out_pr_auc": detector.validation_metrics["pr_auc"],
                "held_out_brier": detector.validation_metrics["brier"],
                "training_data_cutoff": metadata["training_data_cutoff"],
                "case_source_publication_date": source_publication,
                "latest_case_week": latest_case.date().isoformat(),
                "case_data_age_weeks": float((target - latest_case).days / 7),
                "case_report_anchor_week": (
                    report_anchor.date().isoformat()
                    if report_anchor is not None
                    else (
                        (target - pd.Timedelta(weeks=1)).date().isoformat()
                        if variant == "weather_plus_recent_cases"
                        else None
                    )
                ),
                "case_report_age_weeks": (
                    float(row["report_age_weeks"])
                    if variant == "weather_plus_delayed_cases"
                    else (1.0 if variant == "weather_plus_recent_cases" else None)
                ),
                "weather_days_available": int(row["weather_days_available"]),
                "forecast_input_days": int(row["forecast_input_days"]),
                "weather_sources": row["weather_sources"],
                "actual_cases": np.nan,
                "absolute_case_error": np.nan,
                "actual_outbreak": np.nan,
                "evaluated_at_utc": None,
            }
        )
    return pd.DataFrame(rows).reindex(columns=PREDICTION_COLUMNS)


def append_predictions(new_rows: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append alerts idempotently, replacing a same-day rerun of the same model."""

    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=PREDICTION_COLUMNS)
    existing = existing.reindex(columns=PREDICTION_COLUMNS)
    combined = pd.concat(
        [existing, new_rows.reindex(columns=PREDICTION_COLUMNS)], ignore_index=True
    )
    key = ["geography", "as_of_date", "week_start_date", "lead_weeks", "model_version"]
    combined = combined.drop_duplicates(key, keep="last").sort_values(
        ["as_of_date", "week_start_date", "lead_weeks"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.reindex(columns=PREDICTION_COLUMNS).to_csv(path, index=False)
    return combined.reset_index(drop=True)


def reconcile_predictions(
    prediction_path: Path,
    cases: pd.DataFrame,
    as_of: date,
    stabilization_weeks: int,
    metrics_path: Path,
) -> dict[str, Any]:
    """Attach finalized outcomes to prior predictions and calculate alert metrics."""

    if not prediction_path.exists():
        metrics = {"evaluated_rows": 0, "message": "No predictions exist yet"}
        _write_json(metrics_path, metrics)
        return metrics
    predictions = pd.read_csv(prediction_path).reindex(columns=PREDICTION_COLUMNS)
    predictions["week_start_date"] = pd.to_datetime(predictions["week_start_date"])
    stable = finalized_cases(cases, as_of, stabilization_weeks)
    truth = stable.set_index("week_start_date")["total_cases"].to_dict()
    evaluated_at = datetime.now(timezone.utc).isoformat()
    for index, row in predictions.iterrows():
        target = pd.Timestamp(row["week_start_date"])
        if target not in truth:
            continue
        actual_cases = int(truth[target])
        threshold = float(row["historical_outbreak_threshold_cases"])
        predictions.at[index, "actual_cases"] = actual_cases
        if pd.notna(row.get("predicted_cases_p50")):
            predictions.at[index, "absolute_case_error"] = abs(
                actual_cases - float(row["predicted_cases_p50"])
            )
        predictions.at[index, "actual_outbreak"] = int(actual_cases >= threshold)
        predictions.at[index, "evaluated_at_utc"] = evaluated_at

    predictions["week_start_date"] = predictions["week_start_date"].dt.date.astype(str)
    predictions.to_csv(prediction_path, index=False)
    evaluated = predictions[predictions["actual_outbreak"].notna()].copy()
    if evaluated.empty:
        metrics = {"evaluated_rows": 0, "message": "No target weeks are finalized yet"}
    else:
        labels = evaluated["actual_outbreak"].astype(int)
        alerts = evaluated["outbreak_alert"].astype(str).str.lower().eq("true")
        probabilities = evaluated["outbreak_probability"].astype(float)
        metrics = {
            "evaluated_at_utc": evaluated_at,
            "evaluated_rows": int(len(evaluated)),
            "true_positives": int(((alerts == 1) & (labels == 1)).sum()),
            "false_positives": int(((alerts == 1) & (labels == 0)).sum()),
            "true_negatives": int(((alerts == 0) & (labels == 0)).sum()),
            "false_negatives": int(((alerts == 0) & (labels == 1)).sum()),
            "accuracy": float(accuracy_score(labels, alerts)),
            "precision": float(precision_score(labels, alerts, zero_division=0)),
            "recall": float(recall_score(labels, alerts, zero_division=0)),
            "f1": float(f1_score(labels, alerts, zero_division=0)),
            "pr_auc": (
                float(average_precision_score(labels, probabilities))
                if labels.nunique() == 2
                else None
            ),
            "brier": float(brier_score_loss(labels, probabilities)),
        }
        negatives = metrics["true_negatives"] + metrics["false_positives"]
        metrics["specificity"] = (
            float(metrics["true_negatives"] / negatives) if negatives else None
        )
        case_evaluated = evaluated[evaluated["predicted_cases_p50"].notna()].copy()
        if len(case_evaluated):
            actual_cases = case_evaluated["actual_cases"].astype(float)
            p50 = case_evaluated["predicted_cases_p50"].astype(float)
            p80 = case_evaluated["predicted_cases_p80"].astype(float)
            p90 = case_evaluated["predicted_cases_p90"].astype(float)
            actual_outbreak = case_evaluated["actual_outbreak"].astype(int)
            normal = actual_outbreak == 0
            outbreak = actual_outbreak == 1
            metrics["case_forecast"] = {
                "evaluated_rows": int(len(case_evaluated)),
                "mae": float(mean_absolute_error(actual_cases, p50)),
                "median_absolute_error": float(
                    median_absolute_error(actual_cases, p50)
                ),
                "normal_week_mae": (
                    float(mean_absolute_error(actual_cases[normal], p50[normal]))
                    if normal.any()
                    else None
                ),
                "outbreak_week_mae": (
                    float(mean_absolute_error(actual_cases[outbreak], p50[outbreak]))
                    if outbreak.any()
                    else None
                ),
                "p80_coverage": float((actual_cases <= p80).mean()),
                "p90_coverage": float((actual_cases <= p90).mean()),
                "p80_pinball_loss": float(
                    mean_pinball_loss(actual_cases, p80, alpha=0.80)
                ),
                "p90_pinball_loss": float(
                    mean_pinball_loss(actual_cases, p90, alpha=0.90)
                ),
            }
    _write_json(metrics_path, metrics)
    return metrics
