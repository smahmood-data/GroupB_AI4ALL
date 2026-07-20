"""Near-real-time dengue outbreak detection with Open-Meteo data.

This file is intentionally separate from ``dengue_forecast_model.py``.  The
original file evaluates historical weekly case-count forecasts.  This file
solves a different operational problem: train a final outbreak classifier and
use information available now to produce current-week and one-week-ahead
outbreak alerts.

The most important design decision is *training/serving consistency*.  The
historical model used weather columns supplied by the DengAI competition.  A
model trained on those columns should not silently interpret values from a new
weather provider.  This script therefore:

1. keeps the historical DengAI case labels and the project's seasonal outbreak
   definition;
2. downloads historical weather for those same dates from Open-Meteo;
3. retrains the outbreak classifier using the Open-Meteo feature definitions;
4. downloads recent and forecast Open-Meteo weather using the same units and
   column names; and
5. scores the current city-specific epidemiological week and, by default, the
   following week.

Two classifier variants are trained for each city:

``weather_only``
    Uses calendar and lagged weather features.  It is always available.

``weather_plus_recent_cases``
    Adds exact case lags from a user-supplied recent-cases CSV.  It is used only
    when all eight immediately preceding weekly case counts exist.  The script
    never invents future case lags or quietly substitutes old values.

This is a research alert, not an official public-health declaration.  An
"outbreak" retains the project's statistical definition: a weekly case count
at or above the historical, season-specific 75th percentile.

Example:
    python src/near_realtime_outbreak_detection.py --city all

With recent reported cases:
    python src/near_realtime_outbreak_detection.py \
        --city sj \
        --recent-cases data/recent_cases.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

# Some macOS/Python combinations cannot parse the physical-core value returned
# to joblib.  Supplying a conservative logical-core limit avoids a noisy
# warning while keeping scikit-learn's thread limit explicit and portable.
logical_cores = os.cpu_count() or 2
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, logical_cores - 1)))

import numpy as np
import pandas as pd
import joblib
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

# Import the model factory and outbreak-threshold logic from the historical
# project.  Because both files live in ``src/``, this import works when the file
# is run directly with ``python src/near_realtime_outbreak_detection.py``.
from dengue_forecast_model import (
    DATA_BASE,
    fit_outbreak_classifier,
    prepare_raw_data,
    seasonal_threshold_table,
)


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Increment this value whenever feature construction, outbreak labeling, model
# settings, or calibration logic changes.  It prevents an older fitted model
# from being loaded after the code's meaning has changed.
MODEL_SCHEMA_VERSION = 3

# These variables exist with the same names and units in Open-Meteo's archive
# and forecast endpoints.  The underlying archive and operational forecast
# models are not identical, so future weather uncertainty remains.  Keeping the
# schema and provider consistent still removes the much larger mismatch caused
# by feeding new API fields into columns learned from a different dataset.
DAILY_WEATHER_COLUMNS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    # Rain-only totals, rainy hours, dew point, and shallow soil moisture add
    # information about how long water remains available to mosquitoes.  These
    # names are supported by both Open-Meteo's Archive and Forecast APIs.
    "rain_sum",
    "precipitation_hours",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "soil_moisture_0_to_7cm_mean",
    "et0_fao_evapotranspiration",
]

# Four weekly variables are calculated from the seven daily rain values. They
# distinguish one cloudburst, several moderate storms, and an uninterrupted
# dry spell even when weeks have similar total rainfall. ``rain_sum`` itself is
# not modeled because Puerto Rico snowfall is effectively zero, making it
# nearly identical to the already modeled ``precipitation_sum``.
DERIVED_WEEKLY_WEATHER_COLUMNS = [
    "wet_day_count",
    "heavy_rain_day_count",
    "max_daily_rainfall",
    "longest_dry_spell_days",
]
OUTBREAK_MODEL_WEATHER_COLUMNS = [
    column for column in DAILY_WEATHER_COLUMNS if column != "rain_sum"
] + ["wet_day_count", "max_daily_rainfall"]

# Time-held-out ablations showed that heavy-rain frequency and uninterrupted
# dry spells improved case-count MAE, but adding them to the alert classifiers
# reduced weather-only PR-AUC. Generate the union once, then give each model
# family only its validated subset. This is the same principle used to keep the
# MAE-focused forecast separate from the recall-focused alert policy.
CASE_FORECAST_MODEL_WEATHER_COLUMNS = OUTBREAK_MODEL_WEATHER_COLUMNS + [
    "heavy_rain_day_count",
    "longest_dry_spell_days",
]
MODEL_WEATHER_COLUMNS = CASE_FORECAST_MODEL_WEATHER_COLUMNS

# A weekly average is appropriate for mean temperature and humidity.  Weekly
# maximum/minimum preserve extremes, while rainfall and evapotranspiration are
# accumulated because their multi-day totals are biologically meaningful.
WEEKLY_AGGREGATIONS = {
    "temperature_2m_mean": "mean",
    "temperature_2m_max": "max",
    "temperature_2m_min": "min",
    "precipitation_sum": "sum",
    "rain_sum": "sum",
    "precipitation_hours": "sum",
    "relative_humidity_2m_mean": "mean",
    "dew_point_2m_mean": "mean",
    "soil_moisture_0_to_7cm_mean": "mean",
    "et0_fao_evapotranspiration": "sum",
}

# Weather and case lags are deliberately separate.  In particular, adding an
# eight-week rainfall lag must never turn cases_lag_1 into cases_lag_9.
WEATHER_LAGS = (0, 2, 4, 8)
CASE_FEATURE_COLUMNS = [
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_4",
    "cases_mean_4",
    "cases_mean_8",
    "cases_change_1",
]


@dataclass(frozen=True)
class CityConfig:
    """Coordinates and local timezone used for a DengAI city."""

    code: str
    name: str
    latitude: float
    longitude: float
    timezone_name: str


CITY_CONFIGS = {
    "sj": CityConfig(
        code="sj",
        name="San Juan, Puerto Rico",
        latitude=18.4655,
        longitude=-66.1057,
        timezone_name="America/Puerto_Rico",
    ),
    "iq": CityConfig(
        code="iq",
        name="Iquitos, Peru",
        latitude=-3.7437,
        longitude=-73.2516,
        timezone_name="America/Lima",
    ),
}


@dataclass(frozen=True)
class CachedJsonResponse:
    """API payload plus information needed to disclose cache freshness."""

    payload: dict[str, Any]
    cache_state: str
    fetched_at_utc: str


@dataclass
class TrainedOutbreakDetector:
    """A fitted classifier, its probability calibration, and its alert gate."""

    model: Any
    calibrator: LogisticRegression | None
    feature_columns: list[str]
    alert_gate: float
    validation_metrics: dict[str, float]

    def predict_probability(self, feature_row: pd.Series | dict[str, Any]) -> float:
        """Return a calibrated outbreak probability for one weekly row."""

        X = pd.DataFrame([dict(feature_row)]).reindex(columns=self.feature_columns)
        raw_probability = float(self.model.predict_proba(X)[0, 1])

        if self.calibrator is None:
            return raw_probability

        calibrated = self.calibrator.predict_proba(np.array([[raw_probability]]))[0, 1]
        return float(calibrated)


def _detector_to_payload(detector: TrainedOutbreakDetector) -> dict[str, Any]:
    """Convert a detector to built-in containers for stable joblib caching.

    Storing a plain dictionary avoids pickling the custom dataclass as
    ``__main__.TrainedOutbreakDetector`` when this file is run directly.
    """

    return {
        "model": detector.model,
        "calibrator": detector.calibrator,
        "feature_columns": detector.feature_columns,
        "alert_gate": detector.alert_gate,
        "validation_metrics": detector.validation_metrics,
    }


def _detector_from_payload(payload: dict[str, Any]) -> TrainedOutbreakDetector:
    """Reconstruct a detector from a trusted local model-cache dictionary."""

    return TrainedOutbreakDetector(
        model=payload["model"],
        calibrator=payload["calibrator"],
        feature_columns=list(payload["feature_columns"]),
        alert_gate=float(payload["alert_gate"]),
        validation_metrics=dict(payload["validation_metrics"]),
    )


def _cache_file(cache_dir: Path, prefix: str, parameters: dict[str, Any]) -> Path:
    """Create a stable, short cache filename from API parameters."""

    encoded = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return cache_dir / f"{prefix}_{digest}.json"


def fetch_json_with_cache(
    url: str,
    parameters: dict[str, Any],
    cache_dir: Path,
    cache_prefix: str,
    max_cache_age_hours: float | None,
    refresh: bool = False,
    timeout_seconds: int = 60,
) -> CachedJsonResponse:
    """Fetch JSON while preserving a transparent, dated local fallback.

    Historical Open-Meteo requests are immutable enough to cache indefinitely,
    so callers pass ``max_cache_age_hours=None``.  Live forecasts use a short
    cache lifetime.  If a live request fails and an older cache exists, the
    script uses it but returns ``stale_fallback``; the output never presents
    stale data as fresh.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(cache_dir, cache_prefix, parameters)
    now = time.time()

    if cache_path.exists() and not refresh:
        age_hours = (now - cache_path.stat().st_mtime) / 3600.0
        if max_cache_age_hours is None or age_hours <= max_cache_age_hours:
            return CachedJsonResponse(
                payload=json.loads(cache_path.read_text(encoding="utf-8")),
                cache_state="fresh_cache",
                fetched_at_utc=datetime.fromtimestamp(
                    cache_path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            )

    request_url = f"{url}?{urlencode(parameters)}"
    request = Request(request_url, headers={"User-Agent": "dengue-forecasting-model/1.0"})

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        # Open-Meteo may return structured JSON errors.  Treat them as failures
        # instead of allowing a later feature-building error to hide the cause.
        if payload.get("error"):
            raise RuntimeError(payload.get("reason", "Open-Meteo returned an error"))

        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return CachedJsonResponse(
            payload=payload,
            cache_state="live_api",
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        if cache_path.exists():
            return CachedJsonResponse(
                payload=json.loads(cache_path.read_text(encoding="utf-8")),
                cache_state="stale_fallback",
                fetched_at_utc=datetime.fromtimestamp(
                    cache_path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            )
        raise RuntimeError(f"Unable to retrieve {url}: {exc}") from exc


def parse_open_meteo_daily(payload: dict[str, Any], source: str) -> pd.DataFrame:
    """Convert Open-Meteo's column-oriented daily JSON into tidy rows."""

    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise ValueError("Open-Meteo response is missing the daily weather block")

    missing_columns = [column for column in DAILY_WEATHER_COLUMNS if column not in daily]
    if missing_columns:
        raise ValueError(f"Open-Meteo response is missing: {missing_columns}")

    frame = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    expected_length = len(frame)

    for column in DAILY_WEATHER_COLUMNS:
        values = daily[column]
        if len(values) != expected_length:
            raise ValueError(f"Open-Meteo returned a different number of values for {column}")
        frame[column] = pd.to_numeric(values, errors="coerce")

    frame["weather_source"] = source
    return frame.sort_values("date").reset_index(drop=True)


def combine_daily_weather(archive: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Combine archive and live rows, preferring live values on overlap."""

    combined = pd.concat([archive, forecast], ignore_index=True)
    combined = combined.sort_values(["date", "weather_source"])

    # Forecast rows are appended after archive rows.  Keeping the last row makes
    # the frequently refreshed endpoint authoritative for its recent overlap.
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    return combined.sort_values("date").reset_index(drop=True)


def _longest_true_run(values: pd.Series) -> int:
    """Return the longest uninterrupted run of true values in one week."""

    longest = 0
    current = 0
    for value in values.astype(bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def aggregate_daily_to_weeks(
    daily_weather: pd.DataFrame,
    week_starts: list[pd.Timestamp] | pd.Series,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Aggregate seven daily rows into each requested city-specific week.

    We use the original DengAI city's start weekday instead of imposing ISO
    Monday weeks on both locations.  A week must contain all seven daily rows;
    partial rainfall totals would otherwise look artificially dry and could
    produce a false signal.
    """

    daily = daily_weather.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    records: list[dict[str, Any]] = []

    for raw_start in pd.to_datetime(pd.Series(week_starts)).drop_duplicates().sort_values():
        start = pd.Timestamp(raw_start).normalize()
        end = start + pd.Timedelta(days=6)
        week = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()

        record: dict[str, Any] = {
            "week_start_date": start,
            "weather_days_available": int(
                week[DAILY_WEATHER_COLUMNS].notna().all(axis=1).sum()
            )
            if len(week)
            else 0,
            "weather_sources": "+".join(sorted(week["weather_source"].dropna().unique()))
            if len(week)
            else "missing",
        }

        if as_of_date is None:
            record["forecast_input_days"] = 0
        else:
            record["forecast_input_days"] = int(
                (week["date"].dt.date > as_of_date).sum()
            )

        if record["weather_days_available"] < 7:
            # Leave all weekly variables missing together.  The caller can then
            # refuse to score the row instead of using a misleading partial sum.
            for column in DAILY_WEATHER_COLUMNS + DERIVED_WEEKLY_WEATHER_COLUMNS:
                record[column] = np.nan
        else:
            for column, aggregation in WEEKLY_AGGREGATIONS.items():
                values = week[column]
                if aggregation == "mean":
                    record[column] = float(values.mean())
                elif aggregation == "max":
                    record[column] = float(values.max())
                elif aggregation == "min":
                    record[column] = float(values.min())
                elif aggregation == "sum":
                    record[column] = float(values.sum())
                else:  # pragma: no cover - protected by the constant above.
                    raise ValueError(f"Unknown weekly aggregation: {aggregation}")

            # A wet day uses a small 0.1 mm threshold so floating-point traces
            # are not counted as biologically meaningful rain. A heavy-rain day
            # uses 10 mm, a common descriptive threshold rather than a learned
            # cutoff; the tree model still decides whether that count matters.
            # The longest dry spell preserves the sequence of daily values that
            # would otherwise disappear when the week is reduced to totals.
            record["wet_day_count"] = float((week["rain_sum"] >= 0.1).sum())
            record["heavy_rain_day_count"] = float((week["rain_sum"] >= 10.0).sum())
            record["max_daily_rainfall"] = float(week["rain_sum"].max())
            record["longest_dry_spell_days"] = float(
                _longest_true_run(week["rain_sum"] < 0.1)
            )

        records.append(record)

    return pd.DataFrame(records).sort_values("week_start_date").reset_index(drop=True)


def weather_feature_columns() -> list[str]:
    """Return the validated weather schema used by outbreak classifiers."""

    return _weather_feature_columns_for(OUTBREAK_MODEL_WEATHER_COLUMNS)


def case_forecast_weather_feature_columns() -> list[str]:
    """Return weather features selected for MAE-focused case forecasting."""

    return _weather_feature_columns_for(CASE_FORECAST_MODEL_WEATHER_COLUMNS)


def _weather_feature_columns_for(model_columns: list[str]) -> list[str]:
    """Expand raw weekly fields into the common calendar/lag feature names."""

    columns = ["week_sin_1", "week_cos_1", "week_sin_2", "week_cos_2"]
    for weather_column in model_columns:
        columns.extend(
            [f"{weather_column}_lag_{lag}" for lag in WEATHER_LAGS]
        )
        columns.extend(
            [f"{weather_column}_rolling_4", f"{weather_column}_rolling_8"]
        )
    return columns


def add_weather_features(weekly_weather: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, independent lag, and rolling weather features."""

    frame = weekly_weather.sort_values("week_start_date").copy()

    if "weekofyear" not in frame:
        frame["weekofyear"] = frame["week_start_date"].dt.isocalendar().week.astype(int)

    # Accumulate the generated series and concatenate once. Repeatedly inserting
    # more than 100 lag/rolling columns fragments a pandas DataFrame and slows
    # retraining; building a separate feature block keeps the same values and
    # column names without that memory-layout penalty.
    generated: dict[str, pd.Series] = {}
    for harmonic in (1, 2):
        generated[f"week_sin_{harmonic}"] = np.sin(
            2 * np.pi * harmonic * frame["weekofyear"] / 52.0
        )
        generated[f"week_cos_{harmonic}"] = np.cos(
            2 * np.pi * harmonic * frame["weekofyear"] / 52.0
        )

    for column in MODEL_WEATHER_COLUMNS:
        for lag in WEATHER_LAGS:
            generated[f"{column}_lag_{lag}"] = frame[column].shift(lag)

        if WEEKLY_AGGREGATIONS.get(column) == "sum" or column in {
            "wet_day_count",
            "heavy_rain_day_count",
        }:
            generated[f"{column}_rolling_4"] = frame[column].rolling(
                4, min_periods=4
            ).sum()
            generated[f"{column}_rolling_8"] = frame[column].rolling(
                8, min_periods=8
            ).sum()
        elif column == "longest_dry_spell_days":
            generated[f"{column}_rolling_4"] = frame[column].rolling(
                4, min_periods=4
            ).max()
            generated[f"{column}_rolling_8"] = frame[column].rolling(
                8, min_periods=8
            ).max()
        else:
            generated[f"{column}_rolling_4"] = frame[column].rolling(
                4, min_periods=4
            ).mean()
            generated[f"{column}_rolling_8"] = frame[column].rolling(
                8, min_periods=8
            ).mean()

    return pd.concat([frame, pd.DataFrame(generated, index=frame.index)], axis=1)


def add_historical_case_features(cases: pd.DataFrame) -> pd.DataFrame:
    """Add strictly past case features to historical training rows."""

    frame = cases.sort_values("week_start_date").copy()
    shifted = frame["total_cases"].shift(1)

    frame["cases_lag_1"] = frame["total_cases"].shift(1)
    frame["cases_lag_2"] = frame["total_cases"].shift(2)
    frame["cases_lag_4"] = frame["total_cases"].shift(4)
    frame["cases_mean_4"] = shifted.rolling(4, min_periods=4).mean()
    frame["cases_mean_8"] = shifted.rolling(8, min_periods=8).mean()
    frame["cases_change_1"] = frame["cases_lag_1"] - frame["cases_lag_2"]
    return frame


def live_case_features(
    recent_cases: pd.DataFrame,
    target_week_start: pd.Timestamp,
) -> dict[str, float] | None:
    """Build case features only when eight exact preceding weeks are present.

    For a one-week-ahead forecast, the most recent reported case count may not
    yet exist.  Returning ``None`` triggers the weather-only model.  This is the
    key safeguard that prevents the live pipeline from using a case lag that
    would be unavailable in reality.
    """

    if recent_cases.empty:
        return None

    values = recent_cases.set_index("week_start_date")["total_cases"].to_dict()
    required_dates = [target_week_start - pd.Timedelta(weeks=lag) for lag in range(1, 9)]
    if any(required_date not in values for required_date in required_dates):
        return None

    history = [float(values[required_date]) for required_date in reversed(required_dates)]
    lag_1 = float(values[target_week_start - pd.Timedelta(weeks=1)])
    lag_2 = float(values[target_week_start - pd.Timedelta(weeks=2)])

    return {
        "cases_lag_1": lag_1,
        "cases_lag_2": lag_2,
        "cases_lag_4": float(values[target_week_start - pd.Timedelta(weeks=4)]),
        "cases_mean_4": float(np.mean(history[-4:])),
        "cases_mean_8": float(np.mean(history)),
        "cases_change_1": lag_1 - lag_2,
    }


def load_recent_cases(path: Path | None) -> pd.DataFrame:
    """Read and validate optional recent case reports.

    Required columns:
        city, week_start_date, total_cases

    Dates must already represent the same weekly start convention as the city
    being scored.  The script avoids guessing or shifting official reporting
    weeks because an unnoticed one-week error would corrupt every case lag.
    """

    if path is None:
        return pd.DataFrame(columns=["city", "week_start_date", "total_cases"])
    if not path.exists():
        raise FileNotFoundError(f"Recent-case file does not exist: {path}")

    frame = pd.read_csv(path)
    required = {"city", "week_start_date", "total_cases"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Recent-case CSV is missing required columns: {sorted(missing)}")

    frame = frame[list(required)].copy()
    frame["city"] = frame["city"].astype(str).str.lower().str.strip()
    unknown_cities = sorted(set(frame["city"]) - set(CITY_CONFIGS))
    if unknown_cities:
        raise ValueError(f"Unknown city codes in recent-case CSV: {unknown_cities}")

    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"], errors="raise").dt.normalize()
    frame["total_cases"] = pd.to_numeric(frame["total_cases"], errors="raise")
    if (frame["total_cases"] < 0).any():
        raise ValueError("Recent total_cases values cannot be negative")
    if frame.duplicated(["city", "week_start_date"]).any():
        raise ValueError("Recent-case CSV has duplicate city/week rows")

    return frame.sort_values(["city", "week_start_date"]).reset_index(drop=True)


def load_historical_cases(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Load public DengAI labels and cache the joined historical case dates."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "drivendata_historical_cases.csv"

    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, parse_dates=["week_start_date"])
        return frame.sort_values(["city", "week_start_date"]).reset_index(drop=True)

    # Reuse the same public files and key-safe merge used by the historical
    # model.  No Kaggle credentials or Kaggle API are involved.
    features = pd.read_csv(f"{DATA_BASE}/dengue_features_train.csv")
    labels = pd.read_csv(f"{DATA_BASE}/dengue_labels_train.csv")
    joined = prepare_raw_data(features, labels)
    frame = joined[["city", "year", "weekofyear", "week_start_date", "total_cases"]].copy()
    frame.to_csv(cache_path, index=False)
    return frame.sort_values(["city", "week_start_date"]).reset_index(drop=True)


def add_expanding_outbreak_labels(cases: pd.DataFrame, minimum_prior_years: int = 3) -> pd.DataFrame:
    """Label each training year using seasonal thresholds from earlier years.

    This mirrors expanding-window validation.  A row from 2006 cannot help
    define its own 2006 outbreak threshold.  The first few years are omitted
    because they do not yet have a defensible seasonal history.
    """

    frame = cases.sort_values("week_start_date").copy()
    labeled_years: list[pd.DataFrame] = []

    for current_year in sorted(frame["year"].unique()):
        prior = frame[frame["year"] < current_year]
        current = frame[frame["year"] == current_year].copy()
        if prior["year"].nunique() < minimum_prior_years:
            continue

        thresholds = seasonal_threshold_table(prior)
        current = current.merge(
            thresholds[["weekofyear", "outbreak_threshold"]],
            on="weekofyear",
            how="left",
            validate="many_to_one",
        )
        current["outbreak_label"] = (
            current["total_cases"] >= current["outbreak_threshold"]
        ).astype(int)
        labeled_years.append(current)

    if not labeled_years:
        raise ValueError("Not enough historical years to define expanding outbreak labels")

    return pd.concat(labeled_years, ignore_index=True).sort_values("week_start_date")


def _balanced_sample_weights(labels: pd.Series | np.ndarray) -> np.ndarray:
    """Give each class equal total influence without duplicating rows."""

    y = np.asarray(labels, dtype=int)
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        return np.ones(len(y), dtype=float)
    return np.where(y == 1, len(y) / (2 * counts[1]), len(y) / (2 * counts[0]))


def choose_recall_gate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> float:
    """Choose the most precise gate that reaches the requested recall.

    The choice is based only on time-held-out predictions produced below.  If
    no candidate reaches the requested recall, the gate with the best achieved
    recall is used, with precision as the tie-breaker.
    """

    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows: list[dict[str, float]] = []

    for gate in candidates:
        alerts = probabilities >= gate
        rows.append(
            {
                "gate": float(gate),
                "recall": float(recall_score(labels, alerts, zero_division=0)),
                "precision": float(precision_score(labels, alerts, zero_division=0)),
            }
        )

    scores = pd.DataFrame(rows)
    eligible = scores[scores["recall"] >= target_recall]
    if len(eligible):
        chosen = eligible.sort_values(
            ["precision", "gate"], ascending=[False, False]
        ).iloc[0]
    else:
        chosen = scores.sort_values(
            ["recall", "precision", "gate"], ascending=[False, False, False]
        ).iloc[0]
    return float(chosen["gate"])


def fit_time_aware_detector(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    target_recall: float,
    n_validation_years: int = 4,
    validation_filter: tuple[str, float] | None = None,
) -> TrainedOutbreakDetector:
    """Fit, calibrate, and gate a classifier using expanding years.

    The final model trains on all historical rows, but calibration and the alert
    gate use predictions from years that were held out in time.  This is more
    realistic than calibrating probabilities on the same rows used to fit the
    classifier.
    """

    frame = training_frame.dropna(subset=["outbreak_label"]).sort_values("week_start_date")
    years = sorted(frame["year"].unique())
    if len(years) < 4:
        raise ValueError("At least four labeled years are required for time-aware training")

    validation_years = years[-min(n_validation_years, len(years) - 2) :]
    oof_raw: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []

    for validation_year in validation_years:
        train = frame[frame["year"] < validation_year]
        validation = frame[frame["year"] == validation_year]
        if validation_filter is not None:
            column, value = validation_filter
            validation = validation[validation[column] == value]
        y_train = train["outbreak_label"].astype(int)
        if len(train) < 100 or validation.empty or y_train.nunique() < 2:
            continue

        fold_model = fit_outbreak_classifier()
        fold_model.fit(
            train[feature_columns],
            y_train,
            sample_weight=_balanced_sample_weights(y_train),
        )
        oof_raw.append(fold_model.predict_proba(validation[feature_columns])[:, 1])
        oof_labels.append(validation["outbreak_label"].astype(int).to_numpy())

    if not oof_raw:
        raise ValueError("Unable to create time-held-out predictions for calibration")

    raw_probabilities = np.concatenate(oof_raw)
    held_out_labels = np.concatenate(oof_labels)

    calibrator: LogisticRegression | None = None
    calibrated_probabilities = raw_probabilities
    if len(np.unique(held_out_labels)) == 2:
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(raw_probabilities.reshape(-1, 1), held_out_labels)
        calibrated_probabilities = calibrator.predict_proba(
            raw_probabilities.reshape(-1, 1)
        )[:, 1]

    gate = choose_recall_gate(calibrated_probabilities, held_out_labels, target_recall)
    alerts = calibrated_probabilities >= gate

    final_labels = frame["outbreak_label"].astype(int)
    final_model = fit_outbreak_classifier()
    final_model.fit(
        frame[feature_columns],
        final_labels,
        sample_weight=_balanced_sample_weights(final_labels),
    )

    metrics = {
        "held_out_rows": float(len(held_out_labels)),
        "precision": float(precision_score(held_out_labels, alerts, zero_division=0)),
        "recall": float(recall_score(held_out_labels, alerts, zero_division=0)),
        "f1": float(f1_score(held_out_labels, alerts, zero_division=0)),
        "pr_auc": float(average_precision_score(held_out_labels, calibrated_probabilities)),
        "brier": float(brier_score_loss(held_out_labels, calibrated_probabilities)),
    }
    return TrainedOutbreakDetector(
        model=final_model,
        calibrator=calibrator,
        feature_columns=feature_columns,
        alert_gate=gate,
        validation_metrics=metrics,
    )


def _archive_parameters(
    city: CityConfig,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Build one canonical Open-Meteo archive request."""

    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(DAILY_WEATHER_COLUMNS),
        "timezone": city.timezone_name,
    }


def _forecast_parameters(city: CityConfig) -> dict[str, Any]:
    """Request enough recent history for eight-week lag features."""

    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "daily": ",".join(DAILY_WEATHER_COLUMNS),
        "timezone": city.timezone_name,
        "past_days": 92,
        "forecast_days": 16,
    }


def fetch_training_weather(
    city: CityConfig,
    city_cases: pd.DataFrame,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, str]:
    """Download and aggregate historical weather for the labeled case weeks."""

    starts = city_cases["week_start_date"].sort_values().tolist()
    parameters = _archive_parameters(
        city,
        pd.Timestamp(min(starts)).date(),
        (pd.Timestamp(max(starts)) + pd.Timedelta(days=6)).date(),
    )
    response = fetch_json_with_cache(
        OPEN_METEO_ARCHIVE_URL,
        parameters,
        cache_dir,
        cache_prefix=f"{city.code}_training_weather",
        max_cache_age_hours=None,
        refresh=refresh,
    )
    daily = parse_open_meteo_daily(response.payload, source="open_meteo_archive")
    weekly = aggregate_daily_to_weeks(daily, starts)
    return weekly, response.cache_state


def infer_weekday_and_week_offset(city_cases: pd.DataFrame) -> tuple[int, int]:
    """Learn the city's weekly cadence and its epidemiological-week offset."""

    weekdays = city_cases["week_start_date"].dt.weekday
    start_weekday = int(weekdays.mode().iloc[0])

    iso_week = city_cases["week_start_date"].dt.isocalendar().week.astype(int)
    # Calculate the most common circular difference between the dataset's week
    # number and ISO week.  This preserves the historical convention in 2026.
    offsets = ((city_cases["weekofyear"].astype(int) - iso_week + 26) % 53) - 26
    week_offset = int(offsets.mode().iloc[0])
    return start_weekday, week_offset


def epidemiological_week(timestamp: pd.Timestamp, offset: int) -> int:
    """Convert a date to the historical dataset's 1–53 week convention."""

    iso_week = int(timestamp.isocalendar().week)
    return int(((iso_week + offset - 1) % 53) + 1)


def current_week_start(as_of: date, start_weekday: int) -> pd.Timestamp:
    """Return the most recent city-specific week start on or before ``as_of``."""

    days_since_start = (as_of.weekday() - start_weekday) % 7
    return pd.Timestamp(as_of - timedelta(days=days_since_start))


def fetch_live_weekly_weather(
    city: CityConfig,
    target_starts: list[pd.Timestamp],
    as_of: date,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, str]:
    """Combine recent archive values with the refreshed forecast endpoint."""

    earliest_needed = min(target_starts) - pd.Timedelta(weeks=11)
    latest_needed = max(target_starts) + pd.Timedelta(days=6)

    archive_response = fetch_json_with_cache(
        OPEN_METEO_ARCHIVE_URL,
        _archive_parameters(city, earliest_needed.date(), min(latest_needed.date(), as_of)),
        cache_dir,
        cache_prefix=f"{city.code}_recent_archive",
        max_cache_age_hours=24,
        refresh=refresh,
    )
    forecast_response = fetch_json_with_cache(
        OPEN_METEO_FORECAST_URL,
        _forecast_parameters(city),
        cache_dir,
        cache_prefix=f"{city.code}_live_forecast",
        max_cache_age_hours=6,
        refresh=refresh,
    )

    archive = parse_open_meteo_daily(archive_response.payload, source="open_meteo_archive")
    forecast = parse_open_meteo_daily(forecast_response.payload, source="open_meteo_live")
    combined = combine_daily_weather(archive, forecast)

    all_starts = pd.date_range(
        start=earliest_needed,
        end=max(target_starts),
        freq="7D",
    ).tolist()
    weekly = aggregate_daily_to_weeks(combined, all_starts, as_of_date=as_of)

    states = sorted({archive_response.cache_state, forecast_response.cache_state})
    return weekly, "+".join(states)


def train_city_detectors(
    city: CityConfig,
    city_cases: pd.DataFrame,
    cache_dir: Path,
    target_recall: float,
    refresh: bool,
) -> tuple[TrainedOutbreakDetector, TrainedOutbreakDetector, pd.DataFrame, str]:
    """Prepare source-consistent history and train both detector variants."""

    weekly_weather, cache_state = fetch_training_weather(
        city, city_cases, cache_dir, refresh
    )
    weekly_weather = weekly_weather.merge(
        city_cases[["week_start_date", "weekofyear"]],
        on="week_start_date",
        how="left",
        validate="one_to_one",
    )
    weather_features = add_weather_features(weekly_weather)

    labeled_cases = add_expanding_outbreak_labels(city_cases)
    cases_with_lags = add_historical_case_features(city_cases)
    case_lag_columns = ["week_start_date"] + CASE_FEATURE_COLUMNS

    training = labeled_cases.merge(
        weather_features[["week_start_date"] + weather_feature_columns()],
        on="week_start_date",
        how="inner",
        validate="one_to_one",
    ).merge(
        cases_with_lags[case_lag_columns],
        on="week_start_date",
        how="left",
        validate="one_to_one",
    )

    weather_columns = weather_feature_columns()
    weather_detector = fit_time_aware_detector(
        training,
        weather_columns,
        target_recall=target_recall,
    )

    # Case-aware training drops the first eight historical weeks because the
    # model must never learn from median-filled fake case histories.
    case_training = training.dropna(subset=CASE_FEATURE_COLUMNS)
    case_detector = fit_time_aware_detector(
        case_training,
        weather_columns + CASE_FEATURE_COLUMNS,
        target_recall=target_recall,
    )
    return weather_detector, case_detector, training, cache_state


def fitted_model_cache_path(
    city: CityConfig,
    city_cases: pd.DataFrame,
    cache_dir: Path,
    target_recall: float,
) -> Path:
    """Return a versioned path for a city's fitted detector pair.

    The fingerprint changes when the case history, feature schema, recall goal,
    pandas/scikit-learn version, or manual model schema version changes.  This
    keeps the speed benefit of persistence without silently reusing an artifact
    trained under incompatible assumptions.
    """

    fingerprint = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "city": city.code,
        "case_rows": len(city_cases),
        "case_max_date": city_cases["week_start_date"].max().isoformat(),
        "case_total": float(city_cases["total_cases"].sum()),
        "target_recall": float(target_recall),
        "weather_columns": DAILY_WEATHER_COLUMNS,
        "weather_lags": WEATHER_LAGS,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn_version,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"{city.code}_trained_detectors_{digest}.joblib"


def load_or_train_city_detectors(
    city: CityConfig,
    city_cases: pd.DataFrame,
    cache_dir: Path,
    target_recall: float,
    refresh: bool,
) -> tuple[TrainedOutbreakDetector, TrainedOutbreakDetector, str]:
    """Load a compatible fitted model or train and persist a new one.

    Only files created inside this project's cache directory should be loaded;
    joblib files, like pickle files, must never be accepted from untrusted
    sources.  A corrupt local cache simply causes a clean retrain.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = fitted_model_cache_path(
        city, city_cases, cache_dir, target_recall
    )

    if model_path.exists() and not refresh:
        try:
            payload = joblib.load(model_path)
            weather_detector = _detector_from_payload(payload["weather_detector"])
            case_detector = _detector_from_payload(payload["case_detector"])
            return weather_detector, case_detector, "trained_model_cache"
        except (
            KeyError,
            TypeError,
            ValueError,
            EOFError,
            OSError,
            pickle.UnpicklingError,
            AttributeError,
            ImportError,
        ):
            # Do not fail the live run merely because a previous local write was
            # interrupted.  Training below will atomically replace the file.
            pass

    weather_detector, case_detector, _, training_cache_state = train_city_detectors(
        city=city,
        city_cases=city_cases,
        cache_dir=cache_dir,
        target_recall=target_recall,
        refresh=refresh,
    )
    payload = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "weather_detector": _detector_to_payload(weather_detector),
        "case_detector": _detector_to_payload(case_detector),
    }

    # Write to a temporary sibling first so an interrupted process cannot leave
    # a partially serialized file at the final cache path.
    temporary_path = model_path.with_suffix(".tmp")
    joblib.dump(payload, temporary_path)
    temporary_path.replace(model_path)
    return weather_detector, case_detector, training_cache_state


def historical_threshold_for_week(city_cases: pd.DataFrame, weekofyear: int) -> float:
    """Return the final historical q75 threshold displayed beside a live alert."""

    thresholds = seasonal_threshold_table(city_cases)
    match = thresholds.loc[thresholds["weekofyear"] == weekofyear, "outbreak_threshold"]
    return float(match.iloc[0])


def score_city(
    city: CityConfig,
    city_cases: pd.DataFrame,
    recent_cases: pd.DataFrame,
    cache_dir: Path,
    target_recall: float,
    alert_gate_override: float | None,
    weeks_ahead: int,
    refresh: bool,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Train one city's detectors and generate live weekly alert rows."""

    if as_of is None:
        # A UTC date can be one day ahead of Puerto Rico or Peru shortly after
        # midnight UTC.  Week membership must follow the city's local calendar.
        as_of = datetime.now(ZoneInfo(city.timezone_name)).date()

    weather_detector, case_detector, training_cache_state = load_or_train_city_detectors(
        city,
        city_cases,
        cache_dir,
        target_recall,
        refresh,
    )

    start_weekday, week_offset = infer_weekday_and_week_offset(city_cases)
    current_start = current_week_start(as_of, start_weekday)
    target_starts = [
        current_start + pd.Timedelta(weeks=lead) for lead in range(weeks_ahead + 1)
    ]

    live_weekly, live_cache_state = fetch_live_weekly_weather(
        city,
        target_starts,
        as_of,
        cache_dir,
        refresh,
    )
    live_weekly["weekofyear"] = live_weekly["week_start_date"].apply(
        lambda value: epidemiological_week(value, week_offset)
    )
    live_features = add_weather_features(live_weekly).set_index("week_start_date")

    city_recent = recent_cases[recent_cases["city"] == city.code].copy()
    latest_case_date = (
        city_recent["week_start_date"].max() if len(city_recent) else pd.NaT
    )

    output_rows: list[dict[str, Any]] = []
    for lead, target_start in enumerate(target_starts):
        if target_start not in live_features.index:
            raise RuntimeError(f"No live weather row was built for {target_start.date()}")

        row = live_features.loc[target_start].copy()
        if int(row["weather_days_available"]) < 7:
            raise RuntimeError(
                f"Only {int(row['weather_days_available'])}/7 weather days are available "
                f"for {city.code.upper()} week {target_start.date()}"
            )

        exact_case_features = live_case_features(city_recent, target_start)
        if exact_case_features is not None:
            for key, value in exact_case_features.items():
                row[key] = value
            detector = case_detector
            model_variant = "weather_plus_recent_cases"
        else:
            detector = weather_detector
            model_variant = "weather_only"

        probability = detector.predict_probability(row)
        gate = (
            float(alert_gate_override)
            if alert_gate_override is not None
            else detector.alert_gate
        )
        week_number = int(row["weekofyear"])

        if pd.isna(latest_case_date):
            case_age_weeks: float | None = None
            latest_case_text: str | None = None
        else:
            case_age_weeks = float((target_start - latest_case_date).days / 7.0)
            latest_case_text = pd.Timestamp(latest_case_date).date().isoformat()

        output_rows.append(
            {
                "city": city.code,
                "city_name": city.name,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_date": as_of.isoformat(),
                "week_start_date": target_start.date().isoformat(),
                "lead_weeks": lead,
                "time_scope": "current_week" if lead == 0 else "forecast_week",
                "model_variant": model_variant,
                "outbreak_probability": probability,
                "alert_gate": gate,
                "outbreak_alert": bool(probability >= gate),
                # These are time-held-out metrics for the exact model variant
                # used on this row.  Keeping them beside the live alert makes
                # clear, for example, that a high-recall weather-only warning
                # may also generate many false alarms.
                "held_out_precision": detector.validation_metrics["precision"],
                "held_out_recall": detector.validation_metrics["recall"],
                "held_out_f1": detector.validation_metrics["f1"],
                "held_out_pr_auc": detector.validation_metrics["pr_auc"],
                "historical_outbreak_threshold_cases": historical_threshold_for_week(
                    city_cases, week_number
                ),
                "outbreak_definition": "seasonal_training_q75",
                "weather_days_available": int(row["weather_days_available"]),
                "forecast_input_days": int(row["forecast_input_days"]),
                "weather_sources": row["weather_sources"],
                "training_weather_cache_state": training_cache_state,
                "live_weather_cache_state": live_cache_state,
                "latest_case_week": latest_case_text,
                "case_data_age_weeks": case_age_weeks,
            }
        )

    metrics = {
        "weather_only": weather_detector.validation_metrics,
        "weather_plus_recent_cases": case_detector.validation_metrics,
    }
    return pd.DataFrame(output_rows), metrics


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the command-line interface in one testable function."""

    parser = argparse.ArgumentParser(
        description="Retrain and run a near-real-time dengue outbreak detector."
    )
    parser.add_argument(
        "--city",
        choices=["all", "sj", "iq"],
        default="all",
        help="City to score. Default: both San Juan and Iquitos.",
    )
    parser.add_argument(
        "--recent-cases",
        type=Path,
        help="Optional CSV with city, week_start_date, and total_cases columns.",
    )
    parser.add_argument(
        "--weeks-ahead",
        type=int,
        choices=[0, 1],
        default=1,
        help="Score only the current week (0) or current plus next week (1).",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.80,
        help="Held-out recall goal used to choose the default alert gate.",
    )
    parser.add_argument(
        "--alert-gate",
        type=float,
        help="Optional fixed probability cutoff, overriding the learned gate.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/near_realtime"),
        help="Directory for historical and live API caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/near_realtime_outbreak_alerts.csv"),
        help="CSV destination for alert rows.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore fresh caches and request all API data again.",
    )
    return parser


def _validate_probability(name: str, value: float | None) -> None:
    """Reject probability-like CLI values outside the closed unit interval."""

    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1")


def main() -> None:
    """Run the complete retrain-and-score workflow and save auditable output."""

    args = build_argument_parser().parse_args()
    _validate_probability("--target-recall", args.target_recall)
    _validate_probability("--alert-gate", args.alert_gate)

    historical_cases = load_historical_cases(args.cache_dir, refresh=args.refresh)
    recent_cases = load_recent_cases(args.recent_cases)
    city_codes = list(CITY_CONFIGS) if args.city == "all" else [args.city]

    outputs: list[pd.DataFrame] = []
    for city_code in city_codes:
        city = CITY_CONFIGS[city_code]
        city_cases = historical_cases[historical_cases["city"] == city_code].copy()
        print(f"Preparing source-consistent detector for {city.name}...", flush=True)

        city_output, metrics = score_city(
            city=city,
            city_cases=city_cases,
            recent_cases=recent_cases,
            cache_dir=args.cache_dir,
            target_recall=args.target_recall,
            alert_gate_override=args.alert_gate,
            weeks_ahead=args.weeks_ahead,
            refresh=args.refresh,
        )
        outputs.append(city_output)

        print("  Time-held-out alert metrics:")
        for variant, variant_metrics in metrics.items():
            compact = ", ".join(
                f"{name}={value:.3f}"
                for name, value in variant_metrics.items()
                if name != "held_out_rows"
            )
            print(f"    {variant}: {compact}")

    result = pd.concat(outputs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    display_columns = [
        "city",
        "week_start_date",
        "lead_weeks",
        "model_variant",
        "outbreak_probability",
        "alert_gate",
        "outbreak_alert",
        "forecast_input_days",
    ]
    display = result[display_columns].copy()
    display["outbreak_probability"] = display["outbreak_probability"].round(3)
    display["alert_gate"] = display["alert_gate"].round(3)

    print("\nNear-real-time outbreak alerts")
    print(display.to_string(index=False))
    print(f"\nSaved detailed output to {args.output.resolve()}")
    print(
        "Reminder: these are statistical research alerts, not official "
        "public-health outbreak declarations."
    )


if __name__ == "__main__":
    main()
