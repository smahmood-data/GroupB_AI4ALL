"""Build the self-contained Puerto Rico API model explainer.

The explainer is generated from versioned operational artifacts instead of
copying result values into HTML by hand. Re-run this script after a champion is
promoted so charts, live examples, and displayed metrics stay synchronized with
the model that is actually used by the weekly pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    PROJECT_ROOT / "docs" / "assets" / "puerto-rico-api-model-explainer.template.html"
)
OUTPUT_PATH = PROJECT_ROOT / "docs" / "puerto-rico-api-model-explainer.html"
CHAMPION_PATH = PROJECT_ROOT / "models" / "operational" / "pr" / "champion.json"
VALIDATION_PATH = (
    PROJECT_ROOT / "models" / "operational" / "pr" / "validation_predictions.csv"
)
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "operational" / "predictions.csv"


def _safe_script_json(value: Any) -> str:
    """Serialize JSON without allowing a value to close its script element."""

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).replace(
        "</", "<\\/"
    )


def _validation_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Keep only the compact fields used by the interactive validation chart."""

    columns = {
        "week_start_date": "date",
        "year": "year",
        "forecast_variant": "variant",
        "actual_cases": "actual",
        "outbreak_threshold": "threshold",
        "actual_outbreak": "outbreak",
        "predicted_cases_p50": "p50",
        "predicted_cases_p80": "p80",
        "predicted_cases_p90": "p90",
    }
    compact = frame[list(columns)].rename(columns=columns).copy()
    compact["date"] = pd.to_datetime(compact["date"]).dt.date.astype(str)
    compact["year"] = compact["year"].astype(int)
    compact["outbreak"] = compact["outbreak"].astype(int)
    for column in ["actual", "threshold", "p50", "p80", "p90"]:
        compact[column] = compact[column].astype(float).round(3)
    return compact.to_dict(orient="records")


def _latest_live_predictions(
    predictions: pd.DataFrame, model_version: str
) -> list[dict[str, Any]]:
    """Return only the newest row for each lead produced by the champion."""

    current = predictions[predictions["model_version"] == model_version].copy()
    if current.empty:
        return []
    current = current.sort_values("generated_at_utc").drop_duplicates(
        ["week_start_date", "lead_weeks"], keep="last"
    )
    keep = [
        "as_of_date",
        "week_start_date",
        "lead_weeks",
        "model_variant",
        "predicted_cases_p50",
        "predicted_cases_p80",
        "predicted_cases_p90",
        "case_forecast_reliability",
        "historical_outbreak_threshold_cases",
        "outbreak_probability",
        "alert_gate",
        "outbreak_alert",
        "case_data_age_weeks",
        "case_report_anchor_week",
        "case_report_age_weeks",
    ]
    return current[keep].sort_values("lead_weeks").to_dict(orient="records")


def build_explainer(output_path: Path = OUTPUT_PATH) -> Path:
    """Render the template with the committed champion and validation data."""

    metadata = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
    validation = pd.read_csv(VALIDATION_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    variants = set(validation["forecast_variant"])
    expected = {
        "weather_only",
        "weather_plus_delayed_cases",
        "weather_plus_recent_cases",
    }
    if variants != expected:
        raise ValueError(f"Unexpected validation variants: {sorted(variants)}")
    if validation.groupby("forecast_variant").size().to_dict() != {
        "weather_only": 125,
        "weather_plus_delayed_cases": 125,
        "weather_plus_recent_cases": 125,
    }:
        raise ValueError("Explainer expects 125 held-out rows per champion variant")

    replacements = {
        "__MODEL_METADATA__": _safe_script_json(metadata),
        "__VALIDATION_ROWS__": _safe_script_json(_validation_records(validation)),
        "__LIVE_PREDICTIONS__": _safe_script_json(
            _latest_live_predictions(predictions, metadata["model_version"])
        ),
    }
    page = TEMPLATE_PATH.read_text(encoding="utf-8")
    for token, value in replacements.items():
        if token not in page:
            raise ValueError(f"Template token is missing: {token}")
        page = page.replace(token, value)
    if "__MODEL_" in page or "__VALIDATION_" in page or "__LIVE_" in page:
        raise ValueError("One or more explainer tokens were not replaced")

    output_path.write_text(page, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(build_explainer())
