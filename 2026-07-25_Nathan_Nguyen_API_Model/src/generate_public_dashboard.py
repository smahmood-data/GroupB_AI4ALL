"""Build the small, public data snapshot consumed by the dashboard.

The operational pipeline keeps an append-only prediction ledger so that every
issued forecast can later be audited.  A public website should not download
that entire ledger or decide which duplicated row is current in the browser.
This module performs that selection once, strips internal-only fields, and
writes a stable JSON contract under ``docs/app/data``.

The generator deliberately uses only Python's standard library.  GitHub Actions
can therefore refresh the dashboard after every successful weekly model run
without adding another build environment or a second dependency lockfile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data/operational/predictions.csv"
DEFAULT_CASES = PROJECT_ROOT / "data/operational/cases/pr_weekly.csv"
DEFAULT_SOURCE_STATUS = PROJECT_ROOT / "data/operational/source_status.json"
DEFAULT_MONITORING = PROJECT_ROOT / "data/operational/monitoring/latest_metrics.json"
DEFAULT_MODEL_METADATA = PROJECT_ROOT / "models/operational/pr/champion.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/app/data/dashboard.json"
DEFAULT_SCRIPT_OUTPUT = PROJECT_ROOT / "docs/app/data/dashboard-data.js"


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries while preserving the source column names."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path, default: Any) -> Any:
    """Return parsed JSON or a caller-provided fallback when a file is absent."""

    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    """Convert CSV text to a finite float; blank, NaN, and infinity become null."""

    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    """Convert a numeric field to an integer without inventing missing values."""

    parsed = _number(value)
    return None if parsed is None else int(round(parsed))


def _boolean(value: Any) -> bool | None:
    """Parse the explicit boolean strings written by the operational pipeline."""

    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _iso_datetime(value: str | None) -> datetime | None:
    """Parse the pipeline's ISO timestamps, including a trailing ``Z``."""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _select_latest_issue(
    prediction_rows: Iterable[dict[str, str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Select the newest issuance containing both current- and next-week rows.

    ``predictions.csv`` is intentionally append-only.  Sorting individual rows
    would be unsafe because a partially written issuance could contain only one
    horizon.  Grouping by ``generated_at_utc`` ensures the dashboard switches
    atomically to a complete forecast pair.
    """

    issuances: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        if row.get("geography") == "pr" and row.get("generated_at_utc"):
            issuances[row["generated_at_utc"]].append(row)

    for issued_at in sorted(issuances, reverse=True):
        rows_by_lead = {row.get("lead_weeks"): row for row in issuances[issued_at]}
        if "0" in rows_by_lead and "1" in rows_by_lead:
            return rows_by_lead["0"], rows_by_lead["1"]

    raise ValueError(
        "No complete Puerto Rico issuance contains both lead 0 and lead 1 rows."
    )


def _forecast_view(row: dict[str, str]) -> dict[str, Any]:
    """Keep only public, interpretable forecast fields from a ledger row."""

    return {
        "weekStart": row.get("week_start_date"),
        "leadWeeks": _integer(row.get("lead_weeks")),
        "timeScope": row.get("time_scope"),
        "cases": {
            "p50": _number(row.get("predicted_cases_p50")),
            "p80": _number(row.get("predicted_cases_p80")),
            "p90": _number(row.get("predicted_cases_p90")),
            "seasonalThreshold": _number(
                row.get("historical_outbreak_threshold_cases")
            ),
            "riskLevel": row.get("case_risk_level"),
            "reliability": row.get("case_forecast_reliability"),
        },
        "outbreak": {
            "probability": _number(row.get("outbreak_probability")),
            "alertGate": _number(row.get("alert_gate")),
            "alert": _boolean(row.get("outbreak_alert")),
            "definition": row.get("outbreak_definition"),
        },
    }


def _case_history(
    case_rows: Iterable[dict[str, str]], weeks: int | None = None
) -> list[dict]:
    """Return complete official weeks in chronological order.

    The public file is still small enough to include the full island-wide
    history.  Supplying ``weeks`` remains useful for focused tests or future
    consumers that need a smaller window.
    """

    usable = [
        row
        for row in case_rows
        if row.get("geography") == "pr"
        and _boolean(row.get("complete_week")) is True
        and row.get("week_start_date")
    ]
    usable.sort(key=lambda row: row["week_start_date"])
    selected_rows = usable if weeks is None else usable[-weeks:]
    return [
        {
            "weekStart": row["week_start_date"],
            "totalCases": _integer(row.get("total_cases")),
            "pcrCases": _integer(row.get("pcr_cases")),
            "igmCases": _integer(row.get("igm_cases")),
            "hospitalizedCases": _integer(row.get("hospitalized_cases")),
        }
        for row in selected_rows
    ]


def _freshness_level(case_age_weeks: float | None) -> str:
    """Create a simple display state without hiding the underlying age number."""

    if case_age_weeks is None:
        return "unavailable"
    if case_age_weeks <= 2:
        return "current"
    if case_age_weeks <= 4:
        return "delayed"
    return "stale"


def build_dashboard_payload(
    predictions_path: Path = DEFAULT_PREDICTIONS,
    cases_path: Path = DEFAULT_CASES,
    source_status_path: Path = DEFAULT_SOURCE_STATUS,
    monitoring_path: Path = DEFAULT_MONITORING,
    model_metadata_path: Path = DEFAULT_MODEL_METADATA,
) -> dict[str, Any]:
    """Assemble the complete versioned public dashboard payload."""

    current_row, outlook_row = _select_latest_issue(_read_csv(predictions_path))
    case_rows = _read_csv(cases_path)
    source_status = _read_json(source_status_path, {})
    monitoring = _read_json(monitoring_path, {})
    model_metadata = _read_json(model_metadata_path, {})

    generated_at = current_row.get("generated_at_utc")
    generated_dt = _iso_datetime(generated_at)
    generated_date = generated_dt.date() if generated_dt else date.today()
    case_age = _number(current_row.get("case_data_age_weeks"))
    pr_source = source_status.get("puerto_rico", {})

    # Metrics repeated on both forecast rows describe the same fitted champion.
    # Reading them from the current row keeps the public snapshot compact.
    held_out = {
        "caseMae": _number(current_row.get("held_out_case_mae")),
        "normalWeekMae": _number(current_row.get("held_out_normal_week_mae")),
        "outbreakWeekMae": _number(
            current_row.get("held_out_outbreak_week_mae")
        ),
        "precision": _number(current_row.get("held_out_precision")),
        "recall": _number(current_row.get("held_out_recall")),
        "f1": _number(current_row.get("held_out_f1")),
        "prAuc": _number(current_row.get("held_out_pr_auc")),
        "brier": _number(current_row.get("held_out_brier")),
        "basis": "historical_proxy_expanding_time_validation",
    }

    # The prediction ledger stores the point-error and outbreak metrics needed
    # for weekly auditing, while the champion metadata stores the validation
    # results for the two upper case estimates. Include those coverage results
    # only when the metadata describes the exact model version being displayed.
    # This prevents an older model's results from being paired with a new
    # forecast during an incomplete deployment.
    metric_variant = {
        "weather_only": "weather_only",
        "weather_plus_delayed_cases": "weather_plus_delayed_cases",
        "weather_plus_exact_cases": "weather_plus_recent_cases",
    }.get(current_row.get("model_variant"))
    case_metrics: dict[str, Any] = {}
    if (
        metric_variant
        and model_metadata.get("model_version") == current_row.get("model_version")
    ):
        case_metrics = model_metadata.get(
            f"{metric_variant}_case_forecast_metrics", {}
        )
    held_out.update(
        {
            "caseTestWeeks": _integer(case_metrics.get("held_out_rows")),
            "lowerHighEstimateCoverage": _number(
                case_metrics.get("p80_coverage")
            ),
            "upperHighEstimateCoverage": _number(
                case_metrics.get("p90_coverage")
            ),
        }
    )

    prospective = monitoring.get("prospective_information_available", monitoring)

    return {
        "schemaVersion": 1,
        "geography": {"code": "pr", "name": "Puerto Rico"},
        "issuedAt": generated_at,
        "asOfDate": current_row.get("as_of_date"),
        "publication": {
            "date": generated_date.isoformat(),
            "researchUseOnly": True,
            "officialSourceUrl": "https://datos.salud.pr.gov",
        },
        "signal": {
            "currentWeek": _forecast_view(current_row),
            "nextWeek": _forecast_view(outlook_row),
        },
        "history": _case_history(case_rows),
        "freshness": {
            "level": _freshness_level(case_age),
            "latestOfficialCaseWeek": current_row.get("latest_case_week"),
            "officialCaseAgeWeeks": case_age,
            "caseSourcePublicationDate": current_row.get(
                "case_source_publication_date"
            )
            or pr_source.get("publication_date"),
            "sourceCheckedAt": source_status.get("checked_at_utc"),
            "sourceStatus": pr_source.get("status", "unknown"),
            "weatherDaysAvailable": _integer(
                current_row.get("weather_days_available")
            ),
            "forecastInputDays": _integer(current_row.get("forecast_input_days")),
        },
        "model": {
            "version": current_row.get("model_version"),
            "variant": current_row.get("model_variant"),
            "trainingDataCutoff": current_row.get("training_data_cutoff"),
            "inputVintageBasis": current_row.get("input_vintage_basis"),
            "heldOut": held_out,
            "prospective": {
                "evaluatedRows": prospective.get("evaluated_rows", 0),
                "message": prospective.get("message"),
                "basis": prospective.get("evaluation_basis"),
            },
        },
    }


def write_dashboard_payload(payload: dict[str, Any], output_path: Path) -> None:
    """Write deterministic, readable JSON for review and source-control diffs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_dashboard_script(payload: dict[str, Any], output_path: Path) -> None:
    """Write the same snapshot as JavaScript for server-free local viewing.

    Browsers normally block ``fetch()`` calls from a ``file://`` page.  The
    generated assignment lets a reviewer double-click ``docs/app/index.html``
    without running a local web server, while deployed copies can still fetch
    the standalone JSON snapshot normally.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    output_path.write_text(
        f"window.DENGUE_DASHBOARD_DATA = {serialized};\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse the optional output location used by tests and local previews."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON file (default: docs/app/data/dashboard.json)",
    )
    parser.add_argument(
        "--script-output",
        type=Path,
        default=DEFAULT_SCRIPT_OUTPUT,
        help=(
            "Destination browser fallback "
            "(default: docs/app/data/dashboard-data.js)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Generate the dashboard snapshot and print its location."""

    args = parse_args()
    payload = build_dashboard_payload()
    write_dashboard_payload(payload, args.output)
    write_dashboard_script(payload, args.script_output)
    print(f"Wrote public dashboard data to {args.output}")
    print(f"Wrote server-free dashboard data to {args.script_output}")


if __name__ == "__main__":
    main()
