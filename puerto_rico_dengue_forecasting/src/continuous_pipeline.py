"""Command-line orchestration for the continuously updated dengue system.

Commands are deliberately small and composable so the same behavior can run
locally or in GitHub Actions:

``ingest``
    Download official Puerto Rico cases and record source revisions.
``weekly``
    Ingest cases, bootstrap a champion if needed, score two weeks, and monitor.
``monthly``
    Ingest, retrain only when enough finalized labels exist, apply guarded
    promotion, then score and monitor with the resulting champion.
``monitor``
    Reconcile old predictions once their target weeks are finalized.
``import-peru``
    Normalize a manually downloaded official Peru CSV into compact Iquitos
    weekly rows.  This is retained for the legacy city experiment; it does not
    change the Puerto Rico-wide operational target.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from official_case_data import (
    fetch_peru_catalog_status,
    fetch_puerto_rico_snapshot,
    merge_case_snapshot,
    normalize_peru_iquitos_csv,
    write_source_status,
)
from puerto_rico_operational import (
    CASE_FORECAST_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    append_predictions,
    finalized_cases,
    load_champion,
    load_official_cases,
    load_operations_config,
    promote_candidate,
    reconcile_predictions,
    score_champion,
    train_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "operations.json"
DEFAULT_CASES = PROJECT_ROOT / "data" / "operational" / "cases" / "pr_weekly.csv"
DEFAULT_REVISIONS = PROJECT_ROOT / "data" / "operational" / "cases" / "pr_revisions.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "operational" / "predictions.csv"
DEFAULT_SOURCE_STATUS = PROJECT_ROOT / "data" / "operational" / "source_status.json"
DEFAULT_METRICS = PROJECT_ROOT / "data" / "operational" / "monitoring" / "latest_metrics.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "operational" / "pr"
DEFAULT_REPORTS = PROJECT_ROOT / "reports" / "operational"
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "near_realtime"


def champion_metadata_is_compatible(metadata: dict[str, Any] | None) -> bool:
    """Return whether the current code can safely load a registry champion."""

    return bool(
        metadata is not None
        and metadata.get("model_schema_version") == MODEL_SCHEMA_VERSION
        and metadata.get("case_forecast_schema_version")
        == CASE_FORECAST_SCHEMA_VERSION
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic report, creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ingest_sources(config: dict[str, Any]) -> dict[str, Any]:
    """Refresh official source data without letting one optional source block PR."""

    snapshot = fetch_puerto_rico_snapshot(config["puerto_rico"]["catalog_id"])
    revisions = merge_case_snapshot(snapshot.weekly_cases, DEFAULT_CASES, DEFAULT_REVISIONS)
    status: dict[str, Any] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "puerto_rico": {
            "status": "ok",
            "source_file_id": snapshot.source_file_id,
            "publication_date": snapshot.publication_date,
            "retrieved_at_utc": snapshot.retrieved_at_utc,
            "normalized_weeks": int(len(snapshot.weekly_cases)),
            "new_or_revised_weeks": int(len(revisions)),
            "latest_complete_week": snapshot.weekly_cases["week_start_date"].max().date().isoformat(),
        },
    }
    try:
        status["peru"] = fetch_peru_catalog_status()
    except RuntimeError as exc:
        # Peru metadata is informational for the legacy Iquitos experiment.
        # It must not prevent the chosen Puerto Rico-wide pipeline from running.
        status["peru"] = {
            "status": "unavailable",
            "error": str(exc),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    write_source_status(DEFAULT_SOURCE_STATUS, status)
    return status


def train_and_guard(
    config: dict[str, Any],
    as_of: date,
    refresh: bool,
) -> dict[str, Any]:
    """Train only when useful, then apply the champion promotion policy."""

    cases = load_official_cases(DEFAULT_CASES)
    stable = finalized_cases(
        cases, as_of, int(config["label_stabilization_weeks"])
    )
    champion_path = DEFAULT_REGISTRY / "champion.json"
    champion_metadata = (
        json.loads(champion_path.read_text(encoding="utf-8"))
        if champion_path.exists()
        else None
    )
    if (
        champion_metadata_is_compatible(champion_metadata)
    ):
        new_weeks = int(
            (
                stable["week_start_date"].max()
                - pd.Timestamp(champion_metadata["training_data_cutoff"])
            ).days
            // 7
        )
        minimum = int(config["minimum_new_finalized_weeks"])
        if new_weeks < minimum:
            result = {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "promoted": False,
                "candidate_trained": False,
                "scoring_ready": True,
                "reasons": [
                    f"Only {new_weeks} new finalized weeks; {minimum} are required before retraining"
                ],
                "current_champion": champion_metadata,
            }
            name = f"pr_retraining_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            _write_json(DEFAULT_REPORTS / name, result)
            return result

    artifact, metadata = train_candidate(stable, config, DEFAULT_CACHE, refresh)
    decision = promote_candidate(
        artifact, metadata, DEFAULT_REGISTRY, DEFAULT_REPORTS, config
    )
    return {
        "candidate_trained": True,
        "candidate_version": metadata["model_version"],
        "promoted": decision.promote,
        # If a schema migration is rejected, the existing champion may be
        # intentionally incompatible with the new scoring code.  Report that
        # state instead of crashing after the guardrail did its job.
        "scoring_ready": decision.promote
        or champion_metadata_is_compatible(champion_metadata),
        "reasons": decision.reasons,
    }


def predict_and_monitor(
    config: dict[str, Any], as_of: date, refresh: bool
) -> tuple[Any, dict[str, Any]]:
    """Score the champion, append idempotently, and reconcile finalized truth."""

    cases = load_official_cases(DEFAULT_CASES)
    champion = load_champion(DEFAULT_REGISTRY)
    new_rows = score_champion(
        champion, cases, config, DEFAULT_CACHE, as_of=as_of, refresh=refresh
    )
    append_predictions(new_rows, DEFAULT_PREDICTIONS)
    metrics = reconcile_predictions(
        DEFAULT_PREDICTIONS,
        cases,
        as_of,
        int(config["label_stabilization_weeks"]),
        DEFAULT_METRICS,
    )
    return new_rows, metrics


def build_parser() -> argparse.ArgumentParser:
    """Define one documented CLI for local runs and automation."""

    parser = argparse.ArgumentParser(description="Run the operational dengue pipeline")
    parser.add_argument(
        "command", choices=["ingest", "weekly", "monthly", "monitor", "import-peru"]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--as-of", type=date.fromisoformat, help="Local date in YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true", help="Ignore fresh weather caches")
    parser.add_argument("--csv", type=Path, help="Official Peru CSV for import-peru")
    return parser


def main() -> None:
    """Execute the requested operational stage and print a concise audit trail."""

    args = build_parser().parse_args()
    config = load_operations_config(args.config)
    as_of = args.as_of or date.today()

    if args.command == "import-peru":
        if args.csv is None:
            raise ValueError("import-peru requires --csv PATH")
        normalized = normalize_peru_iquitos_csv(args.csv)
        output = PROJECT_ROOT / "data" / "operational" / "cases" / "iq_weekly.csv"
        revisions = PROJECT_ROOT / "data" / "operational" / "cases" / "iq_revisions.csv"
        changes = merge_case_snapshot(normalized, output, revisions)
        print(f"Imported {len(normalized)} Iquitos weeks ({len(changes)} new/revised).")
        return

    if args.command in {"ingest", "weekly", "monthly"}:
        status = ingest_sources(config)
        print(
            "Puerto Rico official cases: "
            f"{status['puerto_rico']['normalized_weeks']} complete weeks, "
            f"latest {status['puerto_rico']['latest_complete_week']}."
        )
    if args.command == "ingest":
        return

    training: dict[str, Any] | None = None
    if args.command == "monthly" or (
        args.command == "weekly" and not (DEFAULT_REGISTRY / "champion.joblib").exists()
    ):
        training = train_and_guard(config, as_of, args.refresh)
        print("Guarded promotion:", json.dumps(training, indent=2))
        if not training["scoring_ready"]:
            print(
                "Prediction skipped: the candidate was rejected and the existing "
                "champion uses an older, incompatible schema. The rejection report "
                "was preserved for review."
            )
            return

    if args.command in {"weekly", "monthly"}:
        predictions, metrics = predict_and_monitor(config, as_of, args.refresh)
        display = predictions[
            [
                "week_start_date",
                "lead_weeks",
                "model_variant",
                "predicted_cases_p50",
                "predicted_cases_p80",
                "predicted_cases_p90",
                "case_forecast_reliability",
                "historical_outbreak_threshold_cases",
                "case_risk_level",
                "outbreak_probability",
                "alert_gate",
                "outbreak_alert",
            ]
        ].copy()
        print("\nPuerto Rico-wide research alerts")
        print(display.to_string(index=False))
        print("\nMonitoring:", json.dumps(metrics, indent=2))
        return

    cases = load_official_cases(DEFAULT_CASES)
    metrics = reconcile_predictions(
        DEFAULT_PREDICTIONS,
        cases,
        as_of,
        int(config["label_stabilization_weeks"]),
        DEFAULT_METRICS,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
