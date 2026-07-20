"""Tests for official ingestion, island aggregation, and promotion safeguards."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import brotli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from official_case_data import (  # noqa: E402
    _decode_payload,
    merge_case_snapshot,
    normalize_puerto_rico_daily_cases,
)
from continuous_pipeline import champion_metadata_is_compatible  # noqa: E402
from puerto_rico_operational import (  # noqa: E402
    CASE_FORECAST_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    CaseForecastBundle,
    _case_forecast_metrics,
    add_delayed_health_features,
    evaluate_promotion,
    finalized_cases,
    live_delayed_health_features,
    spatially_aggregate_daily_weather,
)


class OfficialCaseDataTests(unittest.TestCase):
    """Protect the rules that keep incomplete or revised labels auditable."""

    def test_brotli_encoded_official_payload_is_decoded(self) -> None:
        payload = b'[{"diagnosticDate":"2026-01-05"}]'
        encoded = brotli.compress(payload)

        self.assertEqual(_decode_payload(encoded, {"content-encoding": "br"}), payload)

    def test_pr_normalization_keeps_only_complete_monday_weeks(self) -> None:
        dates = pd.date_range("2026-01-05", periods=14, freq="D")
        records = [
            {
                "diagnosticDate": day.date().isoformat(),
                "totalCasesPcrCount": 2,
                "totalCasesIgMCount": 1,
                "totalHospitalizedCount": 1,
            }
            for day in dates
        ]
        # A missing IgM value makes the second week incomplete even though all
        # dates and PCR values exist.
        records[-1]["totalCasesIgMCount"] = None
        weekly = normalize_puerto_rico_daily_cases(
            records, "file-1", "2026-02-01", "2026-02-02T00:00:00+00:00"
        )

        self.assertEqual(len(weekly), 1)
        self.assertEqual(weekly.loc[0, "week_start_date"], pd.Timestamp("2026-01-05"))
        self.assertEqual(int(weekly.loc[0, "total_cases"]), 21)

    def test_case_merge_records_a_revision_before_replacement(self) -> None:
        base = pd.DataFrame(
            {
                "geography": ["pr"],
                "week_start_date": [pd.Timestamp("2026-01-05")],
                "total_cases": [10],
                "pcr_cases": [7],
                "igm_cases": [3],
                "hospitalized_cases": [2],
                "complete_week": [True],
                "source_file_id": ["old"],
                "source_publication_date": ["2026-02-01"],
                "retrieved_at_utc": ["2026-02-02T00:00:00+00:00"],
                "source_page": ["official"],
            }
        )
        revised = base.copy()
        revised["total_cases"] = 12
        revised["source_file_id"] = "new"
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.csv"
            revisions = Path(directory) / "revisions.csv"
            merge_case_snapshot(base, current, revisions)
            changes = merge_case_snapshot(revised, current, revisions)

            self.assertEqual(changes.loc[0, "change_type"], "revision")
            self.assertEqual(int(changes.loc[0, "old_total_cases"]), 10)
            self.assertEqual(int(changes.loc[0, "new_total_cases"]), 12)
            self.assertEqual(int(pd.read_csv(current).loc[0, "total_cases"]), 12)

    def test_unchanged_case_snapshot_preserves_canonical_retrieval_time(self) -> None:
        snapshot = pd.DataFrame(
            {
                "geography": ["pr"],
                "week_start_date": [pd.Timestamp("2026-01-05")],
                "total_cases": [10],
                "pcr_cases": [7],
                "igm_cases": [3],
                "hospitalized_cases": [2],
                "complete_week": [True],
                "source_file_id": ["same-file"],
                "source_publication_date": ["2026-02-01"],
                "retrieved_at_utc": ["2026-02-02T00:00:00+00:00"],
                "source_page": ["official"],
            }
        )
        refreshed = snapshot.copy()
        refreshed["retrieved_at_utc"] = "2026-02-09T00:00:00+00:00"

        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.csv"
            revisions = Path(directory) / "revisions.csv"
            merge_case_snapshot(snapshot, current, revisions)
            changes = merge_case_snapshot(refreshed, current, revisions)
            stored = pd.read_csv(current)

        self.assertTrue(changes.empty)
        self.assertEqual(
            stored.loc[0, "retrieved_at_utc"], "2026-02-02T00:00:00+00:00"
        )

    def test_same_total_with_revised_components_still_updates_case_table(self) -> None:
        snapshot = pd.DataFrame(
            {
                "geography": ["pr"],
                "week_start_date": [pd.Timestamp("2026-01-05")],
                "total_cases": [10],
                "pcr_cases": [7],
                "igm_cases": [3],
                "hospitalized_cases": [2],
                "complete_week": [True],
                "source_file_id": ["same-file"],
                "source_publication_date": ["2026-02-01"],
                "retrieved_at_utc": ["2026-02-02T00:00:00+00:00"],
                "source_page": ["official"],
            }
        )
        revised_components = snapshot.copy()
        revised_components["pcr_cases"] = 6
        revised_components["igm_cases"] = 4

        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.csv"
            revisions = Path(directory) / "revisions.csv"
            merge_case_snapshot(snapshot, current, revisions)
            merge_case_snapshot(revised_components, current, revisions)
            stored = pd.read_csv(current)

        self.assertEqual(int(stored.loc[0, "pcr_cases"]), 6)
        self.assertEqual(int(stored.loc[0, "igm_cases"]), 4)

    def test_hospitalization_revision_is_audited_even_when_cases_do_not_change(self) -> None:
        snapshot = pd.DataFrame(
            {
                "geography": ["pr"],
                "week_start_date": [pd.Timestamp("2026-01-05")],
                "total_cases": [10],
                "pcr_cases": [7],
                "igm_cases": [3],
                "hospitalized_cases": [2],
                "complete_week": [True],
                "source_file_id": ["same-file"],
                "source_publication_date": ["2026-02-01"],
                "retrieved_at_utc": ["2026-02-02T00:00:00+00:00"],
                "source_page": ["official"],
            }
        )
        revised = snapshot.copy()
        revised["hospitalized_cases"] = 4

        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.csv"
            revisions = Path(directory) / "revisions.csv"
            merge_case_snapshot(snapshot, current, revisions)
            changes = merge_case_snapshot(revised, current, revisions)

        self.assertEqual(int(changes.loc[0, "old_hospitalized_cases"]), 2)
        self.assertEqual(int(changes.loc[0, "new_hospitalized_cases"]), 4)
        self.assertEqual(changes.loc[0, "change_type"], "revision")


class PuertoRicoOperationalTests(unittest.TestCase):
    """Protect geographic coverage, label delay, and promotion policy."""

    def test_champion_compatibility_requires_both_schema_versions(self) -> None:
        current = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "case_forecast_schema_version": CASE_FORECAST_SCHEMA_VERSION,
        }
        self.assertTrue(champion_metadata_is_compatible(current))
        self.assertFalse(
            champion_metadata_is_compatible(
                {
                    **current,
                    "case_forecast_schema_version": CASE_FORECAST_SCHEMA_VERSION - 1,
                }
            )
        )

    def test_delayed_features_use_the_report_anchor_not_the_target_lag_one(self) -> None:
        starts = pd.date_range("2026-01-05", periods=20, freq="7D")
        cases = pd.DataFrame(
            {
                "week_start_date": starts,
                "total_cases": np.arange(100, 120, dtype=float),
                "hospitalized_cases": np.arange(10, 30, dtype=float),
            }
        )

        featured = add_delayed_health_features(cases, [8])
        target = featured[featured["week_start_date"] == starts[15]].iloc[0]

        self.assertEqual(target["delayed_cases_latest"], 107.0)
        self.assertEqual(target["delayed_cases_lag_1"], 106.0)
        self.assertEqual(target["delayed_hospitalized_latest"], 17.0)
        self.assertEqual(target["report_age_weeks"], 8.0)

    def test_live_delayed_features_require_contiguous_history_and_report_age(self) -> None:
        target = pd.Timestamp("2026-07-13")
        anchor = target - pd.Timedelta(weeks=8)
        starts = [anchor - pd.Timedelta(weeks=lag) for lag in range(7, -1, -1)]
        cases = pd.DataFrame(
            {
                "week_start_date": starts,
                "total_cases": np.arange(1, 9, dtype=float),
                "hospitalized_cases": np.arange(0, 8, dtype=float),
            }
        )

        result = live_delayed_health_features(cases, target, 4, 12)
        self.assertIsNotNone(result)
        assert result is not None
        features, selected_anchor = result
        self.assertEqual(selected_anchor, anchor)
        self.assertEqual(features["delayed_cases_latest"], 8.0)
        self.assertEqual(features["delayed_cases_lag_1"], 7.0)
        self.assertEqual(features["report_age_weeks"], 8.0)

        missing_week = cases.drop(index=4)
        self.assertIsNone(
            live_delayed_health_features(missing_week, target, 4, 12)
        )

    def test_case_forecast_bundle_enforces_nonnegative_ordered_quantiles(self) -> None:
        class ConstantModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.array([self.value] * len(frame))

        bundle = CaseForecastBundle(
            models={
                0.50: ConstantModel(20),
                0.80: ConstantModel(15),
                0.90: ConstantModel(-3),
            },
            feature_columns=["feature"],
            validation_metrics={},
            validation_predictions=[],
        )

        forecast = bundle.predict({"feature": 1})

        self.assertEqual(forecast, {"p50": 20.0, "p80": 20.0, "p90": 20.0})

    def test_case_forecast_bundle_adds_residual_to_last_observed_count(self) -> None:
        class ConstantResidualModel:
            def __init__(self, change: float) -> None:
                self.change = change

            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.array([self.change] * len(frame))

        bundle = CaseForecastBundle(
            models={
                0.50: ConstantResidualModel(-5),
                0.80: ConstantResidualModel(2),
                0.90: ConstantResidualModel(8),
            },
            feature_columns=["cases_lag_1", "weather"],
            validation_metrics={},
            validation_predictions=[],
            baseline_column="cases_lag_1",
        )

        forecast = bundle.predict({"cases_lag_1": 40, "weather": 1})

        self.assertEqual(forecast, {"p50": 35.0, "p80": 42.0, "p90": 48.0})

    def test_case_forecast_bundle_shrinks_uncertain_delayed_residuals(self) -> None:
        class ConstantResidualModel:
            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.array([20.0] * len(frame))

        bundle = CaseForecastBundle(
            models={quantile: ConstantResidualModel() for quantile in (0.5, 0.8, 0.9)},
            feature_columns=["delayed_cases_latest"],
            validation_metrics={},
            validation_predictions=[],
            baseline_column="delayed_cases_latest",
            residual_scales={0.5: 0.0, 0.8: 0.5, 0.9: 1.0},
        )

        forecast = bundle.predict({"delayed_cases_latest": 40.0})

        self.assertEqual(forecast, {"p50": 40.0, "p80": 50.0, "p90": 60.0})

    def test_case_forecast_bundle_keeps_calibrated_upper_range(self) -> None:
        class ZeroResidualModel:
            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(frame))

        bundle = CaseForecastBundle(
            models={quantile: ZeroResidualModel() for quantile in (0.5, 0.8, 0.9)},
            feature_columns=["delayed_cases_latest"],
            validation_metrics={},
            validation_predictions=[],
            baseline_column="delayed_cases_latest",
            residual_scales={0.5: 0.0, 0.8: 0.0, 0.9: 0.0},
            upper_offsets={0.8: 12.0, 0.9: 25.0},
        )

        forecast = bundle.predict({"delayed_cases_latest": 40.0})

        self.assertEqual(forecast, {"p50": 40.0, "p80": 52.0, "p90": 65.0})

    def test_case_forecast_metrics_separate_normal_and_outbreak_mae(self) -> None:
        predictions = pd.DataFrame(
            {
                "actual_cases": [10, 20, 100, 120],
                "predicted_cases_p50": [12, 18, 80, 90],
                "predicted_cases_p80": [15, 25, 110, 125],
                "predicted_cases_p90": [20, 30, 130, 140],
                "actual_outbreak": [0, 0, 1, 1],
                "outbreak_threshold": [40, 40, 40, 40],
                "cases_lag_1": [9, 19, 90, 110],
            }
        )

        metrics = _case_forecast_metrics(predictions)

        self.assertAlmostEqual(metrics["mae"], 13.5)
        self.assertAlmostEqual(metrics["normal_week_mae"], 2.0)
        self.assertAlmostEqual(metrics["outbreak_week_mae"], 25.0)
        self.assertEqual(metrics["p90_coverage"], 1.0)

    def test_spatial_weather_uses_all_points_and_preserves_extremes(self) -> None:
        dates = pd.date_range("2026-01-01", periods=2, freq="D")
        frames = []
        for point in range(3):
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "temperature_2m_mean": [20 + point, 21 + point],
                        "temperature_2m_max": [25 + point, 26 + point],
                        "temperature_2m_min": [15 - point, 16 - point],
                        "precipitation_sum": [point, point + 1],
                        "rain_sum": [point, point + 1],
                        "precipitation_hours": [point + 1, point + 2],
                        "relative_humidity_2m_mean": [70 + point, 71 + point],
                        "dew_point_2m_mean": [17 + point, 18 + point],
                        "soil_moisture_0_to_7cm_mean": [0.2 + point / 100] * 2,
                        "et0_fao_evapotranspiration": [1 + point, 2 + point],
                        "weather_source": f"point-{point}",
                        "weather_point": f"point-{point}",
                    }
                )
            )
        island = spatially_aggregate_daily_weather(frames, expected_points=3)

        self.assertEqual(int(island.loc[0, "point_count"]), 3)
        self.assertEqual(island.loc[0, "temperature_2m_mean"], 21)
        self.assertEqual(island.loc[0, "temperature_2m_max"], 27)
        self.assertEqual(island.loc[0, "temperature_2m_min"], 13)
        self.assertEqual(island.loc[0, "precipitation_sum"], 1)

        frames[-1].loc[1, "precipitation_sum"] = np.nan
        incomplete = spatially_aggregate_daily_weather(frames, expected_points=3)
        self.assertTrue(np.isnan(incomplete.loc[1, "temperature_2m_mean"]))

    def test_finalized_cases_exclude_four_week_reporting_buffer(self) -> None:
        starts = pd.date_range("2026-05-04", periods=8, freq="7D")
        cases = pd.DataFrame({"week_start_date": starts, "total_cases": 1})
        stable = finalized_cases(cases, date(2026, 7, 1), stabilization_weeks=4)

        self.assertEqual(stable["week_start_date"].max(), pd.Timestamp("2026-05-25"))

    def test_guarded_promotion_requires_new_labels_and_metric_non_regression(self) -> None:
        config = {
            "minimum_new_finalized_weeks": 13,
            "promotion_guardrails": {
                "precision_tolerance": 0.03,
                "recall_tolerance": 0.03,
                "pr_auc_tolerance": 0.02,
                "brier_tolerance": 0.02,
            },
        }
        champion = {
            "training_data_cutoff": "2025-01-06",
            "weather_only_metrics": {
                "precision": 0.40,
                "recall": 0.80,
                "pr_auc": 0.50,
                "brier": 0.20,
            },
        }
        candidate = {
            "training_data_cutoff": "2025-04-07",
            "weather_only_metrics": {
                "precision": 0.38,
                "recall": 0.78,
                "pr_auc": 0.49,
                "brier": 0.21,
            },
        }
        self.assertTrue(evaluate_promotion(candidate, champion, config).promote)

        candidate["weather_only_metrics"]["recall"] = 0.70
        decision = evaluate_promotion(candidate, champion, config)
        self.assertFalse(decision.promote)
        self.assertTrue(any("recall guardrail: fail" in reason for reason in decision.reasons))

    def test_bootstrap_candidate_promotes_without_a_champion(self) -> None:
        decision = evaluate_promotion(
            {"training_data_cutoff": "2026-01-01"},
            None,
            {"minimum_new_finalized_weeks": 13, "promotion_guardrails": {}},
        )
        self.assertTrue(decision.promote)

    def test_schema_upgrade_can_establish_first_case_mae_baseline(self) -> None:
        common_metrics = {
            "precision": 0.80,
            "recall": 0.90,
            "pr_auc": 0.75,
            "brier": 0.17,
        }
        champion = {
            "model_schema_version": 2,
            "training_data_cutoff": "2026-05-18",
            "weather_only_metrics": common_metrics,
        }
        candidate = {
            "model_schema_version": 3,
            "training_data_cutoff": "2026-05-18",
            "weather_only_metrics": common_metrics,
            "weather_only_case_forecast_metrics": {
                "mae": 20.0,
                "outbreak_week_mae": 30.0,
            },
            "weather_plus_recent_cases_case_forecast_metrics": {
                "mae": 10.0,
                "outbreak_week_mae": 12.0,
                "persistence_mae": 11.0,
            },
        }
        config = {
            "minimum_new_finalized_weeks": 13,
            "promotion_guardrails": {
                "precision_tolerance": 0.03,
                "recall_tolerance": 0.03,
                "pr_auc_tolerance": 0.02,
                "brier_tolerance": 0.02,
                "case_mae_relative_tolerance": 0.10,
                "outbreak_mae_relative_tolerance": 0.15,
            },
        }

        decision = evaluate_promotion(candidate, champion, config)

        self.assertTrue(decision.promote)
        self.assertTrue(any("baselines established" in reason for reason in decision.reasons))

    def test_later_candidate_is_rejected_when_case_mae_regresses(self) -> None:
        classifier_metrics = {
            "precision": 0.80,
            "recall": 0.90,
            "pr_auc": 0.75,
            "brier": 0.17,
        }
        champion = {
            "model_schema_version": 3,
            "case_forecast_schema_version": 1,
            "training_data_cutoff": "2026-01-05",
            "weather_only_metrics": classifier_metrics,
            "weather_only_case_forecast_metrics": {
                "mae": 20.0,
                "outbreak_week_mae": 30.0,
            },
            "weather_plus_recent_cases_case_forecast_metrics": {
                "mae": 10.0,
                "outbreak_week_mae": 12.0,
                "persistence_mae": 11.0,
            },
        }
        candidate = {
            **champion,
            "training_data_cutoff": "2026-04-06",
            "weather_only_case_forecast_metrics": {
                "mae": 23.0,
                "outbreak_week_mae": 30.0,
            },
        }
        config = {
            "minimum_new_finalized_weeks": 13,
            "promotion_guardrails": {
                "precision_tolerance": 0.03,
                "recall_tolerance": 0.03,
                "pr_auc_tolerance": 0.02,
                "brier_tolerance": 0.02,
                "case_mae_relative_tolerance": 0.10,
                "outbreak_mae_relative_tolerance": 0.15,
            },
        }

        decision = evaluate_promotion(candidate, champion, config)

        self.assertFalse(decision.promote)
        self.assertTrue(
            any("case_mae guardrail: fail" in reason for reason in decision.reasons)
        )


if __name__ == "__main__":
    unittest.main()
