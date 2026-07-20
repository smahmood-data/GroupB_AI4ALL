"""Unit tests for the near-real-time outbreak detector's pure logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


# The production file is designed to run directly from ``src/``.  Adding that
# directory here tests the same import path a user gets from the README command.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from near_realtime_outbreak_detection import (  # noqa: E402
    DAILY_WEATHER_COLUMNS,
    add_expanding_outbreak_labels,
    add_weather_features,
    aggregate_daily_to_weeks,
    case_forecast_weather_feature_columns,
    choose_recall_gate,
    live_case_features,
    parse_open_meteo_daily,
    weather_feature_columns,
)


class NearRealtimeOutbreakTests(unittest.TestCase):
    """Protect the temporal and data-quality rules used by the live pipeline."""

    def test_daily_payload_is_aggregated_into_a_complete_week(self) -> None:
        times = pd.date_range("2026-07-06", periods=7, freq="D")
        payload = {
            "daily": {
                "time": [value.date().isoformat() for value in times],
                "temperature_2m_mean": [20, 21, 22, 23, 24, 25, 26],
                "temperature_2m_max": [25, 26, 27, 28, 29, 30, 31],
                "temperature_2m_min": [15, 16, 17, 18, 19, 20, 21],
                "precipitation_sum": [1, 2, 3, 4, 5, 6, 7],
                "rain_sum": [0, 0, 0.2, 0.1, 4, 0, 12],
                "precipitation_hours": [0, 1, 2, 0, 4, 0, 6],
                "relative_humidity_2m_mean": [70, 71, 72, 73, 74, 75, 76],
                "dew_point_2m_mean": [15, 16, 17, 18, 19, 20, 21],
                "soil_moisture_0_to_7cm_mean": [0.3] * 7,
                "et0_fao_evapotranspiration": [0.5] * 7,
            }
        }

        daily = parse_open_meteo_daily(payload, source="test")
        weekly = aggregate_daily_to_weeks(
            daily,
            [pd.Timestamp("2026-07-06")],
            as_of_date=pd.Timestamp("2026-07-08").date(),
        )

        self.assertEqual(int(weekly.loc[0, "weather_days_available"]), 7)
        self.assertEqual(int(weekly.loc[0, "forecast_input_days"]), 4)
        self.assertAlmostEqual(weekly.loc[0, "temperature_2m_mean"], 23.0)
        self.assertAlmostEqual(weekly.loc[0, "temperature_2m_max"], 31.0)
        self.assertAlmostEqual(weekly.loc[0, "temperature_2m_min"], 15.0)
        self.assertAlmostEqual(weekly.loc[0, "precipitation_sum"], 28.0)
        self.assertAlmostEqual(weekly.loc[0, "precipitation_hours"], 13.0)
        self.assertEqual(weekly.loc[0, "wet_day_count"], 4.0)
        self.assertEqual(weekly.loc[0, "heavy_rain_day_count"], 1.0)
        self.assertEqual(weekly.loc[0, "max_daily_rainfall"], 12.0)
        self.assertEqual(weekly.loc[0, "longest_dry_spell_days"], 2.0)
        self.assertAlmostEqual(weekly.loc[0, "et0_fao_evapotranspiration"], 3.5)

    def test_partial_week_does_not_create_a_partial_rainfall_total(self) -> None:
        times = pd.date_range("2026-07-06", periods=6, freq="D")
        payload = {"daily": {"time": [value.date().isoformat() for value in times]}}
        for column in DAILY_WEATHER_COLUMNS:
            payload["daily"][column] = [1.0] * 6

        daily = parse_open_meteo_daily(payload, source="test")
        weekly = aggregate_daily_to_weeks(daily, [pd.Timestamp("2026-07-06")])

        self.assertEqual(int(weekly.loc[0, "weather_days_available"]), 6)
        for column in DAILY_WEATHER_COLUMNS + [
            "wet_day_count",
            "heavy_rain_day_count",
            "max_daily_rainfall",
            "longest_dry_spell_days",
        ]:
            self.assertTrue(np.isnan(weekly.loc[0, column]), msg=column)

    def test_weather_lag_two_is_exactly_two_weather_rows(self) -> None:
        starts = pd.date_range("2026-01-05", periods=10, freq="7D")
        weekly = pd.DataFrame(
            {
                "week_start_date": starts,
                "weekofyear": np.arange(1, 11),
                "weather_days_available": 7,
                "forecast_input_days": 0,
                "weather_sources": "test",
            }
        )
        for column in DAILY_WEATHER_COLUMNS:
            weekly[column] = np.arange(10, dtype=float)
        weekly["wet_day_count"] = np.arange(10, dtype=float)
        weekly["heavy_rain_day_count"] = np.arange(10, dtype=float)
        weekly["max_daily_rainfall"] = np.arange(10, dtype=float)
        weekly["longest_dry_spell_days"] = np.arange(10, dtype=float)

        featured = add_weather_features(weekly)

        self.assertEqual(
            featured.loc[5, "temperature_2m_mean_lag_2"],
            weekly.loc[3, "temperature_2m_mean"],
        )
        self.assertEqual(
            featured.loc[5, "precipitation_sum_lag_0"],
            weekly.loc[5, "precipitation_sum"],
        )
        self.assertEqual(featured.loc[5, "heavy_rain_day_count_rolling_4"], 14.0)
        self.assertEqual(featured.loc[5, "longest_dry_spell_days_rolling_4"], 5.0)

    def test_rain_pattern_features_are_reserved_for_case_forecasting(self) -> None:
        """MAE additions must not silently change the recall-policy inputs."""

        classifier_columns = set(weather_feature_columns())
        case_forecast_columns = set(case_forecast_weather_feature_columns())

        self.assertNotIn("heavy_rain_day_count_lag_0", classifier_columns)
        self.assertNotIn("longest_dry_spell_days_lag_0", classifier_columns)
        self.assertIn("heavy_rain_day_count_lag_0", case_forecast_columns)
        self.assertIn("longest_dry_spell_days_lag_0", case_forecast_columns)
        self.assertTrue(classifier_columns < case_forecast_columns)

    def test_live_case_features_require_eight_exact_preceding_weeks(self) -> None:
        target = pd.Timestamp("2026-07-13")
        starts = [target - pd.Timedelta(weeks=lag) for lag in range(8, 0, -1)]
        cases = pd.DataFrame(
            {
                "city": "sj",
                "week_start_date": starts,
                "total_cases": np.arange(1, 9, dtype=float),
            }
        )

        complete = live_case_features(cases, target)
        self.assertIsNotNone(complete)
        assert complete is not None
        self.assertEqual(complete["cases_lag_1"], 8.0)
        self.assertEqual(complete["cases_lag_2"], 7.0)
        self.assertEqual(complete["cases_lag_4"], 5.0)
        self.assertEqual(complete["cases_mean_4"], 6.5)
        self.assertEqual(complete["cases_mean_8"], 4.5)

        missing_one_week = cases.iloc[:-1]
        self.assertIsNone(live_case_features(missing_one_week, target))

    def test_recall_gate_uses_held_out_probability_cutoff(self) -> None:
        probabilities = np.array([0.05, 0.10, 0.20, 0.40, 0.80, 0.90])
        labels = np.array([0, 0, 1, 0, 1, 1])

        gate = choose_recall_gate(probabilities, labels, target_recall=2 / 3)
        alerts = probabilities >= gate

        self.assertGreaterEqual(recall_score(labels, alerts), 2 / 3)
        self.assertGreaterEqual(gate, 0.20)

    def test_expanding_outbreak_labels_exclude_early_years(self) -> None:
        rows = []
        for year in range(2000, 2005):
            for week in range(1, 53):
                rows.append(
                    {
                        "city": "sj",
                        "year": year,
                        "weekofyear": week,
                        "week_start_date": pd.Timestamp(year, 1, 1)
                        + pd.Timedelta(weeks=week - 1),
                        "total_cases": float(week + (year - 2000)),
                    }
                )
        cases = pd.DataFrame(rows)

        labeled = add_expanding_outbreak_labels(cases, minimum_prior_years=3)

        self.assertEqual(labeled["year"].min(), 2003)
        self.assertIn("outbreak_threshold", labeled.columns)
        self.assertIn("outbreak_label", labeled.columns)


if __name__ == "__main__":
    unittest.main()
