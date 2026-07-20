"""Tests for the generated, self-contained Puerto Rico API model explainer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_api_model_explainer import build_explainer  # noqa: E402


class ApiModelExplainerTests(unittest.TestCase):
    """Keep the published explainer synchronized and structurally valid."""

    def test_committed_explainer_matches_current_model_artifacts(self) -> None:
        committed = PROJECT_ROOT / "docs" / "puerto-rico-api-model-explainer.html"
        with tempfile.TemporaryDirectory() as directory:
            generated = build_explainer(Path(directory) / "explainer.html")
            self.assertEqual(
                generated.read_text(encoding="utf-8"),
                committed.read_text(encoding="utf-8"),
            )

    def test_explainer_contains_api_results_charts_and_file_links(self) -> None:
        page = (
            PROJECT_ROOT / "docs" / "puerto-rico-api-model-explainer.html"
        ).read_text(encoding="utf-8")
        parser = HTMLParser()
        parser.feed(page)
        parser.close()

        required_content = [
            "API-only operational model",
            "MAE-focused case forecast",
            "Recall-focused outbreak alert",
            "The three models, with examples",
            "Weather + delayed cases",
            "newest report is May 18",
            'id="validationChart"',
            'id="maeChart"',
            'id="classifierChart"',
            "Weather + delayed case history",
            "All 375 time-held-out case forecasts",
            "src/puerto_rico_operational.py",
            "models/operational/pr/validation_predictions.csv",
            "data/operational/predictions.csv",
        ]
        for content in required_content:
            self.assertIn(content, page)

        self.assertNotIn("__MODEL_METADATA__", page)
        self.assertNotIn("@tailwindcss", page)
        self.assertNotIn("daisyui", page.lower())

    def test_champion_uses_the_documented_objective_specific_features(self) -> None:
        """Keep serialized model inputs aligned with the published manifest."""

        registry = PROJECT_ROOT / "models" / "operational" / "pr"
        metadata = json.loads(
            (registry / "champion.json").read_text(encoding="utf-8")
        )
        artifact = joblib.load(registry / "champion.joblib")
        manifest = metadata["feature_manifest"]

        self.assertEqual(
            artifact["weather_detector"]["feature_columns"], manifest["weather"]
        )
        self.assertEqual(
            artifact["case_detector"]["feature_columns"],
            manifest["weather"] + manifest["exact_health"],
        )
        self.assertEqual(
            artifact["weather_case_forecaster"]["feature_columns"],
            manifest["weather_case_forecast"],
        )
        self.assertEqual(
            artifact["recent_case_forecaster"]["feature_columns"],
            manifest["weather_case_forecast"] + manifest["exact_health"],
        )
        self.assertNotIn("heavy_rain_day_count_lag_0", manifest["weather"])
        self.assertIn(
            "heavy_rain_day_count_lag_0", manifest["weather_case_forecast"]
        )


if __name__ == "__main__":
    unittest.main()
