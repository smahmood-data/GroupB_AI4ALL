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

from generate_api_model_explainer import (  # noqa: E402
    build_explainer,
    build_focused_explainer,
)


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
            "Score only what was knowable at forecast time",
            "Historical proxy vs. archived issue-time weather",
            "Prospective deployment score",
            "Updating and retraining are separate",
            "Weekly forecast update",
            "Monthly retraining check",
            "No forced training on unchanged data",
            "Failure behavior",
            "data/operational/vintages/",
            "pr_information_available_validation_latest.json",
            "All 375 time-held-out case forecasts",
            "src/puerto_rico_operational.py",
            "models/operational/pr/validation_predictions.csv",
            "data/operational/predictions.csv",
            "automation/puerto-rico-dengue-model.yml",
        ]
        for content in required_content:
            self.assertIn(content, page)

        self.assertNotIn("__MODEL_METADATA__", page)
        self.assertNotIn("@tailwindcss", page)
        self.assertNotIn("daisyui", page.lower())

    def test_focused_explainer_is_generated_and_contains_current_api_results(self) -> None:
        committed = PROJECT_ROOT / "docs" / "puerto-rico-api-model-focused.html"
        with tempfile.TemporaryDirectory() as directory:
            generated = build_focused_explainer(Path(directory) / "focused.html")
            self.assertEqual(
                generated.read_text(encoding="utf-8"),
                committed.read_text(encoding="utf-8"),
            )

        page = committed.read_text(encoding="utf-8")
        parser = HTMLParser()
        parser.feed(page)
        parser.close()
        for content in [
            "How the dengue forecast works.",
            "What “outbreak” means here",
            "How the seasonal outbreak cutoff is calculated",
            "From source data to a monitored forecast",
            "How the app stays current without unsafe self-training",
            "Update every Wednesday",
            "Eligible after 13 new finalized weeks",
            "Forecast update",
            "Bot commit",
            "HistGradientBoostingClassifier + LogisticRegression",
            "HistGradientBoostingRegressor",
            "What the model can see",
            "Train on the past, test on a later period",
            "Results using weather available at forecast time",
            'id="caseChart"',
            "What the current model returns now",
            "Plain-language glossary",
            "Starting estimate",
            "Guarded promotion",
            "src/continuous_pipeline.py",
            "automation/puerto-rico-dengue-model.yml",
            "data/operational/vintages/",
            '"covered_weeks":72',
        ]:
            self.assertIn(content, page)
        self.assertNotIn("__MODEL_METADATA__", page)
        self.assertNotIn("__VALIDATION_ROWS__", page)
        self.assertNotIn("P50 prediction", page)
        self.assertNotIn("recall gate", page.lower())

    def test_active_automation_matches_documented_reference(self) -> None:
        """Keep the runnable root workflow and dated reference synchronized."""

        reference = (
            PROJECT_ROOT / "automation" / "puerto-rico-dengue-model.yml"
        ).read_text(encoding="utf-8")
        active = (
            PROJECT_ROOT.parent
            / ".github"
            / "workflows"
            / "puerto-rico-dengue-model.yml"
        ).read_text(encoding="utf-8")
        self.assertEqual(reference, active)

        for required_content in [
            'cron: "17 14 * * 3"',
            'cron: "17 15 1 * *"',
            "permissions:",
            "contents: write",
            "continuous_pipeline.py",
            "generate_public_dashboard.py",
            "generate_api_model_explainer.py",
            'python -m unittest discover -s tests -p "test_*.py" -v',
            "github-actions[bot]",
            "git diff --cached --quiet",
        ]:
            self.assertIn(required_content, active)

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
