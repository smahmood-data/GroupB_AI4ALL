"""Tests for the public dashboard's data contract and static app shell."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_public_dashboard import (  # noqa: E402
    _select_latest_issue,
    build_dashboard_payload,
    write_dashboard_payload,
    write_dashboard_script,
)


class PublicDashboardDataTests(unittest.TestCase):
    """Protect the small JSON contract consumed by the browser."""

    def test_latest_complete_issue_is_selected_atomically(self) -> None:
        rows = [
            {
                "geography": "pr",
                "generated_at_utc": "2026-07-22T01:00:00+00:00",
                "lead_weeks": "0",
            },
            {
                "geography": "pr",
                "generated_at_utc": "2026-07-22T01:00:00+00:00",
                "lead_weeks": "1",
            },
            {
                "geography": "pr",
                "generated_at_utc": "2026-07-23T01:00:00+00:00",
                "lead_weeks": "0",
            },
        ]

        current, outlook = _select_latest_issue(rows)

        self.assertEqual(current["generated_at_utc"], "2026-07-22T01:00:00+00:00")
        self.assertEqual(outlook["lead_weeks"], "1")

    def test_repository_outputs_build_a_public_snapshot(self) -> None:
        payload = build_dashboard_payload()

        self.assertEqual(payload["schemaVersion"], 1)
        self.assertEqual(payload["geography"]["code"], "pr")
        self.assertIn(payload["model"]["variant"], {
            "weather_only",
            "weather_plus_delayed_cases",
            "weather_plus_exact_cases",
        })
        self.assertEqual(payload["signal"]["currentWeek"]["leadWeeks"], 0)
        self.assertEqual(payload["signal"]["nextWeek"]["leadWeeks"], 1)
        self.assertGreater(len(payload["history"]), 300)
        self.assertGreater(payload["model"]["heldOut"]["caseTestWeeks"], 0)
        self.assertGreater(
            payload["model"]["heldOut"]["lowerHighEstimateCoverage"], 0
        )
        self.assertGreater(
            payload["model"]["heldOut"]["upperHighEstimateCoverage"],
            payload["model"]["heldOut"]["lowerHighEstimateCoverage"],
        )
        self.assertLess(
            payload["history"][0]["weekStart"],
            payload["history"][-1]["weekStart"],
        )

        # Public JSON must not expose the API's opaque file identifier or local
        # input-vintage path.  Those remain in the auditable operational ledger.
        serialized = json.dumps(payload)
        self.assertNotIn("caseSourceFileId", serialized)
        self.assertNotIn("inputVintageId", serialized)
        self.assertNotIn(".cache", serialized)

    def test_written_snapshot_contains_no_nonstandard_nan_tokens(self) -> None:
        payload = build_dashboard_payload()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dashboard.json"
            write_dashboard_payload(payload, output)
            content = output.read_text(encoding="utf-8")

        self.assertNotIn("NaN", content)
        self.assertNotIn("Infinity", content)
        json.loads(content)

    def test_server_free_script_contains_the_same_snapshot(self) -> None:
        payload = build_dashboard_payload()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dashboard-data.js"
            write_dashboard_script(payload, output)
            content = output.read_text(encoding="utf-8")

        prefix = "window.DENGUE_DASHBOARD_DATA = "
        self.assertTrue(content.startswith(prefix))
        self.assertTrue(content.endswith(";\n"))
        embedded_payload = json.loads(content[len(prefix) : -2])
        self.assertEqual(embedded_payload, payload)


class PublicDashboardShellTests(unittest.TestCase):
    """Catch accidental removal of the core approachable chart experience."""

    def test_app_contains_bilingual_interactive_chart_controls(self) -> None:
        html = (PROJECT_ROOT / "docs/app/index.html").read_text(encoding="utf-8")
        script = (PROJECT_ROOT / "docs/app/app.js").read_text(encoding="utf-8")

        self.assertIn('data-language="en"', html)
        self.assertIn('data-language="es"', html)
        self.assertIn('data-range="13"', html)
        self.assertIn('data-range="52"', html)
        self.assertIn('data-range="all"', html)
        self.assertIn('data-metric="hospitalized"', html)
        self.assertIn('id="highRangeBand"', html)
        self.assertIn('id="highRangeDefinition"', html)
        self.assertIn('id="probabilityWeek"', html)
        self.assertIn('id="probabilityExplanation"', html)
        self.assertIn("Chance of an outbreak", script)
        self.assertIn("not an official declaration", script)
        self.assertIn("Starting estimate means", script)
        self.assertIn('id="caseForecastWeek"', html)
        self.assertIn('id="unusualLevelDefinition"', html)
        self.assertIn("Across {weeks} test weeks", script)
        self.assertIn("seasonal top-25% cutoff", script)
        self.assertIn("does not calculate a reliable chance", script)
        self.assertIn("It is not a confirmed count", script)
        self.assertIn('canvas.addEventListener("pointerdown"', script)
        self.assertNotIn('data-mode="advanced"', html)
        self.assertNotIn("Optional notifications", html)
        self.assertNotIn("Email alerts", html)
        self.assertNotIn("In plain language", html)
        self.assertNotIn("Higher but possible", html)
        self.assertNotIn("Highest shown", html)
        self.assertNotIn("Alert gate", html)
        self.assertNotIn("recall-focused", html)
        self.assertNotIn("Model route", html)
        self.assertIn("./data/dashboard-data.js", html)
        self.assertIn("translations = {", script)
        self.assertIn("es: {", script)
        self.assertIn("researchUseOnly", json.dumps(build_dashboard_payload()))


if __name__ == "__main__":
    unittest.main()
