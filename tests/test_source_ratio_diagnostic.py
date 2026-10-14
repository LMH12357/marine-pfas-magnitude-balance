"""Deterministic tests of common-domain mixed-source ratio diagnostics."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code/03_evidence_strengthening/source_ratio_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("source_ratio_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceRatioDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame(
            [
                {
                    "scenario_id": scenario,
                    "row_index": row,
                    "Sampling_Group": f"synthetic-group-{row}",
                    "Sea_Region": "Bohai Sea" if row == 0 else "South China Sea",
                    "outside_source_minmax_any": False,
                    "Long_Balance_delta": 0.8 if row == 0 else -0.4,
                }
                for scenario in MODULE.COMMON_SCENARIOS
                for row in (0, 1)
            ]
        )
        self.features = pd.DataFrame(
            [
                {
                    "scenario_id": MODULE.SCENARIO,
                    "row_index": row,
                    "feature": MODULE.FEATURE,
                    "feature_value_baseline": 1.0,
                    "feature_value_scenario": value,
                }
                for row, value in ((0, 1.02), (1, 1.01))
            ]
        )

    def test_opposing_responses_with_positive_descriptor_changes(self):
        summary, values = MODULE.summarize_descriptor_changes(
            self.predictions, self.features
        )
        self.assertEqual(len(values), 2)
        self.assertTrue((values.descriptor_change_percent > 0).all())
        self.assertAlmostEqual(summary.iloc[0].descriptor_change_median_percent, 2.0)
        self.assertAlmostEqual(summary.iloc[1].descriptor_change_median_percent, 1.0)
        self.assertLess(summary.iloc[1].mean_balance_delta, 0.0)

    def test_support_requires_every_scenario(self):
        altered = self.predictions.copy()
        altered.loc[
            (altered.row_index == 1)
            & (altered.scenario_id == MODULE.COMMON_SCENARIOS[0]),
            "outside_source_minmax_any",
        ] = True
        _, values = MODULE.summarize_descriptor_changes(altered, self.features)
        self.assertEqual(values.row_index.tolist(), [0])

    def test_duplicate_feature_identity_is_rejected(self):
        duplicated = pd.concat(
            [self.features, self.features.iloc[[0]]], ignore_index=True
        )
        with self.assertRaises(ValueError):
            MODULE.summarize_descriptor_changes(self.predictions, duplicated)

    def test_invalid_source_domain_flag_is_rejected(self):
        altered = self.predictions.astype({"outside_source_minmax_any": object})
        altered.loc[0, "outside_source_minmax_any"] = "unknown"
        with self.assertRaises(ValueError):
            MODULE.summarize_descriptor_changes(altered, self.features)

    def test_output_cannot_replace_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source_ratio_descriptor_summary.csv"
            with self.assertRaises(ValueError):
                MODULE.check_output_paths((path,), (path,))

    def test_distinct_derived_output_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.check_output_paths((root / "input.csv",), (root / "derived.csv",))


if __name__ == "__main__":
    unittest.main()
