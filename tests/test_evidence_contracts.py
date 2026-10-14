"""Synthetic integrity and numerical regression tests for fixed-output evidence."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "03_evidence_strengthening"))
from evidence_contract import (
    boolean_flags,
    merge_observation_metadata,
    validate_scenarios,
    validate_selected_shap,
    validate_run_provenance,
)
import wr_evidence_strengthening as evidence
import wr_response_regimes as regimes
import source_ratio_diagnostic as ratio


class EvidenceContracts(unittest.TestCase):
    def test_mixed_generation_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            left.mkdir()
            right.mkdir()
            manifest = dict(
                data_sha256="data",
                feature_table_sha256="features",
                pfas_schema_sha256="schema",
            )
            (left / "Run_Manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (right / "Run_Manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(
                len(
                    validate_run_provenance((left / "a.csv", right / "b.csv"), "schema")
                ),
                2,
            )
            manifest["data_sha256"] = "stale-data"
            (right / "Run_Manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "different data"):
                validate_run_provenance((left / "a.csv", right / "b.csv"), "schema")

    def test_explicit_generator_preserves_stream(self):
        for seed in (42, 20261007):
            np.testing.assert_array_equal(
                np.random.default_rng(seed).integers(0, 17, 100),
                np.random.Generator(np.random.PCG64(seed)).integers(0, 17, 100),
            )

    def test_flags_do_not_silently_accept_missing_values(self):
        self.assertEqual(
            boolean_flags(pd.Series([True, "false", 1, 0])).tolist(),
            [True, False, True, False],
        )
        with self.assertRaises(ValueError):
            boolean_flags(pd.Series(["false", np.nan]))

    def test_scenarios_dynamic_cardinality(self):
        frame = pd.DataFrame(
            [
                {
                    "scenario_id": s,
                    "row_index": i,
                    "Sampling_Group": "a",
                    "Sea_Region": "sea",
                    "Long_Total_baseline": 2.0,
                }
                for s in ("a", "b")
                for i in (2, 7, 9)
            ]
        )
        validate_scenarios(frame, ("a", "b"))
        with self.assertRaises(ValueError):
            validate_scenarios(frame.iloc[:-1], ("a", "b"))
        changed = frame.copy()
        changed.loc[4, "Long_Total_baseline"] = 3
        with self.assertRaisesRegex(ValueError, "Baseline"):
            validate_scenarios(changed, ("a", "b"))

    def test_metadata_join_cannot_drop_unmatched_shap(self):
        meta = pd.DataFrame(
            {"row_index": [1], "Sampling_Group": ["a"], "Sea_Region": ["sea"]}
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            merge_observation_metadata(pd.DataFrame({"row_index": [1, 2]}), meta)

    def test_quartiles_and_historical_random_prefix(self):
        rows = []
        for group in range(3):
            for j in range(8):
                rows.append(
                    dict(
                        row_index=8 * group + j,
                        Sampling_Group=f"g{group}",
                        target="Long_Magnitude",
                        validation="sampling_group",
                        feature="TOC",
                        fold=group,
                        feature_value=float(j),
                        shap_value=-(group + 1) * float(j),
                    )
                )
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "fixture.csv"
            frame.to_csv(path, index=False)
            regimes.run(path, directory / "outputs")
            actual = pd.read_csv(
                directory / "outputs" / "within_group_attribution_contrasts.csv"
            )
        self.assertEqual(actual.paired_groups.tolist(), [3, 3])
        np.testing.assert_allclose(actual.mean_high_minus_low, [-12.0, -12.0])
        rng = np.random.Generator(np.random.PCG64(20261007))
        for _ in range(500):
            rng.integers(0, 3, 3)
        values = np.array([-6.0, -12.0, -18.0])
        boot = values[rng.integers(0, 3, (10000, 3))].mean(axis=1)
        np.testing.assert_allclose(
            actual.loc[0, ["ci_low", "ci_high"]].to_numpy(dtype=float),
            np.quantile(boot, [0.025, 0.975]),
        )

    def test_source_ratio_common_support_is_intersection(self):
        predictions = pd.DataFrame(
            [
                dict(
                    scenario_id=s,
                    row_index=i,
                    Sampling_Group="a",
                    Sea_Region="sea",
                    outside_source_minmax_any=(
                        i == 1 and s == ratio.COMMON_SCENARIOS[0]
                    ),
                    Long_Balance_delta=0.1,
                )
                for s in ratio.COMMON_SCENARIOS
                for i in (0, 1)
            ]
        )
        feature = pd.DataFrame(
            [
                dict(
                    scenario_id=ratio.SCENARIO,
                    row_index=i,
                    feature=ratio.FEATURE,
                    feature_value_baseline=2.0,
                    feature_value_scenario=3.0,
                )
                for i in (0, 1)
            ]
        )
        summary, values = ratio.summarize_descriptor_changes(predictions, feature)
        self.assertEqual(values.row_index.tolist(), [0])
        self.assertEqual(summary.observation_count.tolist(), [1])
        self.assertAlmostEqual(summary.descriptor_change_median_percent.iloc[0], 50.0)


if __name__ == "__main__":
    unittest.main()
