"""Deterministic checks of source attenuation, pairing and provenance contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "02_modeling_and_validation"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_module("scenario_core_test", "_source_attenuation_core.py")
inference = load_module("scenario_inference_test", "03_source_attenuation_inference.py")


def inventory():
    frame = pd.DataFrame({"Grid_Lon": [110.0, 120.0], "Grid_Lat": [20.0, 35.0]})
    for name in core.all_compounds():
        frame[core.flux_column(name)] = [1.0, 2.0]
    frame.loc[1, "PFOS_Flux_Kg"] = 12.0
    frame.loc[1, "PFBS_Flux_Kg"] = 25.0
    return frame


class SourceBoundaryTests(unittest.TestCase):
    def test_panel_is_19_congeners_and_excludes_retired_columns(self):
        self.assertEqual(len(core.all_compounds()), 19)
        self.assertNotIn("PFNS", core.all_compounds())
        self.assertNotIn("PFECHS", core.all_compounds())

    def test_inventory_rejects_missing_negative_duplicate_and_nan(self):
        variants = [inventory().drop(columns="PFOS_Flux_Kg")]
        for value in (-1.0, np.nan):
            changed = inventory()
            changed.loc[0, "PFOS_Flux_Kg"] = value
            variants.append(changed)
        variants.append(
            pd.concat([inventory(), inventory().iloc[[0]]], ignore_index=True)
        )
        for frame in variants:
            with self.subTest(frame=frame.shape), self.assertRaises(ValueError):
                core.validate_inventory(frame)

    def test_mass_closure_and_fixed_node_ratio_does_not_fix_mixed_ratio(self):
        baseline = inventory()
        before, groups = core.aggregate_source_table(baseline)
        multipliers = {
            name: (0.4 if name.startswith("Long_") else 1.0) for name in core.TAXONOMY
        }
        after, _ = core.aggregate_source_table(
            core.apply_multipliers(baseline, multipliers)
        )
        np.testing.assert_allclose(
            before["Load_Total_Legacy_Kg"], groups[list(core.TAXONOMY)].sum(axis=1)
        )
        np.testing.assert_allclose(
            before["Source_Ratio_LC_PFSA_PFCA"], after["Source_Ratio_LC_PFSA_PFCA"]
        )
        self.assertFalse(
            np.allclose(before["Source_Ratio_SC_LC"], after["Source_Ratio_SC_LC"])
        )
        weights = np.array([[0.6, 0.4]])
        base_mixed = inference.route_source_features(weights, before)
        perturbed_mixed = inference.route_source_features(weights, after)
        self.assertNotAlmostEqual(
            base_mixed["Mixed_Source_Ratio_LC_PFSA_PFCA"].iloc[0],
            perturbed_mixed["Mixed_Source_Ratio_LC_PFSA_PFCA"].iloc[0],
        )

    def test_zero_denominator_is_not_imputed(self):
        frame = inventory()
        for name in core.TAXONOMY["Long_PFCA"]:
            frame.loc[0, core.flux_column(name)] = 0
        with self.assertRaises(ValueError):
            core.aggregate_source_table(frame)

    def test_log_path_endpoints(self):
        evidence = {"Long_Synchronized": 0.5, "Long_PFSA": 0.25, "Long_PFCA": 0.64}
        scenario = {
            "legacy_mode": "class_resolved",
            "short_mode": "reference",
            "transition_intensity": 0.5,
        }
        values = core.scenario_multipliers(scenario, evidence)
        self.assertEqual(values["Long_PFSA"], 0.5)
        self.assertEqual(values["Long_PFCA"], 0.8)
        self.assertEqual(values["Short_PFCA"], 1.0)

    def test_config_keeps_all_original_path_levels(self):
        config = core.load_configuration(CODE / "config" / "source_attenuation.json")
        self.assertEqual(len(config["scenarios"]), 14)
        self.assertEqual(len(config["contrasts"]), 6)
        levels = {
            s.get("transition_intensity", 1.0)
            for s in config["scenarios"]
            if s["legacy_mode"] == "synchronized"
        }
        self.assertEqual(levels, {0.1, 0.25, 0.5, 0.75, 1.0})


class InferenceContractTests(unittest.TestCase):
    def test_routing_zero_or_nonfinite_weights_rejected(self):
        sources, _ = core.aggregate_source_table(inventory())
        for weights in (
            np.zeros((1, 2)),
            np.array([[np.nan, 1.0]]),
            np.array([[-1.0, 1.0]]),
            np.array([[0.0, 1.0]]),
        ):
            with self.assertRaises(ValueError):
                inference.route_source_features(weights, sources)

    def test_inference_rejects_missing_or_different_routing_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = directory / "Routing_Manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            for package in (
                {},
                {
                    "training_input_provenance": {
                        "routing_manifest": {"sha256": "stale"}
                    }
                },
            ):
                with self.subTest(package=package), self.assertRaises(ValueError):
                    inference.verify_routing_provenance(package, directory)

    def test_inference_validates_fitted_routing_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = directory / "Routing_Manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            package = {
                "training_input_provenance": {
                    "routing_manifest": {
                        "path": str(manifest),
                        "sha256": inference.sha256(manifest),
                    }
                }
            }
            with patch.object(inference, "load_routing_bundle") as loader:
                self.assertEqual(
                    inference.verify_routing_provenance(package, directory),
                    manifest.resolve(),
                )
                loader.assert_called_once_with(directory)

    def test_anchor_rejects_stale_constant_feature(self):
        base = pd.DataFrame({name: [1.0, 1.0] for name in inference.SOURCE_FEATURES})
        stale = base.copy()
        stale[inference.SOURCE_FEATURES[0]] = 1.001
        with self.assertRaises(ValueError):
            inference.anchor_scenario_features(base, stale, stale)

    def test_anchor_preserves_exact_baseline(self):
        base = pd.DataFrame({name: [1.0, 2.0] for name in inference.SOURCE_FEATURES})
        changed, audit = inference.anchor_scenario_features(
            base, base.copy(), base.copy()
        )
        pd.testing.assert_frame_equal(changed, base)
        self.assertEqual(audit["maximum_absolute_anchor_correction"].max(), 0)

    def test_reconstruction_is_reversible(self):
        components = np.log(np.array([[2.0, 3.0], [0.01, 10.0]]))
        coordinates = np.column_stack(
            (components.sum(axis=1), components[:, 0] - components[:, 1])
        ) / np.sqrt(2)
        outputs = inference.derived_outputs(coordinates)
        np.testing.assert_allclose(outputs["Long_PFSA"], [2.0, 0.01])
        np.testing.assert_allclose(outputs["Long_PFCA"], [3.0, 10.0])
        np.testing.assert_allclose(outputs["Long_Total"], [5.0, 10.01])

    def test_reconstruction_rejects_nan_overflow_and_underflow(self):
        for values in ([[np.nan, 0]], [[2000, 0]], [[-2000, 0]]):
            with self.assertRaises(ValueError):
                inference.derived_outputs(np.array(values))

    def test_nonpositive_percent_denominator_rejected(self):
        for base in (np.array([0.0]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                inference.positive_paired_ratio(base, np.array([1.0]), "Long_Total")

    def test_domain_flags_remain_diagnostics(self):
        frame = pd.DataFrame({name: [0.2, 1.2] for name in inference.SOURCE_FEATURES})
        domain = {
            name: {"minimum": 0, "p02_5": 0.1, "p97_5": 0.9, "maximum": 1.0}
            for name in frame
        }
        flags, _ = inference.domain_diagnostics(frame, domain)
        self.assertEqual(flags["outside_source_minmax_any"].tolist(), [False, True])
        self.assertEqual(len(frame), 2)

    def test_zero_shap_share_is_explicitly_undefined(self):
        frame = pd.DataFrame(
            {
                "scenario_id": ["a", "a"],
                "feature": [inference.SOURCE_FEATURES[0], "TOC"],
                "source_domain_status": ["within_source_minmax"] * 2,
            }
        )
        for output in (
            "Long_Magnitude",
            "Long_Balance",
            "Log_Long_PFSA",
            "Log_Long_PFCA",
            "Log_Long_Ratio",
        ):
            for suffix in ("baseline", "scenario", "delta"):
                frame[f"{output}_SHAP_{suffix}"] = 0.0
        _, category = inference.summarize_shap(frame, inference.SOURCE_FEATURES)
        self.assertFalse(category["share_defined_baseline"].any())
        self.assertTrue(category["absolute_shap_share_baseline_percent"].isna().all())

    def test_product_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.csv"
            path.write_text("1", encoding="utf-8")
            manifest = {
                "pfas_schema_sha256": core.schema_digest(),
                "product_sha256": {path.name: "incorrect"},
            }
            with self.assertRaises(ValueError):
                inference.verify_scenario_products(
                    Path(folder), manifest, pd.DataFrame()
                )

    def test_fixed_seed_cluster_bootstrap_is_reproducible(self):
        frame = pd.DataFrame(
            {
                "Sampling_Group": [1, 1, 2, 2],
                "Long_Total_baseline": [1, 2, 3, 4],
                "Long_Total_scenario": [0.8, 1.5, 2.5, 3.8],
            }
        )
        first = inference.paired_cluster_bootstrap(frame, "Long_Total", True, 20, 42)
        self.assertEqual(
            first, inference.paired_cluster_bootstrap(frame, "Long_Total", True, 20, 42)
        )

    def test_array_bootstrap_matches_dataframe_resampling(self):
        frame = pd.DataFrame(
            {
                "Sampling_Group": [2, 1, 2, 3, 3],
                "Long_Total_baseline": [1, 2, 3, 4, 2],
                "Long_Total_scenario": [0.8, 1.5, 2.5, 3.8, 1.9],
            },
            index=[5, 9, 13, 20, 27],
        )
        actual = inference.paired_cluster_bootstrap(frame, "Long_Total", True, 25, 42)
        grouped = [group for _, group in frame.groupby("Sampling_Group", sort=True)]
        rng, means = np.random.default_rng(42), []
        for _ in range(25):
            selected = rng.integers(0, len(grouped), size=len(grouped))
            sampled = pd.concat(
                [grouped[position] for position in selected], ignore_index=True
            )
            means.append(
                float(
                    (
                        sampled["Long_Total_scenario"] - sampled["Long_Total_baseline"]
                    ).mean()
                )
            )
        self.assertAlmostEqual(
            actual["mean_delta_cluster_CI_lower"], pd.Series(means).quantile(0.025)
        )
        self.assertAlmostEqual(
            actual["mean_delta_cluster_CI_upper"], pd.Series(means).quantile(0.975)
        )


if __name__ == "__main__":
    unittest.main()
