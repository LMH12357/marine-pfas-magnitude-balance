"""Bounded regression checks for current-data preparation and grouped validation."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "02_modeling_and_validation"))
primary = importlib.import_module("01_primary_model_training")
sensitivity = importlib.import_module("04_preprocessing_sensitivity")
metadata = importlib.import_module("metadata_schema")


class PrimaryContracts(unittest.TestCase):
    def test_model_panel_excludes_sparse_nonpanel_compounds(self):
        frame = pd.DataFrame(
            {
                "PFHxS": [1.0],
                "PFOS": [2.0],
                "PFOA": [6.0],
                "PFNS": [1000.0],
                "PFECHS": [2000.0],
            }
        )
        result, audit = primary.construct_long_chain_targets(frame)
        self.assertEqual(result.at[0, "Long_PFSA"], 3.0)
        self.assertEqual(audit.at[0, "eligible_rows"], 1)
        self.assertNotIn("PFECHS", sensitivity.PFAS_COLUMNS)

    def test_invalid_target_does_not_retain_stale_coordinates(self):
        frame = pd.DataFrame(
            {
                "PFOS": [np.nan, 0.0, 1.0],
                "PFOA": [1.0, 2.0, 3.0],
                "Long_Balance": [99.0] * 3,
            }
        )
        result, _ = primary.construct_long_chain_targets(frame)
        self.assertTrue(result.loc[[0, 1], "Long_Balance"].isna().all())
        self.assertTrue(np.isfinite(result.at[2, "Long_Balance"]))

    def test_negative_concentrations_rejected(self):
        with self.assertRaises(ValueError):
            primary.construct_long_chain_targets(
                pd.DataFrame({"PFOS": [-1.0], "PFOA": [2.0]})
            )

    def test_coordinate_roundtrip(self):
        logs = np.log(np.array([[0.2, 2.0], [5.0, 0.003], [4.0, 4.0]]))
        np.testing.assert_allclose(
            primary.magnitude_balance_to_components(
                primary.component_to_magnitude_balance(logs)
            ),
            logs,
            atol=1e-14,
        )

    def test_sampling_group_subsets_keep_identifiers(self):
        frame = pd.DataFrame({"Sampling_Group_ID": [22, 4, 22, 9]})
        normalized = metadata.normalize_sampling_groups(frame)
        self.assertEqual(normalized["Sampling_Group_ID"].tolist(), [22, 4, 22, 9])
        self.assertEqual(
            normalized["Sampling_Group"].tolist(),
            [
                "Sampling_Group_22",
                "Sampling_Group_04",
                "Sampling_Group_22",
                "Sampling_Group_09",
            ],
        )

    def test_legacy_identifiers_are_not_silently_renumbered(self):
        normalized = metadata.normalize_sampling_groups(
            pd.DataFrame({"Study": ["Study_08", "Study_10", "Study_24"]})
        )
        self.assertEqual(normalized["Sampling_Group_ID"].tolist(), [8, 10, 24])

    def test_grouped_cv_never_falls_back_to_random(self):
        frame = pd.DataFrame({"Sampling_Group": ["A"] * 5})
        with self.assertRaisesRegex(ValueError, "at least two training groups"):
            primary.make_inner_splits(frame, "sampling_group", 5, 42)

    def test_grouped_cv_keeps_all_groups_disjoint(self):
        frame = pd.DataFrame({"Sampling_Group": np.repeat(["A", "B", "C", "D"], 4)})
        for train, test, _ in primary.make_splits(frame, "sampling_group", 3, 42):
            self.assertFalse(
                set(frame.iloc[train]["Sampling_Group"])
                & set(frame.iloc[test]["Sampling_Group"])
            )

    def test_raw_data_is_the_default_route(self):
        arguments = primary.parse_arguments(
            ["--data", "DATA.xlsx", "--source-load", "current.csv", "--prepare-only"]
        )
        self.assertIsNone(arguments.feature_table)
        self.assertTrue(arguments.prepare_only)
        self.assertEqual(arguments.seed, 42)

    def test_stable_row_identity(self):
        raw = primary.validate_row_identity(pd.DataFrame({"x": [1, 2, 3]}))
        subset = primary.validate_row_identity(raw.loc[[2, 0]], require_existing=True)
        self.assertEqual(subset.index.tolist(), [2, 0])
        self.assertEqual(subset["excel_row"].tolist(), [4, 2])
        with self.assertRaises(ValueError):
            primary.validate_row_identity(pd.DataFrame({"row_index": [0, 0]}))

    def test_input_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaises(ValueError):
                primary.prepare_output_directory(
                    folder,
                    {"data": {"path": str(folder / "DATA.xlsx"), "sha256": "unused"}},
                )

    def test_feature_table_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            table = folder / "features.csv"
            table.write_text("row_index,x\n0,1\n", encoding="utf-8")
            manifest = {
                "feature_table_sha256": primary.sha256(table),
                "analysis_schema_sha256": primary.schema_digest(),
            }
            (folder / "Feature_Table_Manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            primary.verify_feature_manifest(table)
            table.write_text("row_index,x\n0,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                primary.verify_feature_manifest(table)

    def test_sensitivity_feature_alignment_rejects_same_count_wrong_rows(self):
        data = pd.DataFrame(
            {"Longitude": [120.0, 121.0], "Latitude": [35.0, 34.0], "PFOS": [1.0, 2.0]}
        )
        features = primary.validate_row_identity(data)
        sensitivity.validate_feature_alignment(data, features)
        features.loc[0, "PFOS"] = 100.0
        with self.assertRaisesRegex(ValueError, "PFOS"):
            sensitivity.validate_feature_alignment(data, features)

    def test_censoring_formulas_keep_units(self):
        status, limit = sensitivity.parse_pfas_cell("=6/1000/2^0.5", "f")
        self.assertEqual(status, sensitivity.ND_MDL)
        self.assertAlmostEqual(limit, 0.006)
        status, limit = sensitivity.parse_pfas_cell("=0.01/2", "f")
        self.assertEqual(status, sensitivity.BELOW_MQL)
        self.assertAlmostEqual(limit, 0.01)

    def test_red_numeric_censoring_value_is_unresolved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["PFOS", "PFOA"])
            sheet.append([0.05, "=0.1/2"])
            sheet["A2"].font = Font(color="FF0000")
            sheet["B2"].font = Font(color="FF0000")
            workbook.save(path)
            workbook.close()
            counts, syntax, unresolved = sensitivity.inspect_censoring_evidence(path, 1)
            self.assertEqual(counts["red_value"], 1)
            self.assertEqual(len(syntax), 1)
            self.assertEqual(unresolved["cell"].tolist(), ["A2"])

    def test_bounded_nested_fit_and_shap_with_current_runtime(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            rng.normal(size=(24, len(primary.COVARIATES))), columns=primary.COVARIATES
        )
        frame["Sampling_Group"] = np.repeat([f"Group_{i}" for i in range(6)], 4)
        frame["Sea_Region"] = np.repeat(
            ["Bohai Sea", "Yellow Sea", "South China Sea"], 8
        )
        frame["SamplingYear"] = 2020
        frame["Longitude"] = 120.0
        frame["Latitude"] = 35.0
        frame["Long_Magnitude"] = 0.5 * frame["Temperature"] + 0.2 * frame["TOC"]
        definitions = [
            primary.ModelDefinition(
                definition.name,
                definition.estimator,
                {"n_estimators": [4], "max_depth": [2]},
            )
            for definition in primary.model_definitions(42, True)
        ]
        predictions, folds, shap_values = primary.evaluate_nested(
            frame,
            "Long_Magnitude",
            "sampling_group",
            definitions,
            primary.COVARIATES,
            "all",
            3,
            2,
            42,
            1,
            True,
        )
        self.assertEqual(len(predictions), 48)
        self.assertEqual(len(predictions[predictions["selected_in_fold"]]), 24)
        self.assertEqual(len(folds), 6)
        self.assertTrue(np.isfinite(shap_values["shap_value"]).all())


if __name__ == "__main__":
    unittest.main()
