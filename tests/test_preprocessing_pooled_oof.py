"""Deterministic tests for pooled sensitivity metrics and CLI export."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "code"
    / "02_modeling_and_validation"
    / "04_preprocessing_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pfas_preprocessing_pooled_oof_test_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
PREPROCESSING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPROCESSING
SPEC.loader.exec_module(PREPROCESSING)


def synthetic_predictions() -> pd.DataFrame:
    observed = np.array([1.0, 2.0, 100.0, 101.0])
    records = []
    for scheme, predicted in (
        ("S0_submitted", np.array([1.0, 1.0, 100.0, 100.0])),
        ("S1_half_limit", np.array([0.0, 2.0, 98.0, 101.0])),
    ):
        for index, (value, estimate) in enumerate(zip(observed, predicted)):
            records.append(
                {
                    "evaluation_track": "synthetic_cli_fixture",
                    "row_index": index,
                    "target": "Ratio_Long_PFSA_PFCA_Ocean",
                    "scheme": scheme,
                    "model": "RF",
                    "validation": "sampling_group",
                    "fold": index // 2 + 1,
                    "observed": value,
                    "predicted": estimate,
                }
            )
    return pd.DataFrame(records)


def synthetic_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["target", "scheme", "model", "validation", "fold"]
    for values, group in predictions.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row["scheme_role"] = PREPROCESSING.SCHEMES[row["scheme"]].role
        row.update(
            PREPROCESSING.regression_metrics(
                group["observed"].to_numpy(), group["predicted"].to_numpy()
            )
        )
        records.append(row)
    return pd.DataFrame(records)


class PooledOOFTests(unittest.TestCase):
    def test_pools_observations_instead_of_averaging_fold_metrics(self) -> None:
        predictions = synthetic_predictions()
        before = predictions.copy(deep=True)
        pooled = PREPROCESSING.summarize_pooled_oof_performance(predictions)
        fold_summary, _ = PREPROCESSING.summarize_performance(
            synthetic_fold_metrics(predictions)
        )
        self.assertIn("R2_mean", fold_summary.columns)
        self.assertIn("R2_std", fold_summary.columns)
        self.assertIn("R2_median", fold_summary.columns)
        for _, row in pooled.iterrows():
            group = predictions[predictions["scheme"] == row["scheme"]]
            expected = PREPROCESSING.regression_metrics(
                group["observed"].to_numpy(), group["predicted"].to_numpy()
            )
            self.assertEqual(int(row["N"]), 4)
            for metric, value in expected.items():
                self.assertEqual(row[metric], value)
            old_mean = fold_summary.loc[
                fold_summary["scheme"] == row["scheme"], "R2_mean"
            ].iloc[0]
            self.assertNotAlmostEqual(row["R2"], old_mean, places=6)
        baseline = pooled[pooled["scheme"] == "S0_submitted"].iloc[0]
        self.assertEqual(baseline["Delta_R2"], 0.0)
        self.assertEqual(baseline["Delta_RMSE"], 0.0)
        candidate = pooled[pooled["scheme"] == "S1_half_limit"].iloc[0]
        self.assertEqual(candidate["Delta_R2"], candidate["R2"] - baseline["R2"])
        self.assertEqual(candidate["Delta_RMSE"], candidate["RMSE"] - baseline["RMSE"])
        pd.testing.assert_frame_equal(predictions, before)

    def test_nonfinite_predictions_are_not_silently_dropped(self) -> None:
        predictions = synthetic_predictions()
        predictions.loc[0, "predicted"] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite pooled OOF"):
            PREPROCESSING.summarize_pooled_oof_performance(predictions)

    def test_validation_cli_branch_writes_pooled_export_without_fitting(self) -> None:
        predictions = synthetic_predictions()
        per_fold = synthetic_fold_metrics(predictions)
        pfas = pd.DataFrame(
            {
                "PFBA": [2.0, 2.0, 2.0, 2.0],
                "PFBS": [0.4, 0.5, 0.6, 0.7],
                "PFOA": [1.0, 1.0, 1.0, 1.0],
                "PFOS": [1.0, 2.0, 100.0, 101.0],
            }
        )
        features = pfas.copy()
        for column in PREPROCESSING.COVARIATES:
            features[column] = 1.0
        features["row_index"] = np.arange(len(features))
        features["Longitude"] = [120.0, 120.1, 120.2, 120.3]
        features["Latitude"] = [38.0, 38.1, 38.2, 38.3]
        primary = PREPROCESSING.importlib.import_module("01_primary_model_training")
        with tempfile.TemporaryDirectory(prefix="pfas-pooled-oof-test-") as directory:
            output = Path(directory) / "outputs"
            inputs = Path(directory) / "inputs"
            inputs.mkdir()
            workbook = inputs / "synthetic_input.xlsx"
            features.to_excel(workbook, index=False)
            arguments = SimpleNamespace(
                n_jobs=1,
                prepare_only=False,
                analysis_mode="validation",
                output_dir=output,
                data=workbook,
                feature_table=inputs / "synthetic_features.csv",
                source_load=inputs / "synthetic_source.csv",
                boundary=inputs / "synthetic_boundary.shp",
                routing_directory=None,
                reference_report=output / "unused_legacy_report.csv",
                schemes=["S0_submitted", "S1_half_limit"],
                validation="sampling_group",
            )
            with (
                patch.object(PREPROCESSING, "parse_arguments", return_value=arguments),
                patch.object(
                    primary,
                    "training_input_provenance",
                    return_value={
                        "data": {"path": str(workbook), "sha256": primary.sha256(workbook)}
                    },
                ),
                patch.object(
                    primary, "load_features", return_value=features.copy()
                ),
                patch.object(
                    PREPROCESSING,
                    "fit_sensitivity_validation",
                    return_value=(per_fold, predictions),
                ) as fitting_stub,
                patch.object(
                    PREPROCESSING,
                    "fit_original_protocol",
                    side_effect=AssertionError("Unexpected legacy fitting"),
                ),
                patch.object(
                    PREPROCESSING,
                    "fit_paired_fixed_sensitivity",
                    side_effect=AssertionError("Unexpected legacy fitting"),
                ),
                patch.object(
                    PREPROCESSING.GridSearchCV,
                    "fit",
                    side_effect=AssertionError(
                        "No model fitting is allowed in this test"
                    ),
                ),
            ):
                PREPROCESSING.main()
                fitting_stub.assert_called_once()
            actual = pd.read_csv(output / "Sensitivity_Pooled_OOF_Performance.csv")
            expected = PREPROCESSING.summarize_pooled_oof_performance(predictions)
            pd.testing.assert_frame_equal(
                actual, expected, check_exact=False, rtol=1e-14, atol=1e-14
            )
            old_summary = pd.read_csv(output / "Sensitivity_Performance_Summary.csv")
            self.assertIn("R2_mean", old_summary.columns)
            self.assertNotIn("N", old_summary.columns)

    @unittest.skipUnless(
        os.environ.get("PFAS_POOLED_OOF_REFERENCE_DIR"),
        "Archived reference directory not configured",
    )
    def test_archived_table_s5_reference(self) -> None:
        directory = Path(os.environ["PFAS_POOLED_OOF_REFERENCE_DIR"])
        predictions = pd.read_csv(directory / "Reviewer_Out_of_Fold_Predictions.csv")
        reference = pd.read_csv(directory / "Reviewer_Pooled_OOF_Performance.csv")
        keys = ["target", "scheme", "model", "validation"]
        actual = (
            PREPROCESSING.summarize_pooled_oof_performance(predictions)
            .set_index(keys)
            .sort_index()
        )
        reference = reference.set_index(keys).sort_index()
        pd.testing.assert_index_equal(actual.index, reference.index)
        np.testing.assert_array_equal(actual["N"].to_numpy(), reference["N"].to_numpy())
        metrics = ["R2", "MSE", "RMSE", "MAE", "Delta_R2", "Delta_RMSE"]
        difference = np.abs(actual[metrics].to_numpy() - reference[metrics].to_numpy())
        np.testing.assert_allclose(
            actual[metrics].to_numpy(),
            reference[metrics].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        print(
            f"Archived Table S5: {len(actual)} rows; maximum absolute metric difference = {difference.max():.17g}"
        )


if __name__ == "__main__":
    unittest.main()
