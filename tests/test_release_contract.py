from __future__ import annotations

import hashlib
import importlib.util
import json
import py_compile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleaseContractTests(unittest.TestCase):
    def test_all_python_files_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for source in sorted((PROJECT_ROOT / "code").rglob("*.py")):
                target = Path(directory) / f"{source.stem}.pyc"
                py_compile.compile(str(source), cfile=str(target), doraise=True)

    def test_declared_core_files_match_current_release_hashes(self) -> None:
        metadata = json.loads(
            (PROJECT_ROOT / "release_metadata.json").read_text(encoding="utf-8")
        )
        declared = {
            **metadata["core_source_sha256"],
            **metadata.get("additional_scientific_code_sha256", {}),
        }
        scientific_files = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "code").rglob("*.py")
        }
        self.assertTrue(
            scientific_files.issubset(declared),
            f"Undeclared scientific files: {sorted(scientific_files.difference(declared))}",
        )
        self.assertIn("run_analysis.py", declared)
        self.assertIn("analysis_config.example.json", declared)
        for relative_path, expected in declared.items():
            path = (PROJECT_ROOT / relative_path).resolve()
            self.assertTrue(path.is_relative_to(PROJECT_ROOT.resolve()))
            self.assertRegex(expected, r"^[a-f0-9]{64}$")
            self.assertEqual(sha256(path), expected, relative_path)
        self.assertEqual(metadata["production_run"]["eligible_observations"], 751)
        self.assertEqual(metadata["production_run"]["sampling_groups"], 22)
        for checksum in (
            metadata["data_workbook_sha256"],
            metadata["frozen_model_sha256"],
            *metadata["source_inventory_sha256"].values(),
        ):
            self.assertRegex(checksum, r"^[a-f0-9]{64}$")
        config = json.loads(
            (PROJECT_ROOT / "analysis_config.example.json").read_text(encoding="utf-8")
        )
        for key in ("data", "source_load", "compound_inventory", "boundary",
                    "scenario_config", "routing_config"):
            self.assertFalse(Path(config[key]).is_absolute())
        self.assertEqual(config["data"], "data/raw/DATA.xlsx")
        self.assertFalse((PROJECT_ROOT / config["data"]).exists())
        self.assertFalse((PROJECT_ROOT / "data/fixed_inputs").exists())
        self.assertFalse((PROJECT_ROOT / "data/fixed_outputs").exists())

    def test_step4_is_deterministically_rebuilt_from_synthetic_step3(self) -> None:
        module_path = (
            PROJECT_ROOT
            / "code"
            / "01_data_preparation"
            / "02_source_covariates"
            / "04_coastline_source_descriptors.py"
        )
        spec = importlib.util.spec_from_file_location("coastline_source_descriptors", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        columns = {
            f"{compound}_Flux_Kg": 0.123456789123
            for compounds in module.TAXONOMY.values()
            for compound in compounds
        }
        record = {
            "Year": 2018,
            "Grid_Lon": 120.0,
            "Grid_Lat": 38.0,
            "Expected_Total_Flux_Kg": sum(columns.values()),
            **columns,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Step3_MonteCarlo_Flux_2018.csv"
            pd.DataFrame([record]).to_csv(source, index=False)
            output = Path(directory) / "Step4_Coastline_Source_Load_2018.csv"
            second_output = Path(directory) / "repeated.csv"
            first = module.build_coastline_source_load(source, output)
            second = module.build_coastline_source_load(source, second_output)
            self.assertEqual(sha256(output), sha256(second_output))
            pd.testing.assert_frame_equal(first, second)
            self.assertAlmostEqual(first.Load_Total_Legacy_Kg.iloc[0], sum(columns.values()))

    def test_source_attenuation_configuration_resolves(self) -> None:
        config_path = (
            PROJECT_ROOT
            / "code"
            / "02_modeling_and_validation"
            / "config"
            / "source_attenuation.json"
        )
        configuration = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("usage_data_file", "compound_mapping_file", "release_factor_file"):
            self.assertTrue((config_path.parent / configuration["evidence"][key]).resolve().is_file())

    def test_back_transformed_burden_is_in_analysis_output(self) -> None:
        module_path = (
            PROJECT_ROOT / "code/02_modeling_and_validation/03_source_attenuation_inference.py"
        )
        spec = importlib.util.spec_from_file_location("release_inference", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline_coordinates = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        scenario_coordinates = baseline_coordinates.copy()
        scenario_coordinates[:, 0] -= np.sqrt(2.0) * np.log(2.0)
        baseline = module.derived_outputs(baseline_coordinates)
        scenario = module.derived_outputs(scenario_coordinates)
        frame = pd.DataFrame({
            "scenario_id": ["synthetic_half_concentration"] * 3,
            "Sampling_Group": ["synthetic-1", "synthetic-2", "synthetic-3"],
            "outside_source_minmax_any": [False] * 3,
            **{f"{output}_baseline": value for output, value in baseline.items()},
            **{f"{output}_scenario": value for output, value in scenario.items()},
        })
        summary = module.summarize_predictions(
            frame, "synthetic_fixture", "software_test", bootstrap_iterations=25, seed=42
        )
        columns = {
            "geometric_mean_paired_change_percent",
            "geometric_mean_paired_change_percent_cluster_CI_lower",
            "geometric_mean_paired_change_percent_cluster_CI_upper",
        }
        self.assertTrue(columns.issubset(summary.columns))
        evaluated = summary[
            summary["output"].eq("Long_Total")
            & summary["domain_scope"].eq("within_source_minmax")
        ]
        self.assertGreater(len(evaluated), 0)
        self.assertFalse(evaluated[list(columns)].isna().any().any())
        np.testing.assert_allclose(evaluated[list(columns)].to_numpy(), -50.0)

    def test_plotting_code_is_not_in_analysis_release(self) -> None:
        self.assertFalse((PROJECT_ROOT / "code" / "03_figure_generation").exists())

    def test_no_contributor_specific_paths_in_text_files(self) -> None:
        patterns = ("C:\\Users\\", "D:\\\\", "PycharmProjects")
        suffixes = {".py", ".json", ".md", ".txt", ".csv"}
        offenders: list[str] = []
        for path in PROJECT_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            if any(pattern in text for pattern in patterns):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
