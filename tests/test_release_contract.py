from __future__ import annotations

import hashlib
import importlib.util
import json
import py_compile
import tempfile
import unittest
from pathlib import Path


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

    def test_historical_fixed_inputs_match_manuscript_hashes(self) -> None:
        fixed = PROJECT_ROOT / "data" / "fixed_inputs" / "source_boundary"
        expected = {
            "Step3_MonteCarlo_Flux_2018.csv": "7fe3ab1d98e9b43d8e48dc9110a5e4d5955c6f34fd6876457c6ce2d4404463ff",
            "Step4_Coastline_Source_Load_2018.csv": "38b168027580a0469922fa90ac76030e93621ffd3c76e7c8509f7b55f23617c3",
        }
        for filename, checksum in expected.items():
            self.assertEqual(sha256(fixed / filename), checksum)

        model_input = (
            PROJECT_ROOT
            / "data"
            / "fixed_inputs"
            / "model"
            / "Long_Chain_Model_Feature_Table.csv"
        )
        self.assertEqual(
            sha256(model_input),
            "0057c51827e31e01745ffec19141b48ccd1c2f1332df8219b40a2fc3a7be267d",
        )

    def test_step4_is_deterministically_rebuilt_from_fixed_step3(self) -> None:
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
        fixed = PROJECT_ROOT / "data" / "fixed_inputs" / "source_boundary"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "Step4_Coastline_Source_Load_2018.csv"
            module.build_coastline_source_load(fixed / "Step3_MonteCarlo_Flux_2018.csv", output)
            self.assertEqual(sha256(output), sha256(fixed / output.name))

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
        import pandas as pd

        summary = pd.read_csv(
            PROJECT_ROOT
            / "data"
            / "fixed_outputs"
            / "source_attenuation_inference"
            / "Scenario_Prediction_Summary.csv"
        )
        columns = {
            "geometric_mean_paired_change_percent",
            "geometric_mean_paired_change_percent_cluster_CI_lower",
            "geometric_mean_paired_change_percent_cluster_CI_upper",
        }
        self.assertTrue(columns.issubset(summary.columns))
        evaluated = summary[
            summary["output"].eq("Long_Total")
            & summary["scenario_id"].ne("reference")
            & summary["domain_scope"].eq("within_source_minmax")
        ]
        self.assertGreater(len(evaluated), 0)
        self.assertFalse(evaluated[list(columns)].isna().any().any())

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
