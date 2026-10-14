"""Orchestration checks with temporary files, without scientific model fits."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pipeline_runner_test", ROOT / "run_analysis.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerContracts(unittest.TestCase):
    def inputs(self, directory):
        values = {}
        for name in (
            "data",
            "source_load",
            "compound_inventory",
            "boundary",
            "scenario_config",
            "routing_config",
        ):
            path = directory / (name + ".fixture")
            path.write_text(name, encoding="utf-8")
            values[name] = path
        return values

    def test_relative_config_inputs_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = self.inputs(directory)
            config = directory / "config.json"
            config.write_text(
                json.dumps({name: path.name for name, path in inputs.items()}),
                encoding="utf-8",
            )
            self.assertEqual(
                runner.resolve_inputs(config),
                {key: value.resolve() for key, value in inputs.items()},
            )
            inputs["source_load"].unlink()
            with self.assertRaises(FileNotFoundError):
                runner.resolve_inputs(config)

    def test_stage_commands_isolate_outputs_and_smoke_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = self.inputs(directory)
            output = directory / "outputs"
            production = runner.commands(inputs, output, "production", 3)
            smoke = runner.commands(inputs, output, "smoke", 3)
            self.assertEqual(set(production), set(runner.STAGES))
            for stage, command in production.items():
                self.assertEqual(command[-2], "--output-dir")
                self.assertEqual(Path(command[-1]), output / runner.DIRECTORIES[stage])
                self.assertNotIn("--quick", command)
            for stage in ("primary", "composition", "coordinates"):
                self.assertIn("--quick", smoke[stage])
                position = smoke[stage].index("--bootstrap-iterations")
                self.assertEqual(smoke[stage][position + 1], "50")
            self.assertIn(
                str(output / "01_primary_model" / "Long_Chain_Model_Feature_Table.csv"),
                production["coordinates"],
            )
            routing_directory = str(output / "00_coastal_routing")
            self.assertEqual(runner.STAGES[0], "routing")
            self.assertEqual(runner.DEPENDENCIES["prepare"], ("routing",))
            for stage in ("prepare", "primary", "preprocessing", "inference"):
                position = production[stage].index("--routing-directory")
                self.assertEqual(production[stage][position + 1], routing_directory)
            position = production["routing"].index("--config")
            self.assertEqual(
                production["routing"][position + 1], str(inputs["routing_config"])
            )

    def test_product_reuse_requires_matching_signature_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / runner.DIRECTORIES["prepare"]
            directory.mkdir()
            product = directory / "result.txt"
            product.write_text("fixture", encoding="utf-8")
            records = {
                "prepare": {
                    "status": "completed",
                    "signature": "signature",
                    "products": runner.product_hashes(directory),
                }
            }
            routing = output / runner.DIRECTORIES["routing"]
            routing.mkdir()
            (routing / "Routing_Manifest.json").write_text("{}", encoding="utf-8")
            records["routing"] = {
                "status": "completed",
                "signature": "signature",
                "products": runner.product_hashes(routing),
            }
            runner.validate_completed("prepare", records, output, "signature")
            with self.assertRaisesRegex(ValueError, "matching"):
                runner.validate_completed("prepare", records, output, "different")
            product.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.validate_completed("prepare", records, output, "signature")
            product.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                runner.validate_completed("prepare", records, output, "signature")

    def test_prepare_cannot_reuse_without_frozen_routing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / runner.DIRECTORIES["prepare"]
            directory.mkdir()
            (directory / "features.csv").write_text("fixture", encoding="utf-8")
            records = {
                "prepare": {
                    "status": "completed",
                    "signature": "signature",
                    "products": runner.product_hashes(directory),
                }
            }
            with self.assertRaisesRegex(ValueError, "routing"):
                runner.validate_completed("prepare", records, output, "signature")


if __name__ == "__main__":
    unittest.main()
