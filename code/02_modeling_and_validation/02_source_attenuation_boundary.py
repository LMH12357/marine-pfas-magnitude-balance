from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


VERSION = "v6.0"
CORE_PATH = Path(__file__).resolve().with_name("_source_attenuation_core.py")


def load_transition_core():
    specification = importlib.util.spec_from_file_location("pfas_class_transition_core", CORE_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load scenario core: {CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


CORE = load_transition_core()
TAXONOMY = CORE.TAXONOMY
SOURCE_COLUMNS = CORE.SOURCE_COLUMNS
all_compounds = CORE.all_compounds
flux_column = CORE.flux_column
validate_inventory = CORE.validate_inventory
load_configuration = CORE.load_configuration
load_usage_evidence = CORE.load_usage_evidence
derive_evidence_tables = CORE.derive_evidence_tables
scenario_multipliers = CORE.scenario_multipliers
apply_multipliers = CORE.apply_multipliers
aggregate_source_table = CORE.aggregate_source_table


def run(compound_inventory: Path, configuration: Path, output_directory: Path, overwrite: bool) -> None:
    CORE.run(compound_inventory, configuration, output_directory, overwrite)


def parse_arguments() -> argparse.Namespace:
    project_root = Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
    data_root = Path(os.environ.get("PFAS_DATA_ROOT", project_root / "data")).expanduser().resolve()
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()
    fixed_source = data_root / "fixed_inputs" / "source_boundary"
    parser = argparse.ArgumentParser()
    parser.add_argument("--compound-inventory", type=Path, default=fixed_source / "Step3_MonteCarlo_Flux_2018.csv")
    parser.add_argument("--config", type=Path, default=project_root / "code" / "02_modeling_and_validation" / "config" / "source_attenuation.json")
    parser.add_argument("--output-dir", type=Path, default=output_root / "02_source_attenuation_boundary")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run(arguments.compound_inventory, arguments.config, arguments.output_dir, arguments.overwrite)
