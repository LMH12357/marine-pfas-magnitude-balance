"""Run the coastal-source preparation sequence from explicit local settings."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

from source_common import sha256


def load_module(filename):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path, skip_discharge=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("base_year", 2018) != 2018:
        raise ValueError("This source inventory uses the fixed 2018 baseline")
    output = Path(config["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "execution.log", encoding="utf-8"),
        ],
        force=True,
    )
    inputs = {
        key: Path(config[key])
        for key in ("rivers_file", "estuary_file", "bulletin_file")
    }
    identities = {key: sha256(path) for key, path in inputs.items()}
    modules = {
        key: load_module(name)
        for key, name in [
            ("01", "01_glofas_discharge_extraction.py"),
            ("02", "02_river_fingerprint_construction.py"),
            ("03", "03_monte_carlo_river_flux.py"),
            ("04", "04_coastline_source_descriptors.py"),
        ]
    }
    step1 = output / "Step1_Annual_Runoff_2014_2025.csv"
    if skip_discharge:
        manifest = json.loads(
            step1.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        assert sha256(step1) == manifest["output"]["sha256"]
        assert manifest["start_year"] == config.get("start_year", 2014) and manifest[
            "end_year"
        ] == config.get("end_year", 2025)
        for entry in manifest["inputs"]:
            assert sha256(Path(entry["file"])) == entry["sha256"], entry["file"]
        logging.info("Verified existing current-input annual discharge output")
    else:
        modules["01"].run_discharge(config)
    logging.info("START source topology and regional fingerprints")
    mapping = modules["02"].build_topology_mapping(
        inputs["estuary_file"],
        inputs["rivers_file"],
        step1,
        output / "Step2_River_Sea_Mapping.csv",
    )
    fingerprint = modules["02"].extract_regional_fingerprints(
        inputs["estuary_file"], output / "Step2_Regional_Fingerprints.csv"
    )
    logging.info(
        "START expected flux integration, seed=%s, draws=%s per region",
        config["seed"],
        config["iterations"],
    )
    flux = modules["03"].run_monte_carlo_baseline_flux(
        output,
        output / "Step3_MonteCarlo_Flux_2018.csv",
        iterations=config["iterations"],
        seed=config["seed"],
        rivers_file=inputs["rivers_file"],
        hydrology_diagnostic_iterations=config.get(
            "hydrology_diagnostic_iterations", 0
        ),
    )
    logging.info("START source-class aggregation")
    sources = modules["04"].build_coastline_source_load(
        output / "Step3_MonteCarlo_Flux_2018.csv",
        output / "Step4_Coastline_Source_Load_2018.csv",
    )
    for key, path in inputs.items():
        if sha256(path) != identities[key]:
            raise RuntimeError(f"Input changed during the run: {key}")
    record = {
        "scope": "02 source covariates",
        "base_year": 2018,
        "config": config,
        "input_hashes": identities,
        "code": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "mapping_nodes": len(mapping),
        "regional_fingerprints": len(fingerprint),
        "flux_nodes": len(flux),
        "source_nodes": len(sources),
        "outputs": {
            p.name: sha256(p)
            for p in sorted(output.glob("*.csv"))
            if p.name.startswith("Step") or p.name == "Source_Group_Dictionary.csv"
        },
    }
    (output / "source_execution_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info(
        "DONE: %s baseline source nodes; input workbooks unchanged", len(sources)
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--skip-discharge",
        action="store_true",
        help="Verify and reuse an existing current-input Step1 output",
    )
    args = parser.parse_args()
    run(args.config, args.skip_discharge)
