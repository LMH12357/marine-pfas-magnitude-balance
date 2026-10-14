"""Run the marine PFAS analysis with explicit inputs and isolated output stages.

Smoke runs test software integration; their estimates are not publication results.
Completed stages are reused only when their inputs, code and products match.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "code" / "02_modeling_and_validation"
EVIDENCE = ROOT / "code" / "03_evidence_strengthening"
STAGES = (
    "routing",
    "prepare",
    "primary",
    "boundary",
    "inference",
    "preprocessing",
    "composition",
    "coordinates",
    "evidence",
    "regimes",
    "source-ratios",
)
DEPENDENCIES = {
    "prepare": ("routing",),
    "primary": ("prepare",),
    "inference": ("primary", "boundary"),
    "preprocessing": ("prepare",),
    "composition": ("primary",),
    "coordinates": ("primary",),
    "evidence": ("primary", "inference"),
    "regimes": ("evidence",),
    "source-ratios": ("inference",),
}
DIRECTORIES = dict(
    zip(
        STAGES,
        (
            "00_coastal_routing",
            "00_prepared_inputs",
            "01_primary_model",
            "02_source_attenuation_boundary",
            "03_source_attenuation_inference",
            "04_preprocessing_sensitivity",
            "05_compositional_validation",
            "06_magnitude_balance_validation",
            "evidence_strengthening",
            "response_regimes",
            "source_ratio_diagnostic",
        ),
    )
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def acquire_run_lock(directory):
    """Never let concurrent launchers write the same execution journal."""
    path = directory / ".pipeline.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"An execution lock exists at {path}; verify the recorded process before removing a stale lock"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_utc": utc_now()}, stream)

    def release():
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("pid") == os.getpid():
                path.unlink()

    atexit.register(release)
    return release


def resolve_inputs(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    paths = {}
    for name in (
        "data",
        "source_load",
        "compound_inventory",
        "boundary",
        "scenario_config",
        "routing_config",
    ):
        path = Path(config[name]).expanduser()
        path = (
            (config_path.parent / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        paths[name] = path
    return paths


def commands(inputs, output, mode, jobs):
    destinations = {
        stage: output / directory for stage, directory in DIRECTORIES.items()
    }
    prepared = destinations["prepare"] / "Long_Chain_Model_Feature_Table.csv"
    features = destinations["primary"] / "Long_Chain_Model_Feature_Table.csv"
    scenarios = destinations["inference"] / "Scenario_Paired_Predictions.csv"
    routing_directory = destinations["routing"]
    common = [
        "--data",
        inputs["data"],
        "--source-load",
        inputs["source_load"],
        "--boundary",
        inputs["boundary"],
        "--routing-directory",
        routing_directory,
    ]
    result = {
        "routing": [
            ROOT / "code" / "coastal_routing.py",
            "--data",
            inputs["data"],
            "--source-load",
            inputs["source_load"],
            "--boundary",
            inputs["boundary"],
            "--config",
            inputs["routing_config"],
        ],
        "prepare": [MODEL / "01_primary_model_training.py", *common, "--prepare-only"],
        "primary": [
            MODEL / "01_primary_model_training.py",
            *common,
            "--feature-table",
            prepared,
            "--n-jobs",
            jobs,
        ],
        "boundary": [
            MODEL / "02_source_attenuation_boundary.py",
            "--compound-inventory",
            inputs["compound_inventory"],
            "--config",
            inputs["scenario_config"],
        ],
        "inference": [
            MODEL / "03_source_attenuation_inference.py",
            "--model-package",
            destinations["primary"] / "Long_Chain_Magnitude_Balance_Model_Package.pkl",
            "--scenario-directory",
            destinations["boundary"],
            "--boundary",
            inputs["boundary"],
            "--routing-directory",
            routing_directory,
            "--training-script",
            MODEL / "01_primary_model_training.py",
        ],
        "preprocessing": [
            MODEL / "04_preprocessing_sensitivity.py",
            *common,
            "--feature-table",
            prepared,
            "--n-jobs",
            jobs,
        ],
        "composition": [
            MODEL / "05_compositional_validation.py",
            "--data",
            inputs["data"],
            "--feature-table",
            features,
            "--n-jobs",
            jobs,
        ],
        "coordinates": [
            MODEL / "06_magnitude_balance_validation.py",
            "--feature-table",
            features,
            "--n-jobs",
            jobs,
        ],
        "evidence": [
            EVIDENCE / "wr_evidence_strengthening.py",
            "--scenario-input",
            scenarios,
            "--shap-input",
            destinations["primary"] / "Nested_OOF_SHAP_Values.csv",
        ],
        "regimes": [
            EVIDENCE / "wr_response_regimes.py",
            "--input",
            destinations["evidence"] / "oof_selected_feature_values.csv",
        ],
        "source-ratios": [
            EVIDENCE / "source_ratio_diagnostic.py",
            "--predictions",
            scenarios,
            "--paired-shap",
            destinations["inference"] / "Scenario_Paired_SHAP.csv",
        ],
    }
    if mode == "smoke":
        for stage in ("primary", "composition", "coordinates"):
            result[stage] += [
                "--quick",
                "--outer-folds",
                2,
                "--inner-folds",
                2,
                "--bootstrap-iterations",
                50,
            ]
        result["primary"] += ["--no-shap-interactions"]
        result["inference"] += ["--bootstrap-iterations", 50]
        result["preprocessing"] += ["--prepare-only"]
    return {
        stage: [
            sys.executable,
            "-u",
            *map(str, command),
            "--output-dir",
            str(destinations[stage]),
        ]
        for stage, command in result.items()
    }


def product_hashes(directory):
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def validate_completed(stage, records, output, signature):
    for dependency in DEPENDENCIES.get(stage, ()):
        validate_completed(dependency, records, output, signature)
    record = records.get(stage, {})
    if record.get("status") != "completed" or record.get("signature") != signature:
        raise ValueError(
            f"Stage {stage} has no matching completed execution; use a new output directory if inputs/code changed"
        )
    actual = product_hashes(output / DIRECTORIES[stage])
    if not actual or actual != record.get("products"):
        raise ValueError(f"Stage {stage} products are missing or changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--n-jobs", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    config_path = arguments.config.resolve()
    inputs = resolve_inputs(config_path)
    output = arguments.output_dir.resolve()
    if any(output == path.parent or output in path.parents for path in inputs.values()):
        raise ValueError("Output must be separate from input directories")
    output.mkdir(parents=True, exist_ok=True)
    acquire_run_lock(output)
    journal_path = output / "Pipeline_Status.json"
    journal = (
        json.loads(journal_path.read_text())
        if journal_path.exists()
        else {"created_utc": utc_now(), "stages": {}}
    )
    code_paths = sorted((ROOT / "code").rglob("*.py")) + [Path(__file__).resolve()]
    fingerprints = {
        "inputs": {key: sha256(path) for key, path in inputs.items()},
        "code": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        "reference": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "data" / "reference").rglob("*"))
            if path.is_file()
        },
        "runtime": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "xgboost",
                "shap",
                "geopandas",
                "shapely",
                "pyproj",
                "openpyxl",
                "joblib",
            )
        },
        "python": sys.version,
    }
    for sidecar in inputs["boundary"].parent.glob(inputs["boundary"].stem + ".*"):
        if sidecar.is_file():
            fingerprints["inputs"][f"boundary_{sidecar.suffix}"] = sha256(sidecar)
    fingerprint_payload = {
        **fingerprints,
        "mode": arguments.mode,
        "n_jobs": arguments.n_jobs,
    }
    signature = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    if journal.get("signature", signature) != signature:
        raise ValueError(
            "Output directory belongs to different inputs/code/settings; choose a new directory"
        )
    journal.update(
        {
            "signature": signature,
            **fingerprint_payload,
            "publication_results": arguments.mode == "production",
            "pid": os.getpid(),
        }
    )
    write_json(journal_path, journal)
    environment = os.environ.copy()
    environment.update(
        {
            "PFAS_PROJECT_ROOT": str(ROOT),
            "PFAS_OUTPUT_ROOT": str(output),
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "LOKY_MAX_CPU_COUNT": str(arguments.n_jobs),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    stage_commands = commands(inputs, output, arguments.mode, arguments.n_jobs)
    for stage in STAGES:
        if stage not in arguments.stages:
            continue
        if journal["stages"].get(stage, {}).get("status") == "completed":
            validate_completed(stage, journal["stages"], output, signature)
            print(f"[{utc_now()}] REUSED verified {stage}", flush=True)
            continue
        for dependency in DEPENDENCIES.get(stage, ()):
            validate_completed(dependency, journal["stages"], output, signature)
        destination = output / DIRECTORIES[stage]
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"Incomplete/untracked outputs at {destination}; preserve them and choose a fresh output directory"
            )
        record = {
            "status": "running",
            "started_utc": utc_now(),
            "signature": signature,
            "command": stage_commands[stage],
        }
        journal["stages"][stage] = record
        write_json(journal_path, journal)
        print(f"[{utc_now()}] START {stage}", flush=True)
        started = time.monotonic()
        with (output / f"{stage}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                stage_commands[stage],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record["pid"] = process.pid
            write_json(journal_path, journal)
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(
                            f"[{utc_now()}] RUNNING {stage}: {int(time.monotonic() - started)} s; log={output / (stage + '.log')}",
                            flush=True,
                        )
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                record["status"] = "interrupted"
                write_json(journal_path, journal)
                raise
        record.update(
            {
                "exit_code": process.returncode,
                "finished_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "status": "completed" if process.returncode == 0 else "failed",
            }
        )
        if process.returncode == 0:
            record["products"] = product_hashes(destination)
            if not record["products"]:
                record["status"] = "failed"
        write_json(journal_path, journal)
        print(f"[{utc_now()}] {record['status'].upper()} {stage}", flush=True)
        if record["status"] != "completed":
            raise RuntimeError(
                f"Stage {stage} failed; inspect {output / (stage + '.log')}"
            )
        for key, path in inputs.items():
            if sha256(path) != fingerprints["inputs"][key]:
                raise RuntimeError(f"Input changed during execution: {key}")
    print(
        f"Selected {arguments.mode} stages completed. Status: {journal_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
