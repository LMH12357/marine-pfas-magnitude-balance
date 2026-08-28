from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 42
DEFAULT_ITERATIONS = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_project_root() -> Path:
    configured = os.environ.get("PFAS_PROJECT_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[3]


def run_monte_carlo_baseline_flux(
    input_directory: Path,
    output_file: Path,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    rivers_file: Path | None = None,
) -> pd.DataFrame:
    """Reconstruct the fixed-2018 compound-resolved river flux inventory.

    The archived CSV used by the manuscript is distributed separately as a
    fixed derived input. ``seed`` is nevertheless explicit so any new
    regeneration is deterministic and auditable.
    """
    if iterations < 1:
        raise ValueError("iterations must be a positive integer")

    input_directory = input_directory.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    rivers_file = (rivers_file or input_directory / "100_Rivers_2018.xlsx").expanduser().resolve()
    inputs = {
        "runoff": input_directory / "Step1_Annual_Runoff_2014_2025.csv",
        "mapping": input_directory / "Step2_River_Sea_Mapping.csv",
        "fingerprint": input_directory / "Step2_Regional_Fingerprints.csv",
        "rivers": rivers_file,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + "; ".join(missing))

    df_runoff = pd.read_csv(inputs["runoff"])
    df_runoff = df_runoff[df_runoff["Year"] == 2018]
    df_mapping = pd.read_csv(inputs["mapping"])
    df_fp = pd.read_csv(inputs["fingerprint"])
    pd.read_excel(inputs["rivers"])  # Validate the declared 2018 inventory input.

    pfas_list = [column.replace("_Mean", "") for column in df_fp.columns if "_Mean" in column]
    fp_dict: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in df_fp.iterrows():
        sea = row["Sea_Area"]
        fp_dict[sea] = {
            pfas: {
                "mean": float(row[f"{pfas}_Mean"]),
                "std": float(row[f"{pfas}_Std"]),
            }
            for pfas in pfas_list
        }

    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []
    groups = df_runoff.groupby(["Grid_Lon", "Grid_Lat", "Year"], sort=True)
    for (grid_lon, grid_lat, year), group in groups:
        annual_q = float(group["Annual_Q_10^8m3"].iloc[0])
        q_sd = float(group["Q_SD_10^8m3"].iloc[0])
        if annual_q <= 0:
            continue

        match = df_mapping[
            (df_mapping["Grid_Lon"] == grid_lon) & (df_mapping["Grid_Lat"] == grid_lat)
        ]
        if match.empty:
            continue
        grid_data = match.iloc[0]
        sea_area = grid_data["Sea_Area"]
        if sea_area not in fp_dict or sea_area == "OpenOcean":
            continue

        if "∑PFAAs (ng/L)" in grid_data and pd.notna(grid_data["∑PFAAs (ng/L)"]):
            total_concentration = float(grid_data["∑PFAAs (ng/L)"])
        else:
            total_concentration = sum(
                float(grid_data[pfas])
                for pfas in pfas_list
                if pfas in grid_data and pd.notna(grid_data[pfas])
            )
        if total_concentration <= 0:
            continue

        q_samples = np.maximum(0.0, rng.normal(annual_q, q_sd, iterations))
        total_flux_samples = q_samples * total_concentration * 0.1

        sampled_matrix = np.zeros((iterations, len(pfas_list)))
        for index, pfas in enumerate(pfas_list):
            parameters = fp_dict[sea_area][pfas]
            sampled_matrix[:, index] = np.maximum(
                0.0, rng.normal(parameters["mean"], parameters["std"], iterations)
            )
        row_sums = sampled_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        normalized_matrix = sampled_matrix / row_sums[:, np.newaxis]
        expected_flux = (total_flux_samples[:, np.newaxis] * normalized_matrix).mean(axis=0)

        record: dict[str, object] = {
            "Included_Rivers": ", ".join(sorted(set(group["River"]))),
            "Grid_Lon": grid_lon,
            "Grid_Lat": grid_lat,
            "Sea_Area": sea_area,
            "Year": year,
            "Annual_Q_10^8m3": round(annual_q, 2),
            "Q_SD_10^8m3": round(q_sd, 2),
            "Total_PFAAs_Conc_ng_L": round(total_concentration, 2),
            "Expected_Total_Flux_Kg": round(float(total_flux_samples.mean()), 2),
        }
        record.update({f"{pfas}_Flux_Kg": float(expected_flux[index]) for index, pfas in enumerate(pfas_list)})
        results.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "2018 Monte Carlo river-flux reconstruction",
        "seed": seed,
        "iterations_per_node": iterations,
        "rows": len(frame),
        "inputs": {name: {"file": path.name, "sha256": sha256(path)} for name, path in inputs.items()},
        "output": {"file": output_file.name, "sha256": sha256(output_file)},
    }
    output_file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(frame)} source nodes to {output_file}")
    return frame


def parse_arguments() -> argparse.Namespace:
    root = default_project_root()
    parser = argparse.ArgumentParser(description="Reconstruct the 2018 compound-resolved river flux inventory.")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--rivers-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    arguments.project_root = arguments.project_root.expanduser().resolve()
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", arguments.project_root / "outputs")).expanduser().resolve()
    data_root = Path(os.environ.get("PFAS_DATA_ROOT", arguments.project_root / "data")).expanduser().resolve()
    arguments.input_dir = arguments.input_dir or output_root / "exogenous"
    arguments.rivers_file = arguments.rivers_file or data_root / "raw" / "100_Rivers_2018.xlsx"
    arguments.output = arguments.output or output_root / "exogenous" / "Step3_MonteCarlo_Flux_2018.csv"
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    run_monte_carlo_baseline_flux(args.input_dir, args.output, args.iterations, args.seed, args.rivers_file)
