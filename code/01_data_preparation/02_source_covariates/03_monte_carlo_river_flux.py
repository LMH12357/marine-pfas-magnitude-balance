from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from source_common import (
    PFAS_COMPOUNDS,
    TAXONOMY,
    read_fingerprints,
    read_river_inventory,
    sha256,
)

DEFAULT_SEED = 42
DEFAULT_ITERATIONS = 1_000_000
INTEGRATION_BATCH_SIZE = 100_000
SEED_ALGORITHM = "PCG64; SeedSequence(master_seed, SHA256(namespace key) uint32 words)"


def default_project_root() -> Path:
    configured = os.environ.get("PFAS_PROJECT_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[3]
    )


def keyed_generator(seed: int, *key: str) -> np.random.Generator:
    """Use persistent names, rather than iteration order, to identify streams."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(encoded).digest(), dtype="<u4")
    sequence = np.random.SeedSequence([int(seed), *map(int, words)])
    return np.random.Generator(np.random.PCG64(sequence))


def clipped_normal_mean(mean: float, standard_deviation: float) -> float:
    """Exact expectation of max(0, Normal(mean, standard_deviation))."""
    if not math.isfinite(mean) or not math.isfinite(standard_deviation):
        raise ValueError("Normal parameters must be finite")
    if mean < 0 or standard_deviation < 0:
        raise ValueError("Normal parameters must be non-negative")
    if standard_deviation == 0:
        return mean
    z = mean / standard_deviation
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return mean * cdf + standard_deviation * density


def integrate_regional_fingerprint(
    row: pd.Series, iterations: int, seed: int
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Integrate normalized non-negative fingerprints once for a region.

    Reusing an estimated regional expectation removes numerical differences
    between nodes governed by the same fingerprint distribution. It does not
    imply identical observed composition or estimate cross-node covariance.
    """
    sea = str(row["Sea_Area"])
    available = [p for p in PFAS_COMPOUNDS if row[f"{p}_Origin"] != "Unavailable"]
    if not available or not bool(row["Source_Eligible"]):
        raise ValueError(f"Regional source fingerprint is not eligible: {sea}")
    for group, compounds in TAXONOMY.items():
        if not set(compounds).intersection(available):
            raise ValueError(f"Fingerprint {sea} does not support {group}")
    parameters = {}
    generators = {}
    donors = {str(row.get(f"{p}_Donor_Sea", "")) for p in available}
    complete_borrowing = all(
        row[f"{p}_Origin"] == "Adjacent_Borrowing" for p in available
    )
    integration_sea = (
        next(iter(donors)) if complete_borrowing and len(donors) == 1 else sea
    )
    for compound in available:
        mean = float(row[f"{compound}_Mean"])
        std = float(row[f"{compound}_Std"])
        if not math.isfinite(mean) or not math.isfinite(std) or mean < 0 or std < 0:
            raise ValueError(f"Invalid fingerprint parameters: {sea}, {compound}")
        parameters[compound] = (mean, std)
        generators[compound] = keyed_generator(
            seed, "fingerprint", integration_sea, compound
        )
    sums = np.zeros(len(available), dtype=np.float64)
    square_sums = np.zeros_like(sums)
    for start in range(0, iterations, INTEGRATION_BATCH_SIZE):
        size = min(INTEGRATION_BATCH_SIZE, iterations - start)
        samples = np.empty((size, len(available)), dtype=np.float64)
        for index, compound in enumerate(available):
            mean, std = parameters[compound]
            samples[:, index] = np.maximum(
                0.0, generators[compound].normal(mean, std, size)
            )
        totals = samples.sum(axis=1)
        zero_count = int(np.count_nonzero(totals <= 0))
        if zero_count:
            raise ValueError(
                f"{sea}: {zero_count} fingerprint draws have no positive mass; "
                "a normalized composition is undefined. No mass was discarded."
            )
        samples /= totals[:, None]
        sums += samples.sum(axis=0)
        square_sums += np.square(samples).sum(axis=0)
    estimated_means = sums / iterations
    if not np.isclose(estimated_means.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Regional fingerprint does not conserve mass: {sea}")
    # Correct only floating-point summation residue, not the modeled distribution.
    estimated_means /= estimated_means.sum()
    if iterations > 1:
        variance = np.maximum(
            0.0, (square_sums - np.square(sums) / iterations) / (iterations - 1)
        )
        mcse = np.sqrt(variance / iterations)
    else:
        mcse = np.full(len(available), np.nan)
    fractions = dict.fromkeys(PFAS_COMPOUNDS, 0.0)
    fractions.update(zip(available, map(float, estimated_means)))
    errors = dict(zip(available, map(float, mcse)))
    diagnostics = []
    for compound in PFAS_COMPOUNDS:
        origin = str(row[f"{compound}_Origin"])
        diagnostics.append(
            {
                "Sea_Area": sea,
                "Compound": compound,
                "Allocation_Status": (
                    "Unavailable" if origin == "Unavailable" else "Allocated"
                ),
                "Fingerprint_Origin": origin,
                "Observed_Count": int(row[f"{compound}_Count"]),
                "Donor_Sea": row.get(f"{compound}_Donor_Sea", ""),
                "Donor_Count": int(row.get(f"{compound}_Donor_Count", 0)),
                "Concentration_Mean_ng_L": row[f"{compound}_Mean"],
                "Concentration_SD_ng_L": row[f"{compound}_Std"],
                "Expected_Share": fractions[compound],
                "Share_Integration_MCSE": errors.get(compound, np.nan),
                "Fingerprint_Integration_Draws": iterations,
                "Fingerprint_Integration_Sea_Key": integration_sea,
            }
        )
    return fractions, diagnostics


def _finite_nonnegative(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"{label}: {column} must contain finite non-negative values"
            )


def _consensus_value(group: pd.DataFrame, column: str, key: tuple) -> float:
    values = group[column].to_numpy(dtype=np.float64)
    if not np.allclose(values, values[0], rtol=1e-12, atol=0.0):
        raise ValueError(f"Shared runoff grid {key} has inconsistent {column}")
    # Rivers sharing one discharge grid represent the same grid flow.
    return float(values[0])


def run_monte_carlo_baseline_flux(
    input_directory: Path,
    output_file: Path,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    rivers_file: Path | None = None,
    *,
    hydrology_diagnostic_iterations: int = 0,
) -> pd.DataFrame:
    """Calculate the fixed-2018 compound-resolved expected river inventory.

    Hydrological and chemical variables are independent under the original
    reconstruction model. Thus E[Q * fraction] = E[Q] * E[fraction].
    Hydrological means are evaluated analytically; normalized fingerprint
    means retain the original independent Normal draws and zero clipping.
    """
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer")
    if (
        isinstance(hydrology_diagnostic_iterations, bool)
        or not isinstance(hydrology_diagnostic_iterations, (int, np.integer))
        or hydrology_diagnostic_iterations < 0
    ):
        raise ValueError(
            "hydrology_diagnostic_iterations must be a non-negative integer"
        )
    keyed_generator(seed, "validation")
    input_directory = input_directory.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    rivers_file = (
        (rivers_file or input_directory / "100_Rivers_2018.xlsx").expanduser().resolve()
    )
    inputs = {
        "runoff": input_directory / "Step1_Annual_Runoff_2014_2025.csv",
        "mapping": input_directory / "Step2_River_Sea_Mapping.csv",
        "fingerprint": input_directory / "Step2_Regional_Fingerprints.csv",
        "rivers": rivers_file,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + "; ".join(missing))
    if any("fixed_inputs" in path.parts for path in inputs.values()):
        raise ValueError(
            "Use current Step 1/2 outputs; archived fixed_inputs are not accepted"
        )
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    df_runoff = pd.read_csv(inputs["runoff"])
    df_runoff = df_runoff.loc[df_runoff["Year"] == 2018].copy()
    if df_runoff.empty:
        raise ValueError("Runoff input does not contain the 2018 baseline")
    df_mapping = pd.read_csv(inputs["mapping"])
    df_fp = read_fingerprints(inputs["fingerprint"])
    read_river_inventory(inputs["rivers"])
    _finite_nonnegative(df_runoff, ["Annual_Q_10^8m3", "Q_SD_10^8m3"], "Runoff")
    for frame, label in [(df_runoff, "Runoff"), (df_mapping, "Mapping")]:
        for column in ["Grid_Lon", "Grid_Lat"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"{label}: {column} must be finite")
    if df_mapping.duplicated(["Grid_Lon", "Grid_Lat"]).any():
        raise ValueError(
            "Mapping must have exactly one record per physical source grid"
        )
    if df_fp["Sea_Area"].duplicated().any():
        raise ValueError("Regional fingerprints must have unique sea names")
    mapping = df_mapping.set_index(["Grid_Lon", "Grid_Lat"])
    fingerprints = df_fp.set_index("Sea_Area", drop=False)
    regional_fractions = {}
    fingerprint_diagnostics = []
    for sea in sorted(df_fp.loc[df_fp["Source_Eligible"], "Sea_Area"].astype(str)):
        fractions, diagnostics = integrate_regional_fingerprint(
            fingerprints.loc[sea], iterations, seed
        )
        regional_fractions[sea] = fractions
        fingerprint_diagnostics.extend(diagnostics)
    results = []
    node_diagnostics = []
    exclusions = []
    grouped = df_runoff.groupby(["Grid_Lon", "Grid_Lat", "Year"], sort=True)
    for key, group in grouped:
        grid_lon, grid_lat, year = key
        annual_q = _consensus_value(group, "Annual_Q_10^8m3", key)
        q_sd = _consensus_value(group, "Q_SD_10^8m3", key)
        if (grid_lon, grid_lat) not in mapping.index:
            raise ValueError(f"No topology mapping for source node {key}")
        grid_data = mapping.loc[(grid_lon, grid_lat)]
        sea = str(grid_data["Sea_Area"])
        if sea not in regional_fractions:
            exclusions.append(
                {
                    "grid": [float(grid_lon), float(grid_lat), int(year)],
                    "sea": sea,
                    "reason": "Fingerprint unavailable or outside modeled seas",
                }
            )
            continue
        concentration = float(grid_data["∑PFAAs (ng/L)"])
        if not math.isfinite(concentration) or concentration < 0:
            raise ValueError(f"Invalid source total concentration at {key}")
        if annual_q == 0 or concentration == 0:
            exclusions.append(
                {
                    "grid": [float(grid_lon), float(grid_lat), int(year)],
                    "sea": sea,
                    "reason": "Zero baseline flow or concentration",
                }
            )
            continue
        expected_q = clipped_normal_mean(annual_q, q_sd)
        total_flux = expected_q * concentration * 0.1
        if not math.isfinite(total_flux):
            raise ValueError(f"Source flux exceeds the finite numeric range at {key}")
        fractions = regional_fractions[sea]
        row = fingerprints.loc[sea]
        record = {
            "Included_Rivers": ", ".join(sorted(set(group["River"].astype(str)))),
            "Grid_Lon": float(grid_lon),
            "Grid_Lat": float(grid_lat),
            "Sea_Area": sea,
            "Year": int(year),
            "Annual_Q_10^8m3": annual_q,
            "Q_SD_10^8m3": q_sd,
            "Total_PFAAs_Conc_ng_L": concentration,
            "Expected_Total_Flux_Kg": total_flux,
        }
        for compound in PFAS_COMPOUNDS:
            record[f"{compound}_Flux_Kg"] = total_flux * fractions[compound]
            record[f"{compound}_Allocation_Status"] = (
                "Unavailable"
                if row[f"{compound}_Origin"] == "Unavailable"
                else "Allocated"
            )
        compound_total = math.fsum(
            record[f"{compound}_Flux_Kg"] for compound in PFAS_COMPOUNDS
        )
        if not math.isclose(compound_total, total_flux, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Compound fluxes do not conserve total mass at {key}")
        results.append(record)
        diagnostic = {
            "Grid_Lon": float(grid_lon),
            "Grid_Lat": float(grid_lat),
            "Sea_Area": sea,
            "Year": int(year),
            "Rivers_Sharing_Grid": len(group),
            "Expected_Clipped_Q_10^8m3": expected_q,
            "Hydrology_Diagnostic_Draws": hydrology_diagnostic_iterations,
            "Hydrology_Diagnostic_Seed_Key": json.dumps(
                [
                    "hydrology",
                    float(grid_lon).hex(),
                    float(grid_lat).hex(),
                    str(int(year)),
                ]
            ),
        }
        if hydrology_diagnostic_iterations:
            generator = keyed_generator(
                seed,
                "hydrology",
                float(grid_lon).hex(),
                float(grid_lat).hex(),
                str(int(year)),
            )
            q_draws = np.maximum(
                0.0, generator.normal(annual_q, q_sd, hydrology_diagnostic_iterations)
            )
            diagnostic.update(
                {
                    "Hydrology_MC_Mean_10^8m3": float(q_draws.mean()),
                    "Hydrology_MC_Q025_10^8m3": float(np.quantile(q_draws, 0.025)),
                    "Hydrology_MC_Q975_10^8m3": float(np.quantile(q_draws, 0.975)),
                }
            )
        node_diagnostics.append(diagnostic)
    if not results:
        raise ValueError("No eligible nonzero 2018 source nodes were reconstructed")
    if any(sha256(path) != input_hashes[name] for name, path in inputs.items()):
        raise RuntimeError("A reconstruction input changed during execution")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    regional_file = output_file.with_suffix(".fingerprint_diagnostics.csv")
    pd.DataFrame(fingerprint_diagnostics).to_csv(
        regional_file, index=False, encoding="utf-8-sig"
    )
    node_file = output_file.with_suffix(".hydrology_diagnostics.csv")
    pd.DataFrame(node_diagnostics).to_csv(node_file, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "2018 compound-resolved expected river-flux reconstruction",
        "seed": int(seed),
        "runtime_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "random_generator": SEED_ALGORITHM,
        "fingerprint_iterations_per_region": int(iterations),
        "integration_batch_size": INTEGRATION_BATCH_SIZE,
        "hydrology_diagnostic_iterations_per_node": int(
            hydrology_diagnostic_iterations
        ),
        "expectation": "Independent Q and fingerprint: E[Q*f]=E[Q]*E[f]; E[max(0,Normal)] evaluated analytically",
        "fingerprint_distribution": "Independent compound normal draws clipped at zero, then normalized within each draw",
        "cross_node_covariance": "Not estimated; shared regional integration draws are only a numerical integration device",
        "complete_borrowing_integration": "An entirely borrowed fingerprint uses its donor sea's numerical integration streams, avoiding Monte Carlo differences between identical distributions",
        "unavailable_components": "Zero allocation to unavailable components is a calculation convention, not an observed zero concentration",
        "hydrology_interval_scope": "Optional hydrological diagnostics do not propagate fingerprint uncertainty",
        "compounds": list(PFAS_COMPOUNDS),
        "taxonomy": {name: list(compounds) for name, compounds in TAXONOMY.items()},
        "source_nodes": len(frame),
        "excluded_nodes": exclusions,
        "inputs": {
            name: {"file": str(path), "sha256": input_hashes[name]}
            for name, path in inputs.items()
        },
        "outputs": {
            "flux": {"file": output_file.name, "sha256": sha256(output_file)},
            "fingerprint_diagnostics": {
                "file": regional_file.name,
                "sha256": sha256(regional_file),
            },
            "hydrology_diagnostics": {
                "file": node_file.name,
                "sha256": sha256(node_file),
            },
        },
    }
    output_file.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported {len(frame)} expected-flux source nodes to {output_file}")
    return frame


def parse_arguments() -> argparse.Namespace:
    root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Reconstruct current-input 2018 compound-resolved expected river fluxes."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--rivers-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hydrology-diagnostic-iterations", type=int, default=0)
    arguments = parser.parse_args()
    arguments.project_root = arguments.project_root.expanduser().resolve()
    output_root = (
        Path(os.environ.get("PFAS_OUTPUT_ROOT", arguments.project_root / "outputs"))
        .expanduser()
        .resolve()
    )
    data_root = (
        Path(os.environ.get("PFAS_DATA_ROOT", arguments.project_root / "data"))
        .expanduser()
        .resolve()
    )
    arguments.input_dir = arguments.input_dir or output_root / "exogenous"
    arguments.rivers_file = (
        arguments.rivers_file or data_root / "raw" / "100_Rivers_2018.xlsx"
    )
    arguments.output = (
        arguments.output or output_root / "exogenous" / "Step3_MonteCarlo_Flux_2018.csv"
    )
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    run_monte_carlo_baseline_flux(
        args.input_dir,
        args.output,
        args.iterations,
        args.seed,
        args.rivers_file,
        hydrology_diagnostic_iterations=args.hydrology_diagnostic_iterations,
    )
