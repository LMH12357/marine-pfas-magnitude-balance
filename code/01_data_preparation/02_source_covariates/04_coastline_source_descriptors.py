from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

from source_common import (
    BASE_YEAR,
    PFAS_COMPOUNDS,
    TAXONOMY,
    numeric_column,
    require_columns,
    write_manifest,
)


def default_project_root() -> Path:
    configured = os.environ.get("PFAS_PROJECT_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[3]
    )


def build_coastline_source_load(step3_file: Path, output_file: Path) -> pd.DataFrame:
    step3_file = step3_file.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    if not step3_file.is_file():
        raise FileNotFoundError(f"Required file not found: {step3_file}")

    source = pd.read_csv(step3_file)
    if source.empty:
        raise ValueError(f"No data found in {step3_file}")
    flux_columns = [f"{compound}_Flux_Kg" for compound in PFAS_COMPOUNDS]
    require_columns(
        source,
        ["Year", "Grid_Lon", "Grid_Lat", "Expected_Total_Flux_Kg", *flux_columns],
        step3_file,
    )
    for column in [
        "Year",
        "Grid_Lon",
        "Grid_Lat",
        "Expected_Total_Flux_Kg",
        *flux_columns,
    ]:
        source[column] = numeric_column(
            source,
            column,
            step3_file,
            nonnegative=column not in ("Grid_Lon", "Grid_Lat"),
        )
    if (
        source[
            ["Year", "Grid_Lon", "Grid_Lat", "Expected_Total_Flux_Kg", *flux_columns]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(f"Missing flux or coordinate values in {step3_file}")
    if not source.Year.eq(BASE_YEAR).all():
        raise ValueError(
            f"Source inventory must contain only {BASE_YEAR}: {step3_file}"
        )
    if source.duplicated(["Grid_Lon", "Grid_Lat"]).any():
        raise ValueError(f"Duplicate source nodes in {step3_file}")
    allocated = source[flux_columns].sum(axis=1)
    if not np.allclose(
        allocated, source.Expected_Total_Flux_Kg, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            f"Compound fluxes do not conserve total source flux in {step3_file}"
        )

    records: list[dict[str, float]] = []
    for _, row in source.iterrows():
        totals = {
            group: sum(
                float(row[column])
                for compound in compounds
                if (column := f"{compound}_Flux_Kg") in row and pd.notna(row[column])
            )
            for group, compounds in TAXONOMY.items()
        }
        sum_lc = totals["LC_PFCA"] + totals["LC_PFSA"]
        sum_sc = totals["SC_PFCA"] + totals["SC_PFSA"]
        if any(total <= 0 for total in totals.values()):
            raise ValueError(
                f"A PFAS class lacks positive support at {row['Grid_Lon']}, {row['Grid_Lat']}."
            )
        records.append(
            {
                "Grid_Lon": float(row["Grid_Lon"]),
                "Grid_Lat": float(row["Grid_Lat"]),
                "Year": BASE_YEAR,
                "Load_Total_Legacy_Kg": float(row["Expected_Total_Flux_Kg"]),
                "Source_Ratio_SC_LC": sum_sc / sum_lc,
                "Source_Ratio_LC_PFSA_PFCA": totals["LC_PFSA"] / totals["LC_PFCA"],
                "Source_Ratio_SC_PFSA_PFCA": totals["SC_PFSA"] / totals["SC_PFCA"],
                **{f"{group}_Flux_Kg": total for group, total in totals.items()},
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        pd.DataFrame(records)
        .sort_values(["Grid_Lat", "Grid_Lon"])
        .reset_index(drop=True)
    )
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    write_manifest(
        output_file,
        "2018 coastline source-load aggregation",
        {"compound_flux": step3_file},
        rows=len(frame),
        compound_groups=TAXONOMY,
        precision="Full numeric precision retained for model inputs",
        total_flux_conserved=True,
    )
    logging.info("Exported %s coastline source nodes to %s", len(frame), output_file)
    return frame


def parse_arguments() -> argparse.Namespace:
    root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Aggregate regenerated 2018 compound fluxes into coastal source predictors."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    arguments.project_root = arguments.project_root.expanduser().resolve()
    output_root = (
        Path(os.environ.get("PFAS_OUTPUT_ROOT", arguments.project_root / "outputs"))
        .expanduser()
        .resolve()
    )
    arguments.input = (
        arguments.input or output_root / "exogenous" / "Step3_MonteCarlo_Flux_2018.csv"
    )
    arguments.output = (
        arguments.output
        or output_root / "exogenous" / "Step4_Coastline_Source_Load_2018.csv"
    )
    return arguments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()
    build_coastline_source_load(args.input, args.output)
