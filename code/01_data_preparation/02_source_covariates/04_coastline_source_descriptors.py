from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


BASE_YEAR = 2018
TAXONOMY = {
    "LC_PFCA": ["PFOA", "PFNA", "PFDA", "PFUnDA", "PFDoDA", "PFTrDA", "PFTeDA", "PFHxDA", "PFODA"],
    "LC_PFSA": ["PFHxS", "PFHpS", "PFOS", "PFNS", "PFDS", "PFECHS"],
    "SC_PFCA": ["PFBA", "PFPeA", "PFHxA", "PFHpA"],
    "SC_PFSA": ["PFBS", "PFPeS"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_project_root() -> Path:
    configured = os.environ.get("PFAS_PROJECT_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[3]


def build_coastline_source_load(step3_file: Path, output_file: Path) -> pd.DataFrame:
    step3_file = step3_file.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    if not step3_file.is_file():
        raise FileNotFoundError(f"Required file not found: {step3_file}")

    source = pd.read_csv(step3_file)
    if source.empty:
        raise ValueError(f"No data found in {step3_file}")

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
        if min(sum_lc, totals["LC_PFCA"], totals["SC_PFCA"]) <= 0:
            raise ValueError("A required PFAS class total is zero; source ratios are undefined.")
        records.append({
            "Grid_Lon": float(row["Grid_Lon"]),
            "Grid_Lat": float(row["Grid_Lat"]),
            "Load_Total_Legacy_Kg": round(float(row["Expected_Total_Flux_Kg"]), 4),
            "Source_Ratio_SC_LC": round(sum_sc / sum_lc, 6),
            "Source_Ratio_LC_PFSA_PFCA": round(totals["LC_PFSA"] / totals["LC_PFCA"], 6),
            "Source_Ratio_SC_PFSA_PFCA": round(totals["SC_PFSA"] / totals["SC_PFCA"], 6),
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values(["Grid_Lat", "Grid_Lon"]).reset_index(drop=True)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "2018 coastline source-load aggregation",
        "base_year": BASE_YEAR,
        "rows": len(frame),
        "input": {"file": step3_file.name, "sha256": sha256(step3_file)},
        "output": {"file": output_file.name, "sha256": sha256(output_file)},
    }
    output_file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(frame)} coastline source nodes to {output_file}")
    return frame


def parse_arguments() -> argparse.Namespace:
    root = default_project_root()
    fixed = root / "data" / "fixed_inputs" / "source_boundary"
    parser = argparse.ArgumentParser(description="Aggregate the fixed 2018 compound inventory into model source descriptors.")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    arguments.project_root = arguments.project_root.expanduser().resolve()
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", arguments.project_root / "outputs")).expanduser().resolve()
    arguments.input = arguments.input or fixed / "Step3_MonteCarlo_Flux_2018.csv"
    arguments.output = arguments.output or output_root / "exogenous" / "Step4_Coastline_Source_Load_2018.csv"
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    build_coastline_source_load(args.input, args.output)
