"""Match river mouths and calculate regional PFCA/PFSA fingerprints."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from source_common import (
    BASE_YEAR,
    PFAS_COMPOUNDS,
    SEA_AREAS,
    TAXONOMY,
    canonical_name,
    export_group_dictionary,
    numeric_column,
    read_estuary_inventory,
    read_river_inventory,
    require_columns,
    write_manifest,
)

SEA_BOUNDARIES = {
    "Bohai Sea": {"lat": (37.0, 42.0), "lon": (117.0, 122.5)},
    "Yellow Sea": {"lat": (31.0, 37.0), "lon": (118.0, 127.0)},
    "East China Sea": {"lat": (23.0, 31.0), "lon": (116.0, 131.0)},
    "South China Sea": {"lat": (3.0, 23.0), "lon": (105.0, 121.0)},
}
BORROWING_RULES = {"South China Sea": "East China Sea"}


def get_sea_area_by_topology(lon: float, lat: float) -> str:
    for sea, bounds in SEA_BOUNDARIES.items():
        if (
            bounds["lat"][0] <= lat <= bounds["lat"][1]
            and bounds["lon"][0] <= lon <= bounds["lon"][1]
        ):
            return sea
    # The northern Yellow Sea lies east of the Liaodong Peninsula. The
    # historical southern rectangle omitted its river mouths above 37 N.
    if 122.5 < lon <= 127.0 and 37.0 < lat <= 42.0:
        return "Yellow Sea"
    return "OpenOcean"


def _known_seas(estuary: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for name, group in estuary.groupby(estuary.River.str.casefold(), sort=True):
        seas = sorted(set(group["District I"]))
        if len(seas) != 1:
            raise ValueError(f"Conflicting sea assignments for {name}: {seas}")
        mapping[name] = seas[0]
    return mapping


def _read_runoff(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric = (
        "Year",
        "Orig_Lon",
        "Orig_Lat",
        "Grid_Lon",
        "Grid_Lat",
        "Annual_Q_10^8m3",
        "Q_SD_10^8m3",
    )
    require_columns(frame, ["River", *numeric], path)
    for column in numeric:
        frame[column] = numeric_column(
            frame,
            column,
            path,
            nonnegative=column in ("Annual_Q_10^8m3", "Q_SD_10^8m3"),
        )
    frame = frame.loc[frame.Year.eq(BASE_YEAR)].copy()
    if frame.empty or frame[list(numeric)].isna().any().any():
        raise ValueError(f"Missing or incomplete {BASE_YEAR} runoff records: {path}")
    frame["River"] = frame.River.map(canonical_name)
    if (
        frame.River.eq("").any()
        or frame.duplicated(["River", "Orig_Lon", "Orig_Lat", "Year"]).any()
    ):
        raise ValueError(f"Missing or duplicate river-mouth memberships: {path}")
    return frame.sort_values(
        ["Grid_Lat", "Grid_Lon", "River", "Orig_Lat", "Orig_Lon"], kind="stable"
    )


def _match_inventory(member: pd.Series, rivers: pd.DataFrame) -> tuple[list[int], str]:
    position = np.isclose(
        rivers.Longitude, float(member.Orig_Lon), rtol=0, atol=1e-8
    ) & np.isclose(rivers.Latitude, float(member.Orig_Lat), rtol=0, atol=1e-8)
    candidates = rivers.loc[position & rivers["River Name"].eq(member.River)]
    if "Source_ID" in member.index and pd.notna(member.Source_ID) and "ID" in rivers:
        candidates = rivers.loc[
            position
            & pd.to_numeric(rivers.ID, errors="coerce").eq(float(member.Source_ID))
        ]
        if candidates.empty:
            raise ValueError(
                f"Source_ID {member.Source_ID} disagrees with original coordinates for {member.River}"
            )
        method = "ID_and_coordinates"
    elif not candidates.empty:
        method = "Exact_name_and_coordinates"
    else:
        candidates = rivers.loc[
            position
            & rivers["River Name"].str.casefold().eq(str(member.River).casefold())
        ]
        method = "Unique_case_variant_and_coordinates"
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous river inventory match: {member.River}, {member.Orig_Lon}, {member.Orig_Lat}"
        )
    if candidates.empty:
        if rivers["River Name"].eq(member.River).any():
            raise ValueError(
                f"Original coordinates disagree with the named river inventory: {member.River}"
            )
        return [], "No_empirical_inventory"
    return list(candidates.index), method


def build_topology_mapping(
    estuary_file: Path, rivers_file: Path, step1_file: Path, output_file: Path
) -> pd.DataFrame:
    rivers = read_river_inventory(rivers_file)
    estuary = read_estuary_inventory(estuary_file)
    runoff = _read_runoff(step1_file)
    known_mapping = _known_seas(estuary)
    columns = [
        *[compound for compound in PFAS_COMPOUNDS if compound in rivers],
        "∑PFAAs (ng/L)",
    ]
    records = []
    for (grid_lon, grid_lat), members in runoff.groupby(
        ["Grid_Lon", "Grid_Lat"], sort=True
    ):
        for column in ("Annual_Q_10^8m3", "Q_SD_10^8m3"):
            if not np.allclose(
                members[column], members[column].iloc[0], rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    f"Conflicting shared-node {column}: {grid_lon}, {grid_lat}"
                )
        names = sorted(set(members.River))
        known = sorted(
            {
                known_mapping[name.casefold()]
                for name in names
                if name.casefold() in known_mapping
            }
        )
        if len(known) > 1:
            raise ValueError(
                f"A coastal node combines conflicting seas: {grid_lon}, {grid_lat}, {known}"
            )
        sea = (
            known[0]
            if known
            else get_sea_area_by_topology(float(grid_lon), float(grid_lat))
        )
        indices, methods = [], []
        for _, member in members.iterrows():
            matched, method = _match_inventory(member, rivers)
            indices.extend(matched)
            methods.append(f"{member.River}:{method}")
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"A river inventory row occurs twice in node {grid_lon}, {grid_lat}"
            )
        empirical = rivers.loc[indices]
        profile = {column: np.nan for column in columns}
        for column in columns:
            values, weights = empirical[column], empirical.Runoff
            valid = values.notna() & weights.notna()
            denominator = float(weights.loc[valid].sum())
            if denominator > 0:
                profile[column] = float(
                    (values.loc[valid] * weights.loc[valid]).sum() / denominator
                )
            elif values.notna().any():
                profile[column] = float(values.mean())
        records.append(
            {
                "Grid_Lon": float(grid_lon),
                "Grid_Lat": float(grid_lat),
                "Included_Rivers": ", ".join(names),
                "Sea_Area": sea,
                "Empirical_River_Count": len(indices),
                "Inventory_Row_IDs": (
                    ", ".join(str(value) for value in empirical.ID)
                    if "ID" in empirical
                    else ""
                ),
                "Inventory_Match_Methods": "; ".join(methods),
                **profile,
            }
        )
    frame = (
        pd.DataFrame(records)
        .sort_values(["Grid_Lat", "Grid_Lon"], kind="stable")
        .reset_index(drop=True)
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    write_manifest(
        output_file,
        "River-mouth topology and flow-weighted baseline concentration",
        {"estuary": estuary_file, "rivers": rivers_file, "runoff": step1_file},
        rows=len(frame),
        matching="Original coordinates and source IDs or case-preserving names",
        shared_node_runoff="Validated once per node; never summed across memberships",
        northern_yellow_sea_completion="For otherwise unassigned river mouths, 122.5<longitude<=127 and 37<latitude<=42; fills Dayang and Zhuang coverage omitted by the historical southern rectangle",
    )
    logging.info("Exported %s river-mouth nodes to %s", len(frame), output_file)
    return frame


def extract_regional_fingerprints(
    estuary_file: Path, output_file: Path
) -> pd.DataFrame:
    estuary = read_estuary_inventory(estuary_file)
    global_rsd = {}
    for compound in PFAS_COMPOUNDS:
        values = estuary[compound].dropna()
        mean = float(values.mean()) if len(values) else np.nan
        sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        global_rsd[compound] = (
            sd / mean if np.isfinite(mean) and mean > 0 and np.isfinite(sd) else 0.10
        )
    records = []
    for sea in SEA_AREAS:
        local = estuary.loc[estuary["District I"].eq(sea)]
        record = {"Sea_Area": sea, "Sample_Count": len(local)}
        for compound in PFAS_COMPOUNDS:
            values = local[compound].dropna()
            count = len(values)
            mean = float(values.mean()) if count else np.nan
            if count >= 3:
                sd, origin = float(values.std(ddof=1)), "Observed"
            elif count:
                sd, origin = mean * global_rsd[compound], "Global_RSD"
            else:
                sd, origin = np.nan, "Unavailable"
            record.update(
                {
                    f"{compound}_Mean": mean,
                    f"{compound}_Std": sd,
                    f"{compound}_Count": count,
                    f"{compound}_Origin": origin,
                    f"{compound}_Donor_Count": 0,
                    f"{compound}_Donor_Sea": "",
                }
            )
        records.append(record)
    frame = pd.DataFrame(records).set_index("Sea_Area")
    for target, donor in BORROWING_RULES.items():
        for compound in PFAS_COMPOUNDS:
            if (
                frame.loc[target, f"{compound}_Origin"] == "Unavailable"
                and frame.loc[donor, f"{compound}_Origin"] != "Unavailable"
                and frame.loc[donor, f"{compound}_Mean"] > 0
            ):
                for suffix in ("Mean", "Std"):
                    frame.loc[target, f"{compound}_{suffix}"] = frame.loc[
                        donor, f"{compound}_{suffix}"
                    ]
                frame.loc[target, f"{compound}_Origin"] = "Adjacent_Borrowing"
                frame.loc[target, f"{compound}_Donor_Count"] = int(
                    frame.loc[donor, f"{compound}_Count"]
                )
                frame.loc[target, f"{compound}_Donor_Sea"] = donor
    eligibility, statuses = [], []
    for sea, row in frame.iterrows():
        group_available = {
            group: any(
                row[f"{compound}_Origin"] != "Unavailable"
                and row[f"{compound}_Mean"] > 0
                for compound in compounds
            )
            for group, compounds in TAXONOMY.items()
        }
        for group, available in group_available.items():
            frame.loc[sea, f"{group}_Available"] = available
        eligibility.append(all(group_available.values()))
        borrowed = any(
            row[f"{compound}_Origin"] == "Adjacent_Borrowing"
            for compound in PFAS_COMPOUNDS
        )
        statuses.append(
            "Adjacent_Borrowing_From_East_China_Sea"
            if row.Sample_Count == 0 and borrowed
            else (
                "Independent_Statistics_with_Partial_Borrowing"
                if borrowed
                else "Independent_Statistics" if row.Sample_Count else "No_Data"
            )
        )
    frame = frame.copy()
    frame["Source_Eligible"], frame["Data_Status"] = eligibility, statuses
    frame = frame.reset_index()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    dictionary = output_file.parent / "Source_Group_Dictionary.csv"
    export_group_dictionary(dictionary)
    write_manifest(
        output_file,
        "Regional PFCA/PFSA source fingerprints",
        {"estuary": estuary_file},
        rows=len(frame),
        eligible_seas=list(frame.loc[frame.Source_Eligible, "Sea_Area"]),
        borrowing_rules=BORROWING_RULES,
        missing_parameters="NaN with Origin=Unavailable; not measured zero",
        group_dictionary=dictionary.name,
        chain_length_boundaries="Original PFCA boundary PFOA and PFSA boundary PFHxS",
    )
    logging.info("Exported %s regional fingerprints to %s", len(frame), output_file)
    return frame


def parse_arguments() -> argparse.Namespace:
    root = (
        Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
        .expanduser()
        .resolve()
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--estuary-file", type=Path)
    parser.add_argument("--rivers-file", type=Path)
    parser.add_argument("--runoff-file", type=Path)
    args = parser.parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.data_root = args.data_root or (
        Path(os.environ.get("PFAS_DATA_ROOT", args.project_root / "data")) / "raw"
    )
    args.output_dir = (
        args.output_dir
        or Path(os.environ.get("PFAS_OUTPUT_ROOT", args.project_root / "outputs"))
        / "exogenous"
    )
    args.estuary_file = args.estuary_file or args.data_root / "estuary.xlsx"
    args.rivers_file = args.rivers_file or args.data_root / "100_Rivers_2018.xlsx"
    args.runoff_file = (
        args.runoff_file or args.output_dir / "Step1_Annual_Runoff_2014_2025.csv"
    )
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()
    build_topology_mapping(
        args.estuary_file,
        args.rivers_file,
        args.runoff_file,
        args.output_dir / "Step2_River_Sea_Mapping.csv",
    )
    extract_regional_fingerprints(
        args.estuary_file, args.output_dir / "Step2_Regional_Fingerprints.csv"
    )
