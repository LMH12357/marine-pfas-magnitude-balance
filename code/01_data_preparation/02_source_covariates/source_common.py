"""Shared input definitions for the coastal source inventory."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pfas_schema import SOURCE_GROUPS

BASE_YEAR = 2018
SEA_AREAS = ("Bohai Sea", "Yellow Sea", "East China Sea", "South China Sea")
TAXONOMY = dict(SOURCE_GROUPS)
PFAS_COMPOUNDS = tuple(
    compound for compounds in TAXONOMY.values() for compound in compounds
)
EXCLUDED_COMPOUNDS = {
    "PFNS": "Excluded from the agreed model panel; retained in the original total-PFAS inventory",
    "PFECHS": "Excluded from the agreed model panel; retained in the original total-PFAS inventory",
    "PFOSA": "Sulfonamide precursor, outside the PFCA/PFSA model panel",
    "N-MeFOSA": "Sulfonamide precursor, outside the PFCA/PFSA model panel",
    "N-EtFOSA": "Sulfonamide precursor, outside the PFCA/PFSA model panel",
    "N- MeFOSAA": "Sulfonamidoacetic acid precursor, outside the PFCA/PFSA model panel",
    "N-EtFOSAA": "Sulfonamidoacetic acid precursor, outside the PFCA/PFSA model panel",
}


def canonical_name(value: Any) -> str:
    """Remove surrounding/repeated whitespace without merging case-distinct IDs."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...] | list[str], source: Path | str
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing columns in {source}: {', '.join(missing)}")


def numeric_column(
    frame: pd.DataFrame, column: str, source: Path | str, *, nonnegative: bool = False
) -> pd.Series:
    original = frame[column]
    result = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & result.isna()
    if invalid.any():
        raise ValueError(
            f"Nonnumeric {column} in {source}, rows {list(frame.index[invalid] + 2)}"
        )
    values = result.dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {column} in {source}")
    if nonnegative and (values < 0).any():
        raise ValueError(f"Negative {column} in {source}")
    return result


def read_excel_table(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError(
            f"Duplicate column names after whitespace normalization: {path}"
        )
    return frame


def read_river_inventory(path: Path) -> pd.DataFrame:
    frame = read_excel_table(path)
    require_columns(
        frame, ("River Name", "Longitude", "Latitude", "Runoff", "∑PFAAs (ng/L)"), path
    )
    frame["River Name"] = frame["River Name"].map(canonical_name)
    frame = frame.loc[frame["River Name"].ne("")].copy()
    if frame.empty:
        raise ValueError(f"No river inventory records: {path}")
    for column in ("Longitude", "Latitude", "Runoff", "∑PFAAs (ng/L)"):
        frame[column] = numeric_column(
            frame, column, path, nonnegative=column in ("Runoff", "∑PFAAs (ng/L)")
        )
    if frame[["Longitude", "Latitude"]].isna().any().any():
        raise ValueError(f"A named river lacks coordinates in {path}")
    if (
        not frame.Longitude.between(-180, 180).all()
        or not frame.Latitude.between(-90, 90).all()
    ):
        raise ValueError(f"Invalid geographic coordinates in {path}")
    for compound in PFAS_COMPOUNDS:
        if compound in frame:
            frame[compound] = numeric_column(frame, compound, path, nonnegative=True)
    return frame


def read_estuary_inventory(path: Path) -> pd.DataFrame:
    frame = read_excel_table(path)
    require_columns(frame, ("River", "District I"), path)
    frame["River"] = frame.River.map(canonical_name)
    frame["District I"] = frame["District I"].map(canonical_name)
    if frame.River.eq("").any() or frame["District I"].eq("").any():
        raise ValueError(f"Missing river or sea designation in {path}")
    unexpected = sorted(set(frame["District I"]) - set(SEA_AREAS))
    if unexpected:
        raise ValueError(f"Unknown sea designations in {path}: {unexpected}")
    for compound in PFAS_COMPOUNDS:
        if compound not in frame:
            frame[compound] = np.nan
        else:
            frame[compound] = numeric_column(frame, compound, path, nonnegative=True)
    return frame


def read_fingerprints(path: Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    required = ["Sea_Area", "Source_Eligible"]
    required.extend(
        f"{compound}_{suffix}"
        for compound in PFAS_COMPOUNDS
        for suffix in ("Mean", "Std", "Count", "Origin", "Donor_Count", "Donor_Sea")
    )
    require_columns(frame, required, path)
    if frame.Sea_Area.duplicated().any() or set(frame.Sea_Area) != set(SEA_AREAS):
        raise ValueError(f"Fingerprints must contain one record per sea: {path}")
    booleans = frame.Source_Eligible.astype(str).str.casefold()
    if not booleans.isin(("true", "false")).all():
        raise ValueError(f"Invalid Source_Eligible flag: {path}")
    frame["Source_Eligible"] = booleans.eq("true")
    for compound in PFAS_COMPOUNDS:
        available = frame[f"{compound}_Origin"].ne("Unavailable")
        for suffix in ("Mean", "Std", "Count", "Donor_Count"):
            column = f"{compound}_{suffix}"
            frame[column] = numeric_column(frame, column, path, nonnegative=True)
            if suffix in ("Mean", "Std") and frame.loc[available, column].isna().any():
                raise ValueError(
                    f"Missing available fingerprint parameter {column}: {path}"
                )
        if (
            frame.loc[~available, [f"{compound}_Mean", f"{compound}_Std"]]
            .notna()
            .any()
            .any()
        ):
            raise ValueError(
                f"Unavailable fingerprint encoded as a value: {compound}, {path}"
            )
    return frame


def write_manifest(
    output: Path, analysis: str, inputs: dict[str, Path], **details: Any
) -> None:
    output = Path(output)
    manifest = {
        "analysis": analysis,
        "base_year": BASE_YEAR,
        "inputs": {
            name: {"file": Path(path).name, "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "output": {"file": output.name, "sha256": sha256(output)},
        **details,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def export_group_dictionary(path: Path) -> None:
    records = [
        {
            "Compound": compound,
            "Group": group,
            "Water_Response_Field": compound,
            "Source_Concentration_Field": compound,
            "Source_Flux_Field": f"{compound}_Flux_Kg",
            "Model_Panel_Included": True,
            "Reason": "Original chain-length and functional-group boundaries",
        }
        for group, compounds in TAXONOMY.items()
        for compound in compounds
    ]
    records.extend(
        {
            "Compound": compound,
            "Group": "",
            "Water_Response_Field": compound,
            "Source_Concentration_Field": compound,
            "Source_Flux_Field": "",
            "Model_Panel_Included": False,
            "Reason": reason,
        }
        for compound, reason in EXCLUDED_COMPOUNDS.items()
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
