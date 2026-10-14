"""Shared chemical groups and feature names for the marine PFAS analysis.

The model panel is distinct from the full analytical inventory used in maps.
Concentrations outside the model panel remain available in source workbooks.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

SCHEMA_VERSION = "marine-pfas-analysis-schema/2"
MODEL_GROUPS = {
    "Long_PFCA": (
        "PFOA",
        "PFNA",
        "PFDA",
        "PFUnDA",
        "PFDoDA",
        "PFTrDA",
        "PFTeDA",
        "PFHxDA",
        "PFODA",
    ),
    "Long_PFSA": ("PFHxS", "PFHpS", "PFOS", "PFDS"),
    "Short_PFCA": ("PFBA", "PFPeA", "PFHxA", "PFHpA"),
    "Short_PFSA": ("PFBS", "PFPeS"),
}
SOURCE_GROUPS = {
    name.replace("Long_", "LC_").replace("Short_", "SC_"): compounds
    for name, compounds in MODEL_GROUPS.items()
}
MODEL_COMPOUNDS = tuple(
    compound for group in MODEL_GROUPS.values() for compound in group
)
LONG_PFCA = MODEL_GROUPS["Long_PFCA"]
LONG_PFSA = MODEL_GROUPS["Long_PFSA"]
ENVIRONMENT_COVARIATES = [
    "Temperature",
    "Salinity",
    "Chla",
    "POC",
    "TOC",
    "pH",
    "SPM",
    "DO",
    "Hydro_Bottom_U",
    "Hydro_Bottom_V",
    "Hydro_MLD",
]
SOURCE_COVARIATES = [
    "Diffused_Total_Legacy_Force",
    "Mixed_Source_Ratio_SC_LC",
    "Mixed_Source_Ratio_LC_PFSA_PFCA",
    "Mixed_Source_Ratio_SC_PFSA_PFCA",
]
COVARIATES = ENVIRONMENT_COVARIATES + SOURCE_COVARIATES
FEATURE_ALIASES = {
    "Temperature (°C)": "Temperature",
    "Salinity (‰)": "Salinity",
    "POC (mg/m3)": "POC",
    "TOC (%)": "TOC",
}
FEATURE_UNITS = {
    "Temperature": "degree Celsius",
    "Salinity": "practical salinity, dimensionless",
    "Chla": "mg m-3",
    "POC": "mg m-3",
    "TOC": "%",
    "pH": "dimensionless",
    "SPM": "g m-3",
    "DO": "mg L-1",
    "Hydro_Bottom_U": "m s-1",
    "Hydro_Bottom_V": "m s-1",
    "Hydro_MLD": "m",
    "Diffused_Total_Legacy_Force": "kg yr-1 km-1 (distance-offset weighted)",
    "Mixed_Source_Ratio_SC_LC": "dimensionless",
    "Mixed_Source_Ratio_LC_PFSA_PFCA": "dimensionless",
    "Mixed_Source_Ratio_SC_PFSA_PFCA": "dimensionless",
}


def normalize_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize legacy feature labels without converting numerical values."""
    result = frame.copy()
    if result.columns.duplicated().any():
        raise ValueError("Duplicate input column names are not allowed")
    for alias, canonical in FEATURE_ALIASES.items():
        if alias not in result:
            continue
        if canonical in result:
            left = pd.to_numeric(result[alias], errors="raise")
            right = pd.to_numeric(result[canonical], errors="raise")
            if not np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True):
                raise ValueError(
                    f"Conflicting feature columns: {alias!r}, {canonical!r}"
                )
            result = result.drop(columns=alias)
        else:
            result = result.rename(columns={alias: canonical})
    return result


def sum_group(frame: pd.DataFrame, name: str) -> pd.Series:
    """Sum reported model-panel members; an entirely missing group stays missing."""
    if name not in MODEL_GROUPS:
        raise ValueError(f"Unknown model group: {name}")
    columns = [column for column in MODEL_GROUPS[name] if column in frame]
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float, name=name)
    values = frame[columns].apply(pd.to_numeric, errors="raise")
    present = values.to_numpy(dtype=float)
    if np.isinf(present).any() or (present < 0).any():
        raise ValueError(f"Invalid negative or infinite concentrations in {name}")
    return values.sum(axis=1, min_count=1).rename(name)


def schema_digest() -> str:
    """Fingerprint the chemical membership and model predictor contract."""
    payload = json.dumps(
        {"version": SCHEMA_VERSION, "groups": MODEL_GROUPS, "covariates": COVARIATES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
