"""Input integrity checks shared by fixed-output evidence calculations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_run_provenance(paths, expected_schema: str) -> list[dict]:
    """Require saved-output sources to identify the same upstream feature table."""
    records = []
    for path in paths:
        manifest_path = Path(path).parent / "Run_Manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing current-run provenance: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = ("data_sha256", "feature_table_sha256", "pfas_schema_sha256")
        if any(not manifest.get(key) for key in required):
            raise ValueError(f"Incomplete current-run provenance: {manifest_path}")
        if manifest["pfas_schema_sha256"] != expected_schema:
            raise ValueError(f"Obsolete chemical/feature schema in {manifest_path}")
        records.append(
            {
                **{key: manifest[key] for key in required},
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        )
    for key in ("data_sha256", "feature_table_sha256", "pfas_schema_sha256"):
        if len({record[key] for record in records}) != 1:
            raise ValueError(
                f"Saved outputs come from different data/model generations: {key}"
            )
    return records


def boolean_flags(values: pd.Series) -> pd.Series:
    labels = values.astype(str).str.strip().str.lower()
    if not labels.isin(("true", "false", "1", "0")).all():
        raise ValueError("Boolean flags must be nonmissing true/false or 1/0")
    return labels.isin(("true", "1"))


def require_columns(frame: pd.DataFrame, columns, name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def validate_scenarios(frame: pd.DataFrame, scenario_ids) -> None:
    """Require the same identified observations in each requested scenario."""
    require_columns(
        frame, ("scenario_id", "row_index", "Sampling_Group", "Sea_Region"), "scenarios"
    )
    if (
        frame.empty
        or frame[["scenario_id", "row_index", "Sampling_Group", "Sea_Region"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Scenario identities and group/region labels cannot be empty")
    if frame.duplicated(["scenario_id", "row_index"]).any():
        raise ValueError("Duplicate scenario/observation identities")
    if set(frame.scenario_id) != set(scenario_ids):
        raise ValueError(
            "All requested scenarios are required exactly once per observation"
        )
    expected = set(frame.loc[frame.scenario_id.eq(scenario_ids[0]), "row_index"])
    for scenario_id, group in frame.groupby("scenario_id", sort=False):
        if set(group.row_index) != expected:
            raise ValueError(
                f"Scenario {scenario_id} uses different observation identities"
            )
    labels = frame.groupby("row_index")[["Sampling_Group", "Sea_Region"]].nunique()
    if (labels > 1).any().any():
        raise ValueError("Scenario observation metadata change between scenarios")
    baseline = [column for column in frame if column.endswith("_baseline")]
    if baseline:
        if not np.isfinite(frame[baseline].to_numpy(dtype=float)).all():
            raise ValueError("Scenario baseline values must be finite")
        for _, group in frame.groupby("row_index", sort=False):
            values = group[baseline].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=1e-12, atol=1e-12):
                raise ValueError("Baseline predictions differ across paired scenarios")


def validate_selected_shap(frame: pd.DataFrame) -> None:
    required = (
        "target",
        "validation",
        "feature",
        "row_index",
        "fold",
        "feature_value",
        "shap_value",
    )
    require_columns(frame, required, "selected OOF SHAP")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Selected OOF SHAP inputs cannot be empty or missing")
    if frame.duplicated(["target", "validation", "feature", "row_index"]).any():
        raise ValueError(
            "Duplicate selected OOF SHAP observation/target/feature identities"
        )
    if not np.isfinite(
        frame[["feature_value", "shap_value"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Selected OOF SHAP values must be finite")
    counts = frame.groupby(["target", "validation", "row_index"]).fold.nunique()
    if (counts != 1).any():
        raise ValueError("One observation cannot belong to multiple outer folds")


def merge_observation_metadata(
    shap: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Join every held-out attribution to exactly one current reference row."""
    require_columns(
        metadata, ("row_index", "Sampling_Group", "Sea_Region"), "reference metadata"
    )
    if metadata.row_index.duplicated().any() or metadata.isna().any().any():
        raise ValueError("Reference metadata must have unique, complete row identities")
    result = shap.merge(
        metadata,
        on="row_index",
        how="left",
        validate="many_to_one",
        suffixes=("", "_reference"),
        indicator=True,
    )
    if not result._merge.eq("both").all():
        raise ValueError(
            "SHAP rows are absent from current scenario reference metadata"
        )
    for column in ("Sampling_Group", "Sea_Region"):
        reference = column + "_reference"
        if reference in result:
            if not result[column].astype(str).eq(result[reference].astype(str)).all():
                raise ValueError(
                    f"SHAP metadata conflict with scenario metadata: {column}"
                )
            result = result.drop(columns=reference)
    return result.drop(columns="_merge")
