"""Summarize mixed-source LC ratios on the shared perturbation domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE = "Mixed_Source_Ratio_LC_PFSA_PFCA"
SCENARIO = "synchronized_legacy_control_50pct"
COMMON_SCENARIOS = tuple(
    f"{design}_{position}pct"
    for design in ("synchronized_legacy_control", "class_resolved_legacy_control")
    for position in (10, 25, 50)
)


def _boolean_flags(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    if not text.isin(("true", "false", "1", "0")).all():
        raise ValueError("Source-domain flags must be nonmissing Boolean values")
    return text.isin(("true", "1"))


def summarize_descriptor_changes(
    predictions: pd.DataFrame, paired_shap: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_columns = {
        "scenario_id",
        "row_index",
        "Sea_Region",
        "outside_source_minmax_any",
        "Long_Balance_delta",
    }
    feature_columns = {
        "scenario_id",
        "row_index",
        "feature",
        "feature_value_baseline",
        "feature_value_scenario",
    }
    for required, frame in (
        (prediction_columns, predictions),
        (feature_columns, paired_shap),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
    selected = predictions.loc[predictions.scenario_id.isin(COMMON_SCENARIOS)].copy()
    if set(selected.scenario_id) != set(COMMON_SCENARIOS):
        raise ValueError("All six common-domain scenarios are required")
    if selected.duplicated(["scenario_id", "row_index"]).any():
        raise ValueError("Duplicate paired-prediction scenario/row identities")
    selected["outside"] = _boolean_flags(selected.outside_source_minmax_any)
    support = selected.groupby("row_index").agg(
        scenario_count=("scenario_id", "nunique"), outside=("outside", "any")
    )
    common_rows = support.index[
        (support.scenario_count == len(COMMON_SCENARIOS)) & ~support.outside
    ]
    if len(common_rows) == 0:
        raise ValueError("The fixed common source-feature domain is empty")
    baseline = selected.loc[
        (selected.scenario_id == SCENARIO) & selected.row_index.isin(common_rows),
        ["row_index", "Sea_Region", "Long_Balance_delta"],
    ]
    feature = paired_shap.loc[
        (paired_shap.scenario_id == SCENARIO)
        & (paired_shap.feature == FEATURE)
        & paired_shap.row_index.isin(common_rows),
        ["row_index", "feature_value_baseline", "feature_value_scenario"],
    ]
    if feature.row_index.duplicated().any() or set(feature.row_index) != set(
        common_rows
    ):
        raise ValueError(
            "Feature values must cover every common-domain row exactly once"
        )
    values = baseline.merge(feature, on="row_index", validate="one_to_one")
    if values.Sea_Region.isna().any():
        raise ValueError("Missing sea-region labels")
    numeric = values[
        ["Long_Balance_delta", "feature_value_baseline", "feature_value_scenario"]
    ]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Descriptor and Balance values must be finite")
    if (values.feature_value_baseline <= 0).any():
        raise ValueError("Percentage changes require positive baseline descriptors")
    values["descriptor_change_percent"] = 100.0 * (
        values.feature_value_scenario / values.feature_value_baseline - 1.0
    )
    summary = (
        values.groupby("Sea_Region", sort=True)
        .agg(
            observation_count=("row_index", "size"),
            descriptor_change_median_percent=("descriptor_change_percent", "median"),
            descriptor_change_min_percent=("descriptor_change_percent", "min"),
            descriptor_change_max_percent=("descriptor_change_percent", "max"),
            mean_balance_delta=("Long_Balance_delta", "mean"),
        )
        .reset_index()
    )
    return summary, values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_output_paths(inputs: tuple[Path, ...], outputs: tuple[Path, ...]) -> None:
    """Reject output aliases of input files."""
    sources = tuple(path.expanduser().resolve() for path in inputs)
    for output in outputs:
        destination = output.expanduser().resolve()
        if destination in sources:
            raise ValueError(
                "An output would overwrite an input; select another output directory"
            )
        if destination.exists():
            for source in sources:
                if source.exists() and destination.samefile(source):
                    raise ValueError("An output aliases an input file")


def main() -> None:
    project = Path(
        os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2])
    )
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", project / "outputs"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=output_root
        / "03_source_attenuation_inference/Scenario_Paired_Predictions.csv",
    )
    parser.add_argument(
        "--paired-shap",
        type=Path,
        default=output_root
        / "03_source_attenuation_inference/Scenario_Paired_SHAP.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=output_root / "source_ratio_diagnostic"
    )
    arguments = parser.parse_args()
    check_output_paths(
        (arguments.predictions, arguments.paired_shap),
        (
            arguments.output_dir / "source_ratio_descriptor_summary.csv",
            arguments.output_dir / "source_ratio_diagnostic_manifest.json",
        ),
    )
    predictions = pd.read_csv(arguments.predictions)
    paired_shap = pd.read_csv(
        arguments.paired_shap,
        usecols=[
            "scenario_id",
            "row_index",
            "feature",
            "feature_value_baseline",
            "feature_value_scenario",
        ],
    )
    summary, values = summarize_descriptor_changes(predictions, paired_shap)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        arguments.output_dir / "source_ratio_descriptor_summary.csv", index=False
    )
    manifest = {
        "analysis": "mixed_source_long_chain_ratio_diagnostic",
        "scenario_id": SCENARIO,
        "feature": FEATURE,
        "common_domain_scenarios": list(COMMON_SCENARIOS),
        "domain_rule": "all four source features within training minima/maxima in all six scenarios",
        "common_observations": len(values),
        "descriptor_change_percent": "100 * (scenario / baseline - 1)",
        "all_descriptor_changes_positive": bool(
            (values.descriptor_change_percent > 0).all()
        ),
        "overall_min_percent": float(values.descriptor_change_percent.min()),
        "overall_max_percent": float(values.descriptor_change_percent.max()),
        "inputs": [
            {"filename": path.name, "sha256": sha256(path)}
            for path in (arguments.predictions, arguments.paired_shap)
        ],
        "script_sha256": sha256(Path(__file__)),
        "model_refitted": False,
        "shap_recomputed": False,
        "scope": "mixed_source_long_chain_ratio_descriptor_changes",
    }
    (arguments.output_dir / "source_ratio_diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
