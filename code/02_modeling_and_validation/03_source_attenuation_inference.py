from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfas_schema import COVARIATES, MODEL_GROUPS, SOURCE_COVARIATES, schema_digest
from coastal_routing import compute_routed_features, load_routing_bundle

VERSION = "v6.3"
MODEL_SCHEMA = "pfas-long-chain-magnitude-balance-model-package/v2"
COORDINATES = ("Long_Magnitude", "Long_Balance")
SOURCE_FEATURES = tuple(SOURCE_COVARIATES)
SOURCE_COLUMNS = (
    "Grid_Lon",
    "Grid_Lat",
    "Load_Total_Legacy_Kg",
    "Source_Ratio_SC_LC",
    "Source_Ratio_LC_PFSA_PFCA",
    "Source_Ratio_SC_PFSA_PFCA",
)
ANCHOR_RANGE_TOLERANCE = 1e-10
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_model_package(path: Path) -> dict:
    package = joblib.load(path)
    if package.get("schema_version") != MODEL_SCHEMA:
        raise ValueError(
            f"Unsupported model package schema: {package.get('schema_version')}"
        )
    if (
        tuple(package.get("coordinate_system", {}).get("coordinate_order", ()))
        != COORDINATES
    ):
        raise ValueError("Model package coordinate order is incompatible")
    if set(package.get("models", {})) != set(COORDINATES):
        raise ValueError("Model package must contain both coordinate models")
    features = package.get("feature_names", [])
    if list(features) != list(COVARIATES):
        raise ValueError(
            "Model package feature contract differs from the current analysis schema"
        )
    if package.get("pfas_schema_sha256") != schema_digest():
        raise ValueError(
            "Model package taxonomy/feature schema is stale; retrain with the current data contract"
        )
    if not package.get("data_sha256") or not package.get("feature_table_sha256"):
        raise ValueError("Model package lacks current data and feature-table hashes")
    for key, group in (
        ("long_pfsa_congeners", "Long_PFSA"),
        ("long_pfca_congeners", "Long_PFCA"),
    ):
        if list(package["coordinate_system"].get(key, [])) != list(MODEL_GROUPS[group]):
            raise ValueError(
                f"Model package {key} differs from the current group definition"
            )
    for target in COORDINATES:
        model = package["models"][target]
        if (
            not np.isfinite([model["target_mean"], model["target_scale"]]).all()
            or model["target_scale"] <= 0
        ):
            raise ValueError(f"{target}: invalid target standardization parameters")
    return package


def verify_model_provenance(
    package: dict, training_script: Path, boundary: Path
) -> pd.DataFrame:
    """Reject stale fit inputs before routing or counterfactual prediction."""
    if sha256(training_script) != package.get("training_code_sha256"):
        raise ValueError(
            "Current training code differs from the code recorded by the model package"
        )
    provenance = package.get("training_input_provenance", {})
    if "source_load" not in provenance or "boundary_shp" not in provenance:
        raise ValueError(
            "Model package lacks source-load or routing-boundary provenance"
        )
    for role, record in provenance.items():
        path = (
            boundary.with_suffix("." + role.removeprefix("boundary_"))
            if role.startswith("boundary_")
            else Path(record["path"])
        )
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(
                f"Training input is unavailable or changed after fitting: {role}: {path}"
            )
    return validate_source_table(pd.read_csv(provenance["source_load"]["path"]))


def verify_routing_provenance(package: dict, directory: Path) -> Path:
    """Use exactly the frozen routing artifact that supplied the fitted features."""
    manifest_path = directory.resolve() / "Routing_Manifest.json"
    record = package.get("training_input_provenance", {}).get("routing_manifest")
    if not record or not manifest_path.is_file():
        raise ValueError(
            "Model package or routing directory lacks frozen routing provenance"
        )
    if sha256(manifest_path) != record.get("sha256"):
        raise ValueError(
            "Scenario routing artifact differs from the fitted routing artifact"
        )
    load_routing_bundle(directory)
    return manifest_path


def verify_scenario_products(
    directory: Path, manifest: dict, summary: pd.DataFrame
) -> None:
    if manifest.get("pfas_schema_sha256") != schema_digest():
        raise ValueError(
            "Scenario source taxonomy differs from the current model schema"
        )
    hashes = manifest.get("product_sha256", {})
    if not hashes:
        raise ValueError("Scenario manifest lacks product hashes")
    for name, digest in hashes.items():
        path = (directory / name).resolve()
        if (
            path.parent != directory.resolve()
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError(
                f"Scenario product is missing, changed or outside its output directory: {name}"
            )
    for row in summary.itertuples(index=False):
        if hashes.get(row.aggregated_source_file) != row.aggregated_source_sha256:
            raise ValueError(f"{row.scenario_id}: inconsistent aggregate source hash")
    declared = {item["scenario_id"] for item in manifest["scenario_definitions"]}
    if set(summary["scenario_id"]) != declared:
        raise ValueError("Scenario summary does not match its declared scenario set")


def validate_source_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Source table is missing columns: {missing}")
    result = frame[list(SOURCE_COLUMNS)].copy()
    result[list(SOURCE_COLUMNS)] = result[list(SOURCE_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Source table contains non-finite values")
    if (result[list(SOURCE_COLUMNS[2:])] < 0).any().any():
        raise ValueError("Source loads and ratios must be non-negative")
    if result.duplicated(["Grid_Lon", "Grid_Lat"]).any():
        raise ValueError("Source coordinates must be unique")
    return result.sort_values(["Grid_Lat", "Grid_Lon"]).reset_index(drop=True)


def package_training_frame(package: dict) -> pd.DataFrame:
    features = package["feature_names"]
    X = package["X_data"][features].copy()
    coordinates = package["Coordinates"][["Longitude", "Latitude"]].copy()
    metadata = package["Metadata"].copy()
    if not (X.index.equals(coordinates.index) and X.index.equals(metadata.index)):
        raise ValueError("Package training matrices are not index aligned")
    indices = pd.Index(package["training_row_indices"])
    if (
        not indices.is_unique
        or not X.index.equals(indices)
        or metadata["Sampling_Group"].isna().any()
    ):
        raise ValueError(
            "Package training row identifiers are duplicated, missing or not aligned"
        )
    if (
        not np.isfinite(X.to_numpy(dtype=float)).all()
        or not np.isfinite(coordinates.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Package feature values or coordinates are non-finite")
    frame = pd.concat([metadata, coordinates, X], axis=1)
    frame.insert(0, "row_index", package["training_row_indices"])
    return frame.reset_index(drop=True)


def route_source_features(weights: np.ndarray, source: pd.DataFrame) -> pd.DataFrame:
    if (
        weights.ndim != 2
        or weights.shape[1] != len(source)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "Routing weights must be a finite, positive sample-by-source matrix; disconnected sources are not allowed"
        )
    loads = source["Load_Total_Legacy_Kg"].to_numpy(dtype=float)
    weighted_load = weights * loads[None, :]
    force = weighted_load.sum(axis=1)
    if not np.isfinite(force).all() or np.any(force <= 0):
        raise ValueError("At least one sampling point has zero routed source force")
    ratios = source[
        [
            "Source_Ratio_SC_LC",
            "Source_Ratio_LC_PFSA_PFCA",
            "Source_Ratio_SC_PFSA_PFCA",
        ]
    ].to_numpy(dtype=float)
    mixed = weighted_load @ ratios / force[:, None]
    return pd.DataFrame(
        {
            "Diffused_Total_Legacy_Force": force,
            "Mixed_Source_Ratio_SC_LC": mixed[:, 0],
            "Mixed_Source_Ratio_LC_PFSA_PFCA": mixed[:, 1],
            "Mixed_Source_Ratio_SC_PFSA_PFCA": mixed[:, 2],
        }
    )


def anchor_scenario_features(
    training: pd.DataFrame,
    routed_reference: pd.DataFrame,
    routed_scenario: pd.DataFrame,
):
    if len(training) != len(routed_reference) or len(training) != len(routed_scenario):
        raise ValueError("Routed scenario rows differ from training rows")
    anchored = training.copy()
    correction_rows = []
    for feature in SOURCE_FEATURES:
        correction = training[feature].to_numpy(dtype=float) - routed_reference[
            feature
        ].to_numpy(dtype=float)
        feature_range = float(np.ptp(training[feature].to_numpy(dtype=float)))
        max_correction = float(np.max(np.abs(correction)))
        scale = max(feature_range, float(np.max(np.abs(training[feature]))), 1.0)
        normalized = max_correction / scale
        if not np.isfinite(correction).all() or normalized > ANCHOR_RANGE_TOLERANCE:
            raise ValueError(
                f"{feature}: reference-source reconstruction exceeds anchoring tolerance"
            )
        anchored[feature] = routed_scenario[feature].to_numpy(dtype=float) + correction
        if not np.isfinite(anchored[feature]).all() or (anchored[feature] < 0).any():
            raise ValueError(f"{feature}: invalid routed scenario values")
        correction_rows.append(
            {
                "feature": feature,
                "maximum_absolute_anchor_correction": float(np.max(np.abs(correction))),
                "maximum_correction_divided_by_reference_scale": normalized,
            }
        )
    return anchored, pd.DataFrame(correction_rows)


def coordinate_predictions(package: dict, X: pd.DataFrame) -> np.ndarray:
    if (
        list(X.columns) != list(package["feature_names"])
        or not np.isfinite(X.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Prediction features are invalid or not in model order")
    predictions = []
    for target in COORDINATES:
        model = package["models"][target]
        standardized = np.asarray(model["model_object"].predict(X), dtype=float)
        predictions.append(standardized * model["target_scale"] + model["target_mean"])
    result = np.column_stack(predictions)
    if not np.isfinite(result).all():
        raise ValueError("Model produced non-finite coordinate predictions")
    return result


def derived_outputs(coordinates: np.ndarray) -> dict[str, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError(
            "Magnitude-Balance predictions must be a finite two-column matrix"
        )
    root_two = np.sqrt(2.0)
    log_pfsa = (coordinates[:, 0] + coordinates[:, 1]) / root_two
    log_pfca = (coordinates[:, 0] - coordinates[:, 1]) / root_two
    with np.errstate(over="raise", under="ignore", invalid="raise", divide="raise"):
        try:
            pfsa, pfca = np.exp(log_pfsa), np.exp(log_pfca)
            total = pfsa + pfca
            ratio = np.exp(log_pfsa - log_pfca)
        except FloatingPointError as error:
            raise ValueError(
                "Concentration reconstruction exceeded floating-point limits"
            ) from error
    if np.any(pfsa <= 0) or np.any(pfca <= 0) or np.any(ratio <= 0):
        raise ValueError("Concentration reconstruction underflowed to zero")
    result = {
        "Long_Magnitude": coordinates[:, 0],
        "Long_Balance": coordinates[:, 1],
        "Log_Long_PFSA": log_pfsa,
        "Log_Long_PFCA": log_pfca,
        "Log_Long_Total": np.logaddexp(log_pfsa, log_pfca),
        "Log_Long_Ratio": log_pfsa - log_pfca,
        "Long_PFSA": pfsa,
        "Long_PFCA": pfca,
        "Long_Total": total,
        "Long_Ratio": ratio,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Concentration reconstruction contains non-finite values")
    return result


def coordinate_shap_values(
    package: dict, X: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    values, expected = {}, {}
    for target in COORDINATES:
        model = package["models"][target]
        explainer = shap.TreeExplainer(model["model_object"])
        raw = np.asarray(explainer.shap_values(X), dtype=float)
        values[target] = raw * model["target_scale"]
        raw_expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        expected[target] = raw_expected * model["target_scale"] + model["target_mean"]
    return values, expected


def reconstructed_shap(
    magnitude: np.ndarray, balance: np.ndarray
) -> dict[str, np.ndarray]:
    root_two = np.sqrt(2.0)
    return {
        "Long_Magnitude": magnitude,
        "Long_Balance": balance,
        "Log_Long_PFSA": (magnitude + balance) / root_two,
        "Log_Long_PFCA": (magnitude - balance) / root_two,
        "Log_Long_Ratio": root_two * balance,
    }


def domain_diagnostics(
    X: pd.DataFrame, domain: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(X.to_numpy(dtype=float)).all() or set(X.columns) != set(domain):
        raise ValueError(
            "Feature-domain audit requires finite values and matching domain keys"
        )
    row = pd.DataFrame(index=X.index)
    feature_rows = []
    minmax_flags, central_flags, source_minmax_flags, source_central_flags = (
        [],
        [],
        [],
        [],
    )
    for feature in X.columns:
        limits = domain[feature]
        ordered = [limits[key] for key in ("minimum", "p02_5", "p97_5", "maximum")]
        if not np.isfinite(ordered).all() or ordered != sorted(ordered):
            raise ValueError(f"{feature}: invalid training domain limits")
        values = X[feature].to_numpy(dtype=float)
        outside_minmax = (values < limits["minimum"]) | (values > limits["maximum"])
        outside_central = (values < limits["p02_5"]) | (values > limits["p97_5"])
        minmax_flags.append(outside_minmax)
        central_flags.append(outside_central)
        if feature in SOURCE_FEATURES:
            source_minmax_flags.append(outside_minmax)
            source_central_flags.append(outside_central)
        feature_rows.append(
            {
                "feature": feature,
                "outside_training_minmax_n": int(outside_minmax.sum()),
                "outside_training_central95_n": int(outside_central.sum()),
            }
        )
    row["outside_training_minmax_feature_count"] = np.column_stack(minmax_flags).sum(
        axis=1
    )
    row["outside_training_central95_feature_count"] = np.column_stack(
        central_flags
    ).sum(axis=1)
    row["outside_training_minmax_any"] = (
        row["outside_training_minmax_feature_count"] > 0
    )
    row["outside_training_central95_any"] = (
        row["outside_training_central95_feature_count"] > 0
    )
    row["outside_source_minmax_feature_count"] = np.column_stack(
        source_minmax_flags
    ).sum(axis=1)
    row["outside_source_central95_feature_count"] = np.column_stack(
        source_central_flags
    ).sum(axis=1)
    row["outside_source_minmax_any"] = row["outside_source_minmax_feature_count"] > 0
    row["outside_source_central95_any"] = (
        row["outside_source_central95_feature_count"] > 0
    )
    return row, pd.DataFrame(feature_rows)


def prediction_table(
    metadata: pd.DataFrame, baseline: dict, scenario: dict, scenario_id: str
) -> pd.DataFrame:
    if not metadata["row_index"].is_unique or set(baseline) != set(scenario):
        raise ValueError(
            "Paired predictions require unique row identifiers and identical outputs"
        )
    result = metadata.copy()
    result.insert(0, "scenario_id", scenario_id)
    for output in baseline:
        if len(baseline[output]) != len(metadata) or len(scenario[output]) != len(
            metadata
        ):
            raise ValueError(
                f"{output}: paired prediction lengths differ from metadata"
            )
        result[f"{output}_baseline"] = baseline[output]
        result[f"{output}_scenario"] = scenario[output]
        result[f"{output}_delta"] = scenario[output] - baseline[output]
    return result


def shap_table(
    row_indices, features, X_base, X_scenario, base_values, scenario_values, scenario_id
):
    rows, columns = len(X_base), len(features)
    if (
        len(row_indices) != rows
        or not pd.Index(row_indices).is_unique
        or not X_base.index.equals(X_scenario.index)
        or list(X_base.columns) != list(features)
        or list(X_scenario.columns) != list(features)
    ):
        raise ValueError(
            "Paired SHAP feature matrices or row identifiers are not aligned"
        )
    for matrices in (base_values, scenario_values):
        for target in COORDINATES:
            values = np.asarray(matrices[target])
            if values.shape != (rows, columns) or not np.isfinite(values).all():
                raise ValueError(f"{target}: invalid paired SHAP array")
    result = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "row_index": np.repeat(np.asarray(row_indices), columns),
            "feature": np.tile(features, rows),
            "feature_value_baseline": X_base.to_numpy(dtype=float).reshape(-1),
            "feature_value_scenario": X_scenario.to_numpy(dtype=float).reshape(-1),
        }
    )
    reconstructed_base = reconstructed_shap(
        base_values["Long_Magnitude"], base_values["Long_Balance"]
    )
    reconstructed_scenario = reconstructed_shap(
        scenario_values["Long_Magnitude"], scenario_values["Long_Balance"]
    )
    for output in reconstructed_base:
        base = reconstructed_base[output].reshape(-1)
        perturbed = reconstructed_scenario[output].reshape(-1)
        result[f"{output}_SHAP_baseline"] = base
        result[f"{output}_SHAP_scenario"] = perturbed
        result[f"{output}_SHAP_delta"] = perturbed - base
    return result


def positive_paired_ratio(
    baseline: np.ndarray, scenario: np.ndarray, output: str
) -> np.ndarray:
    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(scenario).all()
        or np.any(baseline <= 0)
        or np.any(scenario <= 0)
    ):
        raise ValueError(
            f"{output}: geometric paired change requires finite positive values"
        )
    with np.errstate(over="raise", under="ignore", invalid="raise", divide="raise"):
        try:
            ratio = scenario / baseline
        except FloatingPointError as error:
            raise ValueError(
                f"{output}: paired ratio exceeded floating-point limits"
            ) from error
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError(f"{output}: invalid paired ratio")
    return ratio


def paired_cluster_bootstrap(
    frame: pd.DataFrame, output: str, include_percent: bool, iterations: int, seed: int
) -> dict:
    if iterations < 1 or frame["Sampling_Group"].isna().any():
        raise ValueError(
            "Cluster bootstrap requires positive iterations and complete group identifiers"
        )
    grouped = [
        np.asarray(indices, dtype=int)
        for indices in frame.groupby("Sampling_Group", sort=True).indices.values()
    ]
    if len(grouped) < 2:
        return {}
    baseline_values = frame[f"{output}_baseline"].to_numpy(dtype=float)
    scenario_values = frame[f"{output}_scenario"].to_numpy(dtype=float)
    if not np.isfinite(baseline_values).all() or not np.isfinite(scenario_values).all():
        raise ValueError(f"{output}: cluster bootstrap inputs must be finite")
    rng, rows = np.random.default_rng(seed), []
    for _ in range(iterations):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        indices = np.concatenate([grouped[position] for position in sampled])
        baseline = baseline_values[indices]
        scenario = scenario_values[indices]
        delta = scenario - baseline
        row = {
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "fraction_increasing": float(np.mean(delta > 0)),
        }
        if include_percent:
            ratio = positive_paired_ratio(baseline, scenario, output)
            row["median_paired_change_percent"] = float(
                np.median((ratio - 1.0) * 100.0)
            )
            row["geometric_mean_paired_change_percent"] = float(
                np.expm1(np.mean(np.log(ratio))) * 100.0
            )
        rows.append(row)
    bootstrap = pd.DataFrame(rows)
    intervals = {}
    for metric in bootstrap:
        intervals[f"{metric}_cluster_CI_lower"] = float(
            bootstrap[metric].quantile(0.025)
        )
        intervals[f"{metric}_cluster_CI_upper"] = float(
            bootstrap[metric].quantile(0.975)
        )
    return intervals


def summarize_predictions(
    frame: pd.DataFrame,
    analysis_role: str,
    claim_scope: str,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rows = []
    outputs = (
        "Long_Magnitude",
        "Long_Balance",
        "Log_Long_PFSA",
        "Log_Long_PFCA",
        "Log_Long_Total",
        "Log_Long_Ratio",
        "Long_PFSA",
        "Long_PFCA",
        "Long_Total",
        "Long_Ratio",
    )
    percent_outputs = {"Long_PFSA", "Long_PFCA", "Long_Total", "Long_Ratio"}
    subsets = [
        ("all_rows", frame),
        ("within_source_minmax", frame[~frame["outside_source_minmax_any"]]),
    ]
    if frame["outside_source_minmax_any"].any():
        subsets.append(
            ("outside_source_minmax", frame[frame["outside_source_minmax_any"]])
        )
    for domain_scope, subset in subsets:
        if subset.empty:
            continue
        for output in outputs:
            baseline = subset[f"{output}_baseline"].to_numpy(dtype=float)
            scenario = subset[f"{output}_scenario"].to_numpy(dtype=float)
            if output in percent_outputs:
                ratio = positive_paired_ratio(baseline, scenario, output)
                paired_percent = (ratio - 1.0) * 100.0
                median_percent = float(np.median(paired_percent))
                geometric_mean_percent = float(np.expm1(np.mean(np.log(ratio))) * 100.0)
            else:
                median_percent = np.nan
                geometric_mean_percent = np.nan
            summary = {
                "scenario_id": frame["scenario_id"].iloc[0],
                "analysis_role": analysis_role,
                "claim_scope": claim_scope,
                "domain_scope": domain_scope,
                "n": len(subset),
                "output": output,
                "baseline_mean": float(np.mean(baseline)),
                "scenario_mean": float(np.mean(scenario)),
                "mean_delta": float(np.mean(scenario - baseline)),
                "median_delta": float(np.median(scenario - baseline)),
                "median_paired_change_percent": median_percent,
                "geometric_mean_paired_change_percent": geometric_mean_percent,
                "fraction_increasing": float(np.mean(scenario > baseline)),
            }
            summary.update(
                paired_cluster_bootstrap(
                    subset,
                    output,
                    output in percent_outputs,
                    bootstrap_iterations,
                    seed,
                )
            )
            rows.append(summary)
    return pd.DataFrame(rows)


def summarize_shap(
    frame: pd.DataFrame, source_features: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = (
        "Long_Magnitude",
        "Long_Balance",
        "Log_Long_PFSA",
        "Log_Long_PFCA",
        "Log_Long_Ratio",
    )
    feature_rows, category_rows = [], []
    subsets = [
        ("all_rows", frame),
        (
            "within_source_minmax",
            frame[frame["source_domain_status"] == "within_source_minmax"],
        ),
    ]
    for domain_scope, domain_frame in subsets:
        if domain_frame.empty:
            continue
        for output in outputs:
            base_column, scenario_column = (
                f"{output}_SHAP_baseline",
                f"{output}_SHAP_scenario",
            )
            grouped = (
                domain_frame.assign(
                    absolute_base=domain_frame[base_column].abs(),
                    absolute_scenario=domain_frame[scenario_column].abs(),
                )
                .groupby("feature", as_index=False)
                .agg(
                    mean_absolute_shap_baseline=("absolute_base", "mean"),
                    mean_absolute_shap_scenario=("absolute_scenario", "mean"),
                    mean_signed_shap_delta=(f"{output}_SHAP_delta", "mean"),
                )
            )
            grouped.insert(0, "domain_scope", domain_scope)
            grouped.insert(0, "output", output)
            grouped.insert(0, "scenario_id", frame["scenario_id"].iloc[0])
            grouped["importance_rank_baseline"] = grouped[
                "mean_absolute_shap_baseline"
            ].rank(method="min", ascending=False)
            grouped["importance_rank_scenario"] = grouped[
                "mean_absolute_shap_scenario"
            ].rank(method="min", ascending=False)
            feature_rows.append(grouped)
            for category, features in (
                ("source", source_features),
                ("environment", tuple(set(frame["feature"]) - set(source_features))),
            ):
                subset = grouped[grouped["feature"].isin(features)]
                base_total = grouped["mean_absolute_shap_baseline"].sum()
                scenario_total = grouped["mean_absolute_shap_scenario"].sum()
                base_share = (
                    subset["mean_absolute_shap_baseline"].sum() / base_total
                    if base_total > 0
                    else np.nan
                )
                scenario_share = (
                    subset["mean_absolute_shap_scenario"].sum() / scenario_total
                    if scenario_total > 0
                    else np.nan
                )
                category_rows.append(
                    {
                        "scenario_id": frame["scenario_id"].iloc[0],
                        "domain_scope": domain_scope,
                        "output": output,
                        "feature_category": category,
                        "absolute_shap_share_baseline_percent": float(100 * base_share),
                        "absolute_shap_share_scenario_percent": float(
                            100 * scenario_share
                        ),
                        "absolute_shap_share_change_percentage_points": float(
                            100 * (scenario_share - base_share)
                        ),
                        "share_defined_baseline": bool(base_total > 0),
                        "share_defined_scenario": bool(scenario_total > 0),
                    }
                )
    return pd.concat(feature_rows, ignore_index=True), pd.DataFrame(category_rows)


def prediction_contrast(
    scenario: pd.DataFrame, comparator: pd.DataFrame, definition: dict
) -> pd.DataFrame:
    if (
        not scenario["row_index"].is_unique
        or not comparator["row_index"].is_unique
        or not scenario["row_index"].equals(comparator["row_index"])
    ):
        raise ValueError(
            f"{definition['contrast_id']}: prediction rows are not aligned"
        )
    metadata = scenario[
        [
            "row_index",
            "Sampling_Group",
            "Sea_Region",
            "SamplingYear",
            "Spatial_Block",
            "Longitude",
            "Latitude",
        ]
    ].copy()
    if not metadata.equals(comparator[metadata.columns]):
        raise ValueError(f"{definition['contrast_id']}: paired metadata differ")
    result = metadata
    result.insert(0, "scenario_id", definition["contrast_id"])
    outputs = (
        "Long_Magnitude",
        "Long_Balance",
        "Log_Long_PFSA",
        "Log_Long_PFCA",
        "Log_Long_Total",
        "Log_Long_Ratio",
        "Long_PFSA",
        "Long_PFCA",
        "Long_Total",
        "Long_Ratio",
    )
    for output in outputs:
        reference = comparator[f"{output}_scenario"].to_numpy(dtype=float)
        changed = scenario[f"{output}_scenario"].to_numpy(dtype=float)
        result[f"{output}_baseline"] = reference
        result[f"{output}_scenario"] = changed
        result[f"{output}_delta"] = changed - reference
    outside = scenario["outside_source_minmax_any"].to_numpy(dtype=bool) | comparator[
        "outside_source_minmax_any"
    ].to_numpy(dtype=bool)
    result["outside_source_minmax_any"] = outside
    return result


def shap_contrast(
    scenario: pd.DataFrame, comparator: pd.DataFrame, definition: dict
) -> pd.DataFrame:
    keys = ["row_index", "feature"]
    if scenario.duplicated(keys).any() or comparator.duplicated(keys).any():
        raise ValueError(
            f"{definition['contrast_id']}: duplicate SHAP row-feature pairs"
        )
    scenario_indexed = scenario.set_index(keys).sort_index()
    comparator_indexed = comparator.set_index(keys).sort_index()
    if not scenario_indexed.index.equals(comparator_indexed.index):
        raise ValueError(f"{definition['contrast_id']}: SHAP rows are not aligned")
    result = pd.DataFrame(index=scenario_indexed.index).reset_index()
    result.insert(0, "scenario_id", definition["contrast_id"])
    result["feature_value_baseline"] = comparator_indexed[
        "feature_value_scenario"
    ].to_numpy()
    result["feature_value_scenario"] = scenario_indexed[
        "feature_value_scenario"
    ].to_numpy()
    outputs = (
        "Long_Magnitude",
        "Long_Balance",
        "Log_Long_PFSA",
        "Log_Long_PFCA",
        "Log_Long_Ratio",
    )
    for output in outputs:
        reference = comparator_indexed[f"{output}_SHAP_scenario"].to_numpy(dtype=float)
        changed = scenario_indexed[f"{output}_SHAP_scenario"].to_numpy(dtype=float)
        result[f"{output}_SHAP_baseline"] = reference
        result[f"{output}_SHAP_scenario"] = changed
        result[f"{output}_SHAP_delta"] = changed - reference
    within = (
        scenario_indexed["source_domain_status"].to_numpy() == "within_source_minmax"
    ) & (
        comparator_indexed["source_domain_status"].to_numpy() == "within_source_minmax"
    )
    result["source_domain_status"] = np.where(
        within, "within_source_minmax", "outside_source_minmax"
    )
    return result


def run(
    model_package_path: Path,
    scenario_directory: Path,
    boundary: Path,
    training_script: Path,
    routing_directory: Path,
    output_directory: Path,
    overwrite: bool,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> None:
    paths = [
        model_package_path.resolve(),
        scenario_directory.resolve(),
        boundary.resolve(),
        training_script.resolve(),
        routing_directory.resolve(),
    ]
    (
        model_package_path,
        scenario_directory,
        boundary,
        training_script,
        routing_directory,
    ) = paths
    output_directory = output_directory.resolve()
    manifest_path = output_directory / "Scenario_Inference_Manifest.json"
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    if (
        output_directory == scenario_directory
        or output_directory in scenario_directory.parents
        or any(
            output_directory == path or output_directory in path.parents
            for path in paths
        )
    ):
        raise ValueError("Inference output directory must not contain input products")
    if output_directory.exists() and any(output_directory.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}. Use --overwrite explicitly"
        )
    package = load_model_package(model_package_path)
    fitted_source = verify_model_provenance(package, training_script, boundary)
    routing_manifest_path = verify_routing_provenance(package, routing_directory)
    training = package_training_frame(package)
    features = package["feature_names"]
    X_base = training[features]
    scenario_summary_path = scenario_directory / "Legacy_Class_Transition_Summary.csv"
    source_manifest_path = scenario_directory / "Legacy_Class_Transition_Manifest.json"
    scenario_summary = pd.read_csv(scenario_summary_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    verify_scenario_products(scenario_directory, source_manifest, scenario_summary)
    if scenario_summary["scenario_id"].duplicated().any() or "reference" not in set(
        scenario_summary["scenario_id"]
    ):
        raise ValueError(
            "Scenario source products require one unique reference scenario"
        )
    reference_row = scenario_summary.set_index("scenario_id").loc["reference"]
    reference_source = validate_source_table(
        pd.read_csv(scenario_directory / reference_row["aggregated_source_file"])
    )
    if fitted_source.shape != reference_source.shape or not np.allclose(
        fitted_source.to_numpy(dtype=float),
        reference_source.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Scenario reference source inventory differs from the source boundary used for fitting"
        )
    routing_rows = training[["row_index", "Longitude", "Latitude"]]
    routed_reference = compute_routed_features(
        routing_rows, reference_source, routing_directory
    )[list(SOURCE_FEATURES)]
    anchor_scenario_features(X_base, routed_reference, routed_reference)
    baseline_coordinates = coordinate_predictions(package, X_base)
    baseline_outputs = derived_outputs(baseline_coordinates)
    baseline_shap, expected_values = coordinate_shap_values(package, X_base)
    output_directory.mkdir(parents=True, exist_ok=True)
    (
        prediction_parts,
        shap_parts,
        prediction_summaries,
        shap_summaries,
        category_summaries,
    ) = ([], [], [], [], [])
    predictions_by_scenario, shap_by_scenario = {}, {}
    domain_parts, domain_feature_parts, anchor_parts = [], [], []
    metadata = training[
        [
            "row_index",
            "Sampling_Group",
            "Sea_Region",
            "SamplingYear",
            "Spatial_Block",
            "Longitude",
            "Latitude",
        ]
    ]
    for row in scenario_summary.itertuples(index=False):
        source_path = scenario_directory / row.aggregated_source_file
        source = validate_source_table(pd.read_csv(source_path))
        if len(source) != len(reference_source) or not np.allclose(
            source[["Grid_Lon", "Grid_Lat"]],
            reference_source[["Grid_Lon", "Grid_Lat"]],
            atol=0,
            rtol=0,
        ):
            raise ValueError(
                f"{row.scenario_id}: source coordinates differ from reference"
            )
        routed = compute_routed_features(routing_rows, source, routing_directory)[
            list(SOURCE_FEATURES)
        ]
        X_scenario, anchor = anchor_scenario_features(X_base, routed_reference, routed)
        coordinates = coordinate_predictions(package, X_scenario)
        outputs = derived_outputs(coordinates)
        scenario_shap, _ = coordinate_shap_values(package, X_scenario)
        predictions = prediction_table(
            metadata, baseline_outputs, outputs, row.scenario_id
        )
        domain_rows, domain_features = domain_diagnostics(
            X_scenario, package["feature_domain"]
        )
        predictions = pd.concat(
            [predictions.reset_index(drop=True), domain_rows.reset_index(drop=True)],
            axis=1,
        )
        shap_values = shap_table(
            training["row_index"],
            features,
            X_base,
            X_scenario,
            baseline_shap,
            scenario_shap,
            row.scenario_id,
        )
        source_domain = np.where(
            domain_rows["outside_source_minmax_any"],
            "outside_source_minmax",
            "within_source_minmax",
        )
        domain_by_index = dict(zip(training["row_index"], source_domain))
        shap_values["source_domain_status"] = shap_values["row_index"].map(
            domain_by_index
        )
        shap_feature, shap_category = summarize_shap(shap_values, SOURCE_FEATURES)
        prediction_parts.append(predictions)
        shap_parts.append(shap_values)
        predictions_by_scenario[row.scenario_id] = predictions
        shap_by_scenario[row.scenario_id] = shap_values
        prediction_summaries.append(
            summarize_predictions(
                predictions,
                row.analysis_role,
                row.claim_scope,
                bootstrap_iterations,
                RANDOM_SEED,
            )
        )
        shap_summaries.append(shap_feature)
        category_summaries.append(shap_category)
        domain_features.insert(0, "scenario_id", row.scenario_id)
        domain_feature_parts.append(domain_features)
        domain_parts.append(
            {
                "scenario_id": row.scenario_id,
                "outside_training_minmax_any_n": int(
                    domain_rows["outside_training_minmax_any"].sum()
                ),
                "outside_training_central95_any_n": int(
                    domain_rows["outside_training_central95_any"].sum()
                ),
                "outside_source_minmax_any_n": int(
                    domain_rows["outside_source_minmax_any"].sum()
                ),
                "outside_source_central95_any_n": int(
                    domain_rows["outside_source_central95_any"].sum()
                ),
                "source_domain_status": (
                    "within_training_minmax"
                    if not domain_rows["outside_source_minmax_any"].any()
                    else (
                        "partly_outside_training_minmax"
                        if not domain_rows["outside_source_minmax_any"].all()
                        else "entirely_outside_training_minmax"
                    )
                ),
            }
        )
        anchor.insert(0, "scenario_id", row.scenario_id)
        anchor_parts.append(anchor)
        print(f"{row.scenario_id}: predictions and paired SHAP completed")
    contrast_predictions, contrast_prediction_summaries = [], []
    contrast_shap_values, contrast_shap_summaries, contrast_category_summaries = (
        [],
        [],
        [],
    )
    contrasts = source_manifest.get("contrasts", [])
    for definition in contrasts:
        scenario_id, comparator_id = (
            definition["scenario_id"],
            definition["comparator_id"],
        )
        if (
            scenario_id not in predictions_by_scenario
            or comparator_id not in predictions_by_scenario
        ):
            raise ValueError(
                f"{definition['contrast_id']}: required scenario output is missing"
            )
        contrasted_predictions = prediction_contrast(
            predictions_by_scenario[scenario_id],
            predictions_by_scenario[comparator_id],
            definition,
        )
        contrasted_shap = shap_contrast(
            shap_by_scenario[scenario_id], shap_by_scenario[comparator_id], definition
        )
        shap_feature, shap_category = summarize_shap(contrasted_shap, SOURCE_FEATURES)
        contrast_predictions.append(contrasted_predictions)
        contrast_prediction_summaries.append(
            summarize_predictions(
                contrasted_predictions,
                "matched_scenario_contrast",
                definition.get("claim_scope", ""),
                bootstrap_iterations,
                RANDOM_SEED,
            )
        )
        contrast_shap_values.append(contrasted_shap)
        contrast_shap_summaries.append(shap_feature)
        contrast_category_summaries.append(shap_category)
        print(f"{definition['contrast_id']}: matched contrast completed")
    products = {
        "predictions": output_directory / "Scenario_Paired_Predictions.csv",
        "shap_values": output_directory / "Scenario_Paired_SHAP.csv",
        "prediction_summary": output_directory / "Scenario_Prediction_Summary.csv",
        "shap_feature_summary": output_directory / "Scenario_SHAP_Feature_Summary.csv",
        "shap_category_summary": output_directory
        / "Scenario_SHAP_Category_Summary.csv",
        "domain_summary": output_directory / "Scenario_Domain_Summary.csv",
        "domain_by_feature": output_directory / "Scenario_Domain_By_Feature.csv",
        "anchor_audit": output_directory / "Scenario_Baseline_Anchor_Audit.csv",
        "contrast_predictions": output_directory / "Scenario_Contrast_Predictions.csv",
        "contrast_prediction_summary": output_directory
        / "Scenario_Contrast_Prediction_Summary.csv",
        "contrast_shap": output_directory / "Scenario_Contrast_SHAP.csv",
        "contrast_shap_feature_summary": output_directory
        / "Scenario_Contrast_SHAP_Feature_Summary.csv",
        "contrast_shap_category_summary": output_directory
        / "Scenario_Contrast_SHAP_Category_Summary.csv",
    }
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        products["predictions"], index=False, encoding="utf-8-sig"
    )
    pd.concat(shap_parts, ignore_index=True).to_csv(
        products["shap_values"], index=False, encoding="utf-8-sig"
    )
    pd.concat(prediction_summaries, ignore_index=True).to_csv(
        products["prediction_summary"], index=False, encoding="utf-8-sig"
    )
    pd.concat(shap_summaries, ignore_index=True).to_csv(
        products["shap_feature_summary"], index=False, encoding="utf-8-sig"
    )
    pd.concat(category_summaries, ignore_index=True).to_csv(
        products["shap_category_summary"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(domain_parts).to_csv(
        products["domain_summary"], index=False, encoding="utf-8-sig"
    )
    pd.concat(domain_feature_parts, ignore_index=True).to_csv(
        products["domain_by_feature"], index=False, encoding="utf-8-sig"
    )
    pd.concat(anchor_parts, ignore_index=True).to_csv(
        products["anchor_audit"], index=False, encoding="utf-8-sig"
    )
    pd.concat(contrast_predictions, ignore_index=True).to_csv(
        products["contrast_predictions"], index=False, encoding="utf-8-sig"
    )
    pd.concat(contrast_prediction_summaries, ignore_index=True).to_csv(
        products["contrast_prediction_summary"], index=False, encoding="utf-8-sig"
    )
    pd.concat(contrast_shap_values, ignore_index=True).to_csv(
        products["contrast_shap"], index=False, encoding="utf-8-sig"
    )
    pd.concat(contrast_shap_summaries, ignore_index=True).to_csv(
        products["contrast_shap_feature_summary"], index=False, encoding="utf-8-sig"
    )
    pd.concat(contrast_category_summaries, ignore_index=True).to_csv(
        products["contrast_shap_category_summary"], index=False, encoding="utf-8-sig"
    )
    manifest = {
        "script_version": VERSION,
        "pfas_schema_sha256": schema_digest(),
        "data_sha256": package["data_sha256"],
        "feature_table_sha256": package["feature_table_sha256"],
        "analysis_label": "Ceteris-paribus source-boundary perturbation",
        "claim_scope": "Dimensionless source-boundary experiments; not calendar-year forecasts or causal effects.",
        "model_package": str(model_package_path),
        "model_package_sha256": sha256(model_package_path),
        "model_schema": package["schema_version"],
        "training_code_sha256": package["training_code_sha256"],
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_evidence": source_manifest.get("claim_scope"),
        "baseline_source_matches_fitted_source": True,
        "training_input_provenance": package["training_input_provenance"],
        "routing_directory": str(routing_directory),
        "routing_manifest_sha256": sha256(routing_manifest_path),
        "routing_matches_fitted_features": True,
        "boundary": str(boundary),
        "boundary_sha256": sha256(boundary),
        "coordinate_expected_values": expected_values,
        "fixed_environmental_features": [
            feature for feature in features if feature not in SOURCE_FEATURES
        ],
        "perturbed_source_features": list(SOURCE_FEATURES),
        "shap_scope": "Paired attribution under one fixed fitted model and one fixed tree baseline; not a causal decomposition.",
        "nonlinear_output_limit": "SHAP is not linearly reconstructed for Log_Long_Total or Long_Total.",
        "paired_summary_uncertainty": (
            f"95% intervals from {bootstrap_iterations} sampling-group-cluster bootstrap resamples; these quantify "
            "sampling heterogeneity of paired model responses, not total model or scenario uncertainty."
        ),
        "matched_contrasts": contrasts,
        "random_seed": RANDOM_SEED,
        "training_rows": len(training),
        "source_nodes": len(reference_source),
        "source_ratio_scope": (
            "Synchronized long-chain attenuation holds the LC PFSA/PFCA ratio fixed at each source node; "
            "SC/LC and the total-load-weighted mixed-source ratios are recalculated rather than fixed."
        ),
        "products": {
            name: {"file": path.name, "sha256": sha256(path)}
            for name, path in products.items()
        },
    }
    write_json(manifest_path, manifest)
    write_json(output_directory / "Run_Manifest.json", manifest)
    print(f"Scenario inference products written to {output_directory}")


def parse_arguments() -> argparse.Namespace:
    project_root = (
        Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        .expanduser()
        .resolve()
    )
    data_root = (
        Path(os.environ.get("PFAS_DATA_ROOT", project_root / "data"))
        .expanduser()
        .resolve()
    )
    output_root = (
        Path(os.environ.get("PFAS_OUTPUT_ROOT", project_root / "outputs"))
        .expanduser()
        .resolve()
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-package",
        type=Path,
        default=output_root
        / "01_primary_model"
        / "Long_Chain_Magnitude_Balance_Model_Package.pkl",
    )
    parser.add_argument(
        "--scenario-directory",
        type=Path,
        default=output_root / "02_source_attenuation_boundary",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=data_root / "reference" / "geospatial" / "china_coastline_boundary.shp",
    )
    parser.add_argument(
        "--routing-directory",
        type=Path,
        required=True,
        help="Frozen routing artifact used to prepare the fitted model's source covariates.",
    )
    parser.add_argument(
        "--training-script",
        type=Path,
        default=project_root
        / "code"
        / "02_modeling_and_validation"
        / "01_primary_model_training.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "03_source_attenuation_inference",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run(
        arguments.model_package,
        arguments.scenario_directory,
        arguments.boundary,
        arguments.training_script,
        arguments.routing_directory,
        arguments.output_dir,
        arguments.overwrite,
        arguments.bootstrap_iterations,
    )
