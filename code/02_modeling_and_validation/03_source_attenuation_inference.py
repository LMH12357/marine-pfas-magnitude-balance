from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.sparse.csgraph import dijkstra


VERSION = "v6.1"
MODEL_SCHEMA = "pfas-long-chain-magnitude-balance-model-package/v2"
COORDINATES = ("Long_Magnitude", "Long_Balance")
SOURCE_FEATURES = (
    "Diffused_Total_Legacy_Force",
    "Mixed_Source_Ratio_SC_LC",
    "Mixed_Source_Ratio_LC_PFSA_PFCA",
    "Mixed_Source_Ratio_SC_PFSA_PFCA",
)
SOURCE_COLUMNS = (
    "Grid_Lon", "Grid_Lat", "Load_Total_Legacy_Kg", "Source_Ratio_SC_LC",
    "Source_Ratio_LC_PFSA_PFCA", "Source_Ratio_SC_PFSA_PFCA",
)
ANCHOR_RANGE_TOLERANCE = 1e-3
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


def load_training_module(path: Path):
    specification = importlib.util.spec_from_file_location("pfas_training_contract", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load training contract: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_model_package(path: Path) -> dict:
    package = joblib.load(path)
    if package.get("schema_version") != MODEL_SCHEMA:
        raise ValueError(f"Unsupported model package schema: {package.get('schema_version')}")
    if tuple(package.get("coordinate_system", {}).get("coordinate_order", ())) != COORDINATES:
        raise ValueError("Model package coordinate order is incompatible")
    if set(package.get("models", {})) != set(COORDINATES):
        raise ValueError("Model package must contain both coordinate models")
    features = package.get("feature_names", [])
    if not features or not set(SOURCE_FEATURES).issubset(features):
        raise ValueError("Model package feature contract is incomplete")
    return package


def validate_source_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Source table is missing columns: {missing}")
    result = frame[list(SOURCE_COLUMNS)].copy()
    result[list(SOURCE_COLUMNS)] = result[list(SOURCE_COLUMNS)].apply(pd.to_numeric, errors="raise")
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
    frame = pd.concat([metadata, coordinates, X], axis=1)
    frame.insert(0, "row_index", package["training_row_indices"])
    return frame.reset_index(drop=True)


def build_routing_weights(training_module, samples: pd.DataFrame, sources: pd.DataFrame, boundary: Path):
    graph, is_land, longitudes, latitudes = training_module.build_coastal_routing_graph(boundary)
    resolution, columns = 0.1, len(longitudes)
    source_rows = np.clip(
        np.rint((sources["Grid_Lat"].to_numpy() - latitudes[0]) / resolution).astype(int),
        0, is_land.shape[0] - 1,
    )
    source_columns = np.clip(
        np.rint((sources["Grid_Lon"].to_numpy() - longitudes[0]) / resolution).astype(int),
        0, is_land.shape[1] - 1,
    )
    weights = np.empty((len(samples), len(sources)), dtype=float)
    for position, row in enumerate(samples.itertuples(index=False), 1):
        grid_column = int(np.clip(round((float(row.Longitude) - longitudes[0]) / resolution), 0, is_land.shape[1] - 1))
        grid_row = int(np.clip(round((float(row.Latitude) - latitudes[0]) / resolution), 0, is_land.shape[0] - 1))
        sea_row, sea_column = training_module.nearest_sea_node(grid_row, grid_column, is_land)
        distances = dijkstra(graph, directed=False, indices=sea_row * columns + sea_column)
        source_distances = distances[source_rows * columns + source_columns]
        weights[position - 1] = np.where(np.isfinite(source_distances), 1.0 / (source_distances + 1.0), 0.0)
        if position % 50 == 0 or position == len(samples):
            print(f"Routing weights: {position}/{len(samples)}")
    return weights


def route_source_features(weights: np.ndarray, source: pd.DataFrame) -> pd.DataFrame:
    loads = source["Load_Total_Legacy_Kg"].to_numpy(dtype=float)
    weighted_load = weights * loads[None, :]
    force = weighted_load.sum(axis=1)
    if np.any(force <= 0):
        raise ValueError("At least one sampling point has zero routed source force")
    ratios = source[[
        "Source_Ratio_SC_LC", "Source_Ratio_LC_PFSA_PFCA", "Source_Ratio_SC_PFSA_PFCA",
    ]].to_numpy(dtype=float)
    mixed = weighted_load @ ratios / force[:, None]
    return pd.DataFrame({
        "Diffused_Total_Legacy_Force": force,
        "Mixed_Source_Ratio_SC_LC": mixed[:, 0],
        "Mixed_Source_Ratio_LC_PFSA_PFCA": mixed[:, 1],
        "Mixed_Source_Ratio_SC_PFSA_PFCA": mixed[:, 2],
    })


def anchor_scenario_features(training: pd.DataFrame, routed_reference: pd.DataFrame, routed_scenario: pd.DataFrame):
    anchored = training.copy()
    correction_rows = []
    for feature in SOURCE_FEATURES:
        correction = training[feature].to_numpy(dtype=float) - routed_reference[feature].to_numpy(dtype=float)
        feature_range = float(np.ptp(training[feature].to_numpy(dtype=float)))
        normalized = float(np.max(np.abs(correction)) / feature_range) if feature_range > 0 else 0.0
        if normalized > ANCHOR_RANGE_TOLERANCE:
            raise ValueError(f"{feature}: reference-source reconstruction exceeds anchoring tolerance")
        anchored[feature] = routed_scenario[feature].to_numpy(dtype=float) + correction
        correction_rows.append({
            "feature": feature, "maximum_absolute_anchor_correction": float(np.max(np.abs(correction))),
            "maximum_correction_divided_by_training_range": normalized,
        })
    return anchored, pd.DataFrame(correction_rows)


def coordinate_predictions(package: dict, X: pd.DataFrame) -> np.ndarray:
    predictions = []
    for target in COORDINATES:
        model = package["models"][target]
        standardized = np.asarray(model["model_object"].predict(X), dtype=float)
        predictions.append(standardized * model["target_scale"] + model["target_mean"])
    return np.column_stack(predictions)


def derived_outputs(coordinates: np.ndarray) -> dict[str, np.ndarray]:
    root_two = np.sqrt(2.0)
    log_pfsa = (coordinates[:, 0] + coordinates[:, 1]) / root_two
    log_pfca = (coordinates[:, 0] - coordinates[:, 1]) / root_two
    pfsa, pfca = np.exp(log_pfsa), np.exp(log_pfca)
    return {
        "Long_Magnitude": coordinates[:, 0], "Long_Balance": coordinates[:, 1],
        "Log_Long_PFSA": log_pfsa, "Log_Long_PFCA": log_pfca,
        "Log_Long_Total": np.log(pfsa + pfca), "Log_Long_Ratio": log_pfsa - log_pfca,
        "Long_PFSA": pfsa, "Long_PFCA": pfca, "Long_Total": pfsa + pfca,
        "Long_Ratio": pfsa / pfca,
    }


def coordinate_shap_values(package: dict, X: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    values, expected = {}, {}
    for target in COORDINATES:
        model = package["models"][target]
        explainer = shap.TreeExplainer(model["model_object"])
        raw = np.asarray(explainer.shap_values(X), dtype=float)
        values[target] = raw * model["target_scale"]
        raw_expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        expected[target] = raw_expected * model["target_scale"] + model["target_mean"]
    return values, expected


def reconstructed_shap(magnitude: np.ndarray, balance: np.ndarray) -> dict[str, np.ndarray]:
    root_two = np.sqrt(2.0)
    return {
        "Long_Magnitude": magnitude,
        "Long_Balance": balance,
        "Log_Long_PFSA": (magnitude + balance) / root_two,
        "Log_Long_PFCA": (magnitude - balance) / root_two,
        "Log_Long_Ratio": root_two * balance,
    }


def domain_diagnostics(X: pd.DataFrame, domain: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    row = pd.DataFrame(index=X.index)
    feature_rows = []
    minmax_flags, central_flags, source_minmax_flags, source_central_flags = [], [], [], []
    for feature in X.columns:
        limits = domain[feature]
        values = X[feature].to_numpy(dtype=float)
        outside_minmax = (values < limits["minimum"]) | (values > limits["maximum"])
        outside_central = (values < limits["p02_5"]) | (values > limits["p97_5"])
        minmax_flags.append(outside_minmax)
        central_flags.append(outside_central)
        if feature in SOURCE_FEATURES:
            source_minmax_flags.append(outside_minmax)
            source_central_flags.append(outside_central)
        feature_rows.append({
            "feature": feature, "outside_training_minmax_n": int(outside_minmax.sum()),
            "outside_training_central95_n": int(outside_central.sum()),
        })
    row["outside_training_minmax_feature_count"] = np.column_stack(minmax_flags).sum(axis=1)
    row["outside_training_central95_feature_count"] = np.column_stack(central_flags).sum(axis=1)
    row["outside_training_minmax_any"] = row["outside_training_minmax_feature_count"] > 0
    row["outside_training_central95_any"] = row["outside_training_central95_feature_count"] > 0
    row["outside_source_minmax_feature_count"] = np.column_stack(source_minmax_flags).sum(axis=1)
    row["outside_source_central95_feature_count"] = np.column_stack(source_central_flags).sum(axis=1)
    row["outside_source_minmax_any"] = row["outside_source_minmax_feature_count"] > 0
    row["outside_source_central95_any"] = row["outside_source_central95_feature_count"] > 0
    return row, pd.DataFrame(feature_rows)


def prediction_table(metadata: pd.DataFrame, baseline: dict, scenario: dict, scenario_id: str) -> pd.DataFrame:
    result = metadata.copy()
    result.insert(0, "scenario_id", scenario_id)
    for output in baseline:
        result[f"{output}_baseline"] = baseline[output]
        result[f"{output}_scenario"] = scenario[output]
        result[f"{output}_delta"] = scenario[output] - baseline[output]
    return result


def shap_table(row_indices, features, X_base, X_scenario, base_values, scenario_values, scenario_id):
    rows, columns = len(X_base), len(features)
    result = pd.DataFrame({
        "scenario_id": scenario_id,
        "row_index": np.repeat(np.asarray(row_indices), columns),
        "feature": np.tile(features, rows),
        "feature_value_baseline": X_base.to_numpy(dtype=float).reshape(-1),
        "feature_value_scenario": X_scenario.to_numpy(dtype=float).reshape(-1),
    })
    reconstructed_base = reconstructed_shap(base_values["Long_Magnitude"], base_values["Long_Balance"])
    reconstructed_scenario = reconstructed_shap(scenario_values["Long_Magnitude"], scenario_values["Long_Balance"])
    for output in reconstructed_base:
        base = reconstructed_base[output].reshape(-1)
        perturbed = reconstructed_scenario[output].reshape(-1)
        result[f"{output}_SHAP_baseline"] = base
        result[f"{output}_SHAP_scenario"] = perturbed
        result[f"{output}_SHAP_delta"] = perturbed - base
    return result


def paired_cluster_bootstrap(frame: pd.DataFrame, output: str, include_percent: bool, iterations: int, seed: int) -> dict:
    grouped = [group for _, group in frame.groupby("Sampling_Group", sort=True)]
    if len(grouped) < 2:
        return {}
    rng, rows = np.random.default_rng(seed), []
    for _ in range(iterations):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        bootstrap = pd.concat([grouped[position] for position in sampled], ignore_index=True)
        baseline = bootstrap[f"{output}_baseline"].to_numpy(dtype=float)
        scenario = bootstrap[f"{output}_scenario"].to_numpy(dtype=float)
        delta = scenario - baseline
        row = {
            "mean_delta": float(np.mean(delta)), "median_delta": float(np.median(delta)),
            "fraction_increasing": float(np.mean(delta > 0)),
        }
        if include_percent:
            ratio = scenario / baseline
            if np.any(ratio <= 0):
                raise ValueError(f"{output}: geometric paired change requires positive values")
            row["median_paired_change_percent"] = float(np.median((ratio - 1.0) * 100.0))
            row["geometric_mean_paired_change_percent"] = float(np.expm1(np.mean(np.log(ratio))) * 100.0)
        rows.append(row)
    bootstrap = pd.DataFrame(rows)
    intervals = {}
    for metric in bootstrap:
        intervals[f"{metric}_cluster_CI_lower"] = float(bootstrap[metric].quantile(0.025))
        intervals[f"{metric}_cluster_CI_upper"] = float(bootstrap[metric].quantile(0.975))
    return intervals


def summarize_predictions(
    frame: pd.DataFrame, analysis_role: str, claim_scope: str,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS, seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rows = []
    outputs = (
        "Long_Magnitude", "Long_Balance", "Log_Long_PFSA", "Log_Long_PFCA",
        "Log_Long_Total", "Log_Long_Ratio", "Long_PFSA", "Long_PFCA", "Long_Total", "Long_Ratio",
    )
    percent_outputs = {"Long_PFSA", "Long_PFCA", "Long_Total", "Long_Ratio"}
    subsets = [("all_rows", frame), ("within_source_minmax", frame[~frame["outside_source_minmax_any"]])]
    if frame["outside_source_minmax_any"].any():
        subsets.append(("outside_source_minmax", frame[frame["outside_source_minmax_any"]]))
    for domain_scope, subset in subsets:
        if subset.empty:
            continue
        for output in outputs:
            baseline = subset[f"{output}_baseline"].to_numpy(dtype=float)
            scenario = subset[f"{output}_scenario"].to_numpy(dtype=float)
            if output in percent_outputs:
                ratio = scenario / baseline
                if np.any(ratio <= 0):
                    raise ValueError(f"{output}: geometric paired change requires positive values")
                paired_percent = (ratio - 1.0) * 100.0
                median_percent = float(np.median(paired_percent))
                geometric_mean_percent = float(np.expm1(np.mean(np.log(ratio))) * 100.0)
            else:
                median_percent = np.nan
                geometric_mean_percent = np.nan
            summary = {
                "scenario_id": frame["scenario_id"].iloc[0], "analysis_role": analysis_role,
                "claim_scope": claim_scope, "domain_scope": domain_scope, "n": len(subset), "output": output,
                "baseline_mean": float(np.mean(baseline)), "scenario_mean": float(np.mean(scenario)),
                "mean_delta": float(np.mean(scenario - baseline)), "median_delta": float(np.median(scenario - baseline)),
                "median_paired_change_percent": median_percent,
                "geometric_mean_paired_change_percent": geometric_mean_percent,
                "fraction_increasing": float(np.mean(scenario > baseline)),
            }
            summary.update(paired_cluster_bootstrap(
                subset, output, output in percent_outputs, bootstrap_iterations, seed,
            ))
            rows.append(summary)
    return pd.DataFrame(rows)


def summarize_shap(frame: pd.DataFrame, source_features: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = ("Long_Magnitude", "Long_Balance", "Log_Long_PFSA", "Log_Long_PFCA", "Log_Long_Ratio")
    feature_rows, category_rows = [], []
    subsets = [("all_rows", frame), ("within_source_minmax", frame[frame["source_domain_status"] == "within_source_minmax"])]
    for domain_scope, domain_frame in subsets:
        if domain_frame.empty:
            continue
        for output in outputs:
            base_column, scenario_column = f"{output}_SHAP_baseline", f"{output}_SHAP_scenario"
            grouped = domain_frame.assign(
                absolute_base=domain_frame[base_column].abs(), absolute_scenario=domain_frame[scenario_column].abs()
            ).groupby("feature", as_index=False).agg(
                mean_absolute_shap_baseline=("absolute_base", "mean"),
                mean_absolute_shap_scenario=("absolute_scenario", "mean"),
                mean_signed_shap_delta=(f"{output}_SHAP_delta", "mean"),
            )
            grouped.insert(0, "domain_scope", domain_scope)
            grouped.insert(0, "output", output)
            grouped.insert(0, "scenario_id", frame["scenario_id"].iloc[0])
            grouped["importance_rank_baseline"] = grouped["mean_absolute_shap_baseline"].rank(method="min", ascending=False)
            grouped["importance_rank_scenario"] = grouped["mean_absolute_shap_scenario"].rank(method="min", ascending=False)
            feature_rows.append(grouped)
            for category, features in (("source", source_features), ("environment", tuple(set(frame["feature"]) - set(source_features)))):
                subset = grouped[grouped["feature"].isin(features)]
                base_share = subset["mean_absolute_shap_baseline"].sum() / grouped["mean_absolute_shap_baseline"].sum()
                scenario_share = subset["mean_absolute_shap_scenario"].sum() / grouped["mean_absolute_shap_scenario"].sum()
                category_rows.append({
                    "scenario_id": frame["scenario_id"].iloc[0], "domain_scope": domain_scope,
                    "output": output, "feature_category": category,
                    "absolute_shap_share_baseline_percent": float(100 * base_share),
                    "absolute_shap_share_scenario_percent": float(100 * scenario_share),
                    "absolute_shap_share_change_percentage_points": float(100 * (scenario_share - base_share)),
                })
    return pd.concat(feature_rows, ignore_index=True), pd.DataFrame(category_rows)


def prediction_contrast(scenario: pd.DataFrame, comparator: pd.DataFrame, definition: dict) -> pd.DataFrame:
    if not scenario["row_index"].equals(comparator["row_index"]):
        raise ValueError(f"{definition['contrast_id']}: prediction rows are not aligned")
    metadata = scenario[["row_index", "Sampling_Group", "Sea_Region", "SamplingYear", "Spatial_Block", "Longitude", "Latitude"]].copy()
    result = metadata
    result.insert(0, "scenario_id", definition["contrast_id"])
    outputs = (
        "Long_Magnitude", "Long_Balance", "Log_Long_PFSA", "Log_Long_PFCA", "Log_Long_Total",
        "Log_Long_Ratio", "Long_PFSA", "Long_PFCA", "Long_Total", "Long_Ratio",
    )
    for output in outputs:
        reference = comparator[f"{output}_scenario"].to_numpy(dtype=float)
        changed = scenario[f"{output}_scenario"].to_numpy(dtype=float)
        result[f"{output}_baseline"] = reference
        result[f"{output}_scenario"] = changed
        result[f"{output}_delta"] = changed - reference
    outside = scenario["outside_source_minmax_any"].to_numpy(dtype=bool) | comparator["outside_source_minmax_any"].to_numpy(dtype=bool)
    result["outside_source_minmax_any"] = outside
    return result


def shap_contrast(scenario: pd.DataFrame, comparator: pd.DataFrame, definition: dict) -> pd.DataFrame:
    keys = ["row_index", "feature"]
    scenario_indexed = scenario.set_index(keys).sort_index()
    comparator_indexed = comparator.set_index(keys).sort_index()
    if not scenario_indexed.index.equals(comparator_indexed.index):
        raise ValueError(f"{definition['contrast_id']}: SHAP rows are not aligned")
    result = pd.DataFrame(index=scenario_indexed.index).reset_index()
    result.insert(0, "scenario_id", definition["contrast_id"])
    result["feature_value_baseline"] = comparator_indexed["feature_value_scenario"].to_numpy()
    result["feature_value_scenario"] = scenario_indexed["feature_value_scenario"].to_numpy()
    outputs = ("Long_Magnitude", "Long_Balance", "Log_Long_PFSA", "Log_Long_PFCA", "Log_Long_Ratio")
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
    result["source_domain_status"] = np.where(within, "within_source_minmax", "outside_source_minmax")
    return result


def run(
    model_package_path: Path, scenario_directory: Path, boundary: Path, training_script: Path,
    output_directory: Path, overwrite: bool, bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> None:
    paths = [model_package_path.resolve(), scenario_directory.resolve(), boundary.resolve(), training_script.resolve()]
    model_package_path, scenario_directory, boundary, training_script = paths
    output_directory = output_directory.resolve()
    manifest_path = output_directory / "Scenario_Inference_Manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {manifest_path}. Use --overwrite to replace this run")
    output_directory.mkdir(parents=True, exist_ok=True)
    package = load_model_package(model_package_path)
    training = package_training_frame(package)
    features = package["feature_names"]
    X_base = training[features]
    scenario_summary_path = scenario_directory / "Legacy_Class_Transition_Summary.csv"
    source_manifest_path = scenario_directory / "Legacy_Class_Transition_Manifest.json"
    scenario_summary = pd.read_csv(scenario_summary_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if scenario_summary["scenario_id"].duplicated().any() or "reference" not in set(scenario_summary["scenario_id"]):
        raise ValueError("Scenario source products require one unique reference scenario")
    reference_row = scenario_summary.set_index("scenario_id").loc["reference"]
    reference_source = validate_source_table(pd.read_csv(scenario_directory / reference_row["aggregated_source_file"]))
    training_module = load_training_module(training_script)
    weights = build_routing_weights(training_module, training[["Longitude", "Latitude"]], reference_source, boundary)
    routed_reference = route_source_features(weights, reference_source)
    baseline_coordinates = coordinate_predictions(package, X_base)
    baseline_outputs = derived_outputs(baseline_coordinates)
    baseline_shap, expected_values = coordinate_shap_values(package, X_base)
    prediction_parts, shap_parts, prediction_summaries, shap_summaries, category_summaries = [], [], [], [], []
    predictions_by_scenario, shap_by_scenario = {}, {}
    domain_parts, domain_feature_parts, anchor_parts = [], [], []
    metadata = training[["row_index", "Sampling_Group", "Sea_Region", "SamplingYear", "Spatial_Block", "Longitude", "Latitude"]]
    for row in scenario_summary.itertuples(index=False):
        source_path = scenario_directory / row.aggregated_source_file
        source = validate_source_table(pd.read_csv(source_path))
        if not np.allclose(source[["Grid_Lon", "Grid_Lat"]], reference_source[["Grid_Lon", "Grid_Lat"]], atol=0, rtol=0):
            raise ValueError(f"{row.scenario_id}: source coordinates differ from reference")
        routed = route_source_features(weights, source)
        X_scenario, anchor = anchor_scenario_features(X_base, routed_reference, routed)
        coordinates = coordinate_predictions(package, X_scenario)
        outputs = derived_outputs(coordinates)
        scenario_shap, _ = coordinate_shap_values(package, X_scenario)
        predictions = prediction_table(metadata, baseline_outputs, outputs, row.scenario_id)
        domain_rows, domain_features = domain_diagnostics(X_scenario, package["feature_domain"])
        predictions = pd.concat([predictions.reset_index(drop=True), domain_rows.reset_index(drop=True)], axis=1)
        shap_values = shap_table(training["row_index"], features, X_base, X_scenario, baseline_shap, scenario_shap, row.scenario_id)
        source_domain = np.where(domain_rows["outside_source_minmax_any"], "outside_source_minmax", "within_source_minmax")
        domain_by_index = dict(zip(training["row_index"], source_domain))
        shap_values["source_domain_status"] = shap_values["row_index"].map(domain_by_index)
        shap_feature, shap_category = summarize_shap(shap_values, SOURCE_FEATURES)
        prediction_parts.append(predictions)
        shap_parts.append(shap_values)
        predictions_by_scenario[row.scenario_id] = predictions
        shap_by_scenario[row.scenario_id] = shap_values
        prediction_summaries.append(summarize_predictions(
            predictions, row.analysis_role, row.claim_scope, bootstrap_iterations, RANDOM_SEED,
        ))
        shap_summaries.append(shap_feature)
        category_summaries.append(shap_category)
        domain_features.insert(0, "scenario_id", row.scenario_id)
        domain_feature_parts.append(domain_features)
        domain_parts.append({
            "scenario_id": row.scenario_id,
            "outside_training_minmax_any_n": int(domain_rows["outside_training_minmax_any"].sum()),
            "outside_training_central95_any_n": int(domain_rows["outside_training_central95_any"].sum()),
            "outside_source_minmax_any_n": int(domain_rows["outside_source_minmax_any"].sum()),
            "outside_source_central95_any_n": int(domain_rows["outside_source_central95_any"].sum()),
            "source_domain_status": (
                "within_training_minmax" if not domain_rows["outside_source_minmax_any"].any()
                else "partly_outside_training_minmax" if not domain_rows["outside_source_minmax_any"].all()
                else "entirely_outside_training_minmax"
            ),
        })
        anchor.insert(0, "scenario_id", row.scenario_id)
        anchor_parts.append(anchor)
        print(f"{row.scenario_id}: predictions and paired SHAP completed")
    contrast_predictions, contrast_prediction_summaries = [], []
    contrast_shap_values, contrast_shap_summaries, contrast_category_summaries = [], [], []
    contrasts = source_manifest.get("contrasts", [])
    for definition in contrasts:
        scenario_id, comparator_id = definition["scenario_id"], definition["comparator_id"]
        if scenario_id not in predictions_by_scenario or comparator_id not in predictions_by_scenario:
            raise ValueError(f"{definition['contrast_id']}: required scenario output is missing")
        contrasted_predictions = prediction_contrast(
            predictions_by_scenario[scenario_id], predictions_by_scenario[comparator_id], definition,
        )
        contrasted_shap = shap_contrast(shap_by_scenario[scenario_id], shap_by_scenario[comparator_id], definition)
        shap_feature, shap_category = summarize_shap(contrasted_shap, SOURCE_FEATURES)
        contrast_predictions.append(contrasted_predictions)
        contrast_prediction_summaries.append(summarize_predictions(
            contrasted_predictions, "matched_scenario_contrast", definition.get("claim_scope", ""),
            bootstrap_iterations, RANDOM_SEED,
        ))
        contrast_shap_values.append(contrasted_shap)
        contrast_shap_summaries.append(shap_feature)
        contrast_category_summaries.append(shap_category)
        print(f"{definition['contrast_id']}: matched contrast completed")
    products = {
        "predictions": output_directory / "Scenario_Paired_Predictions.csv",
        "shap_values": output_directory / "Scenario_Paired_SHAP.csv",
        "prediction_summary": output_directory / "Scenario_Prediction_Summary.csv",
        "shap_feature_summary": output_directory / "Scenario_SHAP_Feature_Summary.csv",
        "shap_category_summary": output_directory / "Scenario_SHAP_Category_Summary.csv",
        "domain_summary": output_directory / "Scenario_Domain_Summary.csv",
        "domain_by_feature": output_directory / "Scenario_Domain_By_Feature.csv",
        "anchor_audit": output_directory / "Scenario_Baseline_Anchor_Audit.csv",
        "contrast_predictions": output_directory / "Scenario_Contrast_Predictions.csv",
        "contrast_prediction_summary": output_directory / "Scenario_Contrast_Prediction_Summary.csv",
        "contrast_shap": output_directory / "Scenario_Contrast_SHAP.csv",
        "contrast_shap_feature_summary": output_directory / "Scenario_Contrast_SHAP_Feature_Summary.csv",
        "contrast_shap_category_summary": output_directory / "Scenario_Contrast_SHAP_Category_Summary.csv",
    }
    pd.concat(prediction_parts, ignore_index=True).to_csv(products["predictions"], index=False, encoding="utf-8-sig")
    pd.concat(shap_parts, ignore_index=True).to_csv(products["shap_values"], index=False, encoding="utf-8-sig")
    pd.concat(prediction_summaries, ignore_index=True).to_csv(products["prediction_summary"], index=False, encoding="utf-8-sig")
    pd.concat(shap_summaries, ignore_index=True).to_csv(products["shap_feature_summary"], index=False, encoding="utf-8-sig")
    pd.concat(category_summaries, ignore_index=True).to_csv(products["shap_category_summary"], index=False, encoding="utf-8-sig")
    pd.DataFrame(domain_parts).to_csv(products["domain_summary"], index=False, encoding="utf-8-sig")
    pd.concat(domain_feature_parts, ignore_index=True).to_csv(products["domain_by_feature"], index=False, encoding="utf-8-sig")
    pd.concat(anchor_parts, ignore_index=True).to_csv(products["anchor_audit"], index=False, encoding="utf-8-sig")
    pd.concat(contrast_predictions, ignore_index=True).to_csv(products["contrast_predictions"], index=False, encoding="utf-8-sig")
    pd.concat(contrast_prediction_summaries, ignore_index=True).to_csv(products["contrast_prediction_summary"], index=False, encoding="utf-8-sig")
    pd.concat(contrast_shap_values, ignore_index=True).to_csv(products["contrast_shap"], index=False, encoding="utf-8-sig")
    pd.concat(contrast_shap_summaries, ignore_index=True).to_csv(products["contrast_shap_feature_summary"], index=False, encoding="utf-8-sig")
    pd.concat(contrast_category_summaries, ignore_index=True).to_csv(products["contrast_shap_category_summary"], index=False, encoding="utf-8-sig")
    manifest = {
        "script_version": VERSION,
        "analysis_label": "Ceteris-paribus source-boundary perturbation",
        "claim_scope": "Dimensionless source-boundary experiments; not calendar-year forecasts or causal effects.",
        "model_package": str(model_package_path), "model_package_sha256": sha256(model_package_path),
        "model_schema": package["schema_version"], "training_code_sha256": package["training_code_sha256"],
        "source_manifest": str(source_manifest_path), "source_manifest_sha256": sha256(source_manifest_path),
        "source_evidence": source_manifest.get("claim_scope"),
        "boundary": str(boundary), "boundary_sha256": sha256(boundary),
        "coordinate_expected_values": expected_values,
        "fixed_environmental_features": [feature for feature in features if feature not in SOURCE_FEATURES],
        "perturbed_source_features": list(SOURCE_FEATURES),
        "shap_scope": "Paired attribution under one fixed fitted model and one fixed tree baseline; not a causal decomposition.",
        "nonlinear_output_limit": "SHAP is not linearly reconstructed for Log_Long_Total or Long_Total.",
        "paired_summary_uncertainty": (
            f"95% intervals from {bootstrap_iterations} sampling-group-cluster bootstrap resamples; these quantify "
            "sampling heterogeneity of paired model responses, not total model or scenario uncertainty."
        ),
        "matched_contrasts": contrasts,
        "products": {name: {"file": path.name, "sha256": sha256(path)} for name, path in products.items()},
    }
    write_json(manifest_path, manifest)
    print(f"Scenario inference products written to {output_directory}")


def parse_arguments() -> argparse.Namespace:
    project_root = Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
    data_root = Path(os.environ.get("PFAS_DATA_ROOT", project_root / "data")).expanduser().resolve()
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package", type=Path, default=data_root / "fixed_outputs" / "primary_model" / "Long_Chain_Magnitude_Balance_Model_Package.pkl")
    parser.add_argument("--scenario-directory", type=Path, default=output_root / "02_source_attenuation_boundary")
    parser.add_argument("--boundary", type=Path, default=data_root / "reference" / "geospatial" / "china_coastline_boundary.shp")
    parser.add_argument("--training-script", type=Path, default=project_root / "code" / "02_modeling_and_validation" / "01_primary_model_training.py")
    parser.add_argument("--output-dir", type=Path, default=output_root / "03_source_attenuation_inference")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run(
        arguments.model_package, arguments.scenario_directory, arguments.boundary,
        arguments.training_script, arguments.output_dir, arguments.overwrite, arguments.bootstrap_iterations,
    )
