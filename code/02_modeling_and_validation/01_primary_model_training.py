from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Point
from shapely.prepared import prep
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut, StratifiedGroupKFold,
    cross_val_predict,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadata_schema import DEFAULT_SEA_REFERENCE, normalize_metadata as normalize_metadata_contract


VERSION = "v6.1"
MODEL_PACKAGE_SCHEMA = "pfas-long-chain-magnitude-balance-model-package/v2"
PRIMARY_TARGETS = ("Long_Magnitude", "Long_Balance")
RANDOM_SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
SPATIAL_BLOCK_DEGREES = 2.0
INTERVAL_ALPHA = 0.05

COVARIATES = [
    "Temperature (°C)", "Salinity (‰)", "Chla", "POC (mg/m3)", "TOC (%)",
    "pH", "SPM", "DO", "Hydro_Bottom_U", "Hydro_Bottom_V", "Hydro_MLD",
    "Diffused_Total_Legacy_Force", "Mixed_Source_Ratio_SC_LC",
    "Mixed_Source_Ratio_LC_PFSA_PFCA", "Mixed_Source_Ratio_SC_PFSA_PFCA",
]

TARGET_ROLES = {
    "Long_Magnitude": "abundance_primary",
    "Long_Balance": "fractionation_primary",
}

LONG_PFSA = ("PFHxS", "PFHpS", "PFOS", "PFNS", "PFDS", "PFECHS")
LONG_PFCA = ("PFOA", "PFNA", "PFDA", "PFUnDA", "PFDoDA", "PFTrDA", "PFTeDA", "PFHxDA", "PFODA")

ENVIRONMENT_COVARIATES = COVARIATES[:11]
SOURCE_COVARIATES = COVARIATES[11:]
FEATURE_BLOCKS = {
    "all": COVARIATES,
    "environment": ENVIRONMENT_COVARIATES,
    "source": SOURCE_COVARIATES,
}

RF_GRID = {
    "n_estimators": [100], "max_depth": [3, 5],
    "max_features": [0.8], "max_samples": [0.8],
}
XGB_GRID = {
    "n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.03, 0.1],
    "subsample": [0.8], "colsample_bytree": [0.8],
}
VALIDATION_SCHEMES = (
    "random", "sampling_group", "sampling_group_stratified", "spatial_block",
    "spatial_block_stratified", "leave_one_sampling_group_out", "leave_one_sea_region_out",
)
DEFAULT_VALIDATIONS = (
    "sampling_group", "spatial_block_stratified", "leave_one_sampling_group_out", "leave_one_sea_region_out",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ScaledTargetRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        values = np.asarray(y, dtype=float)
        self.target_mean_ = float(np.mean(values))
        self.target_scale_ = float(np.std(values))
        if self.target_scale_ <= 0:
            self.target_scale_ = 1.0
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, (values - self.target_mean_) / self.target_scale_)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X), dtype=float) * self.target_scale_ + self.target_mean_


def normalized_rmse_scorer(estimator, X, y):
    observed = np.asarray(y, dtype=float)
    predicted = np.asarray(estimator.predict(X), dtype=float)
    return -float(np.sqrt(np.mean(((observed - predicted) / estimator.target_scale_) ** 2)))


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    estimator: object
    grid: dict[str, list]


def model_definitions(seed, quick):
    rf_grid = RF_GRID if not quick else {key: [values[0]] for key, values in RF_GRID.items()}
    xgb_grid = XGB_GRID if not quick else {key: [values[0]] for key, values in XGB_GRID.items()}
    return [
        ModelDefinition("RF", RandomForestRegressor(random_state=seed, criterion="absolute_error", min_samples_leaf=2, n_jobs=1), rf_grid),
        ModelDefinition("XGB", xgb.XGBRegressor(random_state=seed, objective="reg:pseudohubererror", n_jobs=1), xgb_grid),
    ]


def build_coastal_routing_graph(boundary_path):
    lon_min, lon_max, lat_min, lat_max, resolution = 105.0, 131.0, 18.0, 42.0, 0.1
    longitudes = np.arange(lon_min, lon_max + resolution / 2, resolution)
    latitudes = np.arange(lat_min, lat_max + resolution / 2, resolution)
    rows, columns = len(latitudes), len(longitudes)
    boundary = gpd.read_file(boundary_path)
    geometry = boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") else boundary.geometry.unary_union
    prepared_land = prep(geometry)
    is_land = np.zeros((rows, columns), dtype=bool)
    for row, latitude in enumerate(latitudes):
        for column, longitude in enumerate(longitudes):
            is_land[row, column] = prepared_land.intersects(Point(longitude, latitude))
    graph = dok_matrix((rows * columns, rows * columns), dtype=np.float32)
    earth_radius = 6371.0088
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for row in range(rows):
        latitude_i = np.radians(latitudes[row])
        for column in range(columns):
            if is_land[row, column]:
                continue
            source = row * columns + column
            longitude_i = np.radians(longitudes[column])
            for row_offset, column_offset in neighbors:
                adjacent_row, adjacent_column = row + row_offset, column + column_offset
                if not (0 <= adjacent_row < rows and 0 <= adjacent_column < columns) or is_land[adjacent_row, adjacent_column]:
                    continue
                latitude_j = np.radians(latitudes[adjacent_row])
                longitude_j = np.radians(longitudes[adjacent_column])
                dlat, dlon = latitude_j - latitude_i, longitude_j - longitude_i
                haversine = np.sin(dlat / 2) ** 2 + np.cos(latitude_i) * np.cos(latitude_j) * np.sin(dlon / 2) ** 2
                graph[source, adjacent_row * columns + adjacent_column] = 2 * earth_radius * np.arcsin(np.sqrt(haversine))
    return graph.tocsr(), is_land, longitudes, latitudes


def nearest_sea_node(row, column, is_land):
    if not is_land[row, column]:
        return row, column
    queue, visited = deque([(row, column)]), {(row, column)}
    while queue:
        current_row, current_column = queue.popleft()
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            adjacent_row, adjacent_column = current_row + row_offset, current_column + column_offset
            if not (0 <= adjacent_row < is_land.shape[0] and 0 <= adjacent_column < is_land.shape[1]):
                continue
            if (adjacent_row, adjacent_column) in visited:
                continue
            if not is_land[adjacent_row, adjacent_column]:
                return adjacent_row, adjacent_column
            visited.add((adjacent_row, adjacent_column))
            queue.append((adjacent_row, adjacent_column))
    raise ValueError("No sea node is reachable from the specified grid cell")


def compute_diffused_features(frame, source_load, graph, is_land, longitudes, latitudes):
    result, resolution, columns = frame.copy(), 0.1, len(longitudes)
    source_rows = np.clip(np.rint((source_load["Grid_Lat"].to_numpy() - latitudes[0]) / resolution).astype(int), 0, is_land.shape[0] - 1)
    source_columns = np.clip(np.rint((source_load["Grid_Lon"].to_numpy() - longitudes[0]) / resolution).astype(int), 0, is_land.shape[1] - 1)
    loads = source_load["Load_Total_Legacy_Kg"].to_numpy(dtype=float)
    ratios = source_load[["Source_Ratio_SC_LC", "Source_Ratio_LC_PFSA_PFCA", "Source_Ratio_SC_PFSA_PFCA"]].to_numpy(dtype=float)
    routed = []
    for position, (_, row) in enumerate(result.iterrows(), 1):
        column = int(np.clip(round((float(row["Longitude"]) - longitudes[0]) / resolution), 0, is_land.shape[1] - 1))
        grid_row = int(np.clip(round((float(row["Latitude"]) - latitudes[0]) / resolution), 0, is_land.shape[0] - 1))
        sea_row, sea_column = nearest_sea_node(grid_row, column, is_land)
        distances = dijkstra(graph, directed=False, indices=sea_row * columns + sea_column)
        source_distances = distances[source_rows * columns + source_columns]
        weights = np.where(np.isfinite(source_distances), loads / (source_distances + 1.0), 0.0)
        total = float(weights.sum())
        mixed = (ratios * weights[:, None]).sum(axis=0) / total if total > 0 else np.full(3, np.nan)
        routed.append((total if total > 0 else np.nan, *mixed))
        if position % 50 == 0 or position == len(result):
            print(f"Routing covariates: {position}/{len(result)}")
    result[["Diffused_Total_Legacy_Force", "Mixed_Source_Ratio_SC_LC", "Mixed_Source_Ratio_LC_PFSA_PFCA", "Mixed_Source_Ratio_SC_PFSA_PFCA"]] = pd.DataFrame(routed, index=result.index)
    return result


def normalize_metadata(frame):
    result, _ = normalize_metadata_contract(frame)
    return result


def component_to_magnitude_balance(log_components):
    values = np.asarray(log_components, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Two log-component columns are required")
    return np.column_stack([
        (values[:, 0] + values[:, 1]) / np.sqrt(2.0),
        (values[:, 0] - values[:, 1]) / np.sqrt(2.0),
    ])


def magnitude_balance_to_components(coordinates):
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Magnitude and balance columns are required")
    return np.column_stack([
        (values[:, 0] + values[:, 1]) / np.sqrt(2.0),
        (values[:, 0] - values[:, 1]) / np.sqrt(2.0),
    ])


def derived_outputs(log_components):
    values = np.asarray(log_components, dtype=float)
    long_pfsa, long_pfca = np.exp(values[:, 0]), np.exp(values[:, 1])
    return {
        "Log_Long_PFSA": values[:, 0],
        "Log_Long_PFCA": values[:, 1],
        "Log_Long_Total": np.log(long_pfsa + long_pfca),
        "Log_Long_Ratio": values[:, 0] - values[:, 1],
        "Long_PFSA": long_pfsa,
        "Long_PFCA": long_pfca,
        "Long_Total": long_pfsa + long_pfca,
        "Long_Ratio": long_pfsa / long_pfca,
    }


def construct_long_chain_targets(frame):
    result = frame.copy()
    pfsa_columns = [column for column in LONG_PFSA if column in result.columns]
    pfca_columns = [column for column in LONG_PFCA if column in result.columns]
    if not pfsa_columns or not pfca_columns:
        raise ValueError("Long-chain PFSA and PFCA columns are required")
    result["Long_PFSA"] = result[pfsa_columns].sum(axis=1, min_count=1)
    result["Long_PFCA"] = result[pfca_columns].sum(axis=1, min_count=1)
    eligible = (
        np.isfinite(result["Long_PFSA"]) & np.isfinite(result["Long_PFCA"])
        & (result["Long_PFSA"] > 0) & (result["Long_PFCA"] > 0)
    )
    result["Endpoint_Eligible"] = eligible
    logs = np.column_stack([
        np.log(result.loc[eligible, "Long_PFSA"]),
        np.log(result.loc[eligible, "Long_PFCA"]),
    ])
    coordinates = component_to_magnitude_balance(logs)
    result.loc[eligible, "Log_Long_PFSA"] = logs[:, 0]
    result.loc[eligible, "Log_Long_PFCA"] = logs[:, 1]
    result.loc[eligible, "Long_Magnitude"] = coordinates[:, 0]
    result.loc[eligible, "Long_Balance"] = coordinates[:, 1]
    outputs = derived_outputs(logs)
    result.loc[eligible, "Log_Long_Total"] = outputs["Log_Long_Total"]
    result.loc[eligible, "Log_Long_Ratio"] = outputs["Log_Long_Ratio"]
    audit = pd.DataFrame([{
        "input_rows": len(result),
        "eligible_rows": int(eligible.sum()),
        "long_pfsa_declared": ";".join(LONG_PFSA),
        "long_pfsa_available": ";".join(pfsa_columns),
        "long_pfca_declared": ";".join(LONG_PFCA),
        "long_pfca_available": ";".join(pfca_columns),
        "missing_declared": ";".join(sorted(set(LONG_PFSA + LONG_PFCA).difference(result.columns))),
    }])
    return result, audit


def regression_metrics(observed, predicted):
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    mse = float(mean_squared_error(observed, predicted))
    rmse = float(np.sqrt(mse))
    target_range, target_mean = float(np.ptp(observed)), float(np.mean(np.abs(observed)))
    mae = float(mean_absolute_error(observed, predicted))
    metrics = {
        "R2": float(r2_score(observed, predicted)) if len(observed) > 1 and target_range > 0 else np.nan,
        "MSE": mse, "RMSE": rmse, "MAE": mae, "MedianAE": float(median_absolute_error(observed, predicted)),
        "NRMSE_range": rmse / target_range if target_range > 0 else np.nan,
        "Relative_MAE_mean": mae / target_mean if target_mean > 0 else np.nan, "n": int(len(observed)),
    }
    metrics.update({
        "ModelScale_R2": metrics["R2"], "ModelScale_RMSE": rmse, "ModelScale_MAE": mae,
    })
    return metrics


def make_splits(frame, validation, folds, seed):
    if validation == "random":
        splitter = KFold(n_splits=min(folds, len(frame)), shuffle=True, random_state=seed)
        return [(train, test, str(number)) for number, (train, test) in enumerate(splitter.split(frame), 1)]
    group_column = {
        "sampling_group": "Sampling_Group", "sampling_group_stratified": "Sampling_Group",
        "spatial_block": "Spatial_Block", "spatial_block_stratified": "Spatial_Block",
        "leave_one_sampling_group_out": "Sampling_Group", "leave_one_sea_region_out": "Sea_Region",
    }[validation]
    groups = frame[group_column].astype(str).to_numpy()
    if validation.endswith("_stratified"):
        splitter = StratifiedGroupKFold(
            n_splits=min(folds, len(np.unique(groups))), shuffle=True, random_state=seed
        )
        split_iterator = splitter.split(frame, y=frame["Sea_Region"].astype(str), groups=groups)
    else:
        splitter = LeaveOneGroupOut() if validation.startswith("leave_one") else GroupKFold(n_splits=min(folds, len(np.unique(groups))))
        split_iterator = splitter.split(frame, groups=groups)
    return [(train, test, ";".join(sorted(np.unique(groups[test])))) for train, test in split_iterator]


def make_inner_splits(frame, validation, folds, seed):
    if validation == "random":
        return list(KFold(n_splits=min(folds, len(frame)), shuffle=True, random_state=seed).split(frame))
    group_column = {
        "sampling_group": "Sampling_Group", "sampling_group_stratified": "Sampling_Group",
        "leave_one_sampling_group_out": "Sampling_Group", "spatial_block": "Spatial_Block",
        "spatial_block_stratified": "Spatial_Block", "leave_one_sea_region_out": "Sea_Region",
    }[validation]
    groups = frame[group_column].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        return list(KFold(n_splits=min(folds, len(frame)), shuffle=True, random_state=seed).split(frame))
    if validation.endswith("_stratified"):
        splitter = StratifiedGroupKFold(
            n_splits=min(folds, len(np.unique(groups))), shuffle=True, random_state=seed
        )
        return list(splitter.split(frame, y=frame["Sea_Region"].astype(str), groups=groups))
    return list(GroupKFold(n_splits=min(folds, len(np.unique(groups)))).split(frame, groups=groups))


def conformal_radius(observed, predicted, alpha=INTERVAL_ALPHA):
    residuals = np.abs(np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float))
    probability = min(1.0, np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
    return float(np.quantile(residuals, probability, method="higher"))


def tune_model(definition, X, y, inner_splits, n_jobs):
    wrapped = ScaledTargetRegressor(definition.estimator)
    grid = {f"estimator__{key}": values for key, values in definition.grid.items()}
    search = GridSearchCV(wrapped, grid, scoring=normalized_rmse_scorer, cv=inner_splits, n_jobs=n_jobs, refit=True)
    search.fit(X, y)
    inner_prediction = cross_val_predict(clone(search.best_estimator_), X, y, cv=inner_splits, n_jobs=n_jobs)
    return search, conformal_radius(y, inner_prediction)


def tree_shap_values(model, X):
    explainer = shap.TreeExplainer(model.estimator_)
    values = np.asarray(explainer.shap_values(X)) * model.target_scale_
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != X.shape:
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}; expected {X.shape}")
    return explainer, values


def evaluate_nested(analysis, target, validation, definitions, features, feature_block, outer_folds, inner_folds, seed, n_jobs, collect_shap):
    prediction_rows, fold_rows, shap_rows = [], [], []
    splits = make_splits(analysis, validation, outer_folds, seed)
    for fold_number, (train_position, test_position, fold_label) in enumerate(splits, 1):
        train, test = analysis.iloc[train_position], analysis.iloc[test_position]
        X_train, X_test = train[features], test[features]
        y_train, y_test = train[target].to_numpy(dtype=float), test[target].to_numpy(dtype=float)
        inner_splits = make_inner_splits(train, validation, inner_folds, seed + fold_number)
        fitted = []
        for definition in definitions:
            search, radius = tune_model(definition, X_train, y_train, inner_splits, n_jobs)
            model, prediction = search.best_estimator_, search.best_estimator_.predict(X_test)
            fitted.append((definition.name, search, model, prediction, radius))
        selected_name = max(fitted, key=lambda item: item[1].best_score_)[0]
        for model_name, search, model, prediction, radius in fitted:
            lower, upper = prediction - radius, prediction + radius
            selected = model_name == selected_name
            fold_rows.append({
                "target": target, "feature_block": feature_block, "validation": validation, "fold": fold_number,
                "held_out_groups": fold_label, "model": model_name, "selected_in_fold": selected,
                "train_n": len(train), "test_n": len(test), "inner_best_NRMSE": -float(search.best_score_),
                "conformal_radius": radius, "interval_coverage": float(np.mean((y_test >= lower) & (y_test <= upper))),
                "interval_mean_width": float(np.mean(upper - lower)),
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                **regression_metrics(y_test, prediction),
            })
            fold_shap = tree_shap_values(model, X_test)[1] if collect_shap else None
            for local_position, (index, observed, predicted, low, high) in enumerate(zip(test.index, y_test, prediction, lower, upper)):
                prediction_rows.append({
                    "row_index": index, "Sampling_Group": test.at[index, "Sampling_Group"], "Sea_Region": test.at[index, "Sea_Region"],
                    "SamplingYear": test.at[index, "SamplingYear"], "Longitude": test.at[index, "Longitude"],
                    "Latitude": test.at[index, "Latitude"], "target": target, "feature_block": feature_block,
                    "validation": validation, "fold": fold_number, "held_out_groups": fold_label, "model": model_name,
                    "selected_in_fold": selected, "observed": observed, "predicted": predicted,
                    "residual": observed - predicted, "prediction_interval_lower": low, "prediction_interval_upper": high,
                })
                if fold_shap is None:
                    continue
                for feature_position, feature in enumerate(features):
                    shap_rows.append({
                        "row_index": index, "target": target, "feature_block": feature_block,
                        "validation": validation, "fold": fold_number, "model": model_name,
                        "selected_in_fold": selected, "feature": feature,
                        "feature_value": X_test.iloc[local_position, feature_position],
                        "shap_value": fold_shap[local_position, feature_position],
                    })
        print(f"{target} | {feature_block} | {validation} | fold {fold_number}/{len(splits)}")
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows), pd.DataFrame(shap_rows)


def cluster_bootstrap_intervals(frame, seed, iterations):
    rng, studies, metrics = np.random.default_rng(seed), frame["Sampling_Group"].unique(), []
    for _ in range(iterations):
        sampled = rng.choice(studies, size=len(studies), replace=True)
        bootstrap = pd.concat([frame[frame["Sampling_Group"] == study] for study in sampled], ignore_index=True)
        metrics.append(regression_metrics(bootstrap["observed"], bootstrap["predicted"]))
    return pd.DataFrame(metrics)


def summarize_predictions(predictions, bootstrap_iterations, seed):
    rows, metric_names = [], ("R2", "RMSE", "MAE", "MedianAE", "NRMSE_range", "Relative_MAE_mean")
    groups = [(keys, group) for keys, group in predictions.groupby(["target", "feature_block", "validation", "model"])]
    groups += [((*keys, "NestedSelector"), group) for keys, group in predictions[predictions["selected_in_fold"]].groupby(["target", "feature_block", "validation"])]
    for keys, group in groups:
        row = dict(zip(["target", "feature_block", "validation", "model"], keys))
        row.update(regression_metrics(group["observed"], group["predicted"]))
        bootstrap = cluster_bootstrap_intervals(group, seed, bootstrap_iterations)
        interval_metrics = metric_names + ("ModelScale_R2", "ModelScale_RMSE", "ModelScale_MAE")
        for metric in interval_metrics:
            row[f"{metric}_CI_lower"] = float(bootstrap[metric].quantile(0.025))
            row[f"{metric}_CI_upper"] = float(bootstrap[metric].quantile(0.975))
        row["prediction_interval_coverage"] = float(np.mean((group["observed"] >= group["prediction_interval_lower"]) & (group["observed"] <= group["prediction_interval_upper"])))
        row["prediction_interval_mean_width"] = float(np.mean(group["prediction_interval_upper"] - group["prediction_interval_lower"]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_shap(shap_values):
    selected = shap_values[shap_values["selected_in_fold"]].copy()
    selected["model"] = "NestedSelector"
    shap_values = pd.concat([shap_values, selected], ignore_index=True)
    fold_importance = (
        shap_values.assign(absolute_shap=lambda frame: frame["shap_value"].abs())
        .groupby(["target", "feature_block", "validation", "model", "fold", "feature"], as_index=False)["absolute_shap"]
        .mean().rename(columns={"absolute_shap": "mean_absolute_shap"})
    )
    summary = fold_importance.groupby(["target", "feature_block", "validation", "model", "feature"])["mean_absolute_shap"].agg(
        mean_absolute_shap="mean", fold_sd="std", fold_median="median",
        fold_ci_lower=lambda values: values.quantile(0.025), fold_ci_upper=lambda values: values.quantile(0.975),
    ).reset_index()
    summary["importance_rank"] = summary.groupby(["target", "feature_block", "validation", "model"])["mean_absolute_shap"].rank(method="min", ascending=False)
    return fold_importance, summary


def predictor_provenance(frame):
    rows = []
    source_map = {
        "Temperature (°C)": "Temperature_Source", "Salinity (‰)": "Salinity_Source", "Chla": "Chla_Source",
        "POC (mg/m3)": "POC_Source", "pH": "pH_Source", "SPM": "SPM_Source", "DO": "DO_Source",
        "Hydro_Bottom_U": "Hydro_Source", "Hydro_Bottom_V": "Hydro_Source", "Hydro_MLD": "Hydro_Source",
    }
    for covariate in COVARIATES:
        source_column = source_map.get(covariate)
        if source_column and source_column in frame.columns:
            for source, count in frame[source_column].fillna("Unspecified").value_counts().items():
                rows.append({"covariate": covariate, "source_column": source_column, "source_class": source, "n": int(count)})
        else:
            rows.append({"covariate": covariate, "source_column": "", "source_class": "Model-derived or source column unavailable", "n": len(frame)})
    return pd.DataFrame(rows)


def feature_domain_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary = {}
    for feature in COVARIATES:
        values = frame[feature].to_numpy(dtype=float)
        summary[feature] = {
            "minimum": float(np.min(values)),
            "p02_5": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "p97_5": float(np.quantile(values, 0.975)),
            "maximum": float(np.max(values)),
        }
    return summary


def reconstruct_oof_predictions(predictions):
    selected = predictions[(predictions["selected_in_fold"]) & (predictions["feature_block"] == "all")]
    keys = ["row_index", "Sampling_Group", "Sea_Region", "SamplingYear", "Longitude", "Latitude", "validation", "fold", "held_out_groups"]
    observed = selected.pivot(index=keys, columns="target", values="observed").reset_index()
    predicted = selected.pivot(index=keys, columns="target", values="predicted").reset_index()
    merged = observed.merge(predicted, on=keys, suffixes=("_observed", "_predicted"), validate="one_to_one")
    for bound in ("prediction_interval_lower", "prediction_interval_upper"):
        values = selected.pivot(index=keys, columns="target", values=bound).reset_index()
        merged = merged.merge(values.rename(columns={target: f"{target}_{bound}" for target in PRIMARY_TARGETS}), on=keys, validate="one_to_one")
    for state in ("observed", "predicted"):
        coordinates = merged[[f"Long_Magnitude_{state}", f"Long_Balance_{state}"]].to_numpy()
        components = magnitude_balance_to_components(coordinates)
        outputs = derived_outputs(components)
        for name, values in outputs.items():
            merged[f"{name}_{state}"] = values
    root_two = np.sqrt(2.0)
    magnitude_lower = merged["Long_Magnitude_prediction_interval_lower"]
    magnitude_upper = merged["Long_Magnitude_prediction_interval_upper"]
    balance_lower = merged["Long_Balance_prediction_interval_lower"]
    balance_upper = merged["Long_Balance_prediction_interval_upper"]
    merged["Log_Long_PFSA_interval_lower"] = (magnitude_lower + balance_lower) / root_two
    merged["Log_Long_PFSA_interval_upper"] = (magnitude_upper + balance_upper) / root_two
    merged["Log_Long_PFCA_interval_lower"] = (magnitude_lower - balance_upper) / root_two
    merged["Log_Long_PFCA_interval_upper"] = (magnitude_upper - balance_lower) / root_two
    merged["Log_Long_Ratio_interval_lower"] = root_two * balance_lower
    merged["Log_Long_Ratio_interval_upper"] = root_two * balance_upper
    merged["Log_Long_Total_interval_lower"] = np.log(
        np.exp(merged["Log_Long_PFSA_interval_lower"]) + np.exp(merged["Log_Long_PFCA_interval_lower"])
    )
    merged["Log_Long_Total_interval_upper"] = np.log(
        np.exp(merged["Log_Long_PFSA_interval_upper"]) + np.exp(merged["Log_Long_PFCA_interval_upper"])
    )
    return merged


def summarize_reconstructed_predictions(reconstructed, bootstrap_iterations, seed):
    rows = []
    outputs = ("Log_Long_PFSA", "Log_Long_PFCA", "Log_Long_Total", "Log_Long_Ratio")
    for (validation,), group in reconstructed.groupby(["validation"]):
        for output in outputs:
            evaluation = group.rename(columns={f"{output}_observed": "observed", f"{output}_predicted": "predicted"})
            row = {"validation": validation, "output": output, **regression_metrics(evaluation["observed"], evaluation["predicted"])}
            bootstrap = cluster_bootstrap_intervals(evaluation, seed, bootstrap_iterations)
            for metric in ("R2", "RMSE", "MAE"):
                row[f"{metric}_CI_lower"] = float(bootstrap[metric].quantile(0.025))
                row[f"{metric}_CI_upper"] = float(bootstrap[metric].quantile(0.975))
            row["prediction_interval_coverage"] = float(np.mean(
                (group[f"{output}_observed"] >= group[f"{output}_interval_lower"])
                & (group[f"{output}_observed"] <= group[f"{output}_interval_upper"])
            ))
            row["prediction_interval_mean_width"] = float(np.mean(
                group[f"{output}_interval_upper"] - group[f"{output}_interval_lower"]
            ))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_feature_ablation(performance):
    selected = performance[performance["model"] == "NestedSelector"].copy()
    reference = selected[selected["feature_block"] == "all"][["target", "validation", "R2", "RMSE"]].rename(
        columns={"R2": "All_R2", "RMSE": "All_RMSE"}
    )
    comparison = selected.merge(reference, on=["target", "validation"], validate="many_to_one")
    comparison["Delta_R2_vs_all"] = comparison["R2"] - comparison["All_R2"]
    comparison["Delta_RMSE_vs_all"] = comparison["RMSE"] - comparison["All_RMSE"]
    return comparison


def predict_model_package(package, X):
    if package.get("schema_version") != MODEL_PACKAGE_SCHEMA:
        raise ValueError("Unsupported model package schema")
    features = package["feature_names"]
    matrix = X[features]
    coordinates = np.column_stack([
        np.asarray(package["models"][target]["model_object"].predict(matrix), dtype=float)
        * package["models"][target]["target_scale"]
        + package["models"][target]["target_mean"]
        for target in PRIMARY_TARGETS
    ])
    components = magnitude_balance_to_components(coordinates)
    outputs = derived_outputs(components)
    return pd.DataFrame({
        "Long_Magnitude": coordinates[:, 0],
        "Long_Balance": coordinates[:, 1],
        **outputs,
    }, index=X.index)


def fit_final_model_bundle(
    frame, performance, definitions, output_dir, primary_validation,
    inner_folds, seed, n_jobs, shap_interactions, input_provenance,
):
    primary = performance[
        (performance["validation"] == primary_validation)
        & (performance["feature_block"] == "all")
    ]
    definitions_by_name = {definition.name: definition for definition in definitions}
    analysis = frame.dropna(subset=list(PRIMARY_TARGETS) + COVARIATES).copy()
    models, champion_rows = {}, []
    for target in PRIMARY_TARGETS:
        comparison = primary[
            (primary["target"] == target) & primary["model"].isin(["RF", "XGB"])
        ].sort_values("RMSE")
        if comparison.empty:
            raise ValueError(f"{target}: no primary-validation performance is available for final model selection")
        architecture = comparison.iloc[0]["model"]
        nested = primary[(primary["target"] == target) & (primary["model"] == "NestedSelector")]
        if len(nested) != 1:
            raise ValueError(f"{target}: one nested-selector performance row is required")
        nested = nested.iloc[0]
        inner_splits = make_inner_splits(analysis, primary_validation, inner_folds, seed)
        search, radius = tune_model(definitions_by_name[architecture], analysis[COVARIATES], analysis[target].to_numpy(dtype=float), inner_splits, n_jobs)
        model, X = search.best_estimator_, analysis[COVARIATES]
        explainer, shap_values = tree_shap_values(model, X)
        interaction_values = np.asarray(explainer.shap_interaction_values(X)) * model.target_scale_ if shap_interactions else None
        selected = comparison.iloc[0]
        models[target] = {
            "target_role": TARGET_ROLES[target],
            "model_object": model.estimator_, "model_architecture": architecture,
            "target_preprocessing": "training-fold standardization",
            "target_mean": model.target_mean_, "target_scale": model.target_scale_,
            "shap_values": shap_values, "shap_interaction_values": interaction_values,
            "expected_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]) * model.target_scale_ + model.target_mean_,
            "prediction_interval_radius_95": radius,
            "best_parameters": search.best_params_,
            "selection_metrics": selected.to_dict(),
            "nested_selection_metrics": nested.to_dict(),
        }
        champion_rows.append({
            "Target": target, "Target_role": TARGET_ROLES[target],
            "Architecture": architecture, "Selection_validation": primary_validation,
            "Final_training_n": len(analysis), "Conformal_radius_95": radius,
            "Best_parameters": json.dumps(search.best_params_, sort_keys=True),
            **{f"Candidate_{key}": value for key, value in selected[["R2", "R2_CI_lower", "R2_CI_upper", "RMSE", "RMSE_CI_lower", "RMSE_CI_upper", "MAE", "MedianAE", "NRMSE_range", "Relative_MAE_mean"]].to_dict().items()},
            **{f"Nested_{key}": value for key, value in nested[["R2", "R2_CI_lower", "R2_CI_upper", "RMSE", "RMSE_CI_lower", "RMSE_CI_upper", "MAE", "MedianAE", "NRMSE_range", "Relative_MAE_mean", "prediction_interval_coverage", "prediction_interval_mean_width"]].to_dict().items()},
        })
    package = {
        "schema_version": MODEL_PACKAGE_SCHEMA,
        "training_code_version": VERSION,
        "training_code_sha256": sha256(Path(__file__).resolve()),
        "coordinate_system": {
            "component_order": ["Log_Long_PFSA", "Log_Long_PFCA"],
            "coordinate_order": list(PRIMARY_TARGETS),
            "forward_matrix": [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
            "inverse_matrix": [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
            "long_pfsa_congeners": list(LONG_PFSA), "long_pfca_congeners": list(LONG_PFCA),
        },
        "feature_names": COVARIATES, "feature_domain": feature_domain_summary(analysis[COVARIATES]),
        "models": models, "X_data": analysis[COVARIATES],
        "Coordinates": analysis[["Longitude", "Latitude"]],
        "Metadata": analysis[["Sampling_Group", "Sea_Region", "SamplingYear", "Spatial_Block"]],
        "training_row_indices": analysis.index.to_list(), "training_input_provenance": input_provenance,
        "validation_scheme_for_selection": primary_validation,
    }
    package_path = output_dir / "Long_Chain_Magnitude_Balance_Model_Package.pkl"
    joblib.dump(package, package_path, compress=3)
    champions = pd.DataFrame(champion_rows)
    champions["Model_package_file"] = package_path.name
    champions["Model_package_sha256"] = sha256(package_path)
    champions["Model_package_schema"] = MODEL_PACKAGE_SCHEMA
    return champions, package_path


def load_features(data_path, source_path, boundary_path, feature_table):
    if feature_table is not None:
        frame = pd.read_csv(feature_table) if feature_table.suffix.lower() == ".csv" else pd.read_excel(feature_table)
        return normalize_metadata(frame)
    frame = normalize_metadata(pd.read_excel(data_path))
    source_load = pd.read_csv(source_path)
    graph, is_land, longitudes, latitudes = build_coastal_routing_graph(boundary_path)
    return compute_diffused_features(frame, source_load, graph, is_land, longitudes, latitudes)


def training_input_provenance(arguments) -> dict[str, dict[str, str]]:
    if arguments.feature_table is not None:
        paths = {
            "feature_table": arguments.feature_table.resolve(),
            "sea_region_reference": DEFAULT_SEA_REFERENCE.resolve(),
        }
    else:
        boundary = arguments.boundary.resolve()
        sidecars = sorted(boundary.parent.glob(f"{boundary.stem}.*"))
        paths = {
            "data": arguments.data.resolve(),
            "source_load": arguments.source_load.resolve(),
            "sea_region_reference": DEFAULT_SEA_REFERENCE.resolve(),
            **{f"boundary_{path.suffix.lower().lstrip('.')}": path.resolve() for path in sidecars},
        }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Training inputs are missing: {missing}")
    return {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }


def run_pipeline(arguments):
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = training_input_provenance(arguments)
    frame = load_features(arguments.data, arguments.source_load, arguments.boundary, arguments.feature_table)
    frame, sea_assignment_audit = normalize_metadata_contract(frame)
    frame, target_audit = construct_long_chain_targets(frame)
    frame.to_csv(output_dir / "Long_Chain_Model_Feature_Table.csv", index=False, encoding="utf-8-sig")
    sea_assignment_audit.to_csv(output_dir / "Sea_Region_Assignment_Audit.csv", index=False, encoding="utf-8-sig")
    target_audit.to_csv(output_dir / "Long_Chain_Target_Audit.csv", index=False, encoding="utf-8-sig")
    predictor_provenance(frame).to_csv(output_dir / "Predictor_Provenance.csv", index=False, encoding="utf-8-sig")
    definitions = model_definitions(arguments.seed, arguments.quick)
    validations = ["sampling_group"] if arguments.quick else arguments.validations
    feature_blocks = ["all"] if arguments.quick else arguments.feature_blocks
    analysis = frame.dropna(subset=list(PRIMARY_TARGETS) + COVARIATES).copy()
    if arguments.exclude_sea_boundary_qa:
        analysis = analysis.loc[~analysis["Sea_Boundary_QA_Flag"]].copy()
    prediction_parts, fold_parts, shap_parts = [], [], []
    for target in PRIMARY_TARGETS:
        for feature_block in feature_blocks:
            features = FEATURE_BLOCKS[feature_block]
            for validation in validations:
                predictions, folds, shap_values = evaluate_nested(
                    analysis, target, validation, definitions, features, feature_block,
                    arguments.outer_folds, arguments.inner_folds, arguments.seed, arguments.n_jobs,
                    collect_shap=feature_block == "all",
                )
                prediction_parts.append(predictions)
                fold_parts.append(folds)
                if not shap_values.empty:
                    shap_parts.append(shap_values)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    folds = pd.concat(fold_parts, ignore_index=True)
    shap_values = pd.concat(shap_parts, ignore_index=True) if shap_parts else pd.DataFrame()
    performance = summarize_predictions(predictions, arguments.bootstrap_iterations, arguments.seed)
    ablation = summarize_feature_ablation(performance)
    reconstructed = reconstruct_oof_predictions(predictions)
    reconstructed_performance = summarize_reconstructed_predictions(reconstructed, arguments.bootstrap_iterations, arguments.seed)
    predictions.to_csv(output_dir / "Nested_OOF_Predictions.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output_dir / "Nested_Performance_Per_Fold.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(output_dir / "Nested_Performance_Summary.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output_dir / "Feature_Block_Ablation.csv", index=False, encoding="utf-8-sig")
    reconstructed.to_csv(output_dir / "Nested_OOF_Reconstructed_Outputs.csv", index=False, encoding="utf-8-sig")
    reconstructed_performance.to_csv(output_dir / "Nested_Reconstructed_Performance.csv", index=False, encoding="utf-8-sig")
    if not shap_values.empty:
        shap_folds, shap_summary = summarize_shap(shap_values)
        shap_values.to_csv(output_dir / "Nested_OOF_SHAP_Values.csv", index=False, encoding="utf-8-sig")
        shap_folds.to_csv(output_dir / "SHAP_Importance_Per_Fold.csv", index=False, encoding="utf-8-sig")
        shap_summary.to_csv(output_dir / "SHAP_Importance_Stability.csv", index=False, encoding="utf-8-sig")
    final_validation = "sampling_group" if arguments.quick else arguments.primary_validation
    if not arguments.skip_final_models:
        champions, _ = fit_final_model_bundle(
            frame, performance, definitions, output_dir,
            final_validation, arguments.inner_folds, arguments.seed, arguments.n_jobs,
            arguments.shap_interactions, provenance,
        )
        champions.to_csv(output_dir / f"Long_Chain_Model_Report_{VERSION}.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "version": VERSION, "model_package_schema": MODEL_PACKAGE_SCHEMA,
        "training_code_sha256": sha256(Path(__file__).resolve()),
        "primary_targets": PRIMARY_TARGETS, "target_roles": TARGET_ROLES,
        "data_path": str(arguments.data.resolve()),
        "feature_table": str(arguments.feature_table.resolve()) if arguments.feature_table else None,
        "training_input_provenance": provenance,
        "rows": len(frame), "eligible_rows": len(analysis), "covariates": COVARIATES,
        "feature_blocks": feature_blocks, "validations": validations,
        "primary_validation": final_validation, "outer_folds": arguments.outer_folds,
        "inner_folds": arguments.inner_folds, "spatial_block_degrees": SPATIAL_BLOCK_DEGREES,
        "interval_alpha": INTERVAL_ALPHA,
        "seed": arguments.seed, "bootstrap_iterations": arguments.bootstrap_iterations,
        "shap_interactions": arguments.shap_interactions,
        "skip_final_models": arguments.skip_final_models,
        "exclude_sea_boundary_qa": arguments.exclude_sea_boundary_qa,
        "quick": arguments.quick,
    }
    (output_dir / "Run_Manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Outputs written to {output_dir}")


def parse_arguments():
    project_root = Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
    data_root = Path(os.environ.get("PFAS_DATA_ROOT", project_root / "data")).expanduser().resolve()
    output_root = Path(os.environ.get("PFAS_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()
    fixed_source = data_root / "fixed_inputs" / "source_boundary"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=data_root / "raw" / "DATA.xlsx")
    parser.add_argument("--source-load", type=Path, default=fixed_source / "Step4_Coastline_Source_Load_2018.csv")
    parser.add_argument("--boundary", type=Path, default=data_root / "reference" / "geospatial" / "china_coastline_boundary.shp")
    parser.add_argument("--feature-table", type=Path, default=data_root / "fixed_inputs" / "model" / "Long_Chain_Model_Feature_Table.csv")
    parser.add_argument("--output-dir", type=Path, default=output_root / "01_primary_model")
    parser.add_argument("--validations", nargs="+", choices=VALIDATION_SCHEMES, default=list(DEFAULT_VALIDATIONS))
    parser.add_argument("--feature-blocks", nargs="+", choices=tuple(FEATURE_BLOCKS), default=list(FEATURE_BLOCKS))
    parser.add_argument("--primary-validation", choices=VALIDATION_SCHEMES, default="sampling_group")
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-shap-interactions", dest="shap_interactions", action="store_false")
    parser.add_argument("--skip-final-models", action="store_true")
    parser.add_argument(
        "--exclude-sea-boundary-qa",
        action="store_true",
        help="Sensitivity run excluding observations within 15 km of a competing canonical sea region.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.set_defaults(shap_interactions=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    run_pipeline(parse_arguments())
