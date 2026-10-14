from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadata_schema import normalize_metadata as normalize_metadata_contract
from pfas_schema import (
    COVARIATES,
    MODEL_GROUPS,
    normalize_feature_columns,
    schema_digest,
    sum_group,
)

VERSION = "v1.1"
RANDOM_SEED = 42
SPATIAL_BLOCK_DEGREES = 2.0

GROUP_DEFINITIONS = {
    name: MODEL_GROUPS[name]
    for name in ("Short_PFSA", "Short_PFCA", "Long_PFSA", "Long_PFCA")
}

GROUP_COLUMNS = tuple(GROUP_DEFINITIONS)
ILR_COLUMNS = ("ILR_Chain_Length", "ILR_Short_PFSA_PFCA", "ILR_Long_PFSA_PFCA")
LOG_GDR_COLUMNS = (
    "Log_GDR_Short_Long",
    "Log_GDR_Short_PFSA_PFCA",
    "Log_GDR_Long_PFSA_PFCA",
)
GDR_COLUMNS = ("GDR_Short_Long", "GDR_Short_PFSA_PFCA", "GDR_Long_PFSA_PFCA")

ILR_BASIS = np.asarray(
    [
        [0.5, 0.5, -0.5, -0.5],
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0, 0.0],
        [0.0, 0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
    ]
)

VALIDATIONS = (
    "random",
    "sampling_group",
    "sampling_group_stratified",
    "spatial_block",
    "spatial_block_stratified",
    "leave_one_sampling_group_out",
    "leave_one_sea_region_out",
)
REPRESENTATIONS = ("separate_log_gdr", "independent_ilr", "joint_ilr")


class ScaledTargetRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        values = np.asarray(y, dtype=float)
        self.target_mean_ = np.mean(values, axis=0)
        self.target_scale_ = np.std(values, axis=0)
        self.target_scale_ = np.where(self.target_scale_ > 0, self.target_scale_, 1.0)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, (values - self.target_mean_) / self.target_scale_)
        return self

    def predict(self, X):
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        return prediction * self.target_scale_ + self.target_mean_


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    estimator: object
    grid: dict[str, list]


def model_definitions(seed, quick, joint):
    if quick:
        rf_grid = {
            "n_estimators": [50],
            "max_depth": [5],
            "max_features": [0.8],
            "max_samples": [0.8],
        }
        xgb_grid = {
            "n_estimators": [50],
            "max_depth": [3],
            "learning_rate": [0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }
    else:
        rf_grid = {
            "n_estimators": [100],
            "max_depth": [3, 5],
            "max_features": [0.8],
            "max_samples": [0.8],
        }
        xgb_grid = {
            "n_estimators": [100],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }
    rf = RandomForestRegressor(random_state=seed, criterion="absolute_error", n_jobs=1)
    xgb_base = xgb.XGBRegressor(
        random_state=seed, objective="reg:pseudohubererror", n_jobs=1
    )
    if joint:
        return [
            ModelDefinition(
                "RF",
                rf,
                {f"estimator__{key}": values for key, values in rf_grid.items()},
            ),
            ModelDefinition(
                "XGB",
                MultiOutputRegressor(xgb_base, n_jobs=1),
                {
                    f"estimator__estimator__{key}": values
                    for key, values in xgb_grid.items()
                },
            ),
        ]
    return [
        ModelDefinition(
            "RF", rf, {f"estimator__{key}": values for key, values in rf_grid.items()}
        ),
        ModelDefinition(
            "XGB",
            xgb_base,
            {f"estimator__{key}": values for key, values in xgb_grid.items()},
        ),
    ]


def normalize_metadata(frame):
    result, _ = normalize_metadata_contract(frame)
    return result


def load_analysis_data(data_path, feature_table_path):
    manifest_path = feature_table_path.parent / "Feature_Table_Manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing upstream feature-table provenance: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "feature_table_sha256": hashlib.sha256(
            feature_table_path.read_bytes()
        ).hexdigest(),
        "pfas_schema_sha256": schema_digest(),
    }
    if any(manifest.get(key) != value for key, value in checks.items()):
        raise ValueError(
            "Raw data, prepared feature table or model schema do not match upstream provenance"
        )
    data = normalize_metadata(normalize_feature_columns(pd.read_excel(data_path)))
    routed = normalize_metadata(
        normalize_feature_columns(pd.read_csv(feature_table_path))
    )
    validate_input_alignment(data, routed)
    routing_columns = [column for column in COVARIATES if column not in data.columns]
    missing_routing = sorted(set(routing_columns).difference(routed.columns))
    if missing_routing:
        raise ValueError(f"Missing routed covariates: {missing_routing}")
    for column in routing_columns:
        data[column] = pd.to_numeric(routed[column], errors="raise").to_numpy()
    return data


def validate_input_alignment(data, routed):
    """Reject stale or reordered exports before assigning any routed predictor."""
    if len(data) != len(routed):
        raise ValueError(
            "DATA.xlsx and the routed feature table have different row counts"
        )
    for column, start in (("row_index", 0), ("excel_row", 2)):
        if column in routed:
            identifiers = pd.to_numeric(routed[column], errors="raise").to_numpy()
            if not np.array_equal(identifiers, np.arange(start, start + len(data))):
                raise ValueError(
                    f"Routed {column} does not match raw workbook row order"
                )
    required = ["Sampling_Group_ID", "Longitude", "Latitude"]
    if any(column not in data or column not in routed for column in required):
        raise ValueError(
            "Both inputs must contain canonical group and coordinate identities"
        )
    compounds = [column for group in MODEL_GROUPS.values() for column in group]
    columns = required + [
        column
        for column in compounds + COVARIATES
        if column in data and column in routed
    ]
    for column in columns:
        left = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(routed[column], errors="raise").to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(
                f"Raw data and feature export differ in {column}; regenerate upstream outputs"
            )
    for column in ("Samplingtime", "Sample_ID", "Station", "Station_ID"):
        if column in data and column in routed:
            left = data[column].fillna("").astype(str).str.strip().to_numpy()
            right = routed[column].fillna("").astype(str).str.strip().to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(
                    f"Row identities differ in {column}; regenerate upstream outputs"
                )


def construct_composition(frame):
    result = frame.copy()
    audit_rows = []
    for group, declared_columns in GROUP_DEFINITIONS.items():
        columns = [column for column in declared_columns if column in result.columns]
        if not columns:
            raise ValueError(f"{group}: no measured congeners")
        observed_count = result[columns].notna().sum(axis=1)
        total = sum_group(result, group)
        result[group] = total
        audit_rows.append(
            pd.DataFrame(
                {
                    "row_index": result.index,
                    "Sampling_Group": result["Sampling_Group"],
                    "group": group,
                    "declared_congener_count": len(declared_columns),
                    "available_congener_count": len(columns),
                    "observed_congener_count": observed_count,
                    "positive_group_total": total > 0,
                    "missing_declared_columns": ";".join(
                        column
                        for column in declared_columns
                        if column not in result.columns
                    ),
                }
            )
        )
    group_values = result[list(GROUP_COLUMNS)].to_numpy(dtype=float)
    eligible = np.isfinite(group_values).all(axis=1) & (group_values > 0).all(axis=1)
    composition = np.full_like(group_values, np.nan, dtype=float)
    composition[eligible] = group_values[eligible] / group_values[eligible].sum(
        axis=1, keepdims=True
    )
    result.loc[:, [f"Composition_{column}" for column in GROUP_COLUMNS]] = composition
    if eligible.any():
        result.loc[eligible, list(ILR_COLUMNS)] = ilr_transform(composition[eligible])
        gdr = composition_to_gdr(composition[eligible])
        result.loc[eligible, list(GDR_COLUMNS)] = gdr
        result.loc[eligible, list(LOG_GDR_COLUMNS)] = np.log(gdr)
    result["Composition_Eligible"] = eligible
    return result, pd.concat(audit_rows, ignore_index=True)


def ilr_transform(composition):
    values = np.asarray(composition, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("ILR transformation requires a positive four-part composition")
    closed = values / values.sum(axis=1, keepdims=True)
    return np.log(closed) @ ILR_BASIS.T


def ilr_inverse(coordinates):
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("ILR inverse requires three coordinates")
    clr = values @ ILR_BASIS
    clr -= np.max(clr, axis=1, keepdims=True)
    composition = np.exp(clr)
    return composition / composition.sum(axis=1, keepdims=True)


def composition_to_gdr(composition):
    values = np.asarray(composition, dtype=float)
    short_pfsa, short_pfca, long_pfsa, long_pfca = values.T
    return np.column_stack(
        [
            (short_pfsa + short_pfca) / (long_pfsa + long_pfca),
            short_pfsa / short_pfca,
            long_pfsa / long_pfca,
        ]
    )


def log_gdr_to_composition(log_gdr):
    ratios = np.exp(np.asarray(log_gdr, dtype=float))
    short_long, short_headgroup, long_headgroup = ratios.T
    short_total = short_long / (1.0 + short_long)
    long_total = 1.0 / (1.0 + short_long)
    short_pfsa = short_total * short_headgroup / (1.0 + short_headgroup)
    short_pfca = short_total / (1.0 + short_headgroup)
    long_pfsa = long_total * long_headgroup / (1.0 + long_headgroup)
    long_pfca = long_total / (1.0 + long_headgroup)
    return np.column_stack([short_pfsa, short_pfca, long_pfsa, long_pfca])


def make_splits(frame, validation, folds, seed):
    if validation == "random":
        splitter = KFold(
            n_splits=min(folds, len(frame)), shuffle=True, random_state=seed
        )
        return [
            (train, test, str(number))
            for number, (train, test) in enumerate(splitter.split(frame), 1)
        ]
    group_column = {
        "sampling_group": "Sampling_Group",
        "sampling_group_stratified": "Sampling_Group",
        "spatial_block": "Spatial_Block",
        "spatial_block_stratified": "Spatial_Block",
        "leave_one_sampling_group_out": "Sampling_Group",
        "leave_one_sea_region_out": "Sea_Region",
    }[validation]
    groups = frame[group_column].astype(str).to_numpy()
    if validation.endswith("_stratified"):
        splitter = StratifiedGroupKFold(
            n_splits=min(folds, len(np.unique(groups))), shuffle=True, random_state=seed
        )
        iterator = splitter.split(
            frame, y=frame["Sea_Region"].astype(str), groups=groups
        )
    else:
        splitter = (
            LeaveOneGroupOut()
            if validation.startswith("leave_one")
            else GroupKFold(n_splits=min(folds, len(np.unique(groups))))
        )
        iterator = splitter.split(frame, groups=groups)
    return [
        (train, test, ";".join(sorted(np.unique(groups[test]))))
        for train, test in iterator
    ]


def make_inner_splits(frame, validation, folds, seed):
    if validation == "random":
        return list(
            KFold(
                n_splits=min(folds, len(frame)), shuffle=True, random_state=seed
            ).split(frame)
        )
    group_column = {
        "sampling_group": "Sampling_Group",
        "sampling_group_stratified": "Sampling_Group",
        "spatial_block": "Spatial_Block",
        "spatial_block_stratified": "Spatial_Block",
        "leave_one_sampling_group_out": "Sampling_Group",
        "leave_one_sea_region_out": "Sea_Region",
    }[validation]
    groups = frame[group_column].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError(
            f"{validation} requires at least two inner groups; random-fold fallback is not permitted"
        )
    if validation.endswith("_stratified"):
        splitter = StratifiedGroupKFold(
            n_splits=min(folds, len(np.unique(groups))), shuffle=True, random_state=seed
        )
        return list(
            splitter.split(frame, y=frame["Sea_Region"].astype(str), groups=groups)
        )
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    return list(splitter.split(frame, groups=groups))


def tune_independent(definition, X, targets, inner_splits, n_jobs):
    predictions = []
    searches = []
    for position in range(targets.shape[1]):
        search = GridSearchCV(
            ScaledTargetRegressor(definition.estimator),
            definition.grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_splits,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X, targets[:, position])
        predictions.append(search)
        searches.append(search)
    return searches


def tune_joint(definition, X, targets, inner_splits, n_jobs):
    search = GridSearchCV(
        ScaledTargetRegressor(definition.estimator),
        definition.grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_splits,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, targets)
    return search


def aitchison_distance(observed, predicted):
    observed_clr = np.log(observed) - np.log(observed).mean(axis=1, keepdims=True)
    predicted_clr = np.log(predicted) - np.log(predicted).mean(axis=1, keepdims=True)
    return np.sqrt(np.sum((observed_clr - predicted_clr) ** 2, axis=1))


def evaluate_nested(
    analysis, validation, representations, outer_folds, inner_folds, seed, n_jobs, quick
):
    prediction_rows = []
    splits = make_splits(analysis, validation, outer_folds, seed)
    observed_composition_all = analysis[
        [f"Composition_{column}" for column in GROUP_COLUMNS]
    ].to_numpy(dtype=float)
    for fold_number, (train_position, test_position, held_out) in enumerate(splits, 1):
        train, test = analysis.iloc[train_position], analysis.iloc[test_position]
        X_train, X_test = train[COVARIATES], test[COVARIATES]
        observed_composition = observed_composition_all[test_position]
        inner_splits = make_inner_splits(
            train, validation, inner_folds, seed + fold_number
        )
        for representation in representations:
            target_columns = (
                LOG_GDR_COLUMNS if representation == "separate_log_gdr" else ILR_COLUMNS
            )
            y_train = train[list(target_columns)].to_numpy(dtype=float)
            model_results = []
            for definition in model_definitions(
                seed, quick, representation == "joint_ilr"
            ):
                if representation == "joint_ilr":
                    search = tune_joint(
                        definition, X_train, y_train, inner_splits, n_jobs
                    )
                    predicted_coordinates = search.predict(X_test)
                    inner_score = float(search.best_score_)
                    parameters = search.best_params_
                else:
                    searches = tune_independent(
                        definition, X_train, y_train, inner_splits, n_jobs
                    )
                    predicted_coordinates = np.column_stack(
                        [search.predict(X_test) for search in searches]
                    )
                    inner_score = float(
                        np.mean([search.best_score_ for search in searches])
                    )
                    parameters = [search.best_params_ for search in searches]
                predicted_composition = (
                    log_gdr_to_composition(predicted_coordinates)
                    if representation == "separate_log_gdr"
                    else ilr_inverse(predicted_coordinates)
                )
                model_results.append(
                    (
                        definition.name,
                        inner_score,
                        parameters,
                        predicted_coordinates,
                        predicted_composition,
                    )
                )
            selected_model = max(model_results, key=lambda item: item[1])[0]
            for (
                model_name,
                inner_score,
                parameters,
                predicted_coordinates,
                predicted_composition,
            ) in model_results:
                predicted_gdr = composition_to_gdr(predicted_composition)
                distances = aitchison_distance(
                    observed_composition, predicted_composition
                )
                observed_coordinates = test[list(target_columns)].to_numpy(dtype=float)
                for local_position, index in enumerate(test.index):
                    row = {
                        "row_index": index,
                        "Sampling_Group": test.at[index, "Sampling_Group"],
                        "Sea_Region": test.at[index, "Sea_Region"],
                        "SamplingYear": test.at[index, "SamplingYear"],
                        "Longitude": test.at[index, "Longitude"],
                        "Latitude": test.at[index, "Latitude"],
                        "validation": validation,
                        "fold": fold_number,
                        "held_out_groups": held_out,
                        "representation": representation,
                        "model": model_name,
                        "selected_in_fold": model_name == selected_model,
                        "inner_mean_neg_RMSE": inner_score,
                        "best_parameters": json.dumps(parameters, sort_keys=True),
                        "Aitchison_Distance": distances[local_position],
                    }
                    for position, column in enumerate(target_columns):
                        row[f"Observed_Coordinate_{column}"] = observed_coordinates[
                            local_position, position
                        ]
                        row[f"Predicted_Coordinate_{column}"] = predicted_coordinates[
                            local_position, position
                        ]
                    for position, column in enumerate(GROUP_COLUMNS):
                        row[f"Observed_Composition_{column}"] = observed_composition[
                            local_position, position
                        ]
                        row[f"Predicted_Composition_{column}"] = predicted_composition[
                            local_position, position
                        ]
                    for position, column in enumerate(GDR_COLUMNS):
                        row[f"Observed_{column}"] = composition_to_gdr(
                            observed_composition[[local_position]]
                        )[0, position]
                        row[f"Predicted_{column}"] = predicted_gdr[
                            local_position, position
                        ]
                    prediction_rows.append(row)
            print(f"{validation} | {representation} | fold {fold_number}/{len(splits)}")
    return pd.DataFrame(prediction_rows)


def metric_row(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = float(mean_squared_error(observed, predicted))
    return {
        "R2": float(r2_score(observed, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(observed, predicted)),
        "n": len(observed),
    }


def summarize_metric_group(base, group):
    rows = []
    rows.append(
        {
            **base,
            "metric_space": "composition",
            "target": "Aitchison_Distance",
            "R2": np.nan,
            "MSE": float(np.mean(group["Aitchison_Distance"] ** 2)),
            "RMSE": float(np.sqrt(np.mean(group["Aitchison_Distance"] ** 2))),
            "MAE": float(np.mean(group["Aitchison_Distance"])),
            "n": len(group),
        }
    )
    for target in GDR_COLUMNS:
        observed = group[f"Observed_{target}"].to_numpy(dtype=float)
        predicted = group[f"Predicted_{target}"].to_numpy(dtype=float)
        rows.append(
            {
                **base,
                "metric_space": "original_gdr",
                "target": target,
                **metric_row(observed, predicted),
            }
        )
        rows.append(
            {
                **base,
                "metric_space": "log_gdr",
                "target": target,
                **metric_row(np.log(observed), np.log(predicted)),
            }
        )
    for component in GROUP_COLUMNS:
        observed = group[f"Observed_Composition_{component}"].to_numpy(dtype=float)
        predicted = group[f"Predicted_Composition_{component}"].to_numpy(dtype=float)
        rows.append(
            {
                **base,
                "metric_space": "composition",
                "target": component,
                **metric_row(observed, predicted),
            }
        )
    coordinate_columns = (
        LOG_GDR_COLUMNS if base["representation"] == "separate_log_gdr" else ILR_COLUMNS
    )
    for coordinate in coordinate_columns:
        observed = group[f"Observed_Coordinate_{coordinate}"].to_numpy(dtype=float)
        predicted = group[f"Predicted_Coordinate_{coordinate}"].to_numpy(dtype=float)
        rows.append(
            {
                **base,
                "metric_space": "model_coordinate",
                "target": coordinate,
                **metric_row(observed, predicted),
            }
        )
    return rows


def summarize_predictions(predictions):
    selected = predictions[predictions["selected_in_fold"]].copy()
    selected["model"] = "NestedSelector"
    combined = pd.concat([predictions, selected], ignore_index=True)
    rows = []
    for keys, group in combined.groupby(["validation", "representation", "model"]):
        base = dict(zip(["validation", "representation", "model"], keys))
        rows.extend(summarize_metric_group(base, group))
    return pd.DataFrame(rows)


def summarize_folds(predictions):
    selected = predictions[predictions["selected_in_fold"]].copy()
    selected["model"] = "NestedSelector"
    combined = pd.concat([predictions, selected], ignore_index=True)
    rows = []
    columns = ["validation", "representation", "model", "fold"]
    for keys, fold in combined.groupby(columns):
        base = dict(zip(columns, keys))
        base["held_out_groups"] = fold["held_out_groups"].iloc[0]
        rows.extend(summarize_metric_group(base, fold))
    return pd.DataFrame(rows)


def comparison_metrics(frame):
    metrics = {
        "Aitchison_RMSE": float(np.sqrt(np.mean(frame["Aitchison_Distance"] ** 2)))
    }
    for target in GDR_COLUMNS:
        observed = frame[f"Observed_{target}"].to_numpy(dtype=float)
        predicted = frame[f"Predicted_{target}"].to_numpy(dtype=float)
        metrics[f"Original_R2__{target}"] = float(r2_score(observed, predicted))
        metrics[f"Log_R2__{target}"] = float(
            r2_score(np.log(observed), np.log(predicted))
        )
    return metrics


def paired_cluster_bootstrap(predictions, iterations, seed):
    selected = predictions[predictions["selected_in_fold"]].copy()
    selected["model"] = "NestedSelector"
    combined = pd.concat([predictions, selected], ignore_index=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for (validation, model), model_frame in combined.groupby(["validation", "model"]):
        baseline = model_frame[model_frame["representation"] == "separate_log_gdr"]
        studies = baseline["Sampling_Group"].unique()
        if not len(studies):
            continue
        observed_by_representation = {
            representation: comparison_metrics(group)
            for representation, group in model_frame.groupby("representation")
        }
        bootstrap_differences = {
            candidate: {}
            for candidate in ("independent_ilr", "joint_ilr")
            if candidate in observed_by_representation
        }
        for _ in range(iterations):
            sampled_studies = rng.choice(studies, size=len(studies), replace=True)
            sampled_metrics = {}
            for representation, representation_frame in model_frame.groupby(
                "representation"
            ):
                sampled = pd.concat(
                    [
                        representation_frame[
                            representation_frame["Sampling_Group"] == study
                        ]
                        for study in sampled_studies
                    ],
                    ignore_index=True,
                )
                sampled_metrics[representation] = comparison_metrics(sampled)
            for candidate in bootstrap_differences:
                for metric, baseline_value in sampled_metrics[
                    "separate_log_gdr"
                ].items():
                    bootstrap_differences[candidate].setdefault(metric, []).append(
                        sampled_metrics[candidate][metric] - baseline_value
                    )
        for candidate, metrics in bootstrap_differences.items():
            for metric, differences in metrics.items():
                values = np.asarray(differences, dtype=float)
                observed_difference = (
                    observed_by_representation[candidate][metric]
                    - observed_by_representation["separate_log_gdr"][metric]
                )
                rows.append(
                    {
                        "validation": validation,
                        "model": model,
                        "candidate": candidate,
                        "reference": "separate_log_gdr",
                        "metric": metric,
                        "candidate_minus_reference": observed_difference,
                        "CI_lower": float(np.quantile(values, 0.025)),
                        "CI_upper": float(np.quantile(values, 0.975)),
                        "bootstrap_median": float(np.median(values)),
                        "higher_is_better": not metric.endswith("RMSE"),
                        "cluster_unit": "Sampling_Group",
                        "iterations": iterations,
                    }
                )
    return pd.DataFrame(rows)


def run(arguments):
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_analysis_data(arguments.data, arguments.feature_table)
    frame, audit = construct_composition(frame)
    analysis = frame.dropna(
        subset=list(COVARIATES) + list(ILR_COLUMNS) + list(LOG_GDR_COLUMNS)
    ).copy()
    finite_columns = list(COVARIATES) + list(ILR_COLUMNS) + list(LOG_GDR_COLUMNS)
    analysis = analysis[np.isfinite(analysis[finite_columns]).all(axis=1)].copy()
    prediction_parts = []
    validations = ["sampling_group"] if arguments.quick else arguments.validations
    for validation in validations:
        prediction_parts.append(
            evaluate_nested(
                analysis,
                validation,
                arguments.representations,
                arguments.outer_folds,
                arguments.inner_folds,
                arguments.seed,
                arguments.n_jobs,
                arguments.quick,
            )
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    selected = predictions[predictions["selected_in_fold"]]
    counts = selected.groupby(["validation", "representation"])["row_index"].agg(
        ["count", "nunique"]
    )
    if not (
        (counts["count"] == len(analysis)) & (counts["nunique"] == len(analysis))
    ).all():
        raise RuntimeError(
            "Each selected representation must predict every analysis row exactly once"
        )
    performance = summarize_predictions(predictions)
    fold_performance = summarize_folds(predictions)
    paired_intervals = paired_cluster_bootstrap(
        predictions, arguments.bootstrap_iterations, arguments.seed
    )
    audit.to_csv(
        output_dir / "Composition_Group_Audit.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        output_dir / "Compositional_Nested_OOF_Predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    performance.to_csv(
        output_dir / "Compositional_Performance_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_performance.to_csv(
        output_dir / "Compositional_Performance_Per_Fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired_intervals.to_csv(
        output_dir / "Compositional_Paired_Bootstrap_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "version": VERSION,
        "data": str(arguments.data.resolve()),
        "feature_table": str(arguments.feature_table.resolve()),
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (arguments.data, arguments.feature_table)
        },
        "pfas_schema_sha256": schema_digest(),
        "input_rows": len(frame),
        "analysis_rows": len(analysis),
        "group_definitions": GROUP_DEFINITIONS,
        "ilr_basis": ILR_BASIS.tolist(),
        "representations": arguments.representations,
        "validations": validations,
        "outer_folds": arguments.outer_folds,
        "inner_folds": arguments.inner_folds,
        "seed": arguments.seed,
        "random_generator": "numpy.random.Generator(PCG64)",
        "bootstrap_iterations": arguments.bootstrap_iterations,
        "quick": arguments.quick,
    }
    (output_dir / "Run_Manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Outputs written to {output_dir}")


def parse_arguments():
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
    parser.add_argument("--data", type=Path, default=data_root / "raw" / "DATA.xlsx")
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=output_root / "01_primary_model" / "Long_Chain_Model_Feature_Table.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=output_root / "05_compositional_validation"
    )
    parser.add_argument(
        "--validations",
        nargs="+",
        choices=VALIDATIONS,
        default=[
            "sampling_group",
            "spatial_block_stratified",
            "leave_one_sampling_group_out",
        ],
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
