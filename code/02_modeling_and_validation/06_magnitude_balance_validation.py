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
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadata_schema import normalize_metadata as normalize_metadata_contract
from pfas_schema import (
    COVARIATES,
    LONG_PFSA,
    LONG_PFCA,
    normalize_feature_columns,
    schema_digest,
    sum_group,
)

VERSION = "v1.1"
RANDOM_SEED = 42
SPATIAL_BLOCK_DEGREES = 2.0

VALIDATIONS = (
    "sampling_group",
    "spatial_block_stratified",
    "leave_one_sampling_group_out",
    "leave_one_sea_region_out",
)
REPRESENTATIONS = (
    "independent_components",
    "independent_magnitude_balance",
    "joint_magnitude_balance",
)
OUTPUT_TARGETS = (
    "Log_Long_PFSA",
    "Log_Long_PFCA",
    "Long_Magnitude",
    "Long_Balance",
    "Log_Long_Total",
    "Log_Long_Ratio",
)


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


def normalized_rmse_scorer(estimator, X, y):
    observed = np.asarray(y, dtype=float)
    predicted = np.asarray(estimator.predict(X), dtype=float)
    scale = np.asarray(estimator.target_scale_, dtype=float)
    return -float(np.sqrt(np.mean(((observed - predicted) / scale) ** 2)))


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    estimator: object
    grid: dict[str, list]


def model_definitions(seed, quick, joint):
    if quick:
        rf_grid = {
            "n_estimators": [100],
            "max_depth": [5],
            "max_features": [0.8],
            "max_samples": [0.8],
        }
        xgb_grid = {
            "n_estimators": [100],
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
    rf = RandomForestRegressor(
        random_state=seed, criterion="absolute_error", min_samples_leaf=2, n_jobs=1
    )
    xgb_base = xgb.XGBRegressor(
        random_state=seed, objective="reg:pseudohubererror", n_jobs=1
    )
    if joint:
        return [
            ModelDefinition(
                "RF", rf, {f"estimator__{key}": value for key, value in rf_grid.items()}
            ),
            ModelDefinition(
                "XGB",
                MultiOutputRegressor(xgb_base, n_jobs=1),
                {
                    f"estimator__estimator__{key}": value
                    for key, value in xgb_grid.items()
                },
            ),
        ]
    return [
        ModelDefinition(
            "RF", rf, {f"estimator__{key}": value for key, value in rf_grid.items()}
        ),
        ModelDefinition(
            "XGB",
            xgb_base,
            {f"estimator__{key}": value for key, value in xgb_grid.items()},
        ),
    ]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_metadata(frame):
    result, _ = normalize_metadata_contract(normalize_feature_columns(frame))
    return result


def component_to_magnitude_balance(log_components):
    values = np.asarray(log_components, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Two log-component columns are required")
    return np.column_stack(
        [
            (values[:, 0] + values[:, 1]) / np.sqrt(2.0),
            (values[:, 0] - values[:, 1]) / np.sqrt(2.0),
        ]
    )


def magnitude_balance_to_components(coordinates):
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Magnitude and balance columns are required")
    return np.column_stack(
        [
            (values[:, 0] + values[:, 1]) / np.sqrt(2.0),
            (values[:, 0] - values[:, 1]) / np.sqrt(2.0),
        ]
    )


def construct_targets(frame):
    result = normalize_metadata(frame)
    long_pfsa = sum_group(result, "Long_PFSA")
    long_pfca = sum_group(result, "Long_PFCA")
    pfsa_columns = [column for column in LONG_PFSA if column in result]
    pfca_columns = [column for column in LONG_PFCA if column in result]
    result["Long_PFSA"] = long_pfsa
    result["Long_PFCA"] = long_pfca
    eligible = (
        np.isfinite(long_pfsa)
        & np.isfinite(long_pfca)
        & (long_pfsa > 0)
        & (long_pfca > 0)
    )
    result["Endpoint_Eligible"] = eligible
    log_components = np.column_stack(
        [np.log(long_pfsa[eligible]), np.log(long_pfca[eligible])]
    )
    magnitude_balance = component_to_magnitude_balance(log_components)
    result.loc[eligible, "Log_Long_PFSA"] = log_components[:, 0]
    result.loc[eligible, "Log_Long_PFCA"] = log_components[:, 1]
    result.loc[eligible, "Long_Magnitude"] = magnitude_balance[:, 0]
    result.loc[eligible, "Long_Balance"] = magnitude_balance[:, 1]
    result.loc[eligible, "Log_Long_Total"] = np.log(
        long_pfsa[eligible] + long_pfca[eligible]
    )
    result.loc[eligible, "Log_Long_Ratio"] = log_components[:, 0] - log_components[:, 1]
    audit = {
        "input_rows": len(result),
        "eligible_rows": int(eligible.sum()),
        "long_pfsa_declared": list(LONG_PFSA),
        "long_pfsa_available": pfsa_columns,
        "long_pfca_declared": list(LONG_PFCA),
        "long_pfca_available": pfca_columns,
        "missing_declared": sorted(
            set(LONG_PFSA + LONG_PFCA).difference(result.columns)
        ),
    }
    return result, audit


def make_splits(frame, validation, folds, seed):
    group_column = {
        "sampling_group": "Sampling_Group",
        "spatial_block_stratified": "Spatial_Block",
        "leave_one_sampling_group_out": "Sampling_Group",
        "leave_one_sea_region_out": "Sea_Region",
    }[validation]
    groups = frame[group_column].astype(str).to_numpy()
    if validation == "spatial_block_stratified":
        splitter = StratifiedGroupKFold(
            n_splits=min(folds, len(np.unique(groups))),
            shuffle=True,
            random_state=seed,
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
    group_column = {
        "sampling_group": "Sampling_Group",
        "spatial_block_stratified": "Spatial_Block",
        "leave_one_sampling_group_out": "Sampling_Group",
        "leave_one_sea_region_out": "Sea_Region",
    }[validation]
    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(f"{validation} requires at least two inner groups")
    if validation == "spatial_block_stratified":
        splitter = StratifiedGroupKFold(
            n_splits=min(folds, len(unique_groups)),
            shuffle=True,
            random_state=seed,
        )
        return list(
            splitter.split(frame, y=frame["Sea_Region"].astype(str), groups=groups)
        )
    splitter = GroupKFold(n_splits=min(folds, len(unique_groups)))
    return list(splitter.split(frame, groups=groups))


def representation_targets(frame, representation):
    if representation == "independent_components":
        return frame[["Log_Long_PFSA", "Log_Long_PFCA"]].to_numpy(dtype=float)
    return frame[["Long_Magnitude", "Long_Balance"]].to_numpy(dtype=float)


def tune_independent(definition, X, targets, inner_splits, n_jobs):
    searches = []
    for position in range(targets.shape[1]):
        search = GridSearchCV(
            ScaledTargetRegressor(definition.estimator),
            definition.grid,
            scoring=normalized_rmse_scorer,
            cv=inner_splits,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X, targets[:, position])
        searches.append(search)
    return searches


def tune_joint(definition, X, targets, inner_splits, n_jobs):
    search = GridSearchCV(
        ScaledTargetRegressor(definition.estimator),
        definition.grid,
        scoring=normalized_rmse_scorer,
        cv=inner_splits,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, targets)
    return search


def derived_outputs(log_components):
    values = np.asarray(log_components, dtype=float)
    magnitude_balance = component_to_magnitude_balance(values)
    return {
        "Log_Long_PFSA": values[:, 0],
        "Log_Long_PFCA": values[:, 1],
        "Long_Magnitude": magnitude_balance[:, 0],
        "Long_Balance": magnitude_balance[:, 1],
        "Log_Long_Total": np.logaddexp(values[:, 0], values[:, 1]),
        "Log_Long_Ratio": values[:, 0] - values[:, 1],
    }


def evaluate_nested(
    analysis, validation, representations, outer_folds, inner_folds, seed, n_jobs, quick
):
    rows = []
    splits = make_splits(analysis, validation, outer_folds, seed)
    for fold_number, (train_position, test_position, held_out) in enumerate(splits, 1):
        train = analysis.iloc[train_position]
        test = analysis.iloc[test_position]
        X_train = train[COVARIATES]
        X_test = test[COVARIATES]
        observed = derived_outputs(
            test[["Log_Long_PFSA", "Log_Long_PFCA"]].to_numpy(dtype=float)
        )
        inner_splits = make_inner_splits(
            train, validation, inner_folds, seed + fold_number
        )
        for representation in representations:
            y_train = representation_targets(train, representation)
            joint = representation == "joint_magnitude_balance"
            model_results = []
            for definition in model_definitions(seed, quick, joint):
                if joint:
                    search = tune_joint(
                        definition, X_train, y_train, inner_splits, n_jobs
                    )
                    predicted_coordinates = np.asarray(
                        search.predict(X_test), dtype=float
                    )
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
                predicted_components = (
                    predicted_coordinates
                    if representation == "independent_components"
                    else magnitude_balance_to_components(predicted_coordinates)
                )
                model_results.append(
                    (definition.name, inner_score, parameters, predicted_components)
                )
            selected_model = max(model_results, key=lambda item: item[1])[0]
            for (
                model_name,
                inner_score,
                parameters,
                predicted_components,
            ) in model_results:
                predicted = derived_outputs(predicted_components)
                for local_position, index in enumerate(test.index):
                    row = {
                        "row_index": index,
                        "Sampling_Group": test.at[index, "Sampling_Group"],
                        "Sea_Region": test.at[index, "Sea_Region"],
                        "Longitude": test.at[index, "Longitude"],
                        "Latitude": test.at[index, "Latitude"],
                        "Spatial_Block": test.at[index, "Spatial_Block"],
                        "validation": validation,
                        "fold": fold_number,
                        "held_out_groups": held_out,
                        "representation": representation,
                        "model": model_name,
                        "selected_in_fold": model_name == selected_model,
                        "inner_mean_neg_normalized_RMSE": inner_score,
                        "best_parameters": json.dumps(parameters, sort_keys=True),
                    }
                    for target in OUTPUT_TARGETS:
                        row[f"Observed_{target}"] = observed[target][local_position]
                        row[f"Predicted_{target}"] = predicted[target][local_position]
                    rows.append(row)
            print(f"{validation} | {representation} | fold {fold_number}/{len(splits)}")
    return pd.DataFrame(rows)


def metric_values(observed, predicted):
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


def summarize_predictions(predictions):
    selected = predictions[predictions["selected_in_fold"]].copy()
    selected["model"] = "NestedSelector"
    combined = pd.concat([predictions, selected], ignore_index=True)
    rows = []
    for keys, group in combined.groupby(["validation", "representation", "model"]):
        base = dict(zip(["validation", "representation", "model"], keys))
        for target in OUTPUT_TARGETS:
            rows.append(
                {
                    **base,
                    "target": target,
                    **metric_values(
                        group[f"Observed_{target}"], group[f"Predicted_{target}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_folds(predictions):
    selected = predictions[predictions["selected_in_fold"]].copy()
    selected["model"] = "NestedSelector"
    combined = pd.concat([predictions, selected], ignore_index=True)
    rows = []
    columns = ["validation", "representation", "model", "fold"]
    for keys, group in combined.groupby(columns):
        base = dict(zip(columns, keys))
        base["held_out_groups"] = group["held_out_groups"].iloc[0]
        for target in OUTPUT_TARGETS:
            rows.append(
                {
                    **base,
                    "target": target,
                    **metric_values(
                        group[f"Observed_{target}"], group[f"Predicted_{target}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_intervals(predictions, iterations, seed):
    selected = predictions[predictions["selected_in_fold"]].copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for keys, group in selected.groupby(["validation", "representation"]):
        validation, representation = keys
        studies = group["Sampling_Group"].unique()
        observed_metrics = {}
        bootstrap_metrics = {
            target: {"R2": [], "RMSE": [], "MAE": []} for target in OUTPUT_TARGETS
        }
        for target in OUTPUT_TARGETS:
            observed_metrics[target] = metric_values(
                group[f"Observed_{target}"], group[f"Predicted_{target}"]
            )
        for _ in range(iterations):
            sampled_studies = rng.choice(studies, size=len(studies), replace=True)
            sample = pd.concat(
                [group[group["Sampling_Group"] == study] for study in sampled_studies],
                ignore_index=True,
            )
            for target in OUTPUT_TARGETS:
                metrics = metric_values(
                    sample[f"Observed_{target}"], sample[f"Predicted_{target}"]
                )
                for metric in ("R2", "RMSE", "MAE"):
                    bootstrap_metrics[target][metric].append(metrics[metric])
        for target in OUTPUT_TARGETS:
            for metric in ("R2", "RMSE", "MAE"):
                values = np.asarray(bootstrap_metrics[target][metric], dtype=float)
                rows.append(
                    {
                        "validation": validation,
                        "representation": representation,
                        "model": "NestedSelector",
                        "target": target,
                        "metric": metric,
                        "estimate": observed_metrics[target][metric],
                        "CI_lower": float(np.quantile(values, 0.025)),
                        "CI_upper": float(np.quantile(values, 0.975)),
                        "bootstrap_median": float(np.median(values)),
                        "cluster_unit": "Sampling_Group",
                        "iterations": iterations,
                    }
                )
    return pd.DataFrame(rows)


def paired_representation_bootstrap(predictions, iterations, seed):
    selected = predictions[predictions["selected_in_fold"]].copy()
    reference = "independent_components"
    candidates = [
        value for value in selected["representation"].unique() if value != reference
    ]
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for validation, validation_frame in selected.groupby("validation"):
        studies = validation_frame[validation_frame["representation"] == reference][
            "Sampling_Group"
        ].unique()
        for candidate in candidates:
            candidate_frame = validation_frame[
                validation_frame["representation"] == candidate
            ]
            reference_frame = validation_frame[
                validation_frame["representation"] == reference
            ]
            observed = {}
            distributions = {}
            for target in OUTPUT_TARGETS:
                reference_metrics = metric_values(
                    reference_frame[f"Observed_{target}"],
                    reference_frame[f"Predicted_{target}"],
                )
                candidate_metrics = metric_values(
                    candidate_frame[f"Observed_{target}"],
                    candidate_frame[f"Predicted_{target}"],
                )
                for metric in ("R2", "RMSE", "MAE"):
                    key = (target, metric)
                    observed[key] = (
                        candidate_metrics[metric] - reference_metrics[metric]
                    )
                    distributions[key] = []
            for _ in range(iterations):
                sampled_studies = rng.choice(studies, size=len(studies), replace=True)
                reference_sample = pd.concat(
                    [
                        reference_frame[reference_frame["Sampling_Group"] == study]
                        for study in sampled_studies
                    ],
                    ignore_index=True,
                )
                candidate_sample = pd.concat(
                    [
                        candidate_frame[candidate_frame["Sampling_Group"] == study]
                        for study in sampled_studies
                    ],
                    ignore_index=True,
                )
                for target in OUTPUT_TARGETS:
                    reference_metrics = metric_values(
                        reference_sample[f"Observed_{target}"],
                        reference_sample[f"Predicted_{target}"],
                    )
                    candidate_metrics = metric_values(
                        candidate_sample[f"Observed_{target}"],
                        candidate_sample[f"Predicted_{target}"],
                    )
                    for metric in ("R2", "RMSE", "MAE"):
                        distributions[(target, metric)].append(
                            candidate_metrics[metric] - reference_metrics[metric]
                        )
            for (target, metric), values in distributions.items():
                values = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "validation": validation,
                        "candidate": candidate,
                        "reference": reference,
                        "target": target,
                        "metric": metric,
                        "candidate_minus_reference": observed[(target, metric)],
                        "CI_lower": float(np.quantile(values, 0.025)),
                        "CI_upper": float(np.quantile(values, 0.975)),
                        "bootstrap_median": float(np.median(values)),
                        "cluster_unit": "Sampling_Group",
                        "iterations": iterations,
                    }
                )
    return pd.DataFrame(rows)


def validate_oof(predictions, analysis_rows):
    selected = predictions[predictions["selected_in_fold"]]
    counts = selected.groupby(["validation", "representation"])["row_index"].agg(
        ["count", "nunique"]
    )
    if not (
        (counts["count"] == analysis_rows) & (counts["nunique"] == analysis_rows)
    ).all():
        raise RuntimeError(
            "Each selected representation must predict every analysis row exactly once"
        )


def run(arguments):
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.feature_table.parent / "Feature_Table_Manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing upstream feature-table provenance: {manifest_path}"
        )
    upstream = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        upstream.get("feature_table_sha256") != sha256(arguments.feature_table)
        or upstream.get("pfas_schema_sha256") != schema_digest()
    ):
        raise ValueError(
            "Prepared feature table or model schema differs from upstream provenance"
        )
    frame = pd.read_csv(arguments.feature_table)
    frame, audit = construct_targets(frame)
    required = COVARIATES + list(OUTPUT_TARGETS)
    analysis = frame.dropna(subset=required).copy()
    analysis = analysis[np.isfinite(analysis[required]).all(axis=1)].copy()
    validations = ["sampling_group"] if arguments.quick else arguments.validations
    prediction_parts = []
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
    validate_oof(predictions, len(analysis))
    performance = summarize_predictions(predictions)
    fold_performance = summarize_folds(predictions)
    intervals = cluster_bootstrap_intervals(
        predictions, arguments.bootstrap_iterations, arguments.seed
    )
    comparisons = paired_representation_bootstrap(
        predictions, arguments.bootstrap_iterations, arguments.seed
    )
    predictions.to_csv(
        output_dir / "Magnitude_Balance_Nested_OOF_Predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    performance.to_csv(
        output_dir / "Magnitude_Balance_Performance_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_performance.to_csv(
        output_dir / "Magnitude_Balance_Performance_Per_Fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    intervals.to_csv(
        output_dir / "Magnitude_Balance_Sampling_Group_Cluster_Bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        output_dir / "Magnitude_Balance_Paired_Representation_Differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "version": VERSION,
        "feature_table": str(arguments.feature_table.resolve()),
        "feature_table_sha256": sha256(arguments.feature_table),
        "pfas_schema_sha256": schema_digest(),
        "data_sha256": upstream.get("data_sha256"),
        "input_rows": len(frame),
        "analysis_rows": len(analysis),
        "sampling_group_count": int(analysis["Sampling_Group"].nunique()),
        "spatial_block_count": int(analysis["Spatial_Block"].nunique()),
        "sea_region_count": int(analysis["Sea_Region"].nunique()),
        "target_audit": audit,
        "covariates": COVARIATES,
        "representations": arguments.representations,
        "validations": validations,
        "outer_folds": arguments.outer_folds,
        "inner_folds": arguments.inner_folds,
        "bootstrap_iterations": arguments.bootstrap_iterations,
        "seed": arguments.seed,
        "random_generator": "numpy.random.Generator(PCG64)",
        "quick": arguments.quick,
    }
    (output_dir / "Magnitude_Balance_Run_Manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Outputs written to {output_dir}")


def parse_arguments():
    project_root = (
        Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
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
        "--feature-table",
        type=Path,
        default=output_root / "01_primary_model" / "Long_Chain_Model_Feature_Table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "06_magnitude_balance_validation",
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
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
