"""Regional compositional response diagnostics for a frozen PFAS model package."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone

SEED = 42
FAMILIES = ("class_resolved", "synchronized")
SEA_ORDER = ("Bohai Sea", "Yellow Sea", "East China Sea", "South China Sea")
SCHEMA = "pfas-regional-selectivity-diagnostics/v1"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_percentages(value: str) -> list[int]:
    try:
        levels = sorted(set(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Path positions must be comma-separated integers"
        ) from error
    if not levels or levels[0] < 0 or levels[-1] > 50:
        raise argparse.ArgumentTypeError(
            "Path positions must be in the interval 0 through 50"
        )
    return levels


@dataclass
class AnalysisContext:
    code_root: Path
    production_root: Path
    output_root: Path
    infer: object
    core: object
    package: dict
    train: pd.DataFrame
    common: np.ndarray
    package_path: Path
    seas: tuple[str, ...]
    fingerprint: dict

    @property
    def feature_names(self) -> list[str]:
        return self.package["feature_names"]

    @property
    def routing_root(self) -> Path:
        return self.production_root / "00_coastal_routing"


def context(args) -> AnalysisContext:
    code = args.code_root.resolve()
    prod = args.production_root.resolve()
    output = args.output_root.resolve()
    if (
        output == code
        or output.is_relative_to(code)
        or output == prod
        or output.is_relative_to(prod)
    ):
        raise ValueError(
            "Use an output directory outside the code and frozen production directories"
        )
    sys.path.insert(0, str(code))
    infer = load_module(
        "selectivity_inference",
        code / "02_modeling_and_validation/03_source_attenuation_inference.py",
    )
    core = load_module(
        "selectivity_source_core",
        code / "02_modeling_and_validation/_source_attenuation_core.py",
    )
    package_path = (
        prod / "01_primary_model/Long_Chain_Magnitude_Balance_Model_Package.pkl"
    )
    package = infer.load_model_package(package_path)
    train = infer.package_training_frame(package)
    if train.row_index.duplicated().any():
        raise ValueError("Training row_index must uniquely identify observations")
    paired = pd.read_csv(prod / "evidence_strengthening/paired_class_responses.csv")
    flags = paired.fixed_common_support.astype(str).str.strip().str.lower()
    if not flags.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Invalid fixed_common_support flag in paired class responses")
    ids = pd.Index(paired.loc[flags.isin(["true", "1"]), "row_index"].unique())
    missing = ids.difference(pd.Index(train.row_index))
    if len(missing) or not len(ids):
        raise ValueError(
            f"Fixed-common target population is empty or contains {len(missing)} absent rows"
        )
    common = train.row_index.isin(ids).to_numpy()
    feature_values = train[package["feature_names"]].to_numpy(float)
    if not np.isfinite(feature_values).all() or train.Sampling_Group.isna().any():
        raise ValueError(
            "Training features and sampling-group assignments must be complete"
        )
    observed = set(train.loc[common, "Sea_Region"])
    seas = tuple(
        [sea for sea in SEA_ORDER if sea in observed]
        + sorted(observed.difference(SEA_ORDER))
    )
    inputs = {
        f"production/{str(path.relative_to(prod)).replace(chr(92), '/')}": digest(path)
        for directory in ("00_coastal_routing", "02_source_attenuation_boundary")
        for path in sorted((prod / directory).rglob("*"))
        if path.is_file()
    }
    for relative in (
        "01_primary_model/Long_Chain_Magnitude_Balance_Model_Package.pkl",
        "01_primary_model/Long_Chain_Model_Feature_Table.csv",
        "03_source_attenuation_inference/Scenario_Paired_Predictions.csv",
        "evidence_strengthening/paired_class_responses.csv",
    ):
        inputs[f"production/{relative}"] = digest(prod / relative)
    for path in sorted(code.rglob("*.py")):
        inputs[f"code/{str(path.relative_to(code)).replace(chr(92), '/')}"] = digest(
            path
        )
    fingerprint = {
        "schema": SCHEMA,
        "seed": SEED,
        "input_sha256": inputs,
        "script_sha256": digest(Path(__file__)),
        "path_grid_pct": args.path_pcts,
        "interaction_grid_pct": args.interaction_pcts,
        "proximity_enabled": not args.skip_proximity,
        "common_n": int(common.sum()),
        "common_groups": int(train.loc[common, "Sampling_Group"].nunique()),
        "training_n": len(train),
        "training_groups": int(train.Sampling_Group.nunique()),
        "feature_names": package["feature_names"],
    }
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "run_provenance.json"
    if provenance.exists():
        if json.loads(provenance.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError(
                "Output provenance differs from this invocation; select a new output directory"
            )
    else:
        if any(output.iterdir()):
            raise ValueError(
                "Output directory already contains files without compatible provenance"
            )
        write_json(provenance, fingerprint)
    return AnalysisContext(
        code,
        prod,
        output,
        infer,
        core,
        package,
        train,
        common,
        package_path,
        seas,
        fingerprint,
    )


def completed(ctx: AnalysisContext, stage: str, resume: bool) -> bool:
    path = ctx.output_root / f"{stage}_complete.json"
    if not path.exists():
        return False
    if not resume:
        raise ValueError(
            f"Stage {stage} is already complete; use --resume to verify and reuse it"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest["output_sha256"].items():
        actual = ctx.output_root / relative
        if not actual.is_file() or digest(actual) != expected:
            raise ValueError(
                f"Completed stage output is missing or changed: {relative}"
            )
    print(f"Verified completed stage: {stage}", flush=True)
    return True


def finish(ctx: AnalysisContext, stage: str, outputs: list[str]) -> None:
    expected = ctx.fingerprint["input_sha256"][
        "production/01_primary_model/Long_Chain_Magnitude_Balance_Model_Package.pkl"
    ]
    if digest(ctx.package_path) != expected:
        raise RuntimeError("Frozen production model changed during diagnostics")
    write_json(
        ctx.output_root / f"{stage}_complete.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "output_sha256": {
                relative: digest(ctx.output_root / relative) for relative in outputs
            },
        },
    )


def dense_archive(ctx: AnalysisContext):
    archive = np.load(ctx.output_root / "dense_source_features.npz", allow_pickle=False)
    if not np.array_equal(archive["row_index"], ctx.train.row_index.to_numpy()):
        raise ValueError("Dense scenario row order differs from frozen training frame")
    if not np.array_equal(archive["common"], ctx.common):
        raise ValueError(
            "Dense scenarios use a different fixed-common target population"
        )
    if archive["features"].shape[1:] != (len(ctx.train), len(ctx.feature_names)):
        raise ValueError("Dense scenario feature dimensions are incompatible")
    return archive


def distance_diagnostic(
    base: np.ndarray, query: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Robust-scaled fifth-nearest observation from a different sampling group."""
    if len(query) % len(base):
        raise ValueError("Query rows must contain complete baseline-sized blocks")
    if any(np.sum(groups != group) < 5 for group in np.unique(groups)):
        raise ValueError(
            "At least five observations from other sampling groups are required"
        )
    scale = np.quantile(base, 0.75, axis=0) - np.quantile(base, 0.25, axis=0)
    fallback = np.std(base, axis=0)
    scale = np.where(scale > 0, scale, np.where(fallback > 0, fallback, 1))
    distances = cdist(query / scale, base / scale) / np.sqrt(base.shape[1])
    query_groups = np.tile(groups, len(query) // len(base))
    distances[query_groups[:, None] == groups[None, :]] = np.inf
    return np.partition(distances, 4, axis=1)[:, 4]


def prepare(ctx: AnalysisContext, args) -> None:
    if completed(ctx, "prepare", args.resume):
        return
    infer, core, pkg, train, common = (
        ctx.infer,
        ctx.core,
        ctx.package,
        ctx.train,
        ctx.common,
    )
    out, prod = ctx.output_root, ctx.production_root
    boundary = prod / "02_source_attenuation_boundary"
    manifest = json.loads(
        (boundary / "Legacy_Class_Transition_Manifest.json").read_text(encoding="utf-8")
    )
    summary = pd.read_csv(boundary / "Legacy_Class_Transition_Summary.csv")
    infer.verify_scenario_products(boundary, manifest, summary)
    infer.verify_routing_provenance(pkg, ctx.routing_root)
    inventory = core.validate_inventory(
        pd.read_csv(boundary / "Compound_Source_Flux_reference.csv")
    )
    reference, _ = core.aggregate_source_table(inventory)
    routing_rows = train[["row_index", "Longitude", "Latitude"]]
    source_names = list(infer.SOURCE_FEATURES)
    routed_ref = infer.compute_routed_features(
        routing_rows, reference, ctx.routing_root
    )[source_names]
    X = train[ctx.feature_names].copy()
    environment_names = [name for name in X.columns if name not in source_names]
    base_outputs = infer.derived_outputs(infer.coordinate_predictions(pkg, X))
    frozen = pd.read_csv(
        prod / "03_source_attenuation_inference/Scenario_Paired_Predictions.csv"
    )
    arrays, curves, checks, keys = [], [], [], []
    for family in FAMILIES:
        for percent in args.path_pcts:
            spec = {
                "legacy_mode": family,
                "short_mode": "reference",
                "transition_intensity": percent / 100,
            }
            multipliers = core.scenario_multipliers(
                spec, manifest["evidence_multipliers"]
            )
            source, _ = core.aggregate_source_table(
                core.apply_multipliers(inventory, multipliers)
            )
            routed = infer.compute_routed_features(
                routing_rows, source, ctx.routing_root
            )[source_names]
            xp, _ = infer.anchor_scenario_features(X, routed_ref, routed)
            if not np.array_equal(xp[environment_names], X[environment_names]):
                raise RuntimeError(
                    "Source attenuation changed fixed environmental covariates"
                )
            outputs = infer.derived_outputs(infer.coordinate_predictions(pkg, xp))
            if percent in (0, 10, 25, 50):
                sid = (
                    "reference"
                    if percent == 0
                    else f"{family}_legacy_control_{percent}pct"
                )
                original = (
                    frozen[frozen.scenario_id.eq(sid)]
                    .set_index("row_index")
                    .loc[train.row_index]
                )
                error = max(
                    float(
                        np.max(
                            np.abs(
                                outputs[name] - original[name + "_scenario"].to_numpy()
                            )
                        )
                    )
                    for name in ("Long_Magnitude", "Long_Balance", "Log_Long_Total")
                )
                if error >= 1e-6:
                    raise RuntimeError(
                        f"Frozen scenario predictions do not reproduce: {sid}, error={error}"
                    )
                checks.append(
                    {"path": family, "path_pct": percent, "max_prediction_error": error}
                )
            arrays.append(xp.to_numpy(float))
            keys.append({"path": family, "path_pct": percent})
            for sea in ctx.seas:
                ix = common & train.Sea_Region.eq(sea).to_numpy()
                groups = train.loc[ix, "Sampling_Group"].to_numpy()
                row = {
                    "path": family,
                    "path_pct": percent,
                    "sea": sea,
                    "n": int(ix.sum()),
                    "groups": len(np.unique(groups)),
                }
                for name in (
                    "Long_Magnitude",
                    "Long_Balance",
                    "Log_Long_Ratio",
                    "Log_Long_Total",
                    "Log_Long_PFSA",
                    "Log_Long_PFCA",
                ):
                    delta = outputs[name][ix] - base_outputs[name][ix]
                    row[name + "_mean_delta"] = float(delta.mean())
                    row[name + "_group_equal_delta"] = float(
                        pd.Series(delta).groupby(groups).mean().mean()
                    )
                for feature in source_names:
                    vals = xp.loc[ix, feature]
                    for label, value in (
                        ("median", vals.median()),
                        ("q25", vals.quantile(0.25)),
                        ("q75", vals.quantile(0.75)),
                    ):
                        row[feature + "_" + label] = float(value)
                outside = (xp.loc[ix, source_names] < X[source_names].min() - 1e-10) | (
                    xp.loc[ix, source_names] > X[source_names].max() + 1e-10
                )
                row["outside_source_minmax_fraction"] = float(
                    outside.any(axis=1).mean()
                )
                curves.append(row)
        print(f"Dense source paths reproduced: {family}", flush=True)
    cube = np.array(arrays)
    np.savez_compressed(
        out / "dense_source_features.npz",
        features=cube,
        row_index=train.row_index.to_numpy(),
        common=common,
        path=np.array([key["path"] for key in keys]),
        path_pct=np.array([key["path_pct"] for key in keys]),
    )
    pd.DataFrame(checks).to_csv(out / "frozen_result_reproduction.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "dense_regional_responses.csv", index=False)
    outputs = [
        "dense_source_features.npz",
        "frozen_result_reproduction.csv",
        "dense_regional_responses.csv",
    ]
    if not args.skip_proximity:
        rows, distance_arrays = [], {}
        for block, cols in (("source", source_names), ("all", ctx.feature_names)):
            indices = [ctx.feature_names.index(name) for name in cols]
            baseline = X[cols].to_numpy()
            groups = train.Sampling_Group.to_numpy()
            refdist = distance_diagnostic(baseline, baseline, groups)
            cutoff = float(np.quantile(refdist, 0.95))
            distances = np.vstack(
                [distance_diagnostic(baseline, arr[:, indices], groups) for arr in cube]
            )
            distance_arrays[block] = distances
            for j, key in enumerate(keys):
                for sea in ctx.seas:
                    ix = common & train.Sea_Region.eq(sea).to_numpy()
                    ratio = distances[j, ix] / np.maximum(refdist[ix], 1e-12)
                    rows.append(
                        {
                            **key,
                            "sea": sea,
                            "feature_block": block,
                            "n": int(ix.sum()),
                            "median_d5": float(np.median(distances[j, ix])),
                            "median_paired_distance_ratio": float(np.median(ratio)),
                            "baseline_median_d5": float(np.median(refdist[ix])),
                            "baseline_global_d5_p95": cutoff,
                            "above_baseline_p95_fraction": float(
                                np.mean(distances[j, ix] > cutoff)
                            ),
                        }
                    )
            print(f"Joint proximity diagnostic complete: {block}", flush=True)
        pd.DataFrame(rows).to_csv(out / "joint_proximity_diagnostic.csv", index=False)
        np.savez_compressed(out / "joint_proximity_distances.npz", **distance_arrays)
        outputs.extend(
            ["joint_proximity_diagnostic.csv", "joint_proximity_distances.npz"]
        )
    density = []
    values = X["Mixed_Source_Ratio_SC_LC"]
    for lower in np.arange(0, 3, 0.05):
        ix = (values >= lower) & (values < lower + 0.05)
        density.append(
            {
                "lower": lower,
                "upper": lower + 0.05,
                "n": int(ix.sum()),
                "sampling_groups": int(train.loc[ix, "Sampling_Group"].nunique()),
                "sea_regions": ";".join(sorted(train.loc[ix, "Sea_Region"].unique())),
            }
        )
    pd.DataFrame(density).to_csv(out / "training_source_ratio_density.csv", index=False)
    outputs.append("training_source_ratio_density.csv")
    write_json(
        out / "dense_manifest.json",
        {
            "stage": "dense_paths",
            "seed": SEED,
            "path_grid_pct": args.path_pcts,
            "common_n": int(common.sum()),
            "common_groups": int(train.loc[common, "Sampling_Group"].nunique()),
            "path_rule": "Original group multiplier raised to path intensity; original compound source fluxes, routing and anchoring retained.",
            "distance_rule": "IQR-scaled Euclidean distance/sqrt(p); fifth nearest observation from a different sampling group. Baseline P95 is descriptive.",
            "proximity_enabled": not args.skip_proximity,
            "no_new_exclusions": True,
        },
    )
    outputs.append("dense_manifest.json")
    finish(ctx, "prepare", outputs)


def stability(ctx: AnalysisContext, args) -> None:
    if completed(ctx, "stability", args.resume):
        return
    if not completed(ctx, "prepare", True):
        raise ValueError("The prepare stage must complete before stability diagnostics")
    primary = load_module(
        "selectivity_primary",
        ctx.code_root / "02_modeling_and_validation/01_primary_model_training.py",
    )
    pkg, train, common, out = ctx.package, ctx.train, ctx.common, ctx.output_root
    frame = (
        pd.read_csv(
            ctx.production_root / "01_primary_model/Long_Chain_Model_Feature_Table.csv"
        )
        .set_index("row_index", drop=False)
        .loc[train.row_index]
        .reset_index(drop=True)
    )
    X = train[ctx.feature_names].copy()
    if not np.allclose(X, frame[ctx.feature_names]):
        raise ValueError("Training feature table differs from frozen model package")
    y = frame.Long_Balance.to_numpy()
    definitions = {
        definition.name: definition
        for definition in primary.model_definitions(SEED, False)
    }
    inner = primary.make_inner_splits(frame, "sampling_group", 5, SEED)
    search, _ = primary.tune_model(definitions["XGB"], X, y, inner, 1)
    xgb = search.best_estimator_
    joblib.dump(
        {
            "schema": SCHEMA,
            "estimator": xgb.estimator_,
            "target_mean": xgb.target_mean_,
            "target_scale": xgb.target_scale_,
            "feature_names": ctx.feature_names,
            "best_params": search.best_params_,
        },
        out / "alternative_XGB_Balance.pkl",
    )
    print(
        f"Alternative XGB tuned with original grouped grid: {search.best_params_}",
        flush=True,
    )
    archive = dense_archive(ctx)
    cube = archive["features"]
    flat = pd.DataFrame(cube.reshape(-1, X.shape[1]), columns=X.columns)
    sea_indices = {
        sea: np.flatnonzero(common & train.Sea_Region.eq(sea).to_numpy())
        for sea in ctx.seas
    }

    def summarize(model, architecture, excluded):
        if architecture == "RF" and excluded == "none":
            spec = pkg["models"]["Long_Balance"]
            pred = (
                model.predict(flat).reshape(len(cube), len(X)) * spec["target_scale"]
                + spec["target_mean"]
            )
            base = model.predict(X) * spec["target_scale"] + spec["target_mean"]
        else:
            pred = model.predict(flat).reshape(len(cube), len(X))
            base = model.predict(X)
        delta = pred - base[None, :]
        rows = []
        for j, (family, percent) in enumerate(
            zip(archive["path"], archive["path_pct"])
        ):
            for sea, ix in sea_indices.items():
                groups = train.loc[ix, "Sampling_Group"].to_numpy()
                rows.append(
                    {
                        "model": architecture,
                        "excluded_group": str(excluded),
                        "path": str(family),
                        "path_pct": int(percent),
                        "sea": sea,
                        "mean_delta_balance": float(delta[j, ix].mean()),
                        "group_equal_delta_balance": float(
                            pd.Series(delta[j, ix]).groupby(groups).mean().mean()
                        ),
                    }
                )
        return rows

    records = summarize(
        pkg["models"]["Long_Balance"]["model_object"], "RF", "none"
    ) + summarize(xgb, "XGB", "none")
    groups = sorted(train.Sampling_Group.unique())

    def fit_deletion(architecture, excluded):
        ix = ~train.Sampling_Group.eq(excluded).to_numpy()
        estimator = clone(
            pkg["models"]["Long_Balance"]["model_object"]
            if architecture == "RF"
            else xgb.estimator_
        )
        model = primary.ScaledTargetRegressor(estimator)
        model.fit(X.loc[ix], y[ix])
        return summarize(model, architecture, excluded)

    count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fit_deletion, architecture, group): (architecture, group)
            for architecture in ("RF", "XGB")
            for group in groups
        }
        for future in as_completed(futures):
            records.extend(future.result())
            count += 1
            if count % 4 == 0 or count == len(futures):
                print(f"Group-deletion fits: {count}/{len(futures)}", flush=True)
    result = pd.DataFrame(records).sort_values(
        ["model", "excluded_group", "path", "path_pct", "sea"]
    )
    result.to_csv(out / "model_stability_paths.csv", index=False)
    summaries = []
    for (model, path, percent, sea), group in result[
        result.excluded_group.ne("none")
    ].groupby(["model", "path", "path_pct", "sea"]):
        summaries.append(
            {
                "model": model,
                "path": path,
                "path_pct": percent,
                "sea": sea,
                "refits": len(group),
                "min_delta": group.mean_delta_balance.min(),
                "max_delta": group.mean_delta_balance.max(),
                "median_delta": group.mean_delta_balance.median(),
                "positive_refits": int((group.mean_delta_balance > 0).sum()),
                "negative_refits": int((group.mean_delta_balance < 0).sum()),
            }
        )
    pd.DataFrame(summaries).to_csv(
        out / "group_deletion_stability_summary.csv", index=False
    )
    write_json(
        out / "stability_manifest.json",
        {
            "seed": SEED,
            "architectures": ["original RF", "alternative XGB"],
            "alternative_tuning": search.best_params_,
            "alternative_tuning_design": "Original grid, five-fold sampling-group inner validation, normalized RMSE.",
            "group_deletion_refits": 2 * len(groups),
            "training_groups": len(groups),
            "common_n": int(common.sum()),
            "group_deletion_rule": "Delete each training sampling group once; hold full-fit hyperparameters fixed, recalculate target scaling on retained training rows, evaluate the unchanged fixed-common target population.",
            "scope": "Model-fitting sensitivity; not an unbiased performance estimate, confidence interval or independent replication count.",
        },
    )
    finish(
        ctx,
        "stability",
        [
            "alternative_XGB_Balance.pkl",
            "model_stability_paths.csv",
            "group_deletion_stability_summary.csv",
            "stability_manifest.json",
        ],
    )


def interaction_masks(source: np.ndarray) -> dict[str, np.ndarray]:
    diagonal = np.eye(len(source), dtype=bool)
    masks = {
        "Source main": diagonal & source[:, None],
        "Environment main": diagonal & ~source[:, None],
        "Source-source interaction": ~diagonal & source[:, None] & source[None, :],
        "Environment-environment interaction": ~diagonal
        & ~source[:, None]
        & ~source[None, :],
        "Source-environment interaction": source[:, None] != source[None, :],
    }
    if not np.array_equal(
        np.sum(list(masks.values()), axis=0), np.ones((len(source), len(source)))
    ):
        raise RuntimeError(
            "SHAP interaction masks must partition the full matrix exactly once"
        )
    return masks


def interactions(ctx: AnalysisContext, args) -> None:
    if completed(ctx, "interactions", args.resume):
        return
    if not completed(ctx, "prepare", True) or not completed(ctx, "stability", True):
        raise ValueError(
            "Both prepare and stability stages must complete before interactions"
        )
    import shap

    out, train, pkg, common = ctx.output_root, ctx.train, ctx.package, ctx.common
    xgb = joblib.load(out / "alternative_XGB_Balance.pkl")
    if xgb.get("schema") != SCHEMA or xgb["feature_names"] != ctx.feature_names:
        raise ValueError("Alternative model feature contract is incompatible")
    X = train[ctx.feature_names]
    source = np.array([name in ctx.infer.SOURCE_FEATURES for name in X.columns])
    masks = interaction_masks(source)
    archive = dense_archive(ctx)
    rows, by_group, pairs, errors = [], [], [], []
    missing = set(args.interaction_pcts).difference(archive["path_pct"].tolist())
    if missing:
        raise ValueError(
            f"Interaction positions are absent from prepared grid: {sorted(missing)}"
        )
    for architecture in ("RF", "XGB"):
        if architecture == "RF":
            spec = pkg["models"]["Long_Balance"]
            model, scale = spec["model_object"], spec["target_scale"]
        else:
            model, scale = xgb["estimator"], xgb["target_scale"]
        explainer = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        baseline = np.asarray(explainer.shap_interaction_values(X)) * scale
        for j, (path, percent) in enumerate(zip(archive["path"], archive["path_pct"])):
            if percent not in args.interaction_pcts:
                continue
            xp = pd.DataFrame(archive["features"][j], columns=X.columns)
            delta = np.asarray(explainer.shap_interaction_values(xp)) * scale - baseline
            actual = (model.predict(xp) - model.predict(X)) * scale
            error = float(np.max(np.abs(delta.sum(axis=(1, 2)) - actual)))
            if error >= 2e-5:
                raise RuntimeError(
                    f"Paired SHAP interactions do not retain prediction additivity: {error}"
                )
            errors.append(
                {
                    "model": architecture,
                    "path": str(path),
                    "path_pct": int(percent),
                    "max_additivity_error": error,
                }
            )
            for sea in ctx.seas:
                ix = common & train.Sea_Region.eq(sea).to_numpy()
                groups = train.loc[ix, "Sampling_Group"].to_numpy()
                for label, mask in masks.items():
                    values = delta[ix][:, mask].sum(axis=1)
                    group_mean = pd.Series(values).groupby(groups).mean()
                    rows.append(
                        {
                            "model": architecture,
                            "path": str(path),
                            "path_pct": int(percent),
                            "sea": sea,
                            "block": label,
                            "n": int(ix.sum()),
                            "groups": len(group_mean),
                            "mean_delta_contribution": float(values.mean()),
                            "group_equal_delta_contribution": float(group_mean.mean()),
                        }
                    )
                    by_group.extend(
                        {
                            "model": architecture,
                            "path": str(path),
                            "path_pct": int(percent),
                            "sea": sea,
                            "block": label,
                            "Sampling_Group": group,
                            "mean_delta_contribution": float(value),
                        }
                        for group, value in group_mean.items()
                    )
                mean = delta[ix].mean(axis=0)
                for a in range(len(source)):
                    for b in range(a, len(source)):
                        pairs.append(
                            {
                                "model": architecture,
                                "path": str(path),
                                "path_pct": int(percent),
                                "sea": sea,
                                "feature_1": X.columns[a],
                                "feature_2": X.columns[b],
                                "term": "main" if a == b else "interaction",
                                "mean_delta_contribution": float(
                                    mean[a, b] if a == b else mean[a, b] + mean[b, a]
                                ),
                            }
                        )
            print(
                f"Paired interaction decomposition: {architecture}, {path}, {int(percent)}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(out / "interaction_block_summary.csv", index=False)
    pd.DataFrame(by_group).to_csv(
        out / "interaction_blocks_by_sampling_group.csv", index=False
    )
    pd.DataFrame(pairs).to_csv(out / "interaction_pair_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "interaction_additivity_check.csv", index=False)
    write_json(
        out / "interaction_manifest.json",
        {
            "method": "Tree SHAP interactions; tree_path_dependent, raw output; paired scenario minus baseline using each fitted model's training-path background.",
            "off_diagonal_counting": "Sum both symmetric entries once per feature pair, retaining additivity.",
            "interpretation": "Decomposition of model response; not physical interaction coefficients or statistical interaction tests.",
            "models": ["production RF", "group-tuned alternative XGB"],
            "common_n": int(common.sum()),
            "interaction_grid_pct": args.interaction_pcts,
            "max_error": max(item["max_additivity_error"] for item in errors),
        },
    )
    finish(
        ctx,
        "interactions",
        [
            "interaction_block_summary.csv",
            "interaction_blocks_by_sampling_group.csv",
            "interaction_pair_summary.csv",
            "interaction_additivity_check.csv",
            "interaction_manifest.json",
        ],
    )


def verify(ctx: AnalysisContext, args) -> None:
    if args.reference_root is None:
        raise ValueError("The verify stage requires --reference-root")
    actual = pd.read_csv(ctx.output_root / "dense_regional_responses.csv")
    reference = pd.read_csv(args.reference_root / "dense_regional_responses.csv")
    keys = ["path", "path_pct", "sea"]
    expected = (
        reference.set_index(keys)
        .loc[pd.MultiIndex.from_frame(actual[keys])]
        .reset_index()
    )
    if list(actual.columns) != list(expected.columns):
        raise ValueError("Reference CSV columns differ")
    numeric = [
        name for name in actual.columns if pd.api.types.is_numeric_dtype(actual[name])
    ]
    np.testing.assert_allclose(
        actual[numeric], expected[numeric], rtol=1e-10, atol=1e-10
    )
    for name in set(actual.columns).difference(numeric):
        if not actual[name].equals(expected[name]):
            raise ValueError(f"Reference labels differ: {name}")
    print(
        f"Reference dense responses reproduced: {len(actual)} regional rows", flush=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "stability", "interactions", "verify", "all"]
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        required=True,
        help="Analysis code root containing pfas_schema.py and modeling scripts",
    )
    parser.add_argument(
        "--production-root",
        type=Path,
        required=True,
        help="Frozen pipeline output root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Separate diagnostic output directory",
    )
    parser.add_argument(
        "--path-pcts",
        type=parse_percentages,
        default=list(range(51)),
        help="Path positions, default every integer from 0 to 50; 50 is a path position, not a 50 percent emission reduction",
    )
    parser.add_argument(
        "--interaction-pcts",
        type=parse_percentages,
        default=[10, 25, 50],
        help="Prepared positions for paired interaction decomposition",
    )
    parser.add_argument(
        "--skip-proximity",
        action="store_true",
        help="Omit distance diagnostics for a compact reproduction check",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent group-deletion refits; default 4",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify hashes and reuse completed stages from the same inputs and grid",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Existing diagnostic outputs used by the verify stage",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    ctx = context(args)
    functions = {
        "prepare": prepare,
        "stability": stability,
        "interactions": interactions,
        "verify": verify,
    }
    for stage in (
        ["prepare", "stability", "interactions"]
        if args.stage == "all"
        else [args.stage]
    ):
        functions[stage](ctx, args)


if __name__ == "__main__":
    main()
