"""Compact numerical checks against existing frozen regional diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import regional_selectivity_analysis as analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()
    args.path_pcts = [0, 10, 25, 50]
    args.interaction_pcts = [10, 25, 50]
    args.skip_proximity = True
    args.resume = True
    ctx = analysis.context(args)
    if not analysis.completed(ctx, "prepare", True):
        raise RuntimeError("Run compact prepare before this verification")
    analysis.verify(ctx, args)
    archive = analysis.dense_archive(ctx)
    primary = analysis.load_module(
        "selectivity_primary",
        ctx.code_root / "02_modeling_and_validation/01_primary_model_training.py",
    )
    alternate = joblib.load(args.reference_root / "alternative_XGB_Balance.pkl")
    payload = {
        "schema": analysis.SCHEMA,
        "estimator": alternate.estimator_,
        "target_mean": alternate.target_mean_,
        "target_scale": alternate.target_scale_,
        "feature_names": ctx.feature_names,
    }
    portable_path = ctx.output_root / "alternative_serialization_check.pkl"
    joblib.dump(payload, portable_path)
    restored = joblib.load(portable_path)
    X = ctx.train[ctx.feature_names]
    restored_prediction = (
        np.asarray(restored["estimator"].predict(X), dtype=float)
        * restored["target_scale"]
        + restored["target_mean"]
    )
    np.testing.assert_allclose(
        alternate.predict(X), restored_prediction, rtol=0, atol=0
    )
    source = np.array([name in ctx.infer.SOURCE_FEATURES for name in X.columns])
    masks = analysis.interaction_masks(source)
    expected = pd.read_csv(args.reference_root / "interaction_block_summary.csv")
    errors, decomposition_errors = [], []
    for architecture in ("RF", "XGB"):
        if architecture == "RF":
            spec = ctx.package["models"]["Long_Balance"]
            model, scale = spec["model_object"], spec["target_scale"]
        else:
            model, scale = restored["estimator"], restored["target_scale"]
        explainer = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        baseline = np.asarray(explainer.shap_interaction_values(X)) * scale
        for j, (path, percent) in enumerate(zip(archive["path"], archive["path_pct"])):
            if percent != 50:
                continue
            xp = pd.DataFrame(archive["features"][j], columns=X.columns)
            delta = np.asarray(explainer.shap_interaction_values(xp)) * scale - baseline
            actual = (model.predict(xp) - model.predict(X)) * scale
            error = float(np.max(np.abs(delta.sum(axis=(1, 2)) - actual)))
            if error >= 2e-5:
                raise AssertionError(f"Paired SHAP additivity error: {error}")
            errors.append(
                {
                    "model": architecture,
                    "path": str(path),
                    "max_additivity_error": error,
                }
            )
            for sea in ctx.seas:
                ix = ctx.common & ctx.train.Sea_Region.eq(sea).to_numpy()
                groups = ctx.train.loc[ix, "Sampling_Group"].to_numpy()
                for label, mask in masks.items():
                    values = delta[ix][:, mask].sum(axis=1)
                    reference = expected[
                        (expected.model == architecture)
                        & (expected.path == path)
                        & (expected.path_pct == 50)
                        & (expected.sea == sea)
                        & (expected.block == label)
                    ].iloc[0]
                    means = np.array(
                        [values.mean(), pd.Series(values).groupby(groups).mean().mean()]
                    )
                    refmeans = np.array(
                        [
                            reference.mean_delta_contribution,
                            reference.group_equal_delta_contribution,
                        ]
                    )
                    np.testing.assert_allclose(means, refmeans, rtol=1e-9, atol=1e-9)
                    decomposition_errors.append(float(np.max(np.abs(means - refmeans))))
    rng = np.random.default_rng(analysis.SEED)
    base = rng.normal(size=(24, 4))
    query = base + np.array([0.2, 0.1, -0.1, 0])
    groups = np.repeat(np.arange(6), 4)
    actual = analysis.distance_diagnostic(base, query, groups)
    scale = np.quantile(base, 0.75, axis=0) - np.quantile(base, 0.25, axis=0)
    expected_distances = []
    for row, group in zip(query, groups):
        distance = np.sqrt(
            np.sum(((base[groups != group] - row) / scale) ** 2, axis=1)
        ) / np.sqrt(base.shape[1])
        expected_distances.append(np.sort(distance)[4])
    np.testing.assert_allclose(actual, expected_distances, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual,
        analysis.distance_diagnostic(base * [2, 3, 4, 5], query * [2, 3, 4, 5], groups),
        rtol=1e-12,
        atol=1e-12,
    )
    for count in (4, 7, 15):
        flags = np.arange(count) % 3 == 0
        blocks = analysis.interaction_masks(flags)
        np.testing.assert_array_equal(sum(blocks.values()), np.ones((count, count)))
    report = {
        "checks": [
            "CLI and import",
            "all dense regional columns at both paths and 0/10/25/50 positions",
            "frozen scenario predictions",
            "native-estimator serialization equivalence",
            "RF and XGB paired SHAP blocks at both 50-percent path positions",
            "SHAP additivity",
            "manual fifth-nearest different-group distances",
            "distance unit invariance",
            "interaction masks without fixed feature count",
        ],
        "passed": True,
        "dense_rows": 2 * len(args.path_pcts) * len(ctx.seas),
        "training_n": len(ctx.train),
        "training_groups": int(ctx.train.Sampling_Group.nunique()),
        "common_n": int(ctx.common.sum()),
        "max_shap_block_difference": max(decomposition_errors),
        "paired_additivity": errors,
        "production_model_sha256": analysis.digest(ctx.package_path),
        "full_alternative_search_rerun": False,
        "group_deletion_refits_rerun": False,
    }
    analysis.write_json(ctx.output_root / "validation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
