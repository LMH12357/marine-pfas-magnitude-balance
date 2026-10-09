"""Calculate paired class responses and attribution summaries from saved outputs."""

from pathlib import Path
import argparse
import os
import json
import hashlib
import numpy as np
import pandas as pd

SEED = 20261007
B = 10000
paths = ["synchronized", "class_resolved"]
levels = [10, 25, 50]


def parse_arguments():
    """Resolve inputs without workstation-specific paths; retain science constants."""
    repository = (
        Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        .expanduser()
        .resolve()
    )
    data_root = (
        Path(os.environ.get("PFAS_DATA_ROOT", repository / "data"))
        .expanduser()
        .resolve()
    )
    output_root = (
        Path(os.environ.get("PFAS_OUTPUT_ROOT", repository / "outputs"))
        .expanduser()
        .resolve()
    )
    scenario_generated = (
        output_root
        / "03_source_attenuation_inference"
        / "Scenario_Paired_Predictions.csv"
    )
    scenario_fixed = (
        data_root
        / "fixed_outputs"
        / "source_attenuation_inference"
        / "Scenario_Paired_Predictions.csv"
    )
    shap_generated = output_root / "01_primary_model" / "Nested_OOF_SHAP_Values.csv"
    shap_fixed = (
        data_root / "fixed_outputs" / "primary_model" / "Nested_OOF_SHAP_Values.csv"
    )
    scenario_default = (
        scenario_generated
        if scenario_generated.is_file() or not scenario_fixed.is_file()
        else scenario_fixed
    )
    shap_default = (
        shap_generated
        if shap_generated.is_file() or not shap_fixed.is_file()
        else shap_fixed
    )
    parser = argparse.ArgumentParser(
        description="Post-process existing paired source scenarios and nested OOF SHAP; do not retrain models."
    )
    parser.add_argument(
        "--scenario-input",
        type=Path,
        default=scenario_default,
        help="Existing paired predictions CSV from 03_source_attenuation_inference.py.",
    )
    parser.add_argument(
        "--shap-input",
        type=Path,
        default=shap_default,
        help="Existing held-out SHAP CSV from 01_primary_model_training.py, not an importance summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "evidence_strengthening",
        help="Destination for new derived outputs; inputs are read only.",
    )
    return parser.parse_args()


def summary(g, col):
    """Cluster bootstrap record-weighted log change; group-equal sensitivity."""
    a = g.groupby("Sampling_Group")[col].agg(["sum", "count", "mean"])
    rng = np.random.default_rng(SEED)
    ii = rng.integers(0, len(a), size=(B, len(a)))
    boot = a["sum"].to_numpy()[ii].sum(axis=1) / a["count"].to_numpy()[ii].sum(axis=1)
    mean = g[col].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    loo = (
        (a["sum"].sum() - a["sum"]) / (a["count"].sum() - a["count"])
        if len(a) > 1
        else pd.Series([np.nan])
    )
    return dict(
        n=len(g),
        groups=len(a),
        mean_log_change=mean,
        geometric_change_pct=100 * np.expm1(mean),
        ci_low_pct=100 * np.expm1(lo),
        ci_high_pct=100 * np.expm1(hi),
        group_equal_change_pct=100 * np.expm1(a["mean"].mean()),
        loo_min_pct=100 * np.expm1(loo.min()),
        loo_max_pct=100 * np.expm1(loo.max()),
        fraction_increasing=(g[col] > 1e-12).mean(),
        fraction_decreasing=(g[col] < -1e-12).mean(),
        groups_positive=int((a["mean"] > 1e-12).sum()),
        groups_negative=int((a["mean"] < -1e-12).sum()),
    )


def run(scenario_input: Path, shap_input: Path, output_dir: Path):
    src = scenario_input.expanduser().resolve()
    shp_src = shap_input.expanduser().resolve()
    OUT = output_dir.expanduser().resolve()
    missing = [str(path) for path in (src, shp_src) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing existing output(s): "
            + "; ".join(missing)
            + ". Supply --scenario-input and --shap-input explicitly; this script does not train models or calculate SHAP."
        )
    output_names = (
        "paired_class_responses.csv",
        "regional_class_response_summary.csv",
        "regional_response_patterns.csv",
        "oof_feature_response_bins.csv",
        "oof_feature_direction_by_fold.csv",
        "oof_selected_feature_values.csv",
        "analysis_manifest.json",
        "postprocessing_execution_manifest.json",
    )
    if any(path == OUT / name for path in (src, shp_src) for name in output_names):
        raise ValueError(
            "An output would overwrite an input; select a separate --output-dir."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(src)
    ids = [f"{p}_legacy_control_{l}pct" for p in paths for l in levels]
    d = raw[raw.scenario_id.isin(ids)].copy()
    assert len(d) == 6 * 789
    assert not d.duplicated(["scenario_id", "row_index"]).any()
    d["supported"] = d.outside_source_minmax_any.astype(str).str.lower().eq("false")
    d["path"] = d.scenario_id.str.replace(r"_legacy_control_\d+pct$", "", regex=True)
    d["path_pct"] = d.scenario_id.str.extract(r"_(\d+)pct$").astype(int)
    for c in ["PFSA", "PFCA", "Total"]:
        assert (d[f"Long_{c}_baseline"] > 0).all() and (
            d[f"Long_{c}_scenario"] > 0
        ).all()
        delta = np.log(d[f"Long_{c}_scenario"] / d[f"Long_{c}_baseline"])
        assert np.allclose(delta, d[f"Log_Long_{c}_delta"], atol=1e-10)
    assert np.allclose(
        d.Long_Total_scenario, d.Long_PFSA_scenario + d.Long_PFCA_scenario
    )
    assert np.allclose(
        d.Log_Long_PFSA_delta,
        (d.Long_Magnitude_delta + d.Long_Balance_delta) / np.sqrt(2),
    )
    assert np.allclose(
        d.Log_Long_PFCA_delta,
        (d.Long_Magnitude_delta - d.Long_Balance_delta) / np.sqrt(2),
    )
    fixed = set.intersection(
        *(set(d.loc[d.scenario_id.eq(i) & d.supported, "row_index"]) for i in ids)
    )
    d["fixed_common_support"] = d.row_index.isin(fixed)
    d["paired_path_support"] = False
    for l in levels:
        common = set.intersection(
            *(
                set(d.loc[d.path.eq(p) & d.path_pct.eq(l) & d.supported, "row_index"])
                for p in paths
            )
        )
        d.loc[d.path_pct.eq(l) & d.row_index.isin(common), "paired_path_support"] = True
    d.to_csv(OUT / "paired_class_responses.csv", index=False)
    rows = []
    for scope, mask in [
        ("scenario_supported", d.supported),
        ("paired_paths", d.paired_path_support),
        ("fixed_all_paths", d.fixed_common_support),
    ]:
        for (path, l), g in d[mask].groupby(["path", "path_pct"]):
            for sea, h in [("All", g)] + list(g.groupby("Sea_Region")):
                for c in ["PFSA", "PFCA", "Total", "Ratio"]:
                    rows.append(
                        dict(
                            scope=scope,
                            path=path,
                            path_pct=l,
                            sea=sea,
                            component=c,
                            **summary(h, f"Log_Long_{c}_delta"),
                        )
                    )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "regional_class_response_summary.csv", index=False)
    # Individual/group-level regime classification, not inferred from regional means.
    patterns = []
    for (p, l, sea), g in d[d.fixed_common_support].groupby(
        ["path", "path_pct", "Sea_Region"]
    ):
        sa = g.Log_Long_PFSA_delta.to_numpy()
        ca = g.Log_Long_PFCA_delta.to_numpy()
        for label, m in [
            ("Both decrease, PFSA decreases less", (sa < 0) & (ca < 0) & (sa > ca)),
            ("Both decrease, PFCA decreases less", (sa < 0) & (ca < 0) & (ca > sa)),
            ("PFSA increases, PFCA decreases", (sa > 0) & (ca < 0)),
            ("PFCA increases, PFSA decreases", (ca > 0) & (sa < 0)),
            ("Both increase", (sa > 0) & (ca > 0)),
            (
                "Other incl. ties",
                ~(
                    ((sa < 0) & (ca < 0) & (sa != ca))
                    | ((sa > 0) & (ca < 0))
                    | ((ca > 0) & (sa < 0))
                    | ((sa > 0) & (ca > 0))
                ),
            ),
        ]:
            patterns.append(
                dict(
                    path=p,
                    path_pct=l,
                    sea=sea,
                    pattern=label,
                    n=int(m.sum()),
                    denominator=len(g),
                    fraction=m.mean(),
                )
            )
    pd.DataFrame(patterns).to_csv(OUT / "regional_response_patterns.csv", index=False)
    # OOF diagnostics: report descriptive, group-balanced binned curves, not flux thresholds.
    s = pd.read_csv(shp_src)
    s = s[
        s.selected_in_fold.astype(str).str.lower().eq("true")
        & s.feature_block.eq("all")
        & s.validation.isin(["sampling_group", "spatial_block_stratified"])
    ].copy()
    meta = raw[raw.scenario_id.eq("reference")][
        ["row_index", "Sampling_Group", "Sea_Region"]
    ]
    s = s.merge(meta, on="row_index", validate="many_to_one")
    wanted = [
        "Diffused_Total_Legacy_Force",
        "Hydro_MLD",
        "TOC (%)",
        "Mixed_Source_Ratio_LC_PFSA_PFCA",
    ]
    wanted += [
        x
        for x in s.feature.unique()
        if x.startswith("Salinity") or x.startswith("Temperature")
    ]
    s = s[s.feature.isin(wanted)].copy()
    assert not s.duplicated(["target", "validation", "feature", "row_index"]).any()
    bins = []
    slopes = []
    for (target, validation, feature), g in s.groupby(
        ["target", "validation", "feature"]
    ):
        g = g.copy()
        # Within-fold centering limits different expected-value offsets across outer models.
        g["centered_shap"] = g.shap_value - g.groupby("fold").shap_value.transform(
            "mean"
        )
        edges = np.unique(g.feature_value.quantile(np.linspace(0, 1, 7)).to_numpy())
        g["bin"] = pd.cut(g.feature_value, edges, include_lowest=True, labels=False)
        for bn, h in g.groupby("bin"):
            gm = h.groupby("Sampling_Group").centered_shap.mean()
            foldmeans = h.groupby("fold").centered_shap.mean()
            vals = gm.to_numpy()
            rng = np.random.default_rng(SEED)
            bb = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(axis=1)
            lo, hi = np.quantile(bb, [0.025, 0.975])
            bins.append(
                dict(
                    target=target,
                    validation=validation,
                    feature=feature,
                    bin=int(bn),
                    x_min=h.feature_value.min(),
                    x_max=h.feature_value.max(),
                    x_median=h.feature_value.median(),
                    n=len(h),
                    groups=len(gm),
                    folds=h.fold.nunique(),
                    group_equal_centered_shap=gm.mean(),
                    cluster_ci_low=lo,
                    cluster_ci_high=hi,
                    fold_mean_min=foldmeans.min(),
                    fold_mean_max=foldmeans.max(),
                )
            )
        rhos = []
        for fold, h in g.groupby("fold"):
            rho = h.feature_value.rank().corr(h.shap_value.rank())
            rhos.append(rho)
            slopes.append(
                dict(
                    target=target,
                    validation=validation,
                    feature=feature,
                    fold=fold,
                    n=len(h),
                    groups=h.Sampling_Group.nunique(),
                    spearman_x_shap=rho,
                )
            )
    pd.DataFrame(bins).to_csv(OUT / "oof_feature_response_bins.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUT / "oof_feature_direction_by_fold.csv", index=False)
    s.to_csv(OUT / "oof_selected_feature_values.csv", index=False)
    manifest = dict(
        seed=SEED,
        bootstrap_replicates=B,
        raw_scenario_rows=len(raw),
        selected_scenario_rows=len(d),
        fixed_common_n=len(fixed),
        fixed_common_groups=d[d.fixed_common_support].Sampling_Group.nunique(),
        source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        shap_sha256=hashlib.sha256(shp_src.read_bytes()).hexdigest(),
        notes=[
            "Fixed existing models; no retraining.",
            "10/25/50 percent denote path position, not percent emissions removed.",
            "Support means all four source descriptors within training marginal min/max; not joint support.",
            "Bootstrap resamples sampling groups, preserving their observations; conditional on fixed models.",
            "Intervals are pointwise descriptive, not multiplicity-adjusted or full prediction uncertainty.",
            "Group-equal and leave-one-group-out summaries do not retrain models.",
            "SHAP bins are centered within outer folds and group-balanced within bins; they are attribution descriptions, not causal partial-dependence curves.",
            "No physical desorption threshold, flux, or sediment release is estimated.",
        ],
    )
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    execution = dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={
            src.name: hashlib.sha256(src.read_bytes()).hexdigest(),
            shp_src.name: hashlib.sha256(shp_src.read_bytes()).hexdigest(),
        },
        outputs={
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in output_names
            if name != "postprocessing_execution_manifest.json"
        },
        algorithm="fixed_model_paired_class_responses_and_oof_attribution",
    )
    (OUT / "postprocessing_execution_manifest.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(
        result.query(
            "scope=='fixed_all_paths' and path_pct==50 and component in ['PFSA','PFCA','Ratio']"
        )[
            [
                "path",
                "sea",
                "component",
                "n",
                "groups",
                "geometric_change_pct",
                "ci_low_pct",
                "ci_high_pct",
                "group_equal_change_pct",
                "loo_min_pct",
                "loo_max_pct",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    args = parse_arguments()
    run(args.scenario_input, args.shap_input, args.output_dir)
