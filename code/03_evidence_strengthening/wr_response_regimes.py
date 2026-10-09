"""Calculate within-group attribution contrasts with optional exploratory diagnostics."""

from pathlib import Path
import argparse
import os
import json
import hashlib
import numpy as np
import pandas as pd


def parse_arguments():
    repository = (
        Path(os.environ.get("PFAS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        .expanduser()
        .resolve()
    )
    output_root = (
        Path(os.environ.get("PFAS_OUTPUT_ROOT", repository / "outputs"))
        .expanduser()
        .resolve()
    )
    parser = argparse.ArgumentParser(
        description="Calculate within-group OOF attribution contrasts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=output_root
        / "evidence_strengthening"
        / "oof_selected_feature_values.csv",
        help="Selected OOF feature values exported by wr_evidence_strengthening.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "evidence_strengthening",
        help="Destination for contrast outputs and their provenance.",
    )
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Also fit and export the historical exploratory curve diagnostics.",
    )
    return parser.parse_args()


def fit(x, y, w, knots=None):
    candidates = [None] if knots is None else knots
    best = None
    for k in candidates:
        a = (
            np.column_stack([np.ones(len(x)), x])
            if k is None
            else np.column_stack([np.ones(len(x)), x, np.maximum(x - k, 0)])
        )
        coef = np.linalg.lstsq(a * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        loss = np.average((y - a @ coef) ** 2, weights=w)
        if best is None or loss < best[0]:
            best = (loss, k, coef)
    return best


def predict(x, model):
    _, k, c = model
    a = (
        np.column_stack([np.ones(len(x)), x])
        if k is None
        else np.column_stack([np.ones(len(x)), x, np.maximum(x - k, 0)])
    )
    return a @ c


def run(input_csv: Path, output_dir: Path, include_exploratory: bool = False):
    input_csv = input_csv.expanduser().resolve()
    OUT = output_dir.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(
            "Missing selected OOF values: "
            + str(input_csv)
            + ". Run wr_evidence_strengthening.py first or supply --input explicitly."
        )
    output_names = (
        "within_group_attribution_contrasts.csv",
        "attribution_contrast_manifest.json",
    )
    if include_exploratory:
        output_names = ("exploratory_attribution_knot_stability.csv",) + output_names
    if any(input_csv == OUT / name for name in output_names):
        raise ValueError(
            "An output would overwrite the input; select a separate --output-dir."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(input_csv)
    rows = []
    contrasts = []
    for (target, val, feature), g in d.groupby(["target", "validation", "feature"]):
        g = g.copy()
        g["y"] = g.shap_value - g.groupby("fold").shap_value.transform("mean")
        native = g.feature_value.to_numpy()
        y = g.y.to_numpy()
        groups = g.Sampling_Group.to_numpy()
        rng = np.random.default_rng(20261007)
        ug, gi = np.unique(groups, return_inverse=True)
        if include_exploratory:
            islog = feature == "Diffused_Total_Legacy_Force"
            xt = np.log10(native) if islog else native.copy()
            center = xt.mean()
            scale = xt.std()
            x = (xt - center) / scale
            sizes = g.groupby("Sampling_Group").size()
            w = np.array([1 / sizes[t] for t in groups])
            knots = np.unique(np.quantile(x, np.linspace(0.2, 0.8, 25)))
            m = fit(x, y, w, knots)
            cvlin = []
            cvhinge = []
            loo_knots = []
            for group in np.unique(groups):
                tr = groups != group
                te = ~tr
                kk = np.unique(np.quantile(x[tr], np.linspace(0.2, 0.8, 25)))
                lin = fit(x[tr], y[tr], w[tr])
                hinge = fit(x[tr], y[tr], w[tr], kk)
                cvlin.append(np.mean((y[te] - predict(x[te], lin)) ** 2))
                cvhinge.append(np.mean((y[te] - predict(x[te], hinge)) ** 2))
                loo_knots.append(hinge[1])
            unscale = lambda z: (
                10 ** (np.asarray(z) * scale + center)
                if islog
                else np.asarray(z) * scale + center
            )
            boot = []
            for _ in range(500):
                count = np.bincount(
                    rng.integers(0, len(ug), len(ug)), minlength=len(ug)
                )
                bw = w * count[gi]
                model = fit(x, y, bw, knots)
                boot.append(model[1])
            qlo, qhi = unscale(np.quantile(boot, [0.025, 0.975]))
            rows.append(
                dict(
                    target=target,
                    validation=val,
                    feature=feature,
                    x_transform="log10" if islog else "identity",
                    candidate_knot=float(unscale(m[1])),
                    bootstrap_stability_low=qlo,
                    bootstrap_stability_high=qhi,
                    leave_group_knot_min=float(unscale(min(loo_knots))),
                    leave_group_knot_max=float(unscale(max(loo_knots))),
                    descriptive_group_cv_mse_linear=np.mean(cvlin),
                    descriptive_group_cv_mse_hinge=np.mean(cvhinge),
                    relative_mse_improvement=1 - np.mean(cvhinge) / np.mean(cvlin),
                    lower_search_bound=float(unscale(knots[0])),
                    upper_search_bound=float(unscale(knots[-1])),
                    bootstrap_search_edge_fraction=np.mean(
                        np.isin(boot, [knots[0], knots[-1]])
                    ),
                )
            )
        else:
            for _ in range(500):
                rng.integers(0, len(ug), len(ug))
        # Within-sampling-group high-low attribution contrasts at pooled quartiles.
        q1, q3 = np.quantile(native, [0.25, 0.75])
        for label, column in [("fold_centered", "y"), ("original", "shap_value")]:
            lo = g[g.feature_value <= q1].groupby("Sampling_Group")[column].mean()
            hi = g[g.feature_value >= q3].groupby("Sampling_Group")[column].mean()
            paired = (hi - lo).dropna()
            if len(paired):
                v = paired.to_numpy()
                bootmean = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
                ci = np.quantile(bootmean, [0.025, 0.975])
                contrasts.append(
                    dict(
                        target=target,
                        validation=val,
                        feature=feature,
                        attribution=label,
                        low_upper=q1,
                        high_lower=q3,
                        paired_groups=len(v),
                        mean_high_minus_low=paired.mean(),
                        ci_low=ci[0],
                        ci_high=ci[1],
                        groups_negative=int((v < 0).sum()),
                        groups_positive=int((v > 0).sum()),
                    )
                )
    if include_exploratory:
        pd.DataFrame(rows).to_csv(
            OUT / "exploratory_attribution_knot_stability.csv", index=False
        )
    pd.DataFrame(contrasts).to_csv(
        OUT / "within_group_attribution_contrasts.csv", index=False
    )
    manifest = dict(
        input_file=input_csv.name,
        input_sha256=hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=20261007,
        include_exploratory=include_exploratory,
        exploratory_knot_bootstrap_replicates=500 if include_exploratory else 0,
        historical_rng_prefix_draws_per_feature_group=500,
        rng_prefix_distribution="integers(0, n_unique_groups, size=n_unique_groups)",
        rng_prefix_mode=(
            "knot_bootstrap_and_fit" if include_exploratory else "draws_only"
        ),
        quartile_bootstrap_replicates=10000,
        historical_rng_sequence_preserved=True,
        reported_quartile_calculation_preserved=True,
        full_historical_exploratory_replay=include_exploratory,
        outputs={
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in output_names
            if name != "attribution_contrast_manifest.json"
        },
    )
    (OUT / "attribution_contrast_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if include_exploratory:
        print(
            pd.DataFrame(rows)[
                [
                    "target",
                    "validation",
                    "feature",
                    "candidate_knot",
                    "bootstrap_stability_low",
                    "bootstrap_stability_high",
                    "relative_mse_improvement",
                    "bootstrap_search_edge_fraction",
                ]
            ].to_string(index=False)
        )
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    args = parse_arguments()
    run(args.input, args.output_dir, args.include_exploratory)
