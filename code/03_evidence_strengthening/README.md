# Existing-model evidence postprocessing

This directory adds journal-neutral postprocessing to the existing
Magnitude–Balance workflow. The two historical scripts consume saved paired scenario
predictions and held-out SHAP values. They do not retrain the primary
RF/XGBoost models, recompute SHAP, reconstruct source inventories, or edit
manuscripts.

The original working filenames are retained for provenance. Reported contrast
statistics, sampling constants and the historical random-number sequence are
preserved. Exploratory curve fitting is optional. Portable input/output
arguments, input-overwrite protection and execution-provenance records are
provided separately from the numerical calculations.

## Inputs and defaults

| Script | Argument | Default relative to the repository |
|---|---|---|
| `wr_evidence_strengthening.py` | `--scenario-input` | `outputs/03_source_attenuation_inference/Scenario_Paired_Predictions.csv` |
| `wr_evidence_strengthening.py` | `--shap-input` | `outputs/01_primary_model/Nested_OOF_SHAP_Values.csv` |
| `wr_evidence_strengthening.py` | `--output-dir` | `outputs/evidence_strengthening` |
| `wr_response_regimes.py` | `--input` | `outputs/evidence_strengthening/oof_selected_feature_values.csv` |
| `wr_response_regimes.py` | `--output-dir` | `outputs/evidence_strengthening` |
| `wr_response_regimes.py` | `--include-exploratory` | Off; only reported attribution contrasts are calculated by default |
| `source_ratio_diagnostic.py` | `--predictions` | `outputs/03_source_attenuation_inference/Scenario_Paired_Predictions.csv` |
| `source_ratio_diagnostic.py` | `--paired-shap` | `outputs/03_source_attenuation_inference/Scenario_Paired_SHAP.csv` |
| `source_ratio_diagnostic.py` | `--output-dir` | `outputs/source_ratio_diagnostic` |

`PFAS_PROJECT_ROOT`, `PFAS_DATA_ROOT` and `PFAS_OUTPUT_ROOT` follow the
existing repository path convention. Explicit CLI paths can point to archived
outputs anywhere accessible; no workstation path is embedded in the scripts.

The scripts use only the upstream generated-output locations shown above.
They never silently substitute archived fixed outputs. Source files must have
adjacent `Run_Manifest.json` files with matching `data_sha256`,
`feature_table_sha256` and `pfas_schema_sha256`. Observation identities,
metadata, cardinality and selected-model uniqueness are checked before
combining outputs. Explicit CLI paths remain available, but do not bypass
these generation checks.

The paired CSV is the full output of
`code/02_modeling_and_validation/03_source_attenuation_inference.py`.
It must include reference rows and the synchronized/class-resolved
10%, 25% and 50% path scenarios, their row/group/sea identifiers, source-domain
flags, paired class concentrations and coordinate/log changes.

The SHAP input is the full row-level output of
`code/02_modeling_and_validation/01_primary_model_training.py`.
It needs `row_index`, `target`, `validation`, `feature_block`, `fold`,
`feature`, `feature_value`, `shap_value` and `selected_in_fold`.
Do not substitute `SHAP_Importance_Stability.csv`: that summary cannot recover
row-level attribution contrasts.

Full paired/OOF tables may be external to a compact distribution. If a default
input is absent, the script fails clearly before creating outputs. Supply the
existing full files using CLI arguments, or use the documented upstream
workflow to produce them. These postprocessing scripts do not silently train
models or fabricate missing input rows.

## Commands

With the upstream default output locations:

```sh
python code/03_evidence_strengthening/wr_evidence_strengthening.py
python code/03_evidence_strengthening/wr_response_regimes.py
```

With existing outputs from the repository's upstream default output directories:

```sh
python code/03_evidence_strengthening/wr_evidence_strengthening.py --scenario-input outputs/03_source_attenuation_inference/Scenario_Paired_Predictions.csv --shap-input outputs/01_primary_model/Nested_OOF_SHAP_Values.csv --output-dir outputs/evidence_strengthening
python code/03_evidence_strengthening/wr_response_regimes.py --input outputs/evidence_strengthening/oof_selected_feature_values.csv --output-dir outputs/evidence_strengthening
```

The commands above read existing files. They do not invoke upstream training
or SHAP calculation. A custom first-script output directory requires supplying
its `oof_selected_feature_values.csv` explicitly to the second script.

Optional full historical exploratory replay, in a separate output directory:

```sh
python code/03_evidence_strengthening/wr_response_regimes.py --include-exploratory --output-dir outputs/evidence_strengthening_exploratory
```

## Outputs and fixed calculation contract

The first script retains:

- `paired_class_responses.csv`;
- `regional_class_response_summary.csv`;
- `regional_response_patterns.csv`;
- `oof_feature_response_bins.csv`;
- `oof_feature_direction_by_fold.csv`;
- `oof_selected_feature_values.csv`;
- `analysis_manifest.json`.

It adds `postprocessing_execution_manifest.json`, recording script/input/output
hashes without replacing the historical analysis manifest.

The fixed analysis uses seed **20261007**, **10,000** group-bootstrap resamples,
the same six synchronized/class-resolved 10/25/50 scenarios, and their common
source-feature minimum–maximum domain. Observation and sampling-group counts
are recalculated from the current inputs, not copied from previous runs. Class-specific
percentage changes are derived from mean paired log changes. Group-equal and
leave-one-group-out sensitivities are postprocessing, not new primary-model fits.

The second script writes `within_group_attribution_contrasts.csv` by default.
Only `--include-exploratory` additionally fits and writes
`exploratory_attribution_knot_stability.csv` (descriptive shape diagnostics).

It adds `attribution_contrast_manifest.json`, recording the selected-input hash,
script/output hashes and the retained random-number sequence.

### Historical random-number sequence

For each target/validation/feature group, the generator starts at seed
20261007 using explicit `numpy.random.Generator(PCG64)` and performs **500** successive
`rng.integers(0, n_unique_groups, n_unique_groups)` calls before the **10,000**
fold-centered and **10,000** original-scale quartile draws, in that order.

The default mode only advances those 500 preceding random draws; it does not
fit linear/hinge curves or calculate knot diagnostics. The optional
`--include-exploratory` mode performs the complete historical curve-fitting
flow, including its knot bootstrap. Both modes therefore start the unchanged
quartile calculation at the same random-number state. Reseeding that step or
omitting the preceding random draws would change the frozen intervals.

The manifest explicitly distinguishes `draws_only` from
`knot_bootstrap_and_fit`, with zero exploratory bootstrap fits recorded for the
default mode. The default mode does not delete exploratory files left by an
earlier optional run; use separate output directories to avoid mixing modes.

The exploratory section fits low-dimensional descriptive linear/hinge curves
to existing attributions. These are not the primary RF/XGBoost models.
Knot estimates are **not physical fronts, adsorption/desorption thresholds,
or measured sediment-release rates**. Reported TOC/MLD quartile results use
existing OOF SHAP. Quartiles are pooled across the selected observations;
group-specific high-minus-low means are calculated only for groups with
observations in both tails. Eligible group counts are exported dynamically.

## Verification

Numerical reference checks, software versions and input hashes are recorded
separately in [the verification record](../../docs/evidence_postprocessing_verification.md).

## Mixed-source ratio diagnostic

`source_ratio_diagnostic.py` is an additional diagnostic maintained separately
from the two historical postprocessing scripts. It reads existing paired
predictions and paired SHAP feature-value exports, and summarizes the mixed-source
LC-PFSA/PFCA descriptor change at the synchronized 50% path position on the same
six-scenario common source-feature domain.

```sh
python code/03_evidence_strengthening/source_ratio_diagnostic.py
```

Use `--predictions`, `--paired-shap` and `--output-dir` to point to existing
external outputs. The script writes `source_ratio_descriptor_summary.csv` and
`source_ratio_diagnostic_manifest.json`.

The boundary design fixes each source node's LC-PFSA/PFCA ratio. However,
source-node total-load weights can change, so the receiving-location
mixed-source ratio descriptor need not remain identical. This is a weighting
diagnostic, not proof that environmental covariates alone caused a response,
nor a sediment exchange-flux calculation. It does not refit primary models or
recompute SHAP. Independent numerical and input-integrity regression tests
are maintained under `tests/`.
