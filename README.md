# Marine PFAS analysis code

Analysis code and fixed numerical inputs for examining marine PFAS concentration magnitude and PFSA–PFCA composition under source attenuation in China's marginal seas.

This repository reproduces data preparation, model validation, source-boundary perturbation and numerical results. It intentionally excludes figure-layout and image-export code. Figures are presentation products; the calculations underlying their values are performed and exported by the analysis scripts included here.

## Analysis update

This revision extends the reproducible analysis at commit
`e0347a58d535f6bb56cc2a072aaa22b203dcaacd` with fixed-model evidence
post-processing, pooled sensitivity exports and compact numerical references.
The historical execution manifests are retained with their original identities;
new calculations have separate provenance records. The primary RF/XGB models,
historical Monte Carlo inputs and reported numerical results are unchanged.

## Repository structure

```text
code/
  01_data_preparation/
    01_environmental_covariates/
    02_source_covariates/
  02_modeling_and_validation/
  03_evidence_strengthening/
data/
  fixed_inputs/
  reference/
  fixed_outputs/
  raw/                    # author-shared raw inputs; not duplicated here
docs/
tests/
```

## Analysis order

### 1. Upstream covariates

The scripts in `code/01_data_preparation/` document the environmental and river-source reconstruction. Large third-party satellite products are not redistributed. The exact historical source realization used by the manuscript is fixed under `data/fixed_inputs/source_boundary/`.

### 2. Primary model

```bash
python code/02_modeling_and_validation/01_primary_model_training.py
```

The default input is the metadata-harmonized feature table used for the reported RF/XGB analysis. Default outputs are written to `outputs/01_primary_model/`. The script generates nested out-of-fold performance, reconstructed long-chain outputs, feature-block ablation, SHAP attribution, predictor provenance and the fitted model package.

### 3. Source-attenuation boundary

```bash
python code/02_modeling_and_validation/02_source_attenuation_boundary.py --overwrite
```

This applies the prespecified literature-derived class multipliers to the fixed 2018 compound inventory. Perturbation intensity is dimensionless and does not represent a date, probability or forecast horizon.

### 4. Paired source-attenuation inference

```bash
python code/02_modeling_and_validation/03_source_attenuation_inference.py \
  --model-package data/fixed_outputs/primary_model/Long_Chain_Magnitude_Balance_Model_Package.pkl \
  --scenario-directory outputs/02_source_attenuation_boundary \
  --overwrite
```

This script fixes environmental covariates, routes the altered source boundary, predicts Magnitude and Balance, performs the paired inverse transformation to total long-chain PFAS burden, calculates sampling-group-cluster bootstrap intervals and audits the supported perturbation domain. The burden percentages reported in the abstract and Fig. 4 are therefore generated in the analysis code, not in a plotting script.

### 5. Sensitivity and coordinate validation

```bash
python code/02_modeling_and_validation/04_preprocessing_sensitivity.py --analysis-mode validation
python code/02_modeling_and_validation/05_compositional_validation.py
python code/02_modeling_and_validation/06_magnitude_balance_validation.py
```

The first command requires `data/raw/DATA.xlsx` because it reconstructs censoring
treatments from the workbook. The compositional-validation command (05) also
requires that workbook to reconstruct its group targets and supplements its
predictors from the fixed model feature table. Only command 06 can use the fixed
feature table without first reading the raw workbook.

Command 04 now additionally writes `Sensitivity_Pooled_OOF_Performance.csv`
from its already generated held-out predictions. This evaluates original,
untransformed group ratios and is distinct from the existing per-fold mean
summary in `Sensitivity_Performance_Summary.csv`; neither summary replaces the
other. The pooled export does not refit or modify observations or predictions.

### 6. Existing-output evidence post-processing

```bash
python code/03_evidence_strengthening/wr_evidence_strengthening.py \
  --scenario-input outputs/03_source_attenuation_inference/Scenario_Paired_Predictions.csv \
  --shap-input outputs/01_primary_model/Nested_OOF_SHAP_Values.csv
python code/03_evidence_strengthening/wr_response_regimes.py
python code/03_evidence_strengthening/source_ratio_diagnostic.py
```

The historical `wr_` script names are retained for traceability. The first two
scripts reproduce fixed-common-domain
regional class responses and within-sampling-group TOC/mixed-layer-depth
quartile attribution contrasts, with the original 10,000-resample bootstrap
sequence. Exploratory knot diagnostics remain explicitly separate from the
reported contrasts and are not interpreted as physical thresholds.

The third script summarizes mixed-source long-chain PFSA/PFCA descriptors from
the existing paired feature exports. Its common domain requires all four source
features to remain within training minima/maxima in both designs at 10%, 25%
and 50% path positions. Fixed source-node ratios do not imply fixed mixed-source
descriptors, because attenuation can change relative total-load weights. The
diagnostic is not an isolated test of environmental causality or sediment flux.

The compact reference package does not contain the large, complete paired
prediction and OOF SHAP files. Use existing archived files through the explicit
input arguments, or first generate them with commands 01/03. None of the new
post-processing scripts silently retrains a model or recalculates SHAP when
inputs are missing. See `code/03_evidence_strengthening/README.md` for individual
inputs, outputs and provenance.

## Path configuration

No contributor-specific absolute paths are embedded in the release. Scripts resolve paths from the repository root and accept:

- `PFAS_PROJECT_ROOT` — repository root;
- `PFAS_DATA_ROOT` — data directory, default `<project>/data`;
- `PFAS_OUTPUT_ROOT` — generated-output directory, default `<project>/outputs`.

Raw workbooks and provider-specific files should be placed under `data/raw/` using the names in `data/README.md`.

## Fixed historical Monte Carlo realization

The manuscript's river-flux reconstruction used 1,000 Monte Carlo draws per source node, but the original run did not record a seed. A seed was not assigned retroactively. The exact downstream inputs are archived:

- `Step3_MonteCarlo_Flux_2018.csv`: SHA-256 `7fe3ab1d98e9b43d8e48dc9110a5e4d5955c6f34fd6876457c6ce2d4404463ff`;
- `Step4_Coastline_Source_Load_2018.csv`: SHA-256 `38b168027580a0469922fa90ac76030e93621ffd3c76e7c8509f7b55f23617c3`.

`03_monte_carlo_river_flux.py` now uses `numpy.random.default_rng`, exposes `--seed` (default 42) and records input/output hashes. Seeded reruns test reproducibility but do not replace the fixed historical realization used in the manuscript.

## Fixed outputs

`data/fixed_outputs/` contains compact reference outputs for numerical cross-checking, including model performance, feature-block/SHAP summaries, source-transition products, the back-transformed burden summary and validation summaries. It does not contain submission figures or figure-export code.

The new `Compositional_AllDesigns_Canonical_*` tables recover three-design
canonical figure-source summaries without relabeling the original spatial-only
execution manifest. Their source manifest states precision and field-coverage
boundaries. The `evidence_strengthening/` compact tables retain the reported
regional and attribution-contrast results and identify which rows support the
manuscript figure, rather than treating every exploratory result as a reported
finding.

## Verification

```bash
python -m unittest discover -s tests -v
```

The release tests compile every Python file, verify the fixed-input hashes, deterministically rebuild Step 4 from the fixed Step 3 inventory, resolve the source-transition evidence files and reject contributor-specific paths.

`SHA256SUMS.txt` is the package-level integrity manifest. See `docs/filename_crosswalk.md` for the analysis workflow and file roles.

No license is assigned in this package. The repository owner should select a license before public reuse. Add the immutable GitHub release/archival DOI to the manuscript Code Availability statement only after deposition.
