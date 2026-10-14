# Marine PFAS analysis code

Version 2.0.0 reconstructs environmental and river-source covariates, fits marine
PFAS concentration and composition models, and evaluates source-attenuation
scenarios across China's marginal seas. The release includes research analysis
code, configuration templates, tests and reference inputs. Figure-layout and
image-export code are maintained separately.

This version uses the corrected observation compilation and shared coastal
routing. The verified primary run contains 751 eligible observations from
22 sampling groups. The common comparison population across the specified
attenuation paths contains 606 observations from 18 groups. These counts are
properties of that run; the scripts calculate counts from supplied inputs.

## Version and data provenance

The previous public version is commit
`6685c27e34602e4c207d1316d645cbc24c204499` (v1.1.0). Its fixed numerical inputs
and outputs remain available in Git history. They are not current-release
defaults. Corrected source and model products are provided through the separate
research-data package or regenerated using the commands below.

`release_metadata.json` identifies the corrected input hashes, scientific-code
hashes and preceding version. `docs/release_validation_v2_0_0.json` records the
verification scope. No raw Excel workbook or large provider product is included
in this repository. The data-access information is maintained in `data/README.md`.

## Installation

The verified scientific runtime is Python 3.13.7. Install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
```

The environmental extraction modules additionally use the dependencies listed
in `requirements_environment.txt`. Provider credentials remain local.

## 1. Environmental covariates

Set the corrected observation workbook, TOC reference and provider-product paths
in a copy of `environment_config.example.json`. Run from the repository root:

```sh
python code/01_data_preparation/01_environmental_covariates/run_environment.py --config environment_config.local.json
```

This stages temperature, salinity, pH, Chl-a, POC, SPM, DO, sediment TOC, bottom
currents and mixed-layer depth, with per-record provenance. Source-reported
values are retained. A verified, backed-up workbook update is available through
`--writeback`. `--resume` reuses a checkpoint only when inputs, settings and code
match.

## 2. River and coastal-source covariates

Set river/estuary workbooks, discharge products, output directory and sampling
settings in a copy of `source_config.example.json`:

```sh
python code/01_data_preparation/02_source_covariates/run_sources.py --config source_config.local.json
```

The sequence exports annual discharge, regional concentration profiles,
compound-resolved fluxes and coastline source descriptors. The current source
reconstruction uses seed 42 and 1,000,000 Monte Carlo draws per region. It is
identified separately from the preceding unseeded historical realization.
Each source and its compound fluxes remain identifiable after routing.

## 3. Primary analysis and validation

Set corrected input and reference paths in a copy of
`analysis_config.example.json`. Paths in this file resolve relative to the
configuration file. The complete analysis runs in an isolated output directory:

```sh
python run_analysis.py --config analysis_config.local.json --output-dir outputs/analysis --mode production --n-jobs 8
```

The runner performs 11 stages in dependency order:

1. Shared land-aware coastal routing and frozen attachment distances.
2. Observation metadata, source features and PFAS group targets.
3. Grouped/spatial model selection, validation and OOF SHAP.
4. Compound-resolved source-attenuation boundary construction.
5. Paired scenario predictions and inverse concentration transformation.
6. Left-censoring and preprocessing sensitivity.
7. Compositional validation.
8. Magnitude-Balance representation validation.
9. Regional class-response and environmental attribution summaries.
10. Within-group attribution contrasts.
11. Mixed-source composition diagnostics.

Existing completed stages are reused only when code, inputs and product hashes
match. Use a fresh output directory when the generation changes. `--mode smoke`
tests software integration with reduced settings; its estimates are not
publication results.

The PFAS dictionaries distinguish the full concentration compilation from the
19-compound model panel and the compound-resolved river inventory. Precursor or
replacement measurements in a compilation are not automatically added to the
Magnitude-Balance model groups.

Source routing uses the same frozen graph and attachments in training,
validation and scenario inference. Reviewed receiving-water anchors, land
barriers, source-preservation checks and provenance are defined in
`code/02_modeling_and_validation/config/coastal_routing.json`.

## 4. Regional-response diagnostics

After the production analysis, run the supplementary diagnostic stages:

```sh
python code/03_evidence_strengthening/regional_selectivity_analysis.py prepare --code-root code --production-root outputs/analysis --output-root outputs/regional_selectivity
python code/03_evidence_strengthening/regional_selectivity_analysis.py stability --code-root code --production-root outputs/analysis --output-root outputs/regional_selectivity
python code/03_evidence_strengthening/regional_selectivity_analysis.py interactions --code-root code --production-root outputs/analysis --output-root outputs/regional_selectivity
```

The calculations retain the original compound-source transformations and
environmental settings. Outputs include dense attenuation paths, an alternative
group-tuned XGBoost composition model, training-group deletion refits, paired
Tree SHAP interaction contributions and cross-group feature-distance summaries.
The production model is preserved. Seeds, the evaluated population and input
hashes are recorded in separate provenance files. No figure or document is
generated by these commands.

See `code/03_evidence_strengthening/regional_selectivity_README.txt` for exact
stage requirements, resume behaviour and diagnostic interpretation.

## Reproducibility

```sh
python -m unittest discover -s tests -v
```

Tests use controlled fixtures to exercise schema, source aggregation, protected
workbook updates, routing barriers, scenario transformations, model interfaces
and execution provenance. The completed production run is independently checked
against its frozen inputs and products. Full production execution requires the
author-shared workbooks and specified third-party data.

`SHA256SUMS.txt` provides the release-file integrity manifest. Details belong in
the accompanying research documentation rather than conversational code comments.
The preceding licensing status is retained; no new licence is assigned here.
