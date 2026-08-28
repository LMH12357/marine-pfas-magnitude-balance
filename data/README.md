# Data files used by the analysis release

## Fixed inputs included in GitHub

### Source boundary

- `fixed_inputs/source_boundary/Step3_MonteCarlo_Flux_2018.csv`: fixed compound-resolved 2018 river-flux realization used in the manuscript.
- `fixed_inputs/source_boundary/Step4_Coastline_Source_Load_2018.csv`: deterministic coastline source descriptors derived from Step 3.
- `provenance.json` and `README.md`: hashes and historical-seed explanation.

### Model input

- `fixed_inputs/model/Long_Chain_Model_Feature_Table.csv`: exact metadata-migrated input table used for primary and validation models; SHA-256 `0057c51827e31e01745ffec19141b48ccd1c2f1332df8219b40a2fc3a7be267d`.

### Reference data

- `reference/source_transition/`: literature-extracted use data, compound grouping and release factors used to derive the bounded source-transition multipliers.
- `reference/geospatial/china_marginal_sea_regions.geojson`: canonical sea-region reference.
- `reference/geospatial/china_coastline_boundary.shp/.shx`: coastline geometry used for source routing.

## Compact fixed outputs

- `fixed_outputs/primary_model/`: final model package and compact performance, reconstructed-output, feature-block and SHAP-stability summaries.
- `fixed_outputs/boundary_sensitivity/`: boundary-exclusion performance summary.
- `fixed_outputs/source_attenuation_boundary/`: complete prespecified source-boundary products.
- `fixed_outputs/source_attenuation_inference/Scenario_Prediction_Summary.csv`: final scenario summary, including paired back-transformed total long-chain PFAS burden changes and cluster-bootstrap intervals.
- `fixed_outputs/compositional_validation/` and `magnitude_balance_validation/`: compact SI validation summaries.

These fixed outputs support exact numerical comparison. The analysis scripts can regenerate the full, larger intermediate files.

## External inputs not duplicated

Place author-shared raw files under `data/raw/` when rerunning the corresponding upstream steps:

- `DATA.xlsx`;
- `100_Rivers_2018.xlsx`;
- `estuary.xlsx`;
- `Sediment TOC_metadata.xlsx`;
- optional `China Sediment Bulletin.xlsx`;
- provider-specific NASA, CMEMS and GloFAS NetCDF files/directories named in the extraction scripts.

The harmonized observation workbook should be deposited and cited through the manuscript's data repository. Large third-party satellite products remain subject to provider access and redistribution conditions.
