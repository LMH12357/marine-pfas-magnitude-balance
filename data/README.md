# Data inputs for version 2.0.0

The current code release includes the reference inputs needed to define sea
regions, coastline geometry and literature-derived source-transition factors.
Corrected observation workbooks and generated analysis products are shared as
a separate versioned research-data package.

## Reference inputs

- `reference/geospatial/china_marginal_sea_regions.geojson`
- `reference/geospatial/china_coastline_boundary.shp` and `.shx`
- `reference/source_transition/cai_2025_table_s8_usage.csv`
- `reference/source_transition/cai_2025_gdr_mapping.csv`
- `reference/source_transition/cai_2025_application_release_factors.csv`

## Required external inputs

Configure the actual paths to these author-shared files and provider products:

- Corrected `DATA.xlsx`.
- `100_Rivers_2018.xlsx` and `estuary.xlsx`.
- `Sediment TOC_metadata.xlsx`.
- `China Sediment Bulletin.xlsx`.
- CMEMS, NASA and GloFAS products required by the selected extraction modules.

The environment/source example configurations document their roles. Large
third-party products are not redistributed. Provider identifiers and per-record
extraction details are recorded when the upstream calculations run.

## Generation and provenance

The verified corrected primary run uses 751 eligible observations, 22 sampling
groups and 80 source nodes. The corrected raw workbook, Step3 compound fluxes,
Step4 source loads and final model hashes are recorded in
`../release_metadata.json`. Those hashes identify the completed scientific
generation without including raw workbooks in this code repository.

Run `run_analysis.py` against corrected workbooks and the newly generated Step3
and Step4 products. Do not use v1.1.0 fixed numerical products as substitutes.
The preceding `fixed_inputs` and `fixed_outputs` remain accessible at commit
`6685c27e34602e4c207d1316d645cbc24c204499`.

The separate data record and figure source-data package must use these same
input/result identities. No DOI or accession is asserted until the associated
data record has actually been deposited.
