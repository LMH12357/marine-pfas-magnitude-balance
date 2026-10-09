# Numerical verification of evidence post-processing

Verification date: 2026-10-09. Environment: Python 3.13.7, NumPy 2.3.3 and
pandas 2.3.2. The checks used archived predictions and held-out attributions;
the primary RF/XGB models, SHAP calculations and source inventory were not rerun.

## Reported-output comparisons

The seven retained class-response/attribution outputs and the within-group
contrast CSV reproduced the archived files byte-for-byte by SHA-256: **8/8**
default outputs. The complete 40-row contrast table, including point estimates
and confidence intervals, also passed an exact dataframe comparison.

Both default and optional exploratory modes reproduced that same contrast CSV.
The optional historical replay additionally reproduced the archived 24-row knot
diagnostic CSV. Default processing performs no exploratory fitting, while
preserving the preceding 500 random draws and the reported quartile calculations.

The fixed common domain contains **714 observations and 21 sampling groups**.
Reported within-group contrasts retain **9 eligible TOC groups and 8 MLD groups**.
No physical threshold or sediment exchange flux is estimated by these scripts.

## Pooled sensitivity export

The new pooled export was compared with the 30-row historical Table S5 reference:
all keys and sample sizes matched, and the maximum parsed metric difference was
4.44e-16. The regenerated CSV was also byte-identical to the historical reference.
This original-group-ratio evaluation does not replace the fold-mean summary or
the primary Magnitude–Balance model.

## Source-ratio diagnostic

On the same six-scenario common domain, synchronized attenuation at the 50%
path position increased the mixed-source LC-PFSA/PFCA descriptor by
0.149417535–2.038997272%. Mean aqueous Balance changes remained positive in
the Bohai Sea and negative in the South China Sea. The compact summary and
input hashes are recorded under `data/fixed_outputs/evidence_strengthening/`.

## Software checks

- **17 automated tests passed**, including the archived Table S5 reference,
  default CLI export, original fixed-input hashes and input/output collision guards.
- Four upstream-input default/fallback cases were checked.
- Five newly added Python files were consistently formatted; their ASTs were
  unchanged by formatting.
- Historical execution records were retained with their original identities;
  new execution manifests record the new script hashes separately.

To include the archived pooled-reference check when running the test suite,
set `PFAS_POOLED_OOF_REFERENCE_DIR` to the archive directory containing
`Reviewer_Out_of_Fold_Predictions.csv` and `Reviewer_Pooled_OOF_Performance.csv`.
The CLI-wiring test uses explicitly labelled synthetic fixtures and prevents
model fitting. These fixtures are not research observations.

## Verified input hashes

| Input | SHA-256 |
|---|---|
| Scenario_Paired_Predictions.csv | `5b45aededaa483e48953751a622d5aa3d9b2edee6a04f07a59b1a9f7bff33b54` |
| Nested_OOF_SHAP_Values.csv | `f601d40d4aea6ef2252e767e2fab72b0dc0a85a74f3fea2eff6771faae6b1d39` |
| oof_selected_feature_values.csv | `b1aeff5a102b13bcf4e5f2467f020c20d3630776304a6d018cf31a4335e8be7b` |

The three-design composition references include recovered canonical tables at
their original displayed precision. Their manifest distinguishes those records
from the unchanged spatial-only execution archive; full-precision historical
SG/LOGO execution files were not recovered or fabricated.
