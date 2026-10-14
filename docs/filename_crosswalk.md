# Analysis workflow and file roles

Version 2.0.0 adds the shared `code/pfas_schema.py`, `code/coastal_routing.py`,
upstream `run_environment.py` and `run_sources.py`, and root `run_analysis.py`.
The separate `regional_selectivity_analysis.py` diagnostic consumes completed
production products. The table below retains the sequential workflow names.

| Release path | Role |
|---|---|
| `code/01_data_preparation/01_environmental_covariates/01_marine_physicochemical_interpolation.py` | Interpolates marine physicochemical covariates. |
| `.../02_chlorophyll_a_extraction.py` | Extracts chlorophyll-a covariates. |
| `.../03_particulate_organic_carbon_extraction.py` | Extracts particulate organic carbon covariates. |
| `.../04_suspended_particulate_matter_extraction.py` | Extracts suspended particulate matter covariates. |
| `.../05_dissolved_oxygen_extraction.py` | Extracts dissolved oxygen covariates. |
| `.../06_sediment_toc_interpolation.py` | Interpolates sediment total organic carbon. |
| `.../07_hydrodynamic_extraction.py` | Extracts bottom-current components and mixed-layer depth. |
| `code/01_data_preparation/02_source_covariates/01_glofas_discharge_extraction.py` | Extracts river-discharge covariates. |
| `.../02_river_fingerprint_construction.py` | Constructs compound-resolved river-source profiles. |
| `.../03_monte_carlo_river_flux.py` | Propagates river-concentration uncertainty into compound-resolved fluxes. |
| `.../04_coastline_source_descriptors.py` | Aggregates river fluxes into routed coastal-source descriptors. |
| `code/02_modeling_and_validation/01_primary_model_training.py` | Trains RF/XGB models and exports out-of-fold performance, feature-block and SHAP results. |
| `.../02_source_attenuation_boundary.py` | Applies bounded, literature-derived source-class perturbations. |
| `.../_source_attenuation_core.py` | Provides shared source-attenuation transformations and validation utilities. |
| `.../03_source_attenuation_inference.py` | Predicts Magnitude and Balance under source attenuation and performs paired inverse transformation. |
| `.../04_preprocessing_sensitivity.py` | Evaluates alternative treatments of left-censored observations. |
| `.../05_compositional_validation.py` | Compares compositional coordinate representations and validation designs. |
| `.../06_magnitude_balance_validation.py` | Validates the selected Magnitude–Balance representation. |

Figure-layout and image-export routines are outside the scope of this analysis repository. All numerical transformations supporting the reported results, including the paired inverse transformation and percent change in total long-chain burden, are retained in the analysis scripts.
