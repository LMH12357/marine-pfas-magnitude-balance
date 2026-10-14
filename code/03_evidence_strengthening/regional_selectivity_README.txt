Regional compositional response diagnostics
==========================================

This module extends the frozen PFAS source-attenuation analysis with dense source
paths, a grouped-tuned alternative Balance model, training-group deletion fits,
paired Tree SHAP interaction decompositions and feature proximity diagnostics.
It imports the production analysis functions rather than reimplementing routing,
compound grouping, source aggregation or Magnitude-Balance transformations.

Required input directories
--------------------------
CODE_ROOT contains pfas_schema.py, coastal_routing.py and the existing
02_modeling_and_validation scripts. PRODUCTION_ROOT contains the frozen routing
bundle, primary model and feature table, source boundary products, paired
scenario predictions, and evidence_strengthening/paired_class_responses.csv.
OUTPUT_ROOT must be outside both source directories. Use trusted model packages;
the stored estimators are loaded with joblib.

Example commands (replace uppercase paths with actual directories)
-----------------------------------------------------------------
python regional_selectivity_analysis.py prepare --code-root CODE_ROOT --production-root PRODUCTION_ROOT --output-root OUTPUT_ROOT
python regional_selectivity_analysis.py stability --code-root CODE_ROOT --production-root PRODUCTION_ROOT --output-root OUTPUT_ROOT
python regional_selectivity_analysis.py interactions --code-root CODE_ROOT --production-root PRODUCTION_ROOT --output-root OUTPUT_ROOT

The all stage runs those three stages in sequence. The default dense grid contains
0 through 50 percent path positions. These positions scale the evidence-based
compound multipliers by the original power rule; a 50 percent path position does
not denote a 50 percent source-flux reduction. Every coupled source feature is
rebuilt from the perturbed compound inventory and the original routing network.
Environmental covariates remain fixed.

Compact verification (without alternative tuning or deletion refits)
-------------------------------------------------------------------
python regional_selectivity_analysis.py prepare --code-root CODE_ROOT --production-root PRODUCTION_ROOT --output-root CHECK_ROOT --path-pcts 0,10,25,50 --skip-proximity
python regional_selectivity_analysis.py verify --code-root CODE_ROOT --production-root PRODUCTION_ROOT --output-root CHECK_ROOT --path-pcts 0,10,25,50 --skip-proximity --reference-root REFERENCE_ROOT

The prepare stage always verifies selected 0, 10, 25 and 50 percent positions
against the frozen paired scenario predictions when those positions are present.
The verify stage additionally compares all exported regional response and source
summary columns against an existing dense-response CSV. It checks the numerical
results rather than serialized file bytes.

Resume and provenance
---------------------
All run inputs and Python analysis files are hashed into run_provenance.json using
paths relative to the two input roots. Sampling-group and observation counts are
computed from input data. The target population is selected exclusively from the
fixed_common_support flags in paired_class_responses.csv and kept unchanged for
all stages. The module rejects absent target rows or incompatible feature arrays.

Add --resume to verify and reuse completed stages. Completed stage manifests
record each output hash; missing or changed files are rejected. An incomplete
stage can be rerun with the same invocation and input hashes. Changed inputs,
analysis code or grid require a new output directory. Changes in --workers do not
change scientific inputs; all estimators retain seed 42. The original production
model hash is checked again at each stage completion.

Statistical interpretation
--------------------------
The alternative XGBoost Balance model uses the original full grouped grid search,
five-fold sampling-group inner validation and normalized RMSE scoring. It does not
replace the production Random Forest model. Each training group is deleted once
for each architecture. Full-fit hyperparameters are held fixed and target scaling
is recalculated on retained training rows. Every refit evaluates the same paired
scenario changes in the fixed target population. These refits quantify fitting
sensitivity, not independent replications, a confidence interval or new unbiased
out-of-sample performance.

Tree SHAP interaction matrices use raw output and tree_path_dependent training
backgrounds. Diagonal entries are main effects. Both symmetric off-diagonal
entries are summed once per pair. Scenario-minus-baseline contributions must sum
to the predicted Balance change within the stated numerical tolerance. These
terms decompose model responses and do not estimate physical interaction
coefficients or prove sediment exchange.

Proximity uses robust IQR-scaled Euclidean distance divided by sqrt(feature count).
The fifth-nearest observation must belong to a different sampling group. It is
not the fifth-nearest independent group. The baseline global 95th percentile is
a descriptive reference and does not exclude observations or define a calibrated
applicability domain. Source-only and all-feature distances are exported separately.

Exports
-------
dense_regional_responses.csv: regional paired coordinate and derived-output changes,
observation-weighted and equal-sampling-group summaries, source feature quantiles.
dense_source_features.npz: complete ordered scenario feature arrays and target mask.
frozen_result_reproduction.csv: frozen scenario numerical reproduction errors.
joint_proximity_diagnostic.csv and joint_proximity_distances.npz: distance diagnostics.
training_source_ratio_density.csv: short/long source ratio density by 0.05-wide bins.
model_stability_paths.csv: full-fit and every group-deletion regional paired response.
group_deletion_stability_summary.csv: ranges and sign counts across deletion fits.
alternative_XGB_Balance.pkl: estimator plus target scaling and feature contract;
no dynamically imported wrapper class is serialized.
To reconstruct alternate-model predictions, cast estimator.predict(X) to float64,
multiply by target_scale and add target_mean, retaining the original wrapper's
numerical scaling rule.
interaction_block_summary.csv: main and interaction block contributions.
interaction_blocks_by_sampling_group.csv: corresponding group summaries.
interaction_pair_summary.csv: all individual main and paired interaction terms.
interaction_additivity_check.csv: paired prediction reconstruction errors.

Dependencies: numpy, pandas, scipy, scikit-learn, xgboost, shap and joblib, together
with the versions and source modules required by the frozen production pipeline.
