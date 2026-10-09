# Supplementary Table 7. Structured validation of compositional representations and joint multi-output learning for marine PFAS group diagnostic ratios

## (a) Out-of-fold performance

| Validation | Target representation | N | Short-chain/long-chain R2 | Short-chain PFSA/PFCA R2 | Long-chain PFSA/PFCA R2 | Aitchison RMSE |
|---|---|---:|---:|---:|---:|---:|
| Sampling-grouped five-fold | Separate log GDR | 789 | -0.010 | -0.129 | 0.525 | 2.040 |
| Sampling-grouped five-fold | Independent ILR coordinates | 789 | -0.112 | -0.129 | 0.526 | 1.990 |
| Sampling-grouped five-fold | Joint multi-output ILR | 789 | -0.047 | -0.121 | 0.462 | 2.012 |
| Sea-stratified spatial-block five-fold | Separate log GDR | 789 | -0.035 | -0.025 | 0.444 | 1.602 |
| Sea-stratified spatial-block five-fold | Independent ILR coordinates | 789 | 0.000 | -0.025 | 0.444 | 1.574 |
| Sea-stratified spatial-block five-fold | Joint multi-output ILR | 789 | -0.003 | -0.024 | 0.499 | 1.544 |
| Leave-one-sampling-group-out | Separate log GDR | 789 | -0.021 | -0.120 | 0.488 | 1.965 |
| Leave-one-sampling-group-out | Independent ILR coordinates | 789 | -0.072 | -0.122 | 0.528 | 1.998 |
| Leave-one-sampling-group-out | Joint multi-output ILR | 789 | -0.036 | -0.107 | 0.472 | 1.977 |

## (b) Paired sampling-group-cluster bootstrap differences for joint multi-output ILR relative to separate log-GDR modelling

| Validation | Delta short-chain/long-chain R2 (95% CI) | Delta short-chain PFSA/PFCA R2 (95% CI) | Delta long-chain PFSA/PFCA R2 (95% CI) | Delta Aitchison RMSE (95% CI) |
|---|---:|---:|---:|---:|
| Sampling-grouped five-fold | -0.037 (-0.490, 0.048) | 0.008 (-0.032, 0.081) | -0.063 (-0.160, 0.014) | -0.028 (-0.098, 0.031) |
| Sea-stratified spatial-block five-fold | 0.032 (-0.064, 0.126) | 0.001 (-0.017, 0.081) | 0.056 (-0.023, 0.122) | -0.058 (-0.101, 0.000) |
| Leave-one-sampling-group-out | -0.015 (-0.204, 0.011) | 0.013 (-0.049, 0.106) | -0.016 (-0.063, 0.008) | 0.013 (-0.051, 0.067) |

Note: Four positive group totals were constructed for short-chain PFSA, short-chain PFCA, long-chain PFSA and long-chain PFCA and closed to a four-part composition. No zero replacement or pseudocount was required because every group total was positive in all 789 records. The three orthonormal isometric log-ratio (ILR) coordinates represented chain-length balance, short-chain headgroup balance and long-chain headgroup balance. RF and XGB were selected using training-only inner cross-validation within every outer fold. Reported values are based on unmodified out-of-fold observations. Paired confidence intervals were obtained from 1,000 bootstrap resamples with sampling group as the clustering unit. Positive Delta R2 and negative Delta Aitchison RMSE favour joint ILR modelling. Small improvements under sea-stratified spatial validation were not reproduced under sampling-grouped or leave-one-sampling-group-out validation; therefore, the compositional multi-output architecture was retained as a sensitivity analysis and did not replace the primary long-chain PFSA/PFCA model.
