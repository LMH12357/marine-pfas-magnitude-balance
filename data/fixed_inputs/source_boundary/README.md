# Fixed 2018 source inputs

These two CSV files are the exact derived inputs used by the downstream model
and source-attenuation analyses. They are archived so the reported results do
not depend on regenerating a different Monte Carlo realization.

- `Step3_MonteCarlo_Flux_2018.csv` is the compound-resolved river-flux tensor.
- `Step4_Coastline_Source_Load_2018.csv` is its deterministic aggregation into
  total legacy loading and three source-composition ratios.

The original Step 3 run used 1,000 draws per node but did not record a random
seed. This limitation is stated in `provenance.json`; it is not retroactively
assigned a seed. The included Step 3 script uses an explicit default seed of 42
for any new run and writes a manifest containing the seed and input/output
hashes. New runs are sensitivity regenerations and must not silently replace
the fixed files used for the manuscript.
