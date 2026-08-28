# Reproducibility audit (28 August 2026)

## Historical Monte Carlo output

The source reconstruction used 1,000 Monte Carlo draws per source node, but the original run did not record a random seed. A seed was therefore not assigned retroactively and the source inventory used in the analysis was not regenerated.

The exact downstream inputs are archived under `data/fixed_inputs/source_boundary/`:

| File | SHA-256 | Role |
|---|---|---|
| `Step3_MonteCarlo_Flux_2018.csv` | `7fe3ab1d98e9b43d8e48dc9110a5e4d5955c6f34fd6876457c6ce2d4404463ff` | Compound-resolved fixed 2018 river-flux inventory. |
| `Step4_Coastline_Source_Load_2018.csv` | `38b168027580a0469922fa90ac76030e93621ffd3c76e7c8509f7b55f23617c3` | Deterministic source descriptors used to construct routed model features. |

The Step 3 hash matches the scenario manifests. The archived Step 4 file was rebuilt from the archived Step 3 file and matched byte for byte.

## Deterministic regeneration

`03_monte_carlo_river_flux.py`:

- uses `numpy.random.default_rng`;
- exposes `--seed` (default 42) and `--iterations` (default 1,000);
- records the seed, iteration count and input/output hashes in a JSON manifest;
- accepts configurable source, intermediate and output paths.

Two independent seed-42 runs on the same inputs produced identical SHA-256 hashes (`0c015b7d0cf88f4b104a0b58963a338d4dd56b846bdff3d249d2ad2da37fc0f6`). These deterministic runs verify the regeneration procedure; they do not replace the fixed realization used in the analysis.

## Path portability

Contributor-specific absolute paths were removed from the source tree. Scripts resolve paths from the repository by default and can be redirected through:

- `PFAS_PROJECT_ROOT` for the repository root;
- `PFAS_DATA_ROOT` for source data;
- `PFAS_OUTPUT_ROOT` for generated products;
- script-specific command-line paths where available.

The primary model and scenario scripts default to the archived Step 3 and Step 4 source inputs. The fixed numerical results were not regenerated during path-portability edits.

## Verification

- Python compilation succeeded for the complete code tree.
- No contributor-specific absolute path remains in tracked Python, JSON, Markdown, text or CSV files.
- No unseeded NumPy sampling call remains in tracked Python code.
- All release-contract tests passed.
