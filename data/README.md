# Data directory

**This directory is gitignored.** No data files are committed.

## Expected structure

```
data/
├── raw/          # Original inputs. Never edit.
├── interim/      # Intermediate artefacts from scripts (cleaned benchmarks, etc.)
├── processed/    # Per-species training datasets ready for modelling
└── outputs/      # Model artefacts (fitted models, predicted rasters)
```

## Getting the master table

The master table `combined_data_true_master.csv` is Olga's integrated
Petko et al. 2026 product. It lives on Kristian's laptop at:

    /Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv

To point the pipeline at a different location, either:
- edit `configs/paths.yaml` (committed; changes affect all users), or
- create `configs/paths.local.yaml` (gitignored; per-user override), or
- set the environment variable `SDM_RAW_DATA_PATH`.

## Optional: Petko et al. 2026 Supplementary Table 3 (distributional categories)

To avoid the 'regional' fallback in Task 1 gate 1, provide a CSV with
two columns `species, category` where category is one of
`endemic`, `regional`, `widespread`, e.g.:

```csv
species,category
Austropotamobius_torrentium,regional
Pacifastacus_leniusculus,widespread
```

Pass via `--petko-categories path/to/file.csv` to `run_task1_audit.py`.

## Reproducibility

Never re-save raw data in place. If you reformat or correct something,
write to `data/interim/` with a descriptive name and a README note.
