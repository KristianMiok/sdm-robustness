# SDM Robustness — Stress-testing freshwater SDMs under spatial uncertainty

Companion code repository for:

> *Robustness limits of freshwater species distribution models under spatial uncertainty*

Builds on the Global Crayfish Database of Geospatial Traits (Petko et al. 2026, *Scientific Data*, in revision).

## What this repo does

The paper tests how much positional uncertainty in occurrence data degrades ecological inference in dendritic river networks. This code implements the full experimental pipeline:

1. **Task 1** — Dataset audit and candidate species selection (data-driven feasibility screen)
2. **Task 2** — Final species panel selection (ecological judgement layer)
3. **Task 3** — Locked modelling pipeline specification
4. **Task 4** — Three-tier degradation metrics framework, incl. Inference Stability Threshold (IST)
5. **Task 5** — Stress-test execution (core factorial + benchmark sanity + transferability)
6. **Task 6** — Analysis and synthesis
7. **Task 7** — Manuscript figure-generation scripts

## Repository layout

```
sdm-robustness/
├── configs/                    # YAML configs for frozen design decisions
├── src/sdm_robustness/         # Core Python package
│   ├── io/                     # Data loading, schema validation
│   ├── audit/                  # Task 1: species audit, feasibility gates
│   ├── pipeline/               # Task 3: modelling pipeline (RF, XGBoost, Maxent)
│   ├── metrics/                # Task 4: Tier 1–3 metrics, IST
│   ├── execution/              # Task 5: stress-test runner, contamination
│   ├── analysis/               # Task 6: degradation curves, meta-synthesis
│   └── utils/                  # Logging, reproducibility, geospatial helpers
├── scripts/                    # Entry-point scripts (thin wrappers around src/)
├── notebooks/                  # Exploratory analysis (not committed to runs)
├── tests/                      # pytest — keep this healthy
├── data/                       # Gitignored. See data/README.md
│   ├── raw/                    # Original Petko et al. 2026 master table
│   ├── interim/                # Cleaned, deduplicated benchmark
│   ├── processed/              # Per-species training datasets
│   └── outputs/                # Model artifacts
├── results/                    # Gitignored per-run outputs
├── docs/                       # Protocol documents, technical memos
└── manuscript/                 # Figures, tables, draft sections
```

## Setup (PyCharm)

### 1. Clone and open in PyCharm

```bash
git clone <your-repo-url> sdm-robustness
```

In PyCharm: **File → Open** → select the `sdm-robustness` directory → "Open as Project".

### 2. Create the virtual environment

PyCharm's built-in way: **File → Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv Environment → New**. Set location to `./venv` inside the project, base interpreter to Python 3.10+.

Or from terminal:
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # install the sdm_robustness package in editable mode
```

Editable install (`-e .`) is important — it lets you `import sdm_robustness` from anywhere in the project, and any edits to `src/` take effect immediately without reinstalling.

### 4. Point the pipeline at Olga's data

Edit `configs/paths.yaml`:
```yaml
raw_data_path: /Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv
```

Or set the environment variable `SDM_RAW_DATA_PATH` to override the config.

### 5. Run Task 1

```bash
python scripts/run_task1_audit.py
```

Outputs land in `results/task1_audit/`. See the generated `technical_memo.md` for the one-page summary.

## Development workflow

- **Branch per task**: `task1/audit`, `task3/pipeline-spec`, etc. Main stays releasable.
- **Tests**: `pytest tests/` — run before every push. Audit functions are fully covered; modelling pipeline is scaffolded.
- **Formatting**: `black src/ tests/ scripts/` and `ruff check src/`. Configs in `pyproject.toml`.
- **Reproducibility**: every script accepts `--seed` and logs its seed + config hash in the output directory.

## Frozen design decisions

See `configs/frozen_design.yaml` and `docs/protocols/frozen_design.md`. These are locked — if something looks wrong, raise it before execution, not mid-run.

## License

TBD before submission. Code will likely be MIT or BSD-3; data is owned by Petko et al. 2026.
