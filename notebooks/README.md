# Notebooks

Exploratory and inspection work only. **Notebooks are not part of the
analysis pipeline** — anything that needs to be reproducible lives in
`src/sdm_robustness/` or `scripts/`.

Suggested pattern:
- `01_inspect_master_table.ipynb` — basic counts, NA patterns, sanity plots
- `02_explore_task1_outputs.ipynb` — visualise the feasibility matrix
- `0N_*.ipynb` — one per exploration, numbered in order created

Notebook outputs (`*.ipynb_checkpoints`) are gitignored.
