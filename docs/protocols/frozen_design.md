# Frozen design decisions — protocol

Human-readable companion to `configs/frozen_design.yaml`. If these two
documents disagree, the YAML wins (it's what the code reads). Update both
together.

## Benchmark construction

- Source: Petko et al. 2026 final filtered dataset.
- Accuracy: High only.
- Snapping distance: ≤ 200 m.
- Deduplication: one record per Hydrography90m segment (`subc_id`) per taxon,
  applied once at benchmark construction and carried through every
  contamination run.
- Fallback: if the candidate pool in Task 1 is sparse, relaxation to 500 m
  will be reconsidered. Start with 200 m.

## Distributional categories (from Petko et al. 2026 Supp. Table 3)

- **endemic / narrow-range**: median standardised range ≤ 1.80
- **regional**: 1.80 – 3.14
- **widespread / cosmopolitan**: ≥ 3.14

Minimum clean-benchmark sizes:
- widespread: 500
- regional: 200
- endemic: 80

## Two axes of spatial uncertainty

- **Axis 1 — Snapping**: High-accuracy records where `distance_m` falls in
  one of two bands: 200–500 m (mild) and 500–1000 m (strong). Records with
  `distance_m` > 1000 m are excluded from the experiment and reported only.
- **Axis 2 — Low-accuracy**: records flagged Low by the original contributor.

These axes are kept conceptually separate throughout the analysis.

## Contamination design — substitution, NOT addition

For each contamination level, total sample size *N_experiment* is held
constant. Level `p%` means:

    training set = (1 − p/100) × N_experiment drawn from the clean pool
                 + (p/100)       × N_experiment drawn from the contamination pool

Example with N_experiment = 1000:
- 0 %:  1000 clean + 0 contaminated
- 20 %: 800 clean + 200 contaminated
- 50 %: 500 clean + 500 contaminated

This keeps statistical power constant, so measured degradation reflects
contamination alone.

**Ceiling: 50 %.** Above 50%, contaminated data would outnumber clean data —
that's a different question ("SDM on mostly bad data"), outside scope.

## Contamination levels

`[0, 5, 10, 20, 35, 50]` percent per axis. Six points per curve — sufficient
for segmented regression / changepoint analysis.

## Replicates

- Default: 30 per (species × axis × level × algorithm).
- Escalate to 50 if any 95 % bootstrap CI exceeds 10 % of the metric value.
- **Kristian's recommendation** (awaiting Lucian's sign-off): pre-escalate
  low levels (≤ 10 %) to 50 replicates from the start. At small `p`, the
  number of contaminated records is small and variance dominates.

## Algorithms

- Random Forest (scikit-learn, 500 trees, balanced class weights) — obligatory.
- XGBoost — obligatory.
- Maxent (via elapid or equivalent) — optional; include if feasible.
- Decision Trees — explicitly **NOT** in stress-test (too unstable per run).
  May appear in a benchmark-only interpretability panel.

Hyperparameters are tuned **once** on the clean benchmark per species via
nested CV, then **frozen** for every contamination run of that species.
This prevents conflating degradation with re-tuning noise.

## Predictor handling

- Cleaning rule: drop variables with > 30 % missing; within every correlated
  pair with |r| > 0.98, drop one. Identical to Miok et al. (in review) and
  Miok et al. (CB submission).
- Cleaning is applied **once** on the clean benchmark per species. The
  resulting variable set is **frozen** for every contaminated run of that
  species. Do NOT re-clean after contamination.
- Scale tracks: `local_only` (l_ variables), `upstream_only` (u_ variables),
  and `combined`. All three run in parallel per species — their contrast
  tests hypothesis H3.

## Cross-validation

- Primary: spatial block CV with `basin_id` as grouping variable.
- Secondary: stratified 5-fold CV.
- **Kristian's recommendation**: endemic species with `n_basins < 5` degenerate
  under basin-level CV. Fallback to spatial blocks within basin, keeping
  basin stratification where possible.

## Pseudo-absences

Network-constrained sampling along Hydrography90m segments, within an
accessible-area buffer around known occurrences (radius to be justified in
Task 3 protocol). Presence:absence ratio 1:1 for RF; up to 1:10 for Maxent.

**Kristian's recommendation**: control density per Strahler order, not just
buffer. Otherwise snapping-axis shifts between segments of different orders
induce an absence-density confound.

## Reproducibility

- Master seed: `20260416` (project kickoff date).
- Per-run seeds are derived deterministically from
  `(master_seed, species, axis, level, replicate_idx, algorithm)` via
  `utils.repro.derive_seed()`.
- Every run logs: git commit SHA, working-tree dirty flag, config hash.

## Optional extensions

- **Null model** (Kristian's amendment): "contamination" = random resample of
  clean pool with no positional modification. Gives the structureless-
  contamination reference. ~5 % of core compute.
- **Transferability at 10 / 30 / 50 %** (Kristian's amendment): the briefing
  specified a single 30 % point for Task 5d. Three points give the bias-
  direction shape.
- **2D joint grid** (snapping × low-accuracy): run only if core completes
  on schedule and preliminary results are scientifically interesting.
