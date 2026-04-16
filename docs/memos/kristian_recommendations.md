# Kristian's methodological recommendations

These are the six points raised with Lucian on 2026-04-16 after reading the
project brief. None of them block Task 1; they all affect Tasks 3–5 and are
encoded in `configs/frozen_design.yaml` in the recommended state, pending
Lucian's sign-off.

## 1. Pre-escalate replicates at low contamination levels

**Problem.** At `p = 5 %` on a benchmark of 200 records, we substitute 10
records. Between-replicate variance will be large relative to the effect
size, almost certainly triggering the escalation rule universally.

**Recommendation.** Run 50 replicates from the start for levels ≤ 10 %.
Default 30 for higher levels. Saves re-running cells.

*Config key*: `replicates.preescalate_low_levels`

## 2. Spatial block CV fallback for endemic species

**Problem.** Endemic species with `n_basins = 3–4` collapse to 1–2 CV blocks
under basin-level grouping. Degenerate CV.

**Recommendation.** If `n_basins < 5`, fall back to spatial blocks within
basin, keeping basin stratification where possible. Threshold configurable.

*Config key*: `cross_validation.primary.fallback_n_basins_threshold`

## 3. Robust Tier 2 diagnostic for IST

**Problem.** Plain Spearman rank correlation of variable importance is
dragged around by noise in the long tail (variables with importance near 0
flip ranks between runs).

**Recommendation.** Primary diagnostic: weighted top-K overlap (each variable
weighted by its normalised importance). Plain Spearman reported alongside
for robustness.

*Config key*: (to be specified in Task 4 module)

## 4. Transferability at three levels, not one

**Problem.** The briefing specifies Task 5d at a single contamination level
(30 %). One point does not describe the shape of bias.

**Recommendation.** Run Task 5d at 10 %, 30 %, 50 % for the top 3 species.
Compute cost is modest (3 species × 3 levels × 2 algorithms × 30 replicates).

*Config key*: `optional.transferability_levels_pct`

## 5. Null-model baseline

**Problem.** Current design measures "contaminated vs. clean" but does not
separate structural degradation from pure resampling noise. A reviewer will
ask: "What would we see if we substituted the same number of records with
random draws from the clean pool itself?"

**Recommendation.** Add a null-model run: at each level, replace `p%` of the
training set with random draws from the clean pool (no positional change).
Costs ~5 % of core compute. Anticipates a likely reviewer comment.

*Config key*: `optional.run_null_model`

## 6. Strahler-stratified pseudo-absence density

**Problem.** Under snapping-axis contamination, records shift between
segments of different Strahler orders. If pseudo-absences are sampled
uniformly along the network, segments differ in absence density, and the
axis signal is partly confounded with an absence-density artefact.

**Recommendation.** Stratify pseudo-absence density per Strahler order so
that presence and absence density scale together within each order. Then
snapping shifts sample within the same density regime.

*Config key*: `pseudo_absence.stratify_by_strahler`

---

**Status.** All six points encoded as the default in configs. Lucian to
confirm or reject each before Task 5 execution starts.
