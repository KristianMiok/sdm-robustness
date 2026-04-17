# Metrics framework
## Robustness limits of freshwater SDMs under spatial uncertainty
### Task 4 — three-tier degradation metrics and Inference Stability Threshold

This document defines the frozen metric framework used to quantify degradation under positional uncertainty. It applies to every analytical entity in the locked panel and is shared across all algorithms and spatial-scale tracks.

---

## 1. Purpose

The goal of the metrics framework is to quantify how contamination affects SDM inference at three levels:

1. **Tier 1 — predictive performance**
2. **Tier 2 — ecological signal stability**
3. **Tier 3 — spatial prediction stability**

The framework is designed to distinguish cases where conventional performance remains apparently stable while ecological interpretation or spatial predictions already begin to drift.

---

## 2. General principles

- All metrics are computed per:
  - entity
  - axis
  - contamination level
  - replicate
  - algorithm
  - spatial-scale track

- Benchmark (0%) metrics are the reference condition.

- Degradation is reported relative to benchmark using either:
  - signed change (`Δmetric`)
  - similarity / overlap measure
  - absolute distance / displacement

- Summary statistics across replicates must include:
  - mean
  - standard deviation
  - 95% bootstrap confidence interval

---

## 3. Tier 1 — performance degradation

Tier 1 captures conventional predictive performance on held-out data.

### 3.1 Core metrics
For each run, compute:

- AUC
- TSS
- Boyce index
- Brier score
- sensitivity
- specificity

### 3.2 Derived degradation summaries
Relative to benchmark, compute:

- `ΔAUC`
- `ΔTSS`
- `ΔBoyce`
- `ΔBrier`
- `Δsensitivity`
- `Δspecificity`

### 3.3 Interpretation
Tier 1 answers:
- does predictive discrimination degrade?
- does calibration degrade?
- does thresholded classification degrade?

Tier 1 alone is not sufficient for robustness claims.

---

## 4. Tier 2 — ecological signal instability

Tier 2 quantifies whether the ecological interpretation of the model remains stable.

### 4.1 Variable-importance stability
For each contaminated run versus matched benchmark reference, compute:

- Spearman rank correlation of variable importance vectors
- Jaccard overlap for top-5 predictors
- Jaccard overlap for top-10 predictors

Importance sources:
- RF Gini importance
- SHAP-based importance where available

### 4.2 Response-shape stability
For top variables, compute:

- response-curve distance
- integrated absolute difference (IAD) between partial dependence curves

Recommended scope:
- top-10 predictors from benchmark model
- matched variable set across benchmark and contaminated run

### 4.3 Ecological-space summaries
Compute:

- niche centroid displacement
- niche breadth change

These should be benchmark-relative and defined consistently per entity and track.

### 4.4 Interpretation
Tier 2 answers:
- do the same predictors remain important?
- do their effects remain similar?
- does the inferred ecological envelope move or broaden/narrow?



### 4.4 Domain-level importance aggregation
In addition to variable-level stability, aggregate importance by thematic domain:
- CLI
- TOP
- SOL
- LAC

For each contaminated run, compute:
- domain-level importance share = sum of importances within domain / total importance
- domain importance shift = signed difference in domain share relative to benchmark
- domain rank stability = whether benchmark domain rank order changes under contamination

This is a derived post-processing step from the variable-importance vectors and should be implemented for all algorithms where importance vectors are available.

Tier 2 is central to the paper’s argument about masked uncertainty.

---

## 5. Tier 3 — spatial prediction instability

Tier 3 quantifies how spatial predictions change under contamination.

### 5.1 Continuous-map similarity
Compute:

- Schoener’s D
- Warren’s I

between benchmark and contaminated predicted suitability surfaces.

### 5.2 Binary-range change
Using thresholded predictions, compute:

- predicted range area percent change at threshold 0.5
- predicted range area percent change at MaxSSS

### 5.3 Spatial mismatch
Compute or derive:

- spatial mismatch maps
- cell-level disagreement summaries
- gain / loss / stable area decomposition where feasible

### 5.4 Interpretation
Tier 3 answers:
- do contaminated models predict the same places as suitable?
- does contamination shift the geography of suitability?
- are changes small and local, or large and structurally meaningful?

---

## 6. Inference Stability Threshold (IST)

### 6.1 Definition
For each entity × axis × algorithm × spatial-scale track, define the **Inference Stability Threshold (IST)** as:

> the lowest contamination level at which the mean Spearman rank correlation of variable importance falls below 0.7 across replicates

### 6.2 Sensitivity analysis
Also report:
- IST at threshold 0.8
- IST at threshold 0.6

This provides a robustness band around the primary IST definition.

### 6.3 Practical interpretation
- lower IST = inference destabilises earlier
- higher IST = inference remains stable longer

IST is the headline summary of robustness, but should always be interpreted alongside Tier 1 and Tier 3.

---

## 7. Benchmark matching rules

Where a contaminated run is compared to benchmark, the comparison must be made within the same:

- entity
- algorithm
- spatial-scale track
- CV protocol
- predictor set

The benchmark reference should come from the clean benchmark pipeline for that exact entity-track-algorithm combination.

---

## 8. Aggregation and reporting


### 7.3 Benchmark stability envelope
For each metric, define the benchmark stability envelope using Task 5c as:
- mean ± 2 SD from benchmark-vs-benchmark pairwise comparisons

For each contaminated cell, classify degradation as:
- **within noise**: contaminated mean lies within benchmark envelope
- **marginal**: contaminated mean lies outside the envelope, but contaminated 95% CI overlaps it
- **significant**: contaminated 95% CI does not overlap the benchmark envelope

This classification should be computed automatically for every results cell.

### 8.1 Run-level outputs
Each run must produce a machine-readable record of all Tier 1–3 metrics.

### 8.2 Cell-level summaries
For each entity × axis × level × algorithm × track, summarise:
- mean
- SD
- 95% bootstrap CI
- number of replicates

### 8.3 Entity-level summaries
For each entity, report:
- IST per axis
- earliest Tier 1 drop
- earliest Tier 2 instability
- earliest Tier 3 instability

### 8.4 Global summaries
Across the full panel, compare:
- DUAL-AXIS vs SNAPPING-ONLY entities
- pure native vs pooled vs split invasive entities
- local vs upstream vs combined tracks
- RF vs XGBoost

---

## 9. Expected implementation modules

The reusable Python metrics module should expose functions for:

### Tier 1
- `compute_performance_metrics(...)`
- `compute_delta_performance(...)`

### Tier 2
- `spearman_importance_stability(...)`
- `topk_jaccard(...)`
- `integrated_absolute_difference(...)`
- `centroid_displacement(...)`
- `niche_breadth_change(...)`

### Tier 3
- `schoeners_d(...)`
- `warrens_i(...)`
- `range_area_change(...)`
- `spatial_mismatch_summary(...)`

### IST
- `compute_ist(...)`

---

## 10. Immediate deliverables for Task 4

Task 4 should produce:
- this metrics framework document
- a reusable Python module under `src/sdm_robustness/metrics/`
- unit tests for core metric functions
- a consistent schema for metric outputs

Execution should use only metrics defined here unless a later explicit revision is approved.
