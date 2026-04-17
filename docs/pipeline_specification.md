# Pipeline specification
## Robustness limits of freshwater SDMs under spatial uncertainty
### Task 3 — frozen execution protocol

This document is the frozen protocol for the modelling pipeline used in Tasks 5–7. It follows the locked panel and design defined after Task 1b and is intended to be the operational reference for implementation and execution.

---

## 1. Scope and design status

The analytical panel is locked at 13 analytical entities:

### 1.1 DUAL-AXIS entities
1. Astacus astacus
2. Pontastacus leptodactylus (pooled)
3. Austropotamobius torrentium (pooled)
4. Austropotamobius fulcisianus (pooled)
5. Procambarus clarkii (native)
6. Procambarus clarkii (alien)
7. Pacifastacus leniusculus (alien)
8. Faxonius limosus (alien)

### 1.2 SNAPPING-ONLY entities
9. Lacunicambarus diogenes
10. Cambarus latimanus
11. Cambarus striatus
12. Creaserinus fodiens
13. Faxonius limosus (native)

The analytical unit is **entity**, not species. Each entity is handled independently throughout the pipeline, including predictor cleaning, pseudo-absence generation, benchmark model fitting, contamination experiments, and metrics.

The following design elements are frozen and must not be changed without explicit approval:
- panel composition
- entity treatment (direct / pooled / native-only / alien-only)
- contamination levels
- substitution design
- predictor-cleaning rules
- hyperparameter freeze after benchmark tuning

---

## 2. Experimental design summary

### 2.1 Contamination axes

#### Snapping axis
Levels: 0%, 1%, 2%, 5%

Applicable to:
- all 13 entities

#### Low-accuracy axis
Levels: 0%, 3%, 10%, 20%

Applicable to:
- 8 DUAL-AXIS entities only

### 2.2 Contamination mechanism

Contamination is applied by **substitution**, not addition.

For any entity and contamination level:
- `N_experiment` is held constant
- a fraction of records is drawn from the contamination pool
- the remainder is drawn from the clean benchmark pool

This ensures that model degradation reflects contamination, not sample-size differences.

### 2.3 Replication

Per entity × axis × contamination level × algorithm × spatial-scale track:
- 30 replicate runs
- fixed and logged random seeds
- 95% bootstrap confidence intervals on all summary metrics

Escalation rule:
- if CI width exceeds 10% of the metric value for a given cell, escalate that cell from 30 to 50 replicates

### 2.4 Benchmark and transferability

#### Benchmark sanity check (Task 5c)
- 30 benchmark replicates per entity × algorithm × spatial-scale track

#### Transferability test (Task 5d)
Initial target entities:
- Procambarus clarkii (alien)
- Astacus astacus
- Pacifastacus leniusculus (alien)

These may be revised only after benchmark outputs are inspected.

---

## 3. Benchmark dataset definition

For each entity, the benchmark presence dataset is:

- Accuracy = High
- snapping distance <= 200 m
- deduplicated per Hydrography90m segment (`subc_id`)

This benchmark is the reference dataset for:
- predictor cleaning
- hyperparameter tuning
- benchmark performance estimation
- contamination substitution

No contamination is introduced during benchmark construction.

---

## 4. Predictor space and spatial-scale tracks

Environmental predictors are grouped into three parallel modelling tracks:

### 4.1 Local-only
Use only local predictors:
- `l_*`

### 4.2 Upstream-only
Use only upstream predictors:
- `u_*`

### 4.3 Combined
Use both:
- `l_*`
- `u_*`

Each track is executed independently for every entity and algorithm.

No information is shared across tracks after track definition.

---

## 5. Predictor cleaning

Predictor cleaning is performed **once per entity × spatial-scale track**, on the clean benchmark dataset only, and then frozen for all downstream contamination runs.

### 5.1 Missingness filter
Drop any predictor with:
- more than 30% missing values in the benchmark dataset

### 5.2 Correlation filter
From the remaining predictors:
- compute pairwise correlations
- if `|r| > 0.98`, drop one predictor from the pair

The exact tie-breaking rule must be deterministic and logged. Recommended order:
1. retain predictor with lower missingness
2. if tied, retain predictor with lower mean absolute correlation to all other predictors
3. if still tied, retain alphabetically first predictor

### 5.3 Freezing rule
The cleaned predictor set produced at benchmark stage becomes the fixed predictor set for:
- benchmark runs
- all contamination runs
- all replicate runs for that entity × track

The number of surviving predictors may differ by entity and by track. This is expected and should be logged.

---

## 6. Presence–absence design and pseudo-absence generation

### 6.1 General principle
Pseudo-absences must be generated in a **network-constrained** way, consistent with Hydrography90m topology and the accessible area of the focal entity.

Pseudo-absence generation is done independently for each entity.

### 6.2 Accessible area
The accessible area for an entity is defined over Hydrography90m segments and should be constrained to the hydrographic network represented by that entity’s observed distributional envelope.

Operational accessible-area definition should be deterministic and logged. Recommended implementation:
- begin from all segments occurring within the hydrological basins represented by the benchmark presences
- exclude occupied benchmark segments
- retain only segments with complete predictor coverage for the current spatial-scale track

If a more restrictive network neighborhood rule is used, it must be documented and applied consistently across all entities.

### 6.3 Presence:absence ratio
Use a fixed presence:absence ratio per entity, specified once and then frozen. Recommended default:
- 1:1 for the main implementation

If a different ratio is selected after pilot benchmarking, the change must be justified and applied globally.

### 6.4 Pseudo-absence reproducibility
Pseudo-absence draws must be:
- seed-controlled
- reproducible
- logged per entity × track × replicate

### 6.5 Interaction with contamination design
Pseudo-absences are not part of the contamination mechanism itself. Contamination applies to the presence set by substitution. Pseudo-absence generation remains benchmark-anchored and reproducible across runs.

---

## 7. Cross-validation and train/test splits

### 7.1 Primary CV
Primary evaluation uses **spatial block cross-validation** with:
- `basin_id` as the grouping variable

This is the main evaluation protocol reported in the paper.

### 7.2 Secondary CV
Secondary reference evaluation uses:
- stratified 5-fold cross-validation

This is not the main inferential result, but provides a conventional comparison.

### 7.3 Nested tuning
Hyperparameter tuning is performed once on the clean benchmark using the primary CV strategy. The tuned hyperparameters are then frozen and reused for all contaminated runs for that entity × track × algorithm.

### 7.4 Minimum data requirements
If a specific entity × track combination cannot support a valid primary CV split, the failure must be logged explicitly. No silent fallback is allowed.

---

## 8. Algorithms

### 8.1 Required algorithms
The core paper uses:
- Random Forest
- XGBoost

### 8.2 Optional algorithm
- Maxent may be added later, but it is not part of the frozen core pipeline at this stage

### 8.3 Random Forest
Recommended frozen baseline:
- implementation: scikit-learn
- `n_estimators = 500`
- `class_weight = "balanced"`

Any additional tuned hyperparameters must be selected during benchmark tuning and then frozen.

### 8.4 XGBoost
Use XGBoost classifier with benchmark tuning and frozen post-tuning parameters.

All package versions and final hyperparameter values must be logged.

---

## 9. Hyperparameter policy

Hyperparameters are tuned:
- once per entity
- once per spatial-scale track
- once per algorithm
- on the clean benchmark only

After tuning, hyperparameters are frozen across:
- contamination axes
- contamination levels
- replicate runs

No retuning is allowed on contaminated datasets.

This is essential for interpretability: degradation must reflect contamination, not adaptive retuning.

---

## 10. Output artifacts per run

Every run must emit or make reproducible the following outputs:

### 10.1 Core metadata
- entity
- species
- treatment class
- axis
- contamination level
- replicate index
- random seed
- algorithm
- spatial-scale track
- benchmark / contaminated flag

### 10.2 Model outputs
- fitted model object or serialized reference
- held-out predictions
- held-out labels
- pseudo-absence draw identifier
- predictor set used
- hyperparameter set used

### 10.3 Performance outputs
- AUC
- TSS
- Boyce index
- Brier score
- sensitivity
- specificity

### 10.4 Ecological signal outputs
- variable importance vector
- Gini importance (RF)
- SHAP importance summary
- partial dependence curves for top-10 variables
- niche centroid summary
- niche breadth summary

### 10.5 Spatial outputs
- predicted suitability raster or gridded prediction surface
- binary range predictions under defined threshold rules
- spatial mismatch maps where relevant

All paths and artifact identifiers must be deterministic and recoverable from metadata.

---

## 11. Thresholding and binarisation

For spatial area-change summaries and binary comparisons, report at least:
- threshold = 0.5
- MaxSSS threshold where implemented

Threshold choice must be recorded in the output metadata.

---

## 12. Reproducibility and logging

Every execution unit must log:
- git commit
- config hash
- package versions
- entity
- axis
- contamination level
- replicate index
- seed
- algorithm
- spatial-scale track
- predictor count before and after cleaning
- pseudo-absence pool size
- contamination pool size

All random operations must be seed-controlled.

No hidden randomness is allowed.

---

## 13. Recommended implementation order

1. encode the final 13-entity panel in a machine-readable panel file
2. implement benchmark dataset builder per entity
3. implement predictor cleaning and freeze export
4. implement pseudo-absence generator
5. implement benchmark tuning
6. implement contamination substitution runner
7. implement metrics extraction hooks
8. benchmark runtime on a single DUAL-AXIS entity before large-scale execution

---

## 14. Immediate deliverable for Task 3

The output of Task 3 is this frozen protocol document plus the corresponding code scaffolding that respects these rules.

Execution should not begin until this specification is reviewed and accepted.
