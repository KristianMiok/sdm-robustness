# Instrumented campaign — scope, sizing and order

Draft, 24 August 2026. Numbers from measured fit times, not estimates.

## What the campaign has to deliver

| task | what it needs | delivered by |
|---|---|---|
| T5 | paired benchmark and contaminated arms | replicated benchmark, shared background seed |
| T6.1-6.3 | evaluation not contaminated by the treatment | fixed reference set, fold map from clean benchmark |
| T6.4 | stated threshold, omission rate | inline, done |
| T6.5 | range change at several thresholds | inline: 0.3 / 0.5 / 0.7 / MaxSSS / p10 |
| T6.6 | continuous suitability comparison | inline: MAD and mean shift |
| T8.6 | response curves | models persisted for a subset |
| T1.3 | benchmark at 50 / 100 / 200 m | third arm, benchmark rebuilt per threshold |
| T1.5 | contamination stratified by stream order | fourth arm on the low-accuracy axis |
| T3 | controlled displacement | separate campaign, effective-dose targeted |

## Measured fit times (seconds, one fit, 8 cores)

| algorithm | small entity (~1000 presences) | large (~7700) |
|---|---|---|
| random forest | 1.1 | 4.1 |
| XGBoost | 3.8-5.6 | ~12 |
| Maxent | 220-340 | 340+ |

Maxent trains on 10,000 background points regardless of entity size, so it does
not scale with presences and dominates everything. It is 20-100x random forest.

## Cost

One pass of Grid B at the current design is ~17,300 runs; the replicated
benchmark arm adds 13 x 3 x 3 x n_replicates. At 30 replicates that is 3,510
runs. Each run is ~6 fits (5 CV folds + 1 full refit); the reference-set path
adds one more.

    random forest   ~5,800 runs   ~40 CPU-hours
    XGBoost         ~5,800 runs   ~90 CPU-hours
    Maxent          ~5,800 runs   ~5,800 CPU-hours

Maxent is ~98% of the cost. Total for one instrumented pass: ~6,000 CPU-hours,
roughly 375 node-hours at 16 cores. The extended snapping doses, T1.3 and T1.5
multiply the contaminated side but not the benchmark arm.

## Two decisions this forces

**Maxent replication.** The noise floor for Maxent is comparable to random
forest (background SD 2.9-6.5 vs 2.7-5.8) but a replicate costs 60x more, and
pairing the background buys only 1.3x for Maxent against 2.0x for the tree
models. Running Maxent at 30 replicates is not defensible on cost. Proposal:
30 replicates for RF and XGBoost, 10 for Maxent, pairing modulo 10.

**Replicate allocation by entity.** The floor scales inversely with benchmark
size - background SD is 1.3 for P. clarkii alien (7,711 presences) and 6.5 for
A. fulcisianus (744). Uniform replication over-samples the large entities and
under-powers the small ones. Proposal: allocate to equalise the standard error,
i.e. replicates proportional to the square of the measured floor, floored at 15
and capped at 50.

## Order

1. T6 arm: replicated benchmark, reference set, paired background, extended
   snapping doses (10/20/35/50 where the pool allows). This is the arm every
   Tier 1 and Tier 3 number in the revision comes from.
2. T1.5 stratified low-accuracy contamination. Cheap next to T3 and decisive for
   whether the 29-36% is a stream-order composition effect.
3. T3 displacement, effective-dose targeted, capped at 500 m. Blocked on the
   GeoFRESH lookup.
4. T1.3 benchmark thresholds. Independent of the others, can run last.

## Blocked

T2.1 and T3 both need the subc_id to predictor lookup for every subcatchment in
the extent, not only the occupied ones. Until it exists, the displacement
neutrality result is provisional: it was measured over subcatchments that
contain crayfish records, and the real network has headwater segments with none.
