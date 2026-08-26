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

## Measured fit times (seconds, 8 cores)

| algorithm | small (~360 pres) | ~1000 | ~2000 | large (7711 pres, 48k segments) |
|---|---|---|---|---|
| random forest | - | 1.1 | - | 4.1 |
| XGBoost | - | 3.8-5.6 | - | ~12 |
| Maxent, one fit | 46 | 132 | 110 | 139 |
| Maxent, one run (6 fits + prediction) | 325 | 822 | 694 | 901 |

Maxent trains on 10,000 background points regardless of entity size, so it does
not scale with presences: 21x the data costs 3x the time. RF and XGBoost scale
the other way, drawing as many background points as there are presences.

## Cost

One pass of Grid B at the current design is ~17,300 runs; the replicated
benchmark arm adds 13 x 3 x 3 x n_replicates - 3,510 runs at 30 replicates.
Each run is 5 CV folds plus a full refit, and the reference-set path adds one
more fit.

    random forest   ~5,800 runs   ~40 CPU-hours
    XGBoost         ~5,800 runs   ~90 CPU-hours
    Maxent          ~5,800 runs   ~1,100 CPU-hours   (mean run 685 s)

Total ~1,230 CPU-hours, about 77 node-hours at 16 cores. Maxent is ~90% of it.
An earlier figure of 5,800 CPU-hours for Maxent in this document was wrong by
5x: it multiplied a whole-run time as though it were a per-fit time.

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

## SLURM settings

Use `--cpus-per-task=8 --mem-per-cpu=1800M`. At 3000M the scheduler silently
redirects to `largemem`, which has a longer queue and, during the August 2026
maintenance, 84 of ~190 nodes drained against 38 idle on `cpu`. The ceiling on
`cpu` is somewhere between 14.4 and 24 GB per task. Memory was never the
constraint: the largest entity is 48,267 segments x 211 predictors, about 82 MB.

Maxent branches need `--time=24:00:00` - at 220-340 s per fit an 8 h limit is
hit around replicate 11 of 15. Write results incrementally rather than at the
end; the variance pilot lost two array tasks that way.
