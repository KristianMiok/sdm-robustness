# Pre-registered predictions for the T3 displacement campaign

**Written 24 August 2026, before any T3 run exists.**
Commit this file before the campaign is launched. Do not edit after T3 results
are available; record the outcome in a separate file.

## Why

The headwater-depletion mechanism (benchmark under-representation of first-order
segments relative to the accessible area predicts range inflation under
low-accuracy contamination; Spearman -0.79 / -0.83 / -0.95 for Maxent / RF /
XGBoost) was identified post hoc, and is the strongest of roughly thirty
correlations tested on n = 8 entities. Reported as such it invites a discount.

T3 has not run. Recording what the mechanism predicts, in advance, converts it
from a selected correlation into an out-of-sample test.

## Two rival hypotheses, predicting different orderings

**H1 - composition shift.** Range inflation is driven by the shift in Strahler
composition that contamination induces, not by contamination per se.
Displacement was measured to be compositionally neutral: the reachable
neighbourhood at 100-1000 m has the same first-order share as the benchmark
(within 0.3 points for every entity), and P(lower order | different subcatchment)
is 0.31-0.41, i.e. a displaced record lands on a HIGHER order segment more often
than a lower one.
**H1 therefore predicts that depletion will NOT order the T3 effects.**

**H2 - effective dose.** Effect is driven by how often displacement changes the
subcatchment at all, which is the only quantity displacement varies.

The two predictors are near-independent (Spearman -0.26), so T3 can distinguish
them.

| entity | bench %1st - acc %1st | P(change) @500 m | H1 rank | H2 rank |
|---|---|---|---|---|
| A. torrentium (pooled) | -20.9 | 0.105 | 1 | 7 |
| A. astacus | -18.8 | 0.184 | 2 | 1 |
| F. limosus (alien) | -15.4 | 0.157 | 3 | 4 |
| P. leniusculus (alien) | -12.2 | 0.139 | 4 | 5 |
| P. leptodactylus (pooled) | -1.7 | 0.182 | 5 | 2 |
| P. clarkii (alien) | +2.2 | 0.158 | 6 | 3 |
| A. fulcisianus (pooled) | +4.3 | 0.049 | 7 | 8 |
| P. clarkii (native) | +9.9 | 0.128 | 8 | 6 |

Rank 1 = largest predicted range inflation.

## Predictions, in order of how much they would tell us

**P1.** Spearman between depletion and per-entity T3 range inflation will be
weak, |rho| < 0.50. **Falsified if |rho| > 0.74** (the p<0.05 critical value at
n = 8) with the H1 ordering. Falsification means the depletion mechanism does
not work through Strahler composition, and our stated mechanism is wrong.

**P2.** Per-entity T3 range inflation at 13.2% effective dose (every record
displaced at 500 m) will be smaller than low-accuracy inflation at ~19.7%
effective, for every one of the eight entities, and smaller after normalising
by effective dose. Falsified by any entity where displacement matches or
exceeds substitution per unit effective dose.

**P3.** If anything orders the T3 effects, it will be P(change), the H2 ranking.
Stated as the weaker of the two; H2 was not derived from a causal argument, only
from the observation that it is the only quantity displacement varies.

**P4.** The between-entity spread on T3 will be smaller than the 15.5-fold spread
observed on the low-accuracy axis, because displacement removes the pool-composition
differences that drive it.

## Recorded state of the evidence at the time of writing

Low-accuracy L20, paired background, 15 replicates, combined track, range area
percent change at threshold 0.5:

| entity | RF | XGBoost |
|---|---|---|
| A. torrentium (pooled) | 72.82 | 87.38 |
| P. leniusculus (alien) | 49.50 | 49.08 |
| A. astacus | 39.06 | 52.75 |
| F. limosus (alien) | 36.50 | 34.15 |
| P. clarkii (alien) | 24.64 | 24.95 |
| A. fulcisianus (pooled) | 19.24 | 11.60 |
| P. leptodactylus (pooled) | 13.59 | 15.38 |
| P. clarkii (native) | 4.71 | 4.05 |

Sources: `results/revision/t5_pilot_summary.csv`,
`results/revision/t3_effective_dose.csv`,
`results/revision/t3_effective_dose_substitution.csv`,
`scripts/revision/t3_displacement_bias.py`.

## What we commit to reporting

The outcome of P1-P4 regardless of direction, including the case where the
mechanism fails. If the tightened envelope (fold map fixed, background paired -
SD falls 2-3 fold) plus extended doses place the snapping axis outside the
benchmark envelope, that is reported too, even though it dissolves the two-axis
contrast the submitted manuscript is built on.
