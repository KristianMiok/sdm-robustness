"""Task 5 — contamination sampler.

Implements the substitution design: for a species with clean benchmark pool C
and contamination pool X (snapping or low-accuracy), and target level p%:

    training_set = random sample of (1 - p/100) × N_experiment from C
                 + random sample of (p/100) × N_experiment from X

N_experiment is held constant across all levels for a species.

Status: partial implementation — the sampling math is trivial; the wrapper
lives here so Task 5 execution can use it immediately when the rest of the
pipeline is ready.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdm_robustness.utils import rng


@dataclass
class ContaminationDraw:
    """One contamination draw for one replicate."""

    species: str
    axis: str                   # 'snapping' | 'lowacc' | 'null'
    level_pct: int
    replicate_idx: int
    seed: int
    training_indices: np.ndarray
    n_clean_drawn: int
    n_contam_drawn: int


def draw_substitution_sample(
    clean_pool: pd.DataFrame,
    contam_pool: pd.DataFrame,
    *,
    species: str,
    axis: str,
    level_pct: int,
    replicate_idx: int,
    n_experiment: int,
    master_seed: int,
) -> ContaminationDraw:
    """Draw one contaminated training set under the substitution design.

    Parameters
    ----------
    clean_pool : DataFrame
        Deduplicated High-accuracy ≤200m records for this species.
    contam_pool : DataFrame
        Contamination pool (snapping band or Low-accuracy).
    axis : str
        'snapping' | 'lowacc' | 'null' (null = resample clean_pool as contaminant).
    level_pct : int
        Contamination level in percent (0, 5, 10, 20, 35, 50).
    replicate_idx : int
        0-based replicate index.
    n_experiment : int
        Fixed sample size for this species across all levels.
    master_seed : int
        Project master seed (from configs/frozen_design.yaml).
    """
    generator = rng(
        master_seed, species, axis, level_pct, replicate_idx, "contamination"
    )

    n_contam = int(round(n_experiment * level_pct / 100.0))
    n_clean = n_experiment - n_contam

    if n_clean > len(clean_pool):
        raise ValueError(
            f"Clean pool too small: need {n_clean}, have {len(clean_pool)}"
        )
    if n_contam > len(contam_pool) and axis != "null":
        raise ValueError(
            f"Contamination pool too small: need {n_contam}, have {len(contam_pool)}"
        )

    clean_idx = generator.choice(clean_pool.index.to_numpy(), size=n_clean, replace=False)
    if n_contam > 0:
        # For the null model, draw the "contamination" from clean_pool with
        # replacement disallowed; this gives "structureless contamination".
        pool_to_use = clean_pool if axis == "null" else contam_pool
        contam_idx = generator.choice(
            pool_to_use.index.to_numpy(), size=n_contam, replace=False
        )
    else:
        contam_idx = np.array([], dtype=clean_idx.dtype)

    training = np.concatenate([clean_idx, contam_idx])

    # Derive a deterministic seed for this replicate that downstream fitting
    # functions can reuse (RF random_state, XGBoost seed, etc.).
    from sdm_robustness.utils import derive_seed
    seed = derive_seed(master_seed, species, axis, level_pct, replicate_idx)

    return ContaminationDraw(
        species=species,
        axis=axis,
        level_pct=level_pct,
        replicate_idx=replicate_idx,
        seed=seed,
        training_indices=training,
        n_clean_drawn=n_clean,
        n_contam_drawn=n_contam,
    )
