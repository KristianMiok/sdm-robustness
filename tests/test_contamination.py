"""Tests for execution.contamination sampler.

These lock in the substitution-design invariants: N_experiment constant,
split proportions exact, reproducible seeding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_robustness.execution.contamination import draw_substitution_sample


def _pool(n: int, start: int) -> pd.DataFrame:
    return pd.DataFrame({"id": range(start, start + n)}, index=range(start, start + n))


def test_sample_size_is_fixed():
    """N_experiment must be constant across contamination levels."""
    clean = _pool(500, 0)
    contam = _pool(500, 10_000)
    sizes = []
    for level in (0, 5, 10, 20, 35, 50):
        draw = draw_substitution_sample(
            clean_pool=clean, contam_pool=contam,
            species="Sp", axis="snapping", level_pct=level,
            replicate_idx=0, n_experiment=400, master_seed=42,
        )
        sizes.append(len(draw.training_indices))
    assert len(set(sizes)) == 1, f"Sample sizes varied: {sizes}"
    assert sizes[0] == 400


def test_split_exact():
    """At p%, exactly p% of N_experiment comes from contam_pool."""
    clean = _pool(500, 0)
    contam = _pool(500, 10_000)
    for level in (5, 10, 20, 35, 50):
        draw = draw_substitution_sample(
            clean_pool=clean, contam_pool=contam,
            species="Sp", axis="snapping", level_pct=level,
            replicate_idx=0, n_experiment=400, master_seed=42,
        )
        assert draw.n_contam_drawn == int(round(400 * level / 100))
        assert draw.n_clean_drawn == 400 - draw.n_contam_drawn


def test_zero_percent_is_pure_clean():
    clean = _pool(500, 0)
    contam = _pool(500, 10_000)
    draw = draw_substitution_sample(
        clean_pool=clean, contam_pool=contam,
        species="Sp", axis="snapping", level_pct=0,
        replicate_idx=0, n_experiment=400, master_seed=42,
    )
    assert draw.n_contam_drawn == 0
    # All drawn indices are from the clean pool (0..499)
    assert np.all(draw.training_indices < 500)


def test_reproducibility_same_seed_same_draw():
    clean = _pool(500, 0)
    contam = _pool(500, 10_000)
    kw = dict(
        clean_pool=clean, contam_pool=contam,
        species="Sp", axis="snapping", level_pct=20,
        replicate_idx=3, n_experiment=400, master_seed=42,
    )
    d1 = draw_substitution_sample(**kw)
    d2 = draw_substitution_sample(**kw)
    np.testing.assert_array_equal(d1.training_indices, d2.training_indices)
    assert d1.seed == d2.seed


def test_different_replicates_differ():
    clean = _pool(500, 0)
    contam = _pool(500, 10_000)
    d1 = draw_substitution_sample(
        clean_pool=clean, contam_pool=contam,
        species="Sp", axis="snapping", level_pct=20,
        replicate_idx=0, n_experiment=400, master_seed=42,
    )
    d2 = draw_substitution_sample(
        clean_pool=clean, contam_pool=contam,
        species="Sp", axis="snapping", level_pct=20,
        replicate_idx=1, n_experiment=400, master_seed=42,
    )
    assert not np.array_equal(d1.training_indices, d2.training_indices)


def test_pool_too_small_raises():
    clean = _pool(100, 0)
    contam = _pool(10, 10_000)
    with pytest.raises(ValueError, match="Contamination pool too small"):
        draw_substitution_sample(
            clean_pool=clean, contam_pool=contam,
            species="Sp", axis="snapping", level_pct=50,
            replicate_idx=0, n_experiment=100, master_seed=42,
        )


def test_null_axis_draws_from_clean():
    """Null-model contamination draws 'contam' records from clean pool itself."""
    clean = _pool(500, 0)
    contam = _pool(10, 10_000)  # intentionally tiny — null shouldn't touch it
    draw = draw_substitution_sample(
        clean_pool=clean, contam_pool=contam,
        species="Sp", axis="null", level_pct=50,
        replicate_idx=0, n_experiment=400, master_seed=42,
    )
    # All 400 drawn indices come from clean (0..499), never from contam (10000+)
    assert np.all(draw.training_indices < 500)
