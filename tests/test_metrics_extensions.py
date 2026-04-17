from __future__ import annotations

import numpy as np

from sdm_robustness.metrics import (
    compute_domain_importance_shift,
    domain_rank_stability,
    classify_against_benchmark_envelope,
)


def test_compute_domain_importance_shift():
    b = {"CLI": 0.6, "TOP": 0.4}
    c = {"CLI": 0.3, "TOP": 0.7}
    out = compute_domain_importance_shift(b, c)
    assert "CLI_share_shift" in out
    assert np.isclose(out["CLI_benchmark_share"], 0.6)
    assert np.isclose(out["TOP_current_share"], 0.7)


def test_domain_rank_stability_identity():
    b = {"CLI": 0.6, "TOP": 0.4, "SOL": 0.2}
    c = {"CLI": 0.7, "TOP": 0.5, "SOL": 0.1}
    assert np.isclose(domain_rank_stability(b, c), 1.0)


def test_classify_against_benchmark_envelope():
    assert classify_against_benchmark_envelope(0.50, 0.48, 0.52, 0.50, 0.02) == "within_noise"
    assert classify_against_benchmark_envelope(0.56, 0.53, 0.57, 0.50, 0.02) == "marginal"
    assert classify_against_benchmark_envelope(0.70, 0.68, 0.72, 0.50, 0.02) == "significant"
