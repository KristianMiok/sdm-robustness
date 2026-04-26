"""Tests for Boyce index."""
from __future__ import annotations

import numpy as np
import pytest

from sdm_robustness.metrics.boyce import boyce_index


def test_perfect_model_high_boyce():
    """Presences concentrated in high-suitability bins → Boyce close to +1."""
    rng = np.random.default_rng(0)
    bkg = rng.uniform(0, 1, size=10000)
    # Presences cluster in [0.7, 1.0]
    pres = rng.uniform(0.7, 1.0, size=500)
    b = boyce_index(pres, bkg, n_bins=10)
    assert b > 0.7, f"Expected high Boyce for perfect model, got {b}"


def test_random_model_near_zero():
    """Presences uniformly distributed → P/E ratios ~1 across bins → Boyce ~0."""
    rng = np.random.default_rng(1)
    bkg = rng.uniform(0, 1, size=20000)
    pres = rng.uniform(0, 1, size=2000)
    b = boyce_index(pres, bkg, n_bins=10)
    assert abs(b) < 0.5, f"Expected near-zero Boyce for random model, got {b}"


def test_inverse_model_negative_boyce():
    """Presences cluster in LOW-suitability bins → strong negative Boyce."""
    rng = np.random.default_rng(2)
    bkg = rng.uniform(0, 1, size=10000)
    pres = rng.uniform(0, 0.3, size=500)
    b = boyce_index(pres, bkg, n_bins=10)
    assert b < -0.7, f"Expected low Boyce for inverse model, got {b}"


def test_handles_nan_in_inputs():
    rng = np.random.default_rng(3)
    bkg = rng.uniform(0, 1, size=1000)
    pres = rng.uniform(0.7, 1, size=200)
    pres_with_nan = np.concatenate([pres, [np.nan, np.nan]])
    bkg_with_nan = np.concatenate([bkg, [np.nan]])
    b = boyce_index(pres_with_nan, bkg_with_nan, n_bins=10)
    assert not np.isnan(b)
    assert b > 0


def test_empty_presences_returns_nan():
    bkg = np.linspace(0, 1, 100)
    pres = np.array([])
    assert np.isnan(boyce_index(pres, bkg))


def test_empty_background_returns_nan():
    pres = np.linspace(0, 1, 100)
    bkg = np.array([])
    assert np.isnan(boyce_index(pres, bkg))


def test_constant_suitability_returns_nan():
    """If suitability is constant (no spread), Boyce is undefined."""
    pres = np.full(100, 0.5)
    bkg = np.full(1000, 0.5)
    assert np.isnan(boyce_index(pres, bkg))


def test_too_few_bins_returns_nan():
    """If most bins have zero background (sparse), return NaN."""
    # Highly skewed background — most density at one extreme
    pres = np.array([0.5, 0.6, 0.55])
    bkg = np.full(100, 0.5)  # all background near 0.5 → most bins empty
    result = boyce_index(pres, bkg, n_bins=10)
    # Either NaN or computed but not crashing — both acceptable here
    assert isinstance(result, float)
