"""Tests for utils.repro."""

from __future__ import annotations

from sdm_robustness.utils.repro import derive_seed, rng


def test_derive_seed_deterministic():
    s1 = derive_seed(20260416, "Austropotamobius_torrentium", "snapping", 20, 7, "RF")
    s2 = derive_seed(20260416, "Austropotamobius_torrentium", "snapping", 20, 7, "RF")
    assert s1 == s2


def test_derive_seed_differs_on_component_change():
    s1 = derive_seed(20260416, "SpA", "snapping", 20, 0, "RF")
    s2 = derive_seed(20260416, "SpA", "snapping", 20, 1, "RF")  # different replicate
    assert s1 != s2


def test_derive_seed_fits_uint32():
    s = derive_seed(20260416, "SpA", "axis", 50, 99, "XGB")
    assert 0 <= s < 2**32


def test_rng_deterministic():
    g1 = rng(20260416, "SpA", 0)
    g2 = rng(20260416, "SpA", 0)
    a1 = g1.integers(0, 1_000_000, size=5)
    a2 = g2.integers(0, 1_000_000, size=5)
    assert (a1 == a2).all()
