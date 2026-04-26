"""Tests for domain_map module."""
from __future__ import annotations

import pytest
import pandas as pd

from sdm_robustness.metrics.domain_map import (
    DOMAINS,
    load_domain_map,
    assert_all_predictors_mapped,
    aggregate_to_domain_share,
    domain_shift,
    domain_rank_stable,
)


@pytest.fixture
def tmp_mapping_csv(tmp_path):
    p = tmp_path / "mapping.csv"
    pd.DataFrame({
        "variable": ["l_CLI1", "l_CLI2", "l_TOP1", "l_SOL1", "l_LAC1", "u_CLI1"],
        "domain":   ["CLI",    "CLI",    "TOP",    "SOL",    "LAC",    "CLI"],
    }).to_csv(p, index=False)
    return p


def test_load_basic(tmp_mapping_csv):
    m = load_domain_map(tmp_mapping_csv)
    assert m["l_CLI1"] == "CLI"
    assert m["u_CLI1"] == "CLI"
    assert m["l_TOP1"] == "TOP"
    assert len(m) == 6


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_domain_map(tmp_path / "nope.csv")


def test_load_bad_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="must have columns"):
        load_domain_map(p)


def test_load_unknown_domain_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"variable": ["a"], "domain": ["BOGUS"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Unknown domain"):
        load_domain_map(p)


def test_load_duplicate_variable_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({
        "variable": ["l_CLI1", "l_CLI1"],
        "domain":   ["CLI",    "TOP"],
    }).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_domain_map(p)


def test_assert_all_predictors_mapped_passes(tmp_mapping_csv):
    m = load_domain_map(tmp_mapping_csv)
    assert_all_predictors_mapped(["l_CLI1", "l_TOP1"], m)


def test_assert_all_predictors_mapped_fails():
    m = {"l_CLI1": "CLI"}
    with pytest.raises(KeyError, match="not in domain mapping"):
        assert_all_predictors_mapped(["l_CLI1", "l_NEW1"], m)


def test_aggregate_to_domain_share_basic():
    importance = {
        "l_CLI1": 0.4,
        "l_CLI2": 0.1,  # CLI total = 0.5
        "l_TOP1": 0.3,  # TOP total = 0.3
        "l_SOL1": 0.2,  # SOL total = 0.2
        # LAC = 0
    }
    dmap = {"l_CLI1": "CLI", "l_CLI2": "CLI", "l_TOP1": "TOP", "l_SOL1": "SOL"}
    shares = aggregate_to_domain_share(importance, dmap)
    assert shares["CLI_share"] == pytest.approx(0.5)
    assert shares["TOP_share"] == pytest.approx(0.3)
    assert shares["SOL_share"] == pytest.approx(0.2)
    assert shares["LAC_share"] == pytest.approx(0.0)


def test_aggregate_to_domain_share_negative_importances_use_abs():
    """RF importances are non-negative, but Maxent coefs can be negative."""
    importance = {"v1": -0.3, "v2": 0.7}
    dmap = {"v1": "CLI", "v2": "TOP"}
    shares = aggregate_to_domain_share(importance, dmap)
    assert shares["CLI_share"] == pytest.approx(0.3)
    assert shares["TOP_share"] == pytest.approx(0.7)


def test_aggregate_to_domain_share_unmapped_skipped():
    importance = {"v1": 0.5, "v_unmapped": 1.0}
    dmap = {"v1": "CLI"}
    shares = aggregate_to_domain_share(importance, dmap)
    # unmapped variable ignored, only v1 contributes → CLI = 100%
    assert shares["CLI_share"] == pytest.approx(1.0)


def test_aggregate_to_domain_share_all_zero_returns_nan():
    importance = {"v1": 0.0, "v2": 0.0}
    dmap = {"v1": "CLI", "v2": "TOP"}
    shares = aggregate_to_domain_share(importance, dmap)
    for d in DOMAINS:
        assert pd.isna(shares[f"{d}_share"])


def test_domain_shift_basic():
    bench = {"CLI_share": 0.5, "TOP_share": 0.3, "SOL_share": 0.2, "LAC_share": 0.0}
    cont  = {"CLI_share": 0.6, "TOP_share": 0.2, "SOL_share": 0.2, "LAC_share": 0.0}
    shift = domain_shift(cont, bench)
    assert shift["CLI_shift"] == pytest.approx(0.1)
    assert shift["TOP_shift"] == pytest.approx(-0.1)
    assert shift["SOL_shift"] == pytest.approx(0.0)
    assert shift["LAC_shift"] == pytest.approx(0.0)


def test_domain_rank_stable_true():
    s = {"CLI_share": 0.5, "TOP_share": 0.3, "SOL_share": 0.2, "LAC_share": 0.0}
    s2 = {"CLI_share": 0.6, "TOP_share": 0.25, "SOL_share": 0.1, "LAC_share": 0.05}
    # Same rank order: CLI > TOP > SOL > LAC
    assert domain_rank_stable(s2, s)


def test_domain_rank_stable_false():
    bench = {"CLI_share": 0.5, "TOP_share": 0.3, "SOL_share": 0.2, "LAC_share": 0.0}
    cont  = {"CLI_share": 0.2, "TOP_share": 0.5, "SOL_share": 0.3, "LAC_share": 0.0}
    # Order changed: bench is CLI>TOP>SOL>LAC, cont is TOP>SOL>CLI>LAC
    assert not domain_rank_stable(cont, bench)


def test_domain_rank_stable_with_nan_returns_false():
    s = {"CLI_share": float("nan"), "TOP_share": 0.3, "SOL_share": 0.2, "LAC_share": 0.0}
    s2 = {"CLI_share": 0.6, "TOP_share": 0.25, "SOL_share": 0.1, "LAC_share": 0.05}
    assert not domain_rank_stable(s2, s)
