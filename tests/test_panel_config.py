from __future__ import annotations

from sdm_robustness.config import load_final_panel


def test_final_panel_loads():
    df = load_final_panel()
    assert len(df) == 13
    assert set(df["type"]) == {"DUAL", "SNAP"}


def test_dual_and_snap_counts():
    df = load_final_panel()
    assert (df["type"] == "DUAL").sum() == 8
    assert (df["type"] == "SNAP").sum() == 5


def test_snap_entities_do_not_run_lowacc():
    df = load_final_panel()
    snap = df[df["type"] == "SNAP"]
    assert (snap["run_lowacc"] == 0).all()
