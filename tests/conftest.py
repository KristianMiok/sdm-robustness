"""Shared pytest fixtures.

Provides a small synthetic master-table DataFrame with the expected schema,
designed so the audit outputs are deterministic and hand-verifiable.

Synthetic design:
- 3 species:
    Austropotamobius_mega     — widespread, well-behaved: passes all gates
    Astacus_mini              — endemic: small benchmark but well-spread
    Procambarus_partial       — regional, rich snap pool but sparse lowacc
- Distances chosen to land specific record counts in each band.
- Deliberate NA in upstream predictors for some rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_master_table() -> pd.DataFrame:
    """Deterministic synthetic master table for audit tests."""
    rows: list[dict] = []
    wocid = 1

    def _rec(species, status, acc, dist, basin, subc, strahler, lat=45.7, lon=9.3):
        nonlocal wocid
        rec = {
            "WoCID": f"WA{wocid:04d}",
            "lat_or": lat,
            "long_or": lon,
            "lat_snap": lat,
            "long_snap": lon,
            "Accuracy": acc,
            "Crayfish_scientific_name": species,
            "Status": status,
            "Year_of_record": 2020,
            "basin_id": basin,
            "subc_id": subc,
            "strahler": strahler,
            "distance_m": dist,
            "ab_200m": bool(dist <= 200 and acc == "High"),
            "ab_500m": bool(dist <= 500 and acc == "High"),
            "ab_1000m": bool(dist <= 1000 and acc == "High"),
            "is_coastal": False,
            "l_CLI1": 12.0 + (wocid % 5) * 0.1,
        }
        wocid += 1
        return rec

    # --- Species 1: widespread, passes all gates ---
    # 600 clean ≤200m, 600 snap 200-500, 300 snap 500-1000, 700 lowacc
    # across 6 basins and strahler 2,3,4,5,6,7
    for i in range(600):
        rows.append(_rec(
            "Austropotamobius_mega", "Native", "High", dist=100,
            basin=1000 + (i % 6), subc=10_000 + i,
            strahler=2 + (i % 6), lat=45.0 + (i % 6) * 0.1,
        ))
    for i in range(600):
        rows.append(_rec(
            "Austropotamobius_mega", "Native", "High", dist=350,
            basin=1000 + (i % 6), subc=20_000 + i,
            strahler=2 + (i % 6),
        ))
    for i in range(300):
        rows.append(_rec(
            "Austropotamobius_mega", "Native", "High", dist=750,
            basin=1000 + (i % 6), subc=30_000 + i,
            strahler=2 + (i % 6),
        ))
    for i in range(700):
        rows.append(_rec(
            "Austropotamobius_mega", "Native", "Low", dist=100,
            basin=1000 + (i % 6), subc=40_000 + i,
            strahler=2 + (i % 6),
        ))

    # --- Species 2: endemic but well-spread ---
    # 120 clean ≤200m in 4 basins, strahler 2,3,4 — small but passes endemic gate
    # 140 snap 200-500, 150 lowacc
    for i in range(120):
        rows.append(_rec(
            "Astacus_mini", "Native", "High", dist=150,
            basin=2000 + (i % 4), subc=50_000 + i,
            strahler=2 + (i % 3),
        ))
    for i in range(140):
        rows.append(_rec(
            "Astacus_mini", "Native", "High", dist=400,
            basin=2000 + (i % 4), subc=60_000 + i,
            strahler=2 + (i % 3),
        ))
    for i in range(150):
        rows.append(_rec(
            "Astacus_mini", "Native", "Low", dist=100,
            basin=2000 + (i % 4), subc=70_000 + i,
            strahler=2 + (i % 3),
        ))

    # --- Species 3: regional, passes gates 1/4/5 + gate 2, fails gate 3 ---
    # 250 clean, 300 snap, only 50 lowacc (below benchmark size → PARTIAL)
    for i in range(250):
        rows.append(_rec(
            "Procambarus_partial", "Alien", "High", dist=180,
            basin=3000 + (i % 3), subc=80_000 + i,
            strahler=3 + (i % 3),
        ))
    for i in range(300):
        rows.append(_rec(
            "Procambarus_partial", "Alien", "High", dist=450,
            basin=3000 + (i % 3), subc=90_000 + i,
            strahler=3 + (i % 3),
        ))
    for i in range(50):
        rows.append(_rec(
            "Procambarus_partial", "Alien", "Low", dist=100,
            basin=3000 + (i % 3), subc=95_000 + i,
            strahler=3 + (i % 3),
        ))

    df = pd.DataFrame(rows)
    # Inject some NA in upstream-style column to make sure audit handles it
    # (audit doesn't actually look at u_* columns, but io/loader should).
    df["u_CLI3"] = np.where(df["strahler"] == 2, np.nan, 10.0)
    df["Accuracy"] = df["Accuracy"].astype("category")
    df["Status"] = df["Status"].astype("category")
    return df


@pytest.fixture
def default_gates_config() -> dict:
    """Mirror of configs/task1_gates.yaml used in tests."""
    return {
        "gate_1_minimum_benchmark": {"widespread": 500, "regional": 200, "endemic": 80},
        "gate_2_snapping_pool": {"enabled": True},
        "gate_3_lowacc_pool": {"enabled": True},
        "gate_4_basin_spread": {"min_basins": 3},
        "gate_5_strahler_spread": {"min_distinct_orders": 3},
        "borderline_margin_pct": 10.0,
    }
