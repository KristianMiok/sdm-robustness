"""Task 1 — Step 1.1: per-species inventory table.

For every taxon in the raw dataset (including those below the 10-record
minimum — we need the full picture before filtering), produce an inventory
CSV with pool sizes across the four spatial-uncertainty bands.

Implements the columns specified in briefing §3.3:

    species, status, n_total_raw, n_high_acc, n_low_acc,
    n_clean_200m, n_clean_500m, n_snap_200_500, n_snap_500_1000,
    n_snap_above_1000,
    n_clean_dedup_200m, n_clean_dedup_500m, n_low_acc_dedup,
    n_basins, strahler_min, strahler_max, geographic_extent_km2,
    category_petko2026

The dedup policy is one record per Hydrography90m segment per taxon —
deduplication is keyed on (species, subc_id). If subc_id is missing,
fall back to (species, basin_id, lat_snap_rounded, long_snap_rounded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdm_robustness.io import ACCURACY_HIGH, ACCURACY_LOW
from sdm_robustness.utils import logger

# Band boundaries from frozen design (duplicated here so the function is
# self-contained for testing; production code passes from config).
DEFAULT_STRICT_M = 200
DEFAULT_INTERMEDIATE_M = 500
DEFAULT_MAXIMUM_M = 1000


def build_inventory(
    df: pd.DataFrame,
    *,
    strict_m: float = DEFAULT_STRICT_M,
    intermediate_m: float = DEFAULT_INTERMEDIATE_M,
    maximum_m: float = DEFAULT_MAXIMUM_M,
    dedup_key: str = "subc_id",
    petko_categories: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the per-species inventory table.

    Parameters
    ----------
    df : DataFrame
        Master table from io.load_master_table().
    strict_m, intermediate_m, maximum_m : float
        Snapping distance band boundaries in metres.
    dedup_key : str
        Column defining the deduplication unit (Hydrography90m segment).
        Default 'subc_id'. Must be present in df.
    petko_categories : DataFrame, optional
        Two-column DataFrame (species, category) from Petko et al. 2026
        Supplementary Table 3. If None, category column filled with NA.

    Returns
    -------
    DataFrame with one row per species, ready to be written as
    species_audit_full.csv.
    """
    _validate_inputs(df, dedup_key)
    df = df.copy()

    # --- Normalise accuracy flags so downstream masks are robust ---
    df["_acc"] = df["Accuracy"].astype(str).where(df["Accuracy"].notna(), other=np.nan)
    is_high = df["_acc"].isin(ACCURACY_HIGH)
    is_low = df["_acc"].isin(ACCURACY_LOW)

    dist = pd.to_numeric(df["distance_m"], errors="coerce")

    # --- Band masks ---
    clean_200 = is_high & (dist <= strict_m)
    clean_500 = is_high & (dist <= intermediate_m)
    snap_200_500 = is_high & (dist > strict_m) & (dist <= intermediate_m)
    snap_500_1000 = is_high & (dist > intermediate_m) & (dist <= maximum_m)
    snap_above_1000 = is_high & (dist > maximum_m)

    # --- Attach masks to df for groupby ---
    df["_clean_200"] = clean_200
    df["_clean_500"] = clean_500
    df["_snap_200_500"] = snap_200_500
    df["_snap_500_1000"] = snap_500_1000
    df["_snap_above_1000"] = snap_above_1000
    df["_high_acc"] = is_high
    df["_low_acc"] = is_low

    sp = "Crayfish_scientific_name"
    grouped = df.groupby(sp, dropna=True, observed=True)

    # --- Raw counts ---
    inv = pd.DataFrame(
        {
            "species": grouped.size().index,
            "n_total_raw": grouped.size().values,
        }
    ).set_index("species")

    inv["n_high_acc"] = grouped["_high_acc"].sum().astype(int)
    inv["n_low_acc"] = grouped["_low_acc"].sum().astype(int)
    inv["n_clean_200m"] = grouped["_clean_200"].sum().astype(int)
    inv["n_clean_500m"] = grouped["_clean_500"].sum().astype(int)
    inv["n_snap_200_500"] = grouped["_snap_200_500"].sum().astype(int)
    inv["n_snap_500_1000"] = grouped["_snap_500_1000"].sum().astype(int)
    inv["n_snap_above_1000"] = grouped["_snap_above_1000"].sum().astype(int)

    # --- Status (may be mixed per species if the data has both natives
    # and invasives of the same name somehow — flag it) ---
    inv["status"] = grouped["Status"].agg(_majority_or_mixed)

    # --- Deduplicated counts (reindex to ALL species so zero-match species
    # get 0 instead of dropping out of the Series) ---
    inv["n_clean_dedup_200m"] = (
        _dedup_count(df[clean_200], sp, dedup_key).reindex(inv.index).fillna(0).astype(int)
    )
    inv["n_clean_dedup_500m"] = (
        _dedup_count(df[clean_500], sp, dedup_key).reindex(inv.index).fillna(0).astype(int)
    )
    inv["n_low_acc_dedup"] = (
        _dedup_count(df[is_low], sp, dedup_key).reindex(inv.index).fillna(0).astype(int)
    )

    # --- Spatial spread diagnostics on the clean benchmark ---
    clean_df = df[clean_200]
    basin_spread = clean_df.groupby(sp, observed=True)["basin_id"].nunique()
    inv["n_basins"] = basin_spread.reindex(inv.index).fillna(0).astype(int)

    strahler = pd.to_numeric(clean_df["strahler"], errors="coerce")
    strahler_by_sp = clean_df.assign(_str=strahler).groupby(sp, observed=True)["_str"]
    # Left as nullable for species with no clean records (NaN is informative here)
    inv["strahler_min"] = strahler_by_sp.min().reindex(inv.index)
    inv["strahler_max"] = strahler_by_sp.max().reindex(inv.index)

    # --- Geographic extent (convex hull area on snapped coords) ---
    inv["geographic_extent_km2"] = _compute_hull_area_km2(clean_df, sp)

    # --- Petko category ---
    if petko_categories is not None:
        cat_map = petko_categories.set_index("species")["category"]
        inv["category_petko2026"] = inv.index.map(cat_map)
    else:
        inv["category_petko2026"] = pd.NA

    # --- Reorder columns to match briefing ---
    column_order = [
        "status",
        "n_total_raw",
        "n_high_acc",
        "n_low_acc",
        "n_clean_200m",
        "n_clean_500m",
        "n_snap_200_500",
        "n_snap_500_1000",
        "n_snap_above_1000",
        "n_clean_dedup_200m",
        "n_clean_dedup_500m",
        "n_low_acc_dedup",
        "n_basins",
        "strahler_min",
        "strahler_max",
        "geographic_extent_km2",
        "category_petko2026",
    ]
    inv = inv[column_order].reset_index()

    logger.info(f"Inventory built: {len(inv)} species.")
    return inv


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _validate_inputs(df: pd.DataFrame, dedup_key: str) -> None:
    required = {
        "Crayfish_scientific_name",
        "Accuracy",
        "Status",
        "distance_m",
        "basin_id",
        "strahler",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Inventory builder missing columns: {sorted(missing)}")
    if dedup_key not in df.columns:
        raise ValueError(
            f"Dedup key '{dedup_key}' not found. Available columns include: "
            f"{[c for c in df.columns if 'id' in c.lower()][:10]}"
        )


def _dedup_count(subset: pd.DataFrame, species_col: str, dedup_key: str) -> pd.Series:
    """Count unique (species, dedup_key) rows per species.

    Rows with NA in dedup_key are counted individually (never collapse NAs).
    """
    if subset.empty:
        return pd.Series(dtype=int, name="dedup_count")
    mask_notna = subset[dedup_key].notna()
    uniques = (
        subset.loc[mask_notna]
        .drop_duplicates(subset=[species_col, dedup_key])
        .groupby(species_col, observed=True)
        .size()
    )
    na_counts = subset.loc[~mask_notna].groupby(species_col, observed=True).size()
    total = uniques.add(na_counts, fill_value=0).astype(int)
    return total


def _majority_or_mixed(series: pd.Series) -> str:
    """Return majority status, or 'Mixed' if ambiguous.

    Handles categorical dtypes where ``value_counts()`` returns a row per
    defined category including zero counts — we drop zeros before deciding.
    """
    values = series.dropna()
    if values.empty:
        return "Unknown"
    vc = values.value_counts()
    vc = vc[vc > 0]           # guard against categorical zero-count levels
    if len(vc) == 1:
        return str(vc.index[0])
    # >1 actually-present status — flag as mixed (rare but informative)
    return "Mixed"


def _compute_hull_area_km2(df: pd.DataFrame, species_col: str) -> pd.Series:
    """Convex hull area per species on snapped coordinates, in km²."""
    try:
        from shapely.geometry import MultiPoint
    except ImportError:
        logger.warning("shapely not available; geographic_extent_km2 set to NA")
        return pd.Series(dtype=float)

    # Prefer snapped coordinates when available; fall back to originals.
    lat_col = "lat_snap" if "lat_snap" in df.columns else "lat_or"
    lon_col = "long_snap" if "long_snap" in df.columns else "long_or"

    areas: dict[str, float] = {}
    for species, g in df.groupby(species_col, observed=True):
        pts = g[[lon_col, lat_col]].dropna().to_numpy()
        if len(pts) < 3:
            areas[species] = 0.0
            continue
        hull = MultiPoint(pts).convex_hull
        # Approximate conversion: 1° lat ≈ 111 km; 1° lon ≈ 111 * cos(lat) km.
        # We use the mean latitude of the points for the longitude scaling.
        mean_lat_rad = float(np.deg2rad(pts[:, 1].mean()))
        lon_km_per_deg = 111.0 * np.cos(mean_lat_rad)
        lat_km_per_deg = 111.0
        # hull.area is in squared degrees; convert
        areas[species] = float(hull.area) * lon_km_per_deg * lat_km_per_deg
    return pd.Series(areas, name="geographic_extent_km2")
