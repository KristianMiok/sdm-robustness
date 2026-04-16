"""Data loading and schema validation for the Petko et al. 2026 master table.

The master table (`combined_data_true_master.csv`) contains 115,191 crayfish
occurrence records with Hydrography90m / GeoFRESH environmental variables.

Expected schema (from the file head Kristian inspected on 2026-04-16):
- Identifier / provenance: WoCID, lat_or, long_or, Accuracy, Crayfish_scientific_name,
  Status, Year_of_record
- Network: basin_id, subc_id, strahler, reg_id, hylak_id, is_coastal
- Snap QC: lat_snap, long_snap, distance_m, ab_200m, ab_500m, ab_1000m
- Predictors (Local):    l_CLI1..l_CLI76, l_TOP1..l_TOP144, l_LAC1..l_LAC22, l_SOL1..l_SOL60
- Predictors (Upstream): u_CLI*, u_TOP*, u_LAC*, u_SOL* (subset of Local columns)

Note on NA handling: rows with NA in upstream (u_*) predictors exist when the
record is headwater / tiny catchment — these are NOT malformed. We keep them
and handle imputation / exclusion downstream per scale track.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sdm_robustness.utils import logger, resolve_path


# --- Required columns for Task 1 audit ---
# These MUST be present; loader fails loudly if any are missing.
REQUIRED_AUDIT_COLUMNS: tuple[str, ...] = (
    "WoCID",
    "lat_or",
    "long_or",
    "Accuracy",
    "Crayfish_scientific_name",
    "Status",
    "basin_id",
    "strahler",
    "distance_m",
    "ab_200m",
    "ab_500m",
    "ab_1000m",
)

# Values treated as the two accuracy levels (defensive: the briefing says
# "High / Low" but real data sometimes has casing inconsistencies).
ACCURACY_HIGH = {"High", "high", "HIGH"}
ACCURACY_LOW = {"Low", "low", "LOW"}


@dataclass
class MasterTableInfo:
    """Summary of a loaded master table — logged for reproducibility."""

    n_records: int
    n_species: int
    n_columns: int
    path: Path
    accuracy_values: list[str]
    status_values: list[str]


def load_master_table(
    path: Path | str | None = None,
    *,
    required_columns: tuple[str, ...] = REQUIRED_AUDIT_COLUMNS,
    dtype_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, MasterTableInfo]:
    """Load and schema-validate the Petko et al. 2026 master table.

    Parameters
    ----------
    path : path, optional
        Path to the CSV. If None, resolves from configs/paths.yaml →
        configs/paths.local.yaml → env var SDM_RAW_DATA_PATH.
    required_columns : tuple
        Columns that must be present. Fails loudly if any are missing.
    dtype_overrides : dict, optional
        Column name → dtype string. Useful for forcing boolean cols.

    Returns
    -------
    (df, info) : pandas DataFrame and summary dataclass.
    """
    if path is None:
        from sdm_robustness.utils.config import load_paths
        path = resolve_path(load_paths()["raw_data_path"])
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Master table not found at: {path}\n"
            "Update configs/paths.yaml, create configs/paths.local.yaml, "
            "or set SDM_RAW_DATA_PATH."
        )

    logger.info(f"Loading master table from {path}")

    # Default dtype handling: the boolean ab_* columns come in as 'TRUE'/'FALSE'
    # strings in the file Kristian inspected.
    dtypes: dict[str, str] = {
        "WoCID": "string",
        "Crayfish_scientific_name": "string",
        "Status": "category",
        "Accuracy": "category",
    }
    if dtype_overrides:
        dtypes.update(dtype_overrides)

    # Read CSV. We let pandas infer most columns — 398 predictors would be
    # painful to enumerate, and they're all numeric anyway.
    df = pd.read_csv(path, dtype=dtypes, low_memory=False, na_values=["NA", "", "NaN"])

    # Coerce boolean ab_* flags (TRUE/FALSE strings → bool)
    for col in ("ab_200m", "ab_500m", "ab_1000m", "is_coastal"):
        if col in df.columns:
            df[col] = _coerce_bool(df[col])

    # Schema check
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Master table missing required columns: {missing}\n"
            f"Present columns (first 50): {df.columns.tolist()[:50]}"
        )

    info = MasterTableInfo(
        n_records=len(df),
        n_species=df["Crayfish_scientific_name"].nunique(),
        n_columns=len(df.columns),
        path=path,
        accuracy_values=sorted(df["Accuracy"].dropna().unique().tolist()),
        status_values=sorted(df["Status"].dropna().unique().tolist()),
    )

    logger.info(
        f"Loaded {info.n_records:,} records, {info.n_species} unique taxa, "
        f"{info.n_columns} columns."
    )
    logger.info(f"Accuracy values present: {info.accuracy_values}")
    logger.info(f"Status values present: {info.status_values}")

    return df, info


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce TRUE/FALSE/True/False/1/0 strings to nullable boolean."""
    if series.dtype == bool:
        return series
    mapping = {
        "TRUE": True, "True": True, "true": True, "1": True, 1: True, True: True,
        "FALSE": False, "False": False, "false": False, "0": False, 0: False, False: False,
    }
    return series.map(mapping).astype("boolean")


def get_predictor_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify predictor columns by scale (Local/Upstream) and domain.

    Returns
    -------
    dict with keys: 'local_CLI', 'local_TOP', 'local_LAC', 'local_SOL',
                    'upstream_CLI', 'upstream_TOP', 'upstream_LAC', 'upstream_SOL'.
    """
    cols = df.columns.tolist()
    out: dict[str, list[str]] = {}
    for scale_prefix, scale_name in (("l_", "local"), ("u_", "upstream")):
        for domain in ("CLI", "TOP", "LAC", "SOL"):
            key = f"{scale_name}_{domain}"
            out[key] = sorted(
                [c for c in cols if c.startswith(f"{scale_prefix}{domain}")],
                key=_numeric_suffix_sort_key,
            )
    return out


def _numeric_suffix_sort_key(col: str) -> tuple[str, int]:
    """Sort 'l_CLI1', 'l_CLI2', ..., 'l_CLI10' correctly (not lexicographic)."""
    # Find the numeric tail
    i = len(col)
    while i > 0 and col[i - 1].isdigit():
        i -= 1
    prefix = col[:i]
    num_str = col[i:]
    num = int(num_str) if num_str else 0
    return (prefix, num)
