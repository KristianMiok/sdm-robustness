"""Variable → domain mapping (CLI / TOP / SOL / LAC).

Loads the canonical mapping from CSV (default: data/variable_domain_mapping.csv)
and provides utilities to assert all predictors used in a track are mapped.

The mapping was extracted from Petko et al. 2026 Supplementary Table S2
and is the single source of truth — do not derive domains from variable
name prefixes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DOMAINS = ("CLI", "TOP", "SOL", "LAC")


def load_domain_map(
    path: str | Path = "data/variable_domain_mapping.csv",
) -> dict[str, str]:
    """Load variable → domain mapping from CSV.

    Parameters
    ----------
    path : str or Path
        Path to the mapping CSV. Must have columns 'variable' and 'domain'.

    Returns
    -------
    dict[str, str]
        Mapping from variable name (e.g. 'l_CLI1') to domain code (one of CLI/TOP/SOL/LAC).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file has unexpected columns or contains unknown domain codes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Domain mapping file not found: {p}")

    df = pd.read_csv(p)

    expected = {"variable", "domain"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Domain mapping must have columns {expected}, got {list(df.columns)}"
        )

    unknown = set(df["domain"].unique()) - set(DOMAINS)
    if unknown:
        raise ValueError(f"Unknown domain codes in mapping: {unknown}")

    duplicates = df[df["variable"].duplicated()]["variable"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate variables in mapping: {duplicates[:5]}")

    return dict(zip(df["variable"], df["domain"]))


def assert_all_predictors_mapped(
    predictors: list[str], domain_map: dict[str, str]
) -> None:
    """Verify every predictor has a domain assignment. Raise on any missing.

    Parameters
    ----------
    predictors : list[str]
        Variable names used in the cleaned predictor set.
    domain_map : dict[str, str]
        The loaded mapping (from load_domain_map).

    Raises
    ------
    KeyError
        If any predictor is not in the mapping.
    """
    missing = [p for p in predictors if p not in domain_map]
    if missing:
        raise KeyError(
            f"{len(missing)} predictor(s) not in domain mapping: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )


def aggregate_to_domain_share(
    importance_vec: dict[str, float], domain_map: dict[str, str]
) -> dict[str, float]:
    """Aggregate per-variable importance to per-domain share.

    Each domain's share = sum of |importance| for variables in that domain,
    divided by sum across all domains. Output keys are
    'CLI_share', 'TOP_share', 'SOL_share', 'LAC_share'.

    Variables with importance = 0 do not contribute. If the total is 0 (degenerate
    model), all shares are NaN.

    Parameters
    ----------
    importance_vec : dict[str, float]
        Per-variable importance (e.g. RF feature_importances_ as {var: value}).
    domain_map : dict[str, str]
        Variable → domain.

    Returns
    -------
    dict[str, float]
        {'CLI_share': ..., 'TOP_share': ..., 'SOL_share': ..., 'LAC_share': ...}
    """
    sums = {d: 0.0 for d in DOMAINS}
    for var, imp in importance_vec.items():
        if var not in domain_map:
            continue
        sums[domain_map[var]] += abs(float(imp))

    total = sum(sums.values())
    if total == 0:
        return {f"{d}_share": float("nan") for d in DOMAINS}

    return {f"{d}_share": sums[d] / total for d in DOMAINS}


def domain_shift(
    contaminated_share: dict[str, float], benchmark_share: dict[str, float]
) -> dict[str, float]:
    """Signed difference between contaminated and benchmark domain shares.

    Output keys: 'CLI_shift', 'TOP_shift', 'SOL_shift', 'LAC_shift'.
    Positive means the contaminated model leans more on that domain.
    """
    return {
        f"{d}_shift": contaminated_share.get(f"{d}_share", float("nan"))
        - benchmark_share.get(f"{d}_share", float("nan"))
        for d in DOMAINS
    }


def domain_rank_stable(
    contaminated_share: dict[str, float], benchmark_share: dict[str, float]
) -> bool:
    """Whether the rank order of domains is identical between benchmark and contaminated.

    Compares ordering of CLI/TOP/SOL/LAC by share value, descending. Returns False
    if either input contains NaN.
    """
    bench = [(benchmark_share.get(f"{d}_share", float("nan")), d) for d in DOMAINS]
    cont = [(contaminated_share.get(f"{d}_share", float("nan")), d) for d in DOMAINS]

    if any(pd.isna(v) for v, _ in bench + cont):
        return False

    bench_order = [d for _, d in sorted(bench, reverse=True)]
    cont_order = [d for _, d in sorted(cont, reverse=True)]
    return bench_order == cont_order
