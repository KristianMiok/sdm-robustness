"""Task 4 — three-tier degradation metrics framework + IST."""

from sdm_robustness.metrics.ist import IST, compute_ist
from sdm_robustness.metrics.tier1_performance import Tier1Metrics, compute_tier1, delta_tier1
from sdm_robustness.metrics.tier2_ecological import (
    importance_rank_correlation,
    niche_breadth_change,
    niche_centroid_displacement,
    response_curve_distance,
    top_k_jaccard,
)
from sdm_robustness.metrics.tier3_spatial import (
    range_area_percent_change,
    schoeners_d,
    spatial_mismatch_map,
    warrens_i,
)

__all__ = [
    "IST",
    "Tier1Metrics",
    "compute_ist",
    "compute_tier1",
    "delta_tier1",
    "importance_rank_correlation",
    "niche_breadth_change",
    "niche_centroid_displacement",
    "range_area_percent_change",
    "response_curve_distance",
    "schoeners_d",
    "spatial_mismatch_map",
    "top_k_jaccard",
    "warrens_i",
]
