"""Reusable metrics for robustness analysis."""

from sdm_robustness.metrics.core import (
    compute_performance_metrics,
    compute_delta_performance,
    spearman_importance_stability,
    topk_jaccard,
    integrated_absolute_difference,
    centroid_displacement,
    niche_breadth_change,
    schoeners_d,
    warrens_i,
    range_area_change,
    spatial_mismatch_summary,
    compute_ist,
    compute_domain_importance_shift,
    domain_rank_stability,
    classify_against_benchmark_envelope,
)

__all__ = [
    "boyce_index",
    "compute_performance_metrics",
    "compute_delta_performance",
    "spearman_importance_stability",
    "topk_jaccard",
    "integrated_absolute_difference",
    "centroid_displacement",
    "niche_breadth_change",
    "schoeners_d",
    "warrens_i",
    "range_area_change",
    "spatial_mismatch_summary",
    "compute_ist",
    "compute_domain_importance_shift",
    "domain_rank_stability",
    "classify_against_benchmark_envelope",
]

from sdm_robustness.metrics.schema import RunMetadata, MetricRecord
