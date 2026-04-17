from __future__ import annotations

from sdm_robustness.metrics.schema import RunMetadata, MetricRecord


def test_run_metadata_constructs():
    meta = RunMetadata(
        entity="Astacus astacus",
        axis="snap",
        contamination_level=1.0,
        replicate=1,
        algorithm="rf",
        spatial_scale="local",
        seed=42,
        benchmark=False,
    )
    assert meta.entity == "Astacus astacus"


def test_metric_record_to_dict():
    rec = MetricRecord(
        entity="Astacus astacus",
        axis="snap",
        contamination_level=1.0,
        replicate=1,
        algorithm="rf",
        spatial_scale="local",
        seed=42,
        benchmark=False,
        metric_name="auc",
        metric_value=0.81,
        metric_tier="tier1",
    )
    d = rec.to_dict()
    assert d["metric_name"] == "auc"
    assert d["metric_tier"] == "tier1"
