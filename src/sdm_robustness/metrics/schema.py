from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RunMetadata:
    entity: str
    axis: str
    contamination_level: float
    replicate: int
    algorithm: str
    spatial_scale: str
    seed: int
    benchmark: bool


@dataclass
class MetricRecord:
    entity: str
    axis: str
    contamination_level: float
    replicate: int
    algorithm: str
    spatial_scale: str
    seed: int
    benchmark: bool
    metric_name: str
    metric_value: float
    metric_tier: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
