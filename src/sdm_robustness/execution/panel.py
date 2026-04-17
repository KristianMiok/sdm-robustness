from __future__ import annotations

import pandas as pd

from sdm_robustness.config import load_final_panel


def get_panel_entity(entity_name: str) -> pd.Series:
    df = load_final_panel()
    hit = df.loc[df["entity"] == entity_name]
    if hit.empty:
        raise ValueError(f"Entity not found in final panel: {entity_name}")
    if len(hit) > 1:
        raise ValueError(f"Entity appears multiple times in final panel: {entity_name}")
    return hit.iloc[0].copy()
