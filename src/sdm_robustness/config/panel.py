from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "entity",
    "type",
    "class_group",
    "treatment",
    "category",
    "run_snapping",
    "run_lowacc",
    "notes",
}


def load_final_panel(path: str | Path = "config/final_panel.csv") -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required panel columns: {sorted(missing)}")

    if df["entity"].duplicated().any():
        dups = df.loc[df["entity"].duplicated(), "entity"].tolist()
        raise ValueError(f"Duplicate entities in panel file: {dups}")

    allowed_types = {"DUAL", "SNAP"}
    bad_types = set(df["type"]) - allowed_types
    if bad_types:
        raise ValueError(f"Unexpected panel types: {sorted(bad_types)}")

    for col in ["run_snapping", "run_lowacc"]:
        bad = set(df[col].dropna().astype(int)) - {0, 1}
        if bad:
            raise ValueError(f"{col} must contain only 0/1 values, found {sorted(bad)}")

    # Logical consistency
    invalid = df[(df["type"] == "SNAP") & (df["run_lowacc"] != 0)]
    if not invalid.empty:
        raise ValueError("SNAP entities cannot have run_lowacc=1")

    return df.copy()
