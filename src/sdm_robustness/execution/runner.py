from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sdm_robustness.config import load_final_panel
from sdm_robustness.execution import get_panel_entity
from sdm_robustness.io import load_master_table
from sdm_robustness.pipeline import fit_cv_cell, prepare_accessible_area
from sdm_robustness.utils.repro import derive_seed


HIGH_VALUES = {"high", "High", "HIGH"}
NATIVE_VALUES = {"Native", "Type locality"}
ALIEN_VALUES = {"Alien", "Introduced"}


def _is_high_accuracy(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(HIGH_VALUES)


def _dedup_by_subc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()


def _species_from_entity_name(entity_name: str) -> str:
    if " (" in entity_name:
        return entity_name.split(" (", 1)[0]
    return entity_name


def _prepare_entity_data(
    master_table: pd.DataFrame,
    entity_name: str,
) -> dict[str, object]:
    panel_row = get_panel_entity(entity_name)

    species = panel_row["species"] if "species" in panel_row.index else _species_from_entity_name(entity_name)
    category = str(panel_row["category"]).lower() if "category" in panel_row.index else "pooled"
    entity_type = str(panel_row["type"]).upper() if "type" in panel_row.index else "DUAL-AXIS"

    base = master_table[master_table["Crayfish_scientific_name"] == species].copy()

    if category == "native":
        base = base[base["Status"].isin(NATIVE_VALUES)].copy()
    elif category == "alien":
        base = base[base["Status"].isin(ALIEN_VALUES)].copy()

    benchmark = base[
        _is_high_accuracy(base["Accuracy"])
        & (base["distance_m"] <= 200)
    ].copy()
    benchmark = _dedup_by_subc(benchmark)

    if benchmark.empty:
        raise RuntimeError(f"Benchmark set is empty for entity={entity_name}")

    snap_pool = base[
        _is_high_accuracy(base["Accuracy"])
        & (base["distance_m"] > 200)
        & (base["distance_m"] <= 1000)
    ].copy()
    snap_pool = _dedup_by_subc(snap_pool)

    lowacc_pool = base[~_is_high_accuracy(base["Accuracy"])].copy()
    lowacc_pool = _dedup_by_subc(lowacc_pool)

    accessible_area = prepare_accessible_area(master_table, benchmark)

    return {
        "panel_row": panel_row,
        "species": species,
        "category": category,
        "entity_type": entity_type,
        "benchmark": benchmark,
        "snap_pool": snap_pool,
        "lowacc_pool": lowacc_pool,
        "accessible_area": accessible_area,
    }


def _axes_for_entity_type(entity_type: str) -> tuple[str, ...]:
    entity_type = str(entity_type).upper()
    if entity_type == "DUAL-AXIS":
        return ("snapping", "lowacc")
    if entity_type == "SNAPPING-ONLY":
        return ("snapping",)
    return ("snapping",)


def _checkpoint_rows(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def run_core_factorial(
    species_panel: pd.DataFrame,
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    algorithms: tuple[str, ...] = ("random_forest", "xgboost", "maxent"),
    axes: tuple[str, ...] = ("snapping", "lowacc"),
    levels_pct: tuple[int, ...] = (0, 5, 10, 20, 35, 50),
    scale_tracks: tuple[str, ...] = ("local_only", "upstream_only", "combined"),
    n_replicates_default: int = 30,
    n_replicates_low_levels: int = 50,
    low_level_threshold_pct: int = 10,
    master_seed: int = 20260416,
    maxent_background_n: int = 10_000,
    rf_xgb_pa_ratio: float = 1.0,
    n_splits: int = 5,
    looo_threshold: int = 15,
    checkpoint_every: int = 50,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    checkpoint_path = output_dir / "results_raw.checkpoint.parquet"
    final_path = output_dir / "results_raw.parquet"

    entity_names = species_panel["entity"].tolist()

    for entity_name in entity_names:
        prepared = _prepare_entity_data(master_table, entity_name)
        entity_type = prepared["entity_type"]
        valid_axes = tuple(a for a in _axes_for_entity_type(entity_type) if a in axes)

        for axis in valid_axes:
            contamination_pool = prepared["snap_pool"] if axis == "snapping" else prepared["lowacc_pool"]

            for level in levels_pct:
                n_replicates = (
                    n_replicates_low_levels
                    if level <= low_level_threshold_pct
                    else n_replicates_default
                )

                for replicate in range(n_replicates):
                    seed = derive_seed(master_seed, entity_name, axis, level, replicate)

                    for algorithm in algorithms:
                        for track in scale_tracks:
                            try:
                                row = fit_cv_cell(
                                    benchmark=prepared["benchmark"],
                                    contamination_pool=contamination_pool,
                                    accessible_area=prepared["accessible_area"],
                                    entity=entity_name,
                                    algorithm=algorithm,
                                    track=track,
                                    axis=axis,
                                    level=level,
                                    replicate=replicate,
                                    seed=seed,
                                    n_splits=n_splits,
                                    looo_threshold=looo_threshold,
                                    maxent_background_n=maxent_background_n,
                                    rf_xgb_pa_ratio=rf_xgb_pa_ratio,
                                    maxent_n_cpus=1,
                                )
                            except Exception as e:
                                row = {
                                    "entity": entity_name,
                                    "algorithm": algorithm,
                                    "track": track,
                                    "axis": axis,
                                    "level": level,
                                    "replicate": replicate,
                                    "seed": seed,
                                    "status": "error",
                                    "error_message": str(e),
                                }

                            row["species"] = prepared["species"]
                            row["category"] = prepared["category"]
                            row["entity_type"] = prepared["entity_type"]
                            rows.append(row)

                            if len(rows) % checkpoint_every == 0:
                                _checkpoint_rows(rows, checkpoint_path)

    _checkpoint_rows(rows, final_path)
    return final_path


def run_benchmark_sanity_check(
    species_panel: pd.DataFrame,
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    n_replicates: int = 30,
    master_seed: int = 20260416,
    algorithms: tuple[str, ...] = ("random_forest", "xgboost", "maxent"),
    scale_tracks: tuple[str, ...] = ("local_only", "upstream_only", "combined"),
    maxent_background_n: int = 10_000,
    rf_xgb_pa_ratio: float = 1.0,
    n_splits: int = 5,
    looo_threshold: int = 15,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    final_path = output_dir / "benchmark_stability.parquet"

    entity_names = species_panel["entity"].tolist()

    metric_cols = ["auc", "brier", "sensitivity", "specificity", "tss"]

    for entity_name in entity_names:
        prepared = _prepare_entity_data(master_table, entity_name)

        for replicate in range(n_replicates):
            seed = derive_seed(master_seed, entity_name, "benchmark", 0, replicate)

            for algorithm in algorithms:
                for track in scale_tracks:
                    try:
                        row = fit_cv_cell(
                            benchmark=prepared["benchmark"],
                            contamination_pool=prepared["benchmark"],
                            accessible_area=prepared["accessible_area"],
                            entity=entity_name,
                            algorithm=algorithm,
                            track=track,
                            axis="benchmark",
                            level=0,
                            replicate=replicate,
                            seed=seed,
                            n_splits=n_splits,
                            looo_threshold=looo_threshold,
                            maxent_background_n=maxent_background_n,
                            rf_xgb_pa_ratio=rf_xgb_pa_ratio,
                            maxent_n_cpus=1,
                        )
                    except Exception as e:
                        row = {
                            "entity": entity_name,
                            "algorithm": algorithm,
                            "track": track,
                            "axis": "benchmark",
                            "level": 0,
                            "replicate": replicate,
                            "seed": seed,
                            "status": "error",
                            "error_message": str(e),
                        }

                    row["species"] = prepared["species"]
                    row["category"] = prepared["category"]
                    row["entity_type"] = prepared["entity_type"]
                    raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    ok_df = raw_df[raw_df["status"] == "ok"].copy()

    if ok_df.empty:
        pd.DataFrame(summary_rows).to_parquet(final_path, index=False)
        return final_path

    group_cols = ["entity", "species", "category", "entity_type", "algorithm", "track"]

    for keys, sub in ok_df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))
        n_ok = int(len(sub))

        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            summary_rows.append(
                {
                    **key_map,
                    "metric_name": metric,
                    "benchmark_mean": float(vals.mean()) if len(vals) else np.nan,
                    "benchmark_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n_replicates_ok": n_ok,
                }
            )

    pd.DataFrame(summary_rows).to_parquet(final_path, index=False)
    return final_path


def run_transferability_test(
    top_species: list[str],
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    levels_pct: tuple[int, ...] = (10, 30, 50),
    master_seed: int = 20260416,
) -> Path:
    raise NotImplementedError("Task 5d — implement later, after core factorial is running.")


def run_null_model(
    species_panel: pd.DataFrame,
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    levels_pct: tuple[int, ...] = (0, 5, 10, 20, 35, 50),
    n_replicates: int = 30,
    master_seed: int = 20260416,
) -> Path:
    raise NotImplementedError("Null-model — implement later, after core factorial is running.")


def load_panel_and_master(
    data_path: str | None = None,
    entity: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_final_panel()
    if entity is not None:
        panel = panel.loc[panel["entity"] == entity].copy()
        if panel.empty:
            raise ValueError(f"Entity not found in final panel: {entity}")

    master_df, _ = load_master_table(data_path)
    return panel, master_df
