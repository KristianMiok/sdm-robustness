from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sdm_robustness.config import load_final_panel
from sdm_robustness.execution import get_panel_entity
from sdm_robustness.io import load_master_table
from sdm_robustness.pipeline import fit_cv_cell, prepare_accessible_area
from sdm_robustness.metrics.domain_map import load_domain_map
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

    species = _species_from_entity_name(entity_name)
    category = str(panel_row.get("category", "")).lower() if "category" in panel_row.index else ""
    entity_type = str(panel_row.get("type", "DUAL")).upper() if "type" in panel_row.index else "DUAL"
    treatment = str(panel_row.get("treatment", "")).lower() if "treatment" in panel_row.index else ""

    base = master_table[master_table["Crayfish_scientific_name"] == species].copy()

    if treatment == "native_only":
        base = base[base["Status"].isin(NATIVE_VALUES)].copy()
    elif treatment == "alien_only":
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
        "treatment": treatment,
        "benchmark": benchmark,
        "snap_pool": snap_pool,
        "lowacc_pool": lowacc_pool,
        "accessible_area": accessible_area,
    }


def _axes_for_panel_row(panel_row: pd.Series, allowed_axes: tuple[str, ...]) -> tuple[str, ...]:
    """Respect the explicit run_snapping / run_lowacc columns if present,
    otherwise fall back to type-based dispatch."""
    axes = []

    if "run_snapping" in panel_row.index:
        if int(panel_row["run_snapping"]) == 1 and "snapping" in allowed_axes:
            axes.append("snapping")
    else:
        if "snapping" in allowed_axes:
            axes.append("snapping")

    if "run_lowacc" in panel_row.index:
        if int(panel_row["run_lowacc"]) == 1 and "lowacc" in allowed_axes:
            axes.append("lowacc")
    else:
        entity_type = str(panel_row.get("type", "DUAL")).upper()
        if entity_type in ("DUAL", "DUAL-AXIS") and "lowacc" in allowed_axes:
            axes.append("lowacc")

    return tuple(axes)


def _compute_n_experiment(
    benchmark_n: int,
    snap_pool_n: int,
    lowacc_pool_n: int,
    axes: tuple[str, ...],
    max_level_pct: int,
) -> int:
    """Compute a single n_experiment that is feasible across all axes and levels.

    The constraint is: for each active axis, n_experiment * max_level/100 <= pool_size.
    Also n_experiment <= benchmark_n (can't sample more clean records than we have).
    """
    frac = max_level_pct / 100.0
    caps = [benchmark_n]

    if "snapping" in axes and frac > 0:
        caps.append(int(snap_pool_n / frac))
    if "lowacc" in axes and frac > 0:
        caps.append(int(lowacc_pool_n / frac))

    return max(10, min(caps))


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
    n_experiment_override: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    checkpoint_path = output_dir / "results_raw.checkpoint.parquet"
    final_path = output_dir / "results_raw.parquet"

    entity_names = species_panel["entity"].tolist()
    max_level = max(levels_pct)

    for entity_name in entity_names:
        prepared = _prepare_entity_data(master_table, entity_name)
        panel_row = prepared["panel_row"]
        valid_axes = _axes_for_panel_row(panel_row, axes)

        benchmark_n = len(prepared["benchmark"])
        snap_pool_n = len(prepared["snap_pool"])
        lowacc_pool_n = len(prepared["lowacc_pool"])

        if n_experiment_override is not None:
            n_experiment = n_experiment_override
        else:
            n_experiment = _compute_n_experiment(
                benchmark_n=benchmark_n,
                snap_pool_n=snap_pool_n,
                lowacc_pool_n=lowacc_pool_n,
                axes=valid_axes,
                max_level_pct=max_level,
            )

        print(
            f"[{entity_name}] axes={valid_axes} benchmark_n={benchmark_n} "
            f"snap_pool={snap_pool_n} lowacc_pool={lowacc_pool_n} n_experiment={n_experiment}",
            flush=True,
        )

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
                                    n_experiment=n_experiment,
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
                            row["n_experiment"] = n_experiment
                            rows.append(row)

                            if len(rows) % checkpoint_every == 0:
                                _checkpoint_rows(rows, checkpoint_path)

    _checkpoint_rows(rows, final_path)
    return final_path


def run_grid_b_factorial(
    species_panel: pd.DataFrame,
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    algorithms: tuple[str, ...] = ("random_forest", "xgboost", "maxent"),
    snap_levels_pct: tuple[int, ...] = (0, 1, 2, 5),
    lowacc_levels_pct: tuple[int, ...] = (0, 3, 10, 20),
    scale_tracks: tuple[str, ...] = ("local_only", "upstream_only", "combined"),
    n_replicates: int = 30,
    master_seed: int = 20260426,
    maxent_background_n: int = 10_000,
    rf_xgb_pa_ratio: float = 1.0,
    n_splits: int = 5,
    looo_threshold: int = 15,
    checkpoint_every: int = 50,
    domain_map_path: str | Path = "data/variable_domain_mapping.csv",
    save_surfaces: bool = True,
) -> Path:
    """Grid B execution: full benchmark N, asymmetric levels, Tier 2/3 metrics.

    Differences from run_core_factorial:
      - N_experiment = full benchmark (no subsampling cap)
      - Snapping levels: 0/1/2/5 ; lowacc levels: 0/3/10/20
      - Pre-computes benchmark importance + surface once per (entity, alg, track),
        passes them into all contaminated cells for Tier 2/3 metrics
      - Saves full importance vectors as separate parquet
      - Saves benchmark suitability surfaces and one representative replicate
        per (entity x axis x max-level) for spatial mismatch figures
      - Every row tagged with grid_id="B"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir = output_dir / "surfaces"
    if save_surfaces:
        surfaces_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    importance_rows: list[dict] = []
    checkpoint_path = output_dir / "results_raw.checkpoint.parquet"
    final_path = output_dir / "results_raw.parquet"
    importance_path = output_dir / "variable_importance_vectors.parquet"

    domain_map = load_domain_map(domain_map_path)
    entity_names = species_panel["entity"].tolist()

    for entity_name in entity_names:
        prepared = _prepare_entity_data(master_table, entity_name)
        panel_row = prepared["panel_row"]
        valid_axes = _axes_for_panel_row(panel_row, ("snapping", "lowacc"))
        benchmark_n = len(prepared["benchmark"])
        n_experiment = benchmark_n  # Grid B: full benchmark, no cap

        print(
            f"[{entity_name}] Grid B: axes={valid_axes} benchmark_n={benchmark_n} "
            f"n_experiment={n_experiment}",
            flush=True,
        )

        # ===== STEP 1: Pre-compute benchmark artifacts per (algorithm, track) =====
        bench_artifacts: dict[tuple[str, str], dict] = {}
        for algorithm in algorithms:
            for track in scale_tracks:
                bench_seed = derive_seed(master_seed, entity_name, "benchmark", 0, 0)
                # Use the snap_pool as a placeholder contamination pool — won't be used at level=0
                pool_for_bench = prepared["snap_pool"] if "snapping" in valid_axes else prepared["lowacc_pool"]
                try:
                    bench_row = fit_cv_cell(
                        benchmark=prepared["benchmark"],
                        contamination_pool=pool_for_bench,
                        accessible_area=prepared["accessible_area"],
                        entity=entity_name,
                        algorithm=algorithm,
                        track=track,
                        axis="benchmark",
                        level=0,
                        replicate=0,
                        seed=bench_seed,
                        n_splits=n_splits,
                        looo_threshold=looo_threshold,
                        maxent_background_n=maxent_background_n,
                        rf_xgb_pa_ratio=rf_xgb_pa_ratio,
                        maxent_n_cpus=1,
                        n_experiment=n_experiment,
                        return_artifacts=True,
                        domain_map=domain_map,
                    )
                except Exception as e:
                    print(f"[{entity_name}/{algorithm}/{track}] benchmark FAILED: {e}", flush=True)
                    bench_artifacts[(algorithm, track)] = {"importance": None, "surface": None}
                    continue

                if bench_row.get("status") != "ok":
                    print(f"[{entity_name}/{algorithm}/{track}] benchmark status={bench_row.get('status')}", flush=True)
                    bench_artifacts[(algorithm, track)] = {"importance": None, "surface": None}
                    continue

                # Capture artifacts before they're stripped
                bench_imp = bench_row.pop("_run_importance", None)
                bench_surf = bench_row.pop("_run_surface", None)
                bench_artifacts[(algorithm, track)] = {
                    "importance": bench_imp,
                    "surface": bench_surf,
                }

                # Persist benchmark surface
                if save_surfaces and bench_surf is not None and len(bench_surf) > 0:
                    acc_subc = prepared["accessible_area"]["subc_id"].values
                    surface_df = pd.DataFrame({
                        "subc_id": acc_subc[: len(bench_surf)],
                        "predicted_probability": bench_surf,
                    })
                    safe_entity = entity_name.replace(" ", "_").replace("(", "").replace(")", "")
                    surf_path = surfaces_dir / f"{safe_entity}_{algorithm}_{track}_benchmark.parquet"
                    surface_df.to_parquet(surf_path, index=False)

                # Persist importance vector
                if bench_imp is not None:
                    for var, imp in bench_imp.items():
                        importance_rows.append({
                            "entity": entity_name,
                            "grid_id": "B",
                            "axis": "benchmark",
                            "level": 0,
                            "replicate": 0,
                            "algorithm": algorithm,
                            "track": track,
                            "variable": var,
                            "importance": float(imp) if imp is not None else float("nan"),
                        })

                # Save benchmark row to results
                bench_row["species"] = prepared["species"]
                bench_row["category"] = prepared["category"]
                bench_row["entity_type"] = prepared["entity_type"]
                bench_row["n_experiment"] = n_experiment
                bench_row["grid_id"] = "B"
                rows.append(bench_row)

        # ===== STEP 2: Loop over contaminated cells =====
        for axis in valid_axes:
            levels = snap_levels_pct if axis == "snapping" else lowacc_levels_pct
            contamination_pool = prepared["snap_pool"] if axis == "snapping" else prepared["lowacc_pool"]
            max_level = max(levels)

            for level in levels:
                if level == 0:
                    continue  # benchmark already captured above

                for replicate in range(n_replicates):
                    seed = derive_seed(master_seed, entity_name, axis, level, replicate)

                    for algorithm in algorithms:
                        for track in scale_tracks:
                            artifacts = bench_artifacts.get((algorithm, track), {})
                            bench_imp = artifacts.get("importance")
                            bench_surf = artifacts.get("surface")

                            # Save artifacts only for one representative replicate at max level
                            return_artifacts = (
                                save_surfaces
                                and level == max_level
                                and replicate == 0
                            )

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
                                    n_experiment=n_experiment,
                                    benchmark_importance=bench_imp,
                                    benchmark_surface=bench_surf,
                                    domain_map=domain_map,
                                    return_artifacts=return_artifacts,
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

                            # Capture run importance for separate parquet
                            run_imp = row.pop("_run_importance", None) if isinstance(row, dict) else None
                            run_surf = row.pop("_run_surface", None) if isinstance(row, dict) else None

                            if run_imp is not None:
                                for var, imp in run_imp.items():
                                    importance_rows.append({
                                        "entity": entity_name,
                                        "grid_id": "B",
                                        "axis": axis,
                                        "level": level,
                                        "replicate": replicate,
                                        "algorithm": algorithm,
                                        "track": track,
                                        "variable": var,
                                        "importance": float(imp) if imp is not None else float("nan"),
                                    })

                            # Save representative contaminated surface
                            if save_surfaces and return_artifacts and run_surf is not None and len(run_surf) > 0:
                                acc_subc = prepared["accessible_area"]["subc_id"].values
                                surf_df = pd.DataFrame({
                                    "subc_id": acc_subc[: len(run_surf)],
                                    "predicted_probability": run_surf,
                                })
                                safe_entity = entity_name.replace(" ", "_").replace("(", "").replace(")", "")
                                surf_path = surfaces_dir / f"{safe_entity}_{algorithm}_{track}_{axis}_max.parquet"
                                surf_df.to_parquet(surf_path, index=False)

                            row["species"] = prepared["species"]
                            row["category"] = prepared["category"]
                            row["entity_type"] = prepared["entity_type"]
                            row["n_experiment"] = n_experiment
                            row["grid_id"] = "B"
                            rows.append(row)

                            if len(rows) % checkpoint_every == 0:
                                _checkpoint_rows(rows, checkpoint_path)

    _checkpoint_rows(rows, final_path)
    if importance_rows:
        pd.DataFrame(importance_rows).to_parquet(importance_path, index=False)
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
        benchmark_n = len(prepared["benchmark"])

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
                            n_experiment=benchmark_n,
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
