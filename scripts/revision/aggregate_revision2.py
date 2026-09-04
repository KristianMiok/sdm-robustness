"""Agregacija revizionih kampanja -> results/revision/tables/."""
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path("results/revision/tables"); OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260828); B = 3000
KEY = ["entity","algorithm","track","axis","level","replicate"]
GRP = ["entity","algorithm","track"]
ALGS = ("random_forest","xgboost","maxent")

# --- 1. planirani dizajn iz tasks.tsv (imenilac za izvodljivost) -----------
T = pd.read_csv("run_scripts/revision/campaign/tasks.tsv", sep="\t", header=None,
                names=["entity","slug","grp","algs","tracks","snap","low","reps"],
                dtype=str).fillna("")
T["reps"] = T["reps"].astype(int)
SLUG2ENT = dict(zip(T.slug, T.entity))
plan = set()
for r in T.itertuples(index=False):
    for a in r.algs.split():
        for tr in r.tracks.split():
            for axis, levels in (("snapping", r.snap), ("lowacc", r.low)):
                for lv in (int(x) for x in levels.split()):
                    ax = "benchmark" if lv == 0 else axis
                    for rep in range(r.reps):
                        plan.add((r.entity, a, tr, ax, lv, rep))
PLAN = pd.DataFrame(sorted(plan), columns=KEY)
print(f"planirano celija: {len(PLAN)} | entiteta: {PLAN.entity.nunique()}")

# --- 2. kampanja -----------------------------------------------------------
files = sorted(glob.glob("results/campaign/*/results_raw.parquet"))
D = pd.concat([pd.read_parquet(f, engine="fastparquet").assign(_src=Path(f).parent.name)
               for f in files], ignore_index=True)
print(f"parqueta: {len(files)} | redova: {len(D)} | ok: {(D.status=='ok').sum()}")
OK = D[D.status == "ok"].copy()
NUM = [c for c in OK.select_dtypes("number").columns if c not in KEY]

# --- 3. duplikati benchmarka: prosek, raspon zabelezen ---------------------
dup = OK.duplicated(KEY, keep=False)
if dup.any():
    dd = OK[dup]
    sp = dd.groupby(KEY)[NUM].agg(lambda s: s.max()-s.min()).reset_index()
    src = dd.groupby(KEY)["_src"].apply(lambda s: "|".join(sorted(set(s)))).reset_index()
    sp.merge(src, on=KEY).to_csv(OUT/"benchmark_duplicate_spread.csv", index=False)
    print(f"dupliranih celija: {dd.groupby(KEY).ngroups} na osama {sorted(dd.axis.unique())}"
          f" -> uprosecene, raspon u benchmark_duplicate_spread.csv")
OK = OK.groupby(KEY, as_index=False)[NUM].mean()

# --- 4. izvodljivost prema dizajnu ----------------------------------------
FEAS = pd.concat([PLAN.groupby(["axis","level"]).size().rename("planned"),
                  OK.groupby(["axis","level"]).size().rename("ok")], axis=1).fillna(0).astype(int)
FEAS["pct"] = (100*FEAS.ok/FEAS.planned).round(1)
FEAS.reset_index().to_csv(OUT/"dose_feasibility.csv", index=False)
print("\n=== izvodljivost (imenilac = dizajn) ===\n", FEAS)

# --- 5. sastav panela po dozi ---------------------------------------------
C = OK[OK.axis != "benchmark"]
pan = C.groupby(["axis","level"])["entity"].agg(
        n_entities="nunique", entities=lambda s: "; ".join(sorted(s.unique())))
PAN = pd.concat([PLAN[PLAN.axis!="benchmark"].groupby(["axis","level"])["entity"]
                 .nunique().rename("planned_entities"), pan], axis=1).fillna(0)
PAN["n_entities"] = PAN.n_entities.astype(int)
PAN["tier"] = np.where(PAN.n_entities == 0, "not_executed",
               np.where(PAN.n_entities < PAN.planned_entities, "reduced_panel", "full_panel"))
PAN.reset_index().to_csv(OUT/"panel_composition.csv", index=False)
print("\n=== sastav panela ===\n", PAN[["planned_entities","n_entities","tier"]])

# --- 6. delte S4/S5 sa hijerarhijskim bootstrapom --------------------------
MET = [c for c in ("auc","tss","brier","boyce","sensitivity","specificity","omission_rate")
       if c in OK.columns]
# Tier 2: vec su poredjenja sa benchmarkom - izvestavaju se kao nivoi, ne kao delte
T2 = [c for c in ("importance_spearman","importance_jaccard_top5","importance_jaccard_top10",
                  "env_centroid_disp","env_dispersion_change","domain_rank_stability")
      if c in OK.columns]
def hboot(g, col):
    by = {e: v.dropna().to_numpy() for e, v in g.groupby("entity")[col]}
    by = {e: v for e, v in by.items() if len(v)}
    if len(by) < 2: return (np.nan,)*3
    ks = list(by)
    out = np.array([np.mean([by[ks[p]][RNG.integers(0,len(by[ks[p]]),len(by[ks[p]]))].mean()
                             for p in RNG.choice(len(ks), len(ks), True)]) for _ in range(B)])
    return (float(np.mean([v.mean() for v in by.values()])),
            float(np.percentile(out,2.5)), float(np.percentile(out,97.5)))

bench = OK[OK.axis=="benchmark"].groupby(GRP, as_index=False)[MET].mean()
M = C.merge(bench, on=GRP, suffixes=("","_b"))
for m in MET: M[m+"_d"] = M[m] - M[m+"_b"]
rows = []
for (ax,lv,alg,tr), g in M.groupby(["axis","level","algorithm","track"]):
    for m in MET:
        mu,lo,hi = hboot(g, m+"_d")
        rows.append(dict(axis=ax, level=lv, algorithm=alg, track=tr, metric=m,
                         delta=mu, ci_lo=lo, ci_hi=hi,
                         n=int(g[m+"_d"].notna().sum()), entities=g.entity.nunique()))
S45 = pd.DataFrame(rows).merge(PAN.reset_index()[["axis","level","tier"]], on=["axis","level"])
S45.to_csv(OUT/"S4_S5_delta.csv", index=False)
CB2 = C[C.track == "combined"]
if T2:
    # jedno pravilo: combined track, prosek u entitetu, medijana preko entiteta
    R2 = (CB2.groupby(["axis","level","entity"])[T2].mean()
             .groupby(["axis","level"]).median().round(4).join(PAN["tier"]))
    R2.to_csv(OUT/"T8_tier2.csv")
    (CB2.groupby(["axis","level","algorithm","entity"])[T2].mean()
        .groupby(["axis","level","algorithm"]).median().round(4)
        .to_csv(OUT/"T8_tier2_by_alg.csv"))
    print("\n=== Tier 2 (entitetske medijane, nivoi ne delte) ===\n", R2)
print("\n=== S4/S5, AUC, combined ===")
print(S45[(S45.metric=="auc") & (S45.track=="combined")]
      .pivot_table(index=["axis","level","tier"], columns="algorithm", values="delta").round(4))

# --- 7. pragovi, kontinuirane, po entitetu --------------------------------
THR  = [c for c in OK.columns if c.startswith("range_area_pct_change")]
CONT = [c for c in ("suitability_mad","suitability_mean_shift","schoener_d","warren_i")
        if c in OK.columns]
(CB2.groupby(["axis","level","entity"])[THR].mean().groupby(["axis","level"])
    .median().round(4).join(PAN["tier"]).to_csv(OUT/"T6_5_thresholds.csv"))
(CB2.groupby(["axis","level","entity"])[CONT].mean().groupby(["axis","level"])
    .median().round(4).join(PAN["tier"]).to_csv(OUT/"T6_6_continuous.csv"))
C.pivot_table(index="entity", columns=["axis","level"],
              values="range_area_pct_change_05", aggfunc="mean").round(2).to_csv(OUT/"per_entity_range.csv")
print("\n=== T6.5 pragovi ===\n", C.groupby(["axis","level"])[THR].mean().round(2))

# --- 8. T3 (t3_ + t3ext_, bez effective_dose fajlova) ---------------------
def split_alg(stem, pref):
    n = stem[len(pref):]
    for a in ALGS:
        if n.endswith("_"+a): return n[:-(len(a)+1)], a
    return n, None
rows = []
for f in sorted(glob.glob("results/revision/t3*_*.csv")):
    nm = Path(f).name
    pref = "t3ext_" if nm.startswith("t3ext_") else "t3_"
    slug, alg = split_alg(Path(f).stem, pref)
    if slug not in SLUG2ENT:
        print(f"  T3 preskacem (nije entitet panela): {nm}"); continue
    rows.append(pd.read_csv(f).assign(entity=SLUG2ENT[slug], algorithm=alg))
T3 = pd.concat(rows, ignore_index=True).drop_duplicates(["entity","algorithm","rep","dist_m","frac"])
T3.to_csv(OUT/"T3_raw.csv", index=False)
T3M = [c for c in ("auc","tss","range_area_pct_change_05","range_area_pct_change_maxsss",
                   "suitability_mad","schoener_d","eff_dose") if c in T3.columns]
T3S = T3[T3.dist_m>0].groupby(["frac","dist_m"])[T3M].mean().round(4)
T3S.to_csv(OUT/"T3_displacement.csv")
T3[T3.dist_m>0].groupby(["entity","frac"])[T3M].mean().round(4).to_csv(OUT/"T3_per_entity.csv")
print(f"\n=== T3: {T3.entity.nunique()} entiteta, dist_m={sorted(T3.dist_m.unique())} ===\n", T3S)

# --- 9. T1.5 --------------------------------------------------------------
rows = []
for f in sorted(glob.glob("results/revision/t1_5_*.csv")):
    slug, alg = split_alg(Path(f).stem, "t1_5_")
    if slug not in SLUG2ENT:
        print(f"  T1.5 preskacem: {Path(f).name}"); continue
    rows.append(pd.read_csv(f).assign(entity=SLUG2ENT[slug], algorithm=alg))
if rows:
    T15 = pd.concat(rows, ignore_index=True)
    T15.to_csv(OUT/"T1_5_raw.csv", index=False)
    M15 = [c for c in ("d_str1","auc","tss","range_area_pct_change_05",
                       "range_area_pct_change_maxsss","suitability_mad","schoener_d")
           if c in T15.columns]
    S = T15.groupby(["level","mode"])[M15].mean().round(4)
    S.to_csv(OUT/"T1_5_stratified.csv")
    W = S["range_area_pct_change_05"].unstack("mode")
    W["abs_reduction"] = (W["plain"] - W["stratified"]).round(4)
    W["pct_of_plain"] = (100*(W["plain"]-W["stratified"])/W["plain"]).round(1)
    W.to_csv(OUT/"T1_5_strahler_effect.csv")
    T15.groupby(["entity","level","mode"])[M15].mean().round(4).to_csv(OUT/"T1_5_per_entity.csv")
    print(f"\n=== T1.5: {T15.entity.nunique()} entiteta ===\n", S)
    print("\n=== T1.5 udeo inflacije objasnjen Strahlerom ===\n", W)

print("\n--- tabele ---")
for p in sorted(OUT.glob("*.csv")): print(f"  {p.name:38s} {p.stat().st_size:>8d} B")
