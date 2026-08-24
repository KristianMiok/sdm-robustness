"""Per-entity decomposition of the 29-36% headline, and which entities
enter the mean at each contamination level."""
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R

RG, rng = "range_area_pct_change_05", np.random.default_rng(20260822)
B = pd.read_parquet("results/grid_b_merged/grid_b_results_raw_merged.parquet",
                    engine="fastparquet")
B = B[B.status == "ok"].drop_duplicates(
    ["entity","algorithm","track","axis","level","replicate"])
C = B[(B.axis != "benchmark") & (B.track == "combined")].copy()

print("=== A. ko ulazi u prosek, po osi i nivou ===")
for ax in ("lowacc","snapping"):
    s = C[C.axis == ax]
    print(f"\n--- {ax} ---")
    t = s.pivot_table(index="entity", columns="level", values=RG,
                      aggfunc=lambda x: x.notna().sum())
    print(t.fillna(0).astype(int).to_string())
    print("  entiteta po nivou:", {int(l): int((t[l].fillna(0) > 0).sum()) for l in t.columns})
    print("  isti skup na svim nivoima:",
          all(set(t.index[t[l].fillna(0) > 0]) == set(t.index[t[t.columns[0]].fillna(0) > 0])
              for l in t.columns))

print("\n=== B. range promena po entitetu (lowacc) ===")
L = C[C.axis == "lowacc"]
for alg in sorted(L.algorithm.unique()):
    s = L[L.algorithm == alg]
    t = s.pivot_table(index="entity", columns="level", values=RG, aggfunc="mean")
    sd = s.pivot_table(index="entity", columns="level", values=RG, aggfunc="std")
    t.columns = [f"L{int(c)}" for c in t.columns]
    sd.columns = [f"sd{int(c)}" for c in sd.columns]
    n = s.groupby("entity").n_experiment.first().rename("n_exp")
    print(f"\n--- {alg} ---")
    print(t.join(sd).join(n).round(2).to_string())

print("\n=== C. pooled prosek vs prosek po entitetu ===")
for ax, lv in (("lowacc", 20), ("snapping", 5)):
    s = C[(C.axis == ax) & (C.level == lv)]
    print(f"\n--- {ax} L{lv} ---")
    for alg in sorted(s.algorithm.unique()):
        a = s[s.algorithm == alg]
        pooled = a[RG].mean()
        per_ent = a.groupby("entity")[RG].mean()
        print(f"  {alg:<14} pooled {pooled:>8.3f}   po entitetu {per_ent.mean():>8.3f}"
              f"   median {per_ent.median():>8.3f}   raspon {per_ent.min():>7.2f}..{per_ent.max():>7.2f}")

print("\n=== D. leave-one-entity-out na headline (lowacc L20) ===")
s = C[(C.axis == "lowacc") & (C.level == 20)]
for alg in sorted(s.algorithm.unique()):
    a = s[s.algorithm == alg]
    full = a[RG].mean()
    print(f"\n--- {alg}: pun panel {full:.3f}% ---")
    rows = []
    for e in sorted(a.entity.unique()):
        d = a[a.entity != e]
        rows.append({"izbacen": e, "prosek_bez": d[RG].mean(),
                     "delta": d[RG].mean() - full,
                     "sopstveni": a[a.entity == e][RG].mean()})
    print(pd.DataFrame(rows).sort_values("delta").round(2).to_string(index=False))

print("\n=== E. hijerarhijski bootstrap headline-a (entiteti pa replike) ===")
for ax, lv in (("lowacc", 20), ("snapping", 5)):
    s = C[(C.axis == ax) & (C.level == lv)]
    print(f"\n--- {ax} L{lv} ---")
    for alg in sorted(s.algorithm.unique()):
        by = {e: v.dropna().to_numpy() for e, v in s[s.algorithm == alg].groupby("entity")[RG]}
        by = {e: v for e, v in by.items() if len(v)}
        ks = list(by)
        out = np.array([np.mean([by[ks[p]][rng.integers(0, len(by[ks[p]]), len(by[ks[p]]))].mean()
                                 for p in rng.choice(len(ks), len(ks), True)]) for _ in range(3000)])
        print(f"  {alg:<14} {np.mean([v.mean() for v in by.values()]):>8.3f}"
              f"  [{np.percentile(out,2.5):>7.3f}, {np.percentile(out,97.5):>7.3f}]  ent={len(ks)}")

print("\n=== F. efekat po entitetu vs osobine entiteta ===")
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False,
                usecols=["subc_id","basin_id","strahler","distance_m","Accuracy",
                         "Status","Crayfish_scientific_name"])
P = pd.read_csv("config/final_panel.csv")
rows = []
for _, r in P.iterrows():
    sp = r.entity.split(" (")[0]
    d = M[M.Crayfish_scientific_name == sp]
    if r.treatment == "native_only": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif r.treatment == "alien_only": d = d[d.Status.isin(R.ALIEN_VALUES)]
    hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m <= 200)])
    l = R._dedup_by_subc(d[~hi])
    rows.append(dict(entity=r.entity, bench=len(b), low=len(l),
                     low_pct=len(l)/len(b)*100, basena=b.basin_id.nunique(),
                     po_basenu=len(b)/b.basin_id.nunique(),
                     str1=(b.strahler == 1).mean()*100,
                     low_str1=(l.strahler == 1).mean()*100 if len(l) else np.nan))
E = pd.DataFrame(rows).set_index("entity")
eff = s[s.algorithm == "xgboost"].groupby("entity")[RG].mean().rename("range20")
J = E.join(eff, how="inner")
print(J.round(2).to_string())
print("\n  Spearman sa range20:")
print(J.corr(method="spearman")["range20"].drop("range20").round(3).to_string())
