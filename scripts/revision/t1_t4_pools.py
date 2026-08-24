"""T1.4 + T4: pool composition with effect sizes, and Low-accuracy characterisation.
Rebuilds pools exactly as runner.py does, and tries to reproduce R1's Cramer's V."""
import re, pathlib
import numpy as np, pandas as pd

print("=== 0. konstante i panel ===")
src = pathlib.Path("src/sdm_robustness")
for f in sorted(src.rglob("*.py")):
    t = f.read_text()
    for nm in ("HIGH_VALUES","NATIVE_VALUES","ALIEN_VALUES"):
        m = re.search(rf"^{nm}\s*=\s*(.+)$", t, re.M)
        if m: print(f"  {f.relative_to(src)}: {nm} = {m.group(1).strip()}")
for d in ("configs","config"):
    for p in sorted(pathlib.Path(d).rglob("*")) if pathlib.Path(d).exists() else []:
        if p.is_file(): print("  config:", p)

from sdm_robustness.execution.runner import (
    _is_high_accuracy, _dedup_by_subc, NATIVE_VALUES, ALIEN_VALUES)

ENT = [("Astacus astacus",""), ("Austropotamobius fulcisianus","pooled"),
       ("Austropotamobius torrentium","pooled"), ("Cambarus latimanus",""),
       ("Cambarus striatus",""), ("Creaserinus fodiens",""),
       ("Faxonius limosus","alien"), ("Faxonius limosus","native"),
       ("Lacunicambarus diogenes",""), ("Pacifastacus leniusculus","alien"),
       ("Pontastacus leptodactylus","pooled"), ("Procambarus clarkii","alien"),
       ("Procambarus clarkii","native")]

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False, usecols=[
    "WoCID","subc_id","reg_id","basin_id","strahler","area_sqm","sum_area_sqm",
    "hylak_id","lat_snap","long_snap","distance_m","Accuracy","Status",
    "Crayfish_scientific_name","Year_of_record"])

def pools(species, treat, dedup=True):
    b = M[M.Crayfish_scientific_name == species].copy()
    if treat == "native": b = b[b.Status.isin(NATIVE_VALUES)]
    elif treat == "alien": b = b[b.Status.isin(ALIEN_VALUES)]
    hi = _is_high_accuracy(b.Accuracy)
    out = {"benchmark": b[hi & (b.distance_m <= 200)],
           "snap": b[hi & (b.distance_m > 200) & (b.distance_m <= 1000)],
           "lowacc": b[~hi]}
    return {k: (_dedup_by_subc(v) if dedup else v) for k, v in out.items()}

def cramers_v(a, b):
    t = pd.crosstab(pd.concat([a, b]), ["A"]*len(a) + ["B"]*len(b))
    if t.shape[0] < 2: return np.nan, np.nan
    n = t.values.sum()
    e = np.outer(t.sum(1), t.sum(0)) / n
    chi2 = float(((t.values - e)**2 / np.where(e == 0, np.nan, e)).sum())
    return np.sqrt(chi2 / (n * (min(t.shape) - 1))), chi2

def d_cohen(x, y):
    x, y = np.log10(np.asarray(x, float)+1), np.log10(np.asarray(y, float)+1)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 5 or len(y) < 5: return np.nan
    s = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) / (len(x)+len(y)-2))
    return (y.mean() - x.mean()) / s if s else np.nan

for dedup in (True, False):
    P = {k: [] for k in ("benchmark","snap","lowacc")}
    for sp, tr in ENT:
        d = pools(sp, tr, dedup)
        for k in P: P[k].append(d[k])
    P = {k: pd.concat(v, ignore_index=True) for k, v in P.items()}
    print(f"\n{'='*60}\n=== POOL-OVI, dedup={dedup} ===")
    print("  n:", {k: len(v) for k, v in P.items()})
    print("\n  --- Strahler, udeo 1. reda ---")
    for k, v in P.items():
        print(f"    {k:<10} {(v.strahler == 1).mean()*100:5.1f}%   median {v.strahler.median():.0f}")
    for k in ("snap","lowacc"):
        V, chi2 = cramers_v(P["benchmark"].strahler, P[k].strahler)
        print(f"    benchmark vs {k:<7} Cramer's V = {V:.4f}   (chi2={chi2:.0f})")

print(f"\n{'='*60}\n=== T1.4 mrezni atributi (dedup=True) ===")
P = {k: pd.concat([pools(sp, tr)[k] for sp, tr in ENT], ignore_index=True)
     for k in ("benchmark","snap","lowacc")}
print(f"{'atribut':<16}{'benchmark':>12}{'snap':>12}{'lowacc':>12}{'d(snap)':>10}{'d(low)':>10}")
for col in ("area_sqm","sum_area_sqm","distance_m"):
    b, s, l = P["benchmark"][col], P["snap"][col], P["lowacc"][col]
    print(f"{col:<16}{b.median():>12.0f}{s.median():>12.0f}{l.median():>12.0f}"
          f"{d_cohen(b,s):>10.3f}{d_cohen(b,l):>10.3f}")
print(f"{'lake assoc %':<16}" + "".join(f"{P[k].hylak_id.notna().mean()*100:>12.2f}"
      for k in ("benchmark","snap","lowacc")))
print("\n  Strahler raspodela (%):")
print(pd.DataFrame({k: P[k].strahler.value_counts(normalize=True).mul(100)
                    for k in P}).sort_index().round(1).head(9).to_string())

print(f"\n{'='*60}\n=== T4 low-accuracy pool ===")
print("  --- godina zapisa ---")
print(pd.DataFrame({k: P[k].Year_of_record.describe(percentiles=[.1,.5,.9])
                    for k in P}).round(0).to_string())
print("\n  --- Status ---")
print(pd.DataFrame({k: P[k].Status.value_counts(normalize=True).mul(100)
                    for k in P}).round(1).to_string())
print("\n  --- geografski obuhvat ---")
for k, v in P.items():
    print(f"    {k:<10} basena {v.basin_id.nunique():>5}  subc {v.subc_id.nunique():>6}"
          f"  regiona {v.reg_id.nunique():>3}"
          f"  lat {v.lat_snap.min():.1f}..{v.lat_snap.max():.1f}"
          f"  lon {v.long_snap.min():.1f}..{v.long_snap.max():.1f}")
print("\n  --- snapping rastojanje ---")
print(pd.DataFrame({k: P[k].distance_m.describe(percentiles=[.5,.75,.9,.99])
                    for k in P}).round(1).to_string())

print(f"\n{'='*60}\n=== velicine pool-ova po entitetu ===")
rows = []
for sp, tr in ENT:
    d = pools(sp, tr)
    rows.append({"entity": f"{sp} {tr}".strip(), **{k: len(v) for k, v in d.items()},
                 "bench_str1_%": round((d["benchmark"].strahler == 1).mean()*100, 1),
                 "low_str1_%": round((d["lowacc"].strahler == 1).mean()*100, 1)})
print(pd.DataFrame(rows).to_string(index=False))
