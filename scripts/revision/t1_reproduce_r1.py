"""Two things: reproduce R1's 39.5 / 71.4, and verify Grid A sample sizes."""
import itertools, inspect
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R

ENT = [("Astacus astacus",""),("Austropotamobius fulcisianus","pooled"),
       ("Austropotamobius torrentium","pooled"),("Cambarus latimanus",""),
       ("Cambarus striatus",""),("Creaserinus fodiens",""),
       ("Faxonius limosus","alien"),("Faxonius limosus","native"),
       ("Lacunicambarus diogenes",""),("Pacifastacus leniusculus","alien"),
       ("Pontastacus leptodactylus","pooled"),("Procambarus clarkii","alien"),
       ("Procambarus clarkii","native")]
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False, usecols=[
    "WoCID","subc_id","basin_id","strahler","distance_m","Accuracy","Status",
    "Crayfish_scientific_name"])
hi = R._is_high_accuracy(M.Accuracy)

def V(a, b):
    t = pd.crosstab(pd.concat([a,b]), ["A"]*len(a)+["B"]*len(b))
    if t.shape[0] < 2: return np.nan
    n = t.values.sum(); e = np.outer(t.sum(1), t.sum(0))/n
    chi2 = ((t.values-e)**2/np.where(e==0, np.nan, e)).sum()
    return float(np.sqrt(chi2/(n*(min(t.shape)-1))))

def ent_mask():
    m = pd.Series(False, index=M.index)
    for sp, tr in ENT:
        s = M.Crayfish_scientific_name == sp
        if tr == "native": s &= M.Status.isin(R.NATIVE_VALUES)
        elif tr == "alien": s &= M.Status.isin(R.ALIEN_VALUES)
        m |= s
    return m

POP = {"13 entiteta": ent_mask(),
       "vrste 13 entiteta": M.Crayfish_scientific_name.isin([s for s,_ in ENT]),
       "svi zapisi": pd.Series(True, index=M.index)}
BENCH = {"High<=200": lambda d: d[R._is_high_accuracy(d.Accuracy) & (d.distance_m<=200)],
         "svi<=200":  lambda d: d[d.distance_m<=200]}
CONT = {"High 200-1000": lambda d: d[R._is_high_accuracy(d.Accuracy)&(d.distance_m>200)&(d.distance_m<=1000)],
        "High >200":     lambda d: d[R._is_high_accuracy(d.Accuracy)&(d.distance_m>200)],
        "lowacc":        lambda d: d[~R._is_high_accuracy(d.Accuracy)],
        "snap+lowacc":   lambda d: d[(~R._is_high_accuracy(d.Accuracy))|
                                     (R._is_high_accuracy(d.Accuracy)&(d.distance_m>200)&(d.distance_m<=1000))],
        "svi >200":      lambda d: d[d.distance_m>200]}

print(f"{'populacija':<19}{'benchmark':<11}{'kontaminacija':<15}{'dedup':<7}"
      f"{'bench%':>8}{'cont%':>8}{'V':>8}{'gap':>8}")
best = []
for (pn,pm),(bn,bf),(cn,cf),dd in itertools.product(POP.items(),BENCH.items(),CONT.items(),(True,False)):
    d = M[pm]
    b, c = bf(d), cf(d)
    if dd:
        b, c = R._dedup_by_subc(b), R._dedup_by_subc(c)
    if len(b) < 50 or len(c) < 50: continue
    pb, pc = (b.strahler==1).mean()*100, (c.strahler==1).mean()*100
    gap = abs(pb-39.5)+abs(pc-71.4)
    best.append((gap, pn, bn, cn, dd, pb, pc, V(b.strahler, c.strahler)))
for gap,pn,bn,cn,dd,pb,pc,v in sorted(best)[:14]:
    print(f"{pn:<19}{bn:<11}{cn:<15}{str(dd):<7}{pb:>8.1f}{pc:>8.1f}{v:>8.4f}{gap:>8.1f}")

print("\n" + "="*70)
print("=== _compute_n_experiment ===")
print(inspect.getsource(R._compute_n_experiment))

print("\n=== rekonstrukcija n_experiment ===")
rows = []
for sp, tr in ENT:
    d = M[M.Crayfish_scientific_name == sp]
    if tr == "native": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif tr == "alien": d = d[d.Status.isin(R.ALIEN_VALUES)]
    h = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[h & (d.distance_m<=200)])
    s = R._dedup_by_subc(d[h & (d.distance_m>200) & (d.distance_m<=1000)])
    l = R._dedup_by_subc(d[~h])
    rows.append({"entity": f"{sp} {tr}".strip(), "bench": len(b), "snap": len(s), "low": len(l),
                 "capA_50": min(len(b), int(len(s)/0.5), int(len(l)/0.5)) if len(l) else min(len(b), int(len(s)/0.5)),
                 "capB_5_20": min(len(b), int(len(s)/0.05), int(len(l)/0.20)) if len(l) else min(len(b), int(len(s)/0.05))})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
print("\nGrid A opazeni n_experiment:", sorted([40,104,122,134,208,222,224,302,514,660,700,704,1938]))
print("rekonstruisani capA_50:      ", sorted(t.capA_50.tolist()))

print("\n=== iscrpljenost snap pool-a po nivou (Grid A) ===")
for _, r in t.iterrows():
    n = r.capA_50
    line = f"  {r.entity:<36} n={n:>5}"
    for lv in (20, 35, 50):
        need = int(round(n*lv/100))
        line += f" | L{lv}: {need}/{r.snap}" + ("  ISCRPLJEN" if need >= r.snap else "")
    print(line)
