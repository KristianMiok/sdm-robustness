import sys, pandas as pd, numpy as np
CSV=sys.argv[1]; ENV=sys.argv[2]
d=pd.read_csv(CSV,low_memory=False)
d=d[(d.get("grid_id","B")=="B")&(d["status"]=="ok")&(d["track"]=="combined")]
env=pd.read_csv(ENV); env=env[(env["track"]=="combined")&(env["metric_name"]=="auc")]
E={(r.entity,r.algorithm):(r.benchmark_mean,r.benchmark_sd) for r in env.itertuples()}
print("Per-entity benchmark SD (AUC) and one-sided z>2 exceedance, MAXENT, lowacc L20:")
print(f"{'entity':40s} {'bSD':>7s} {'mean_z':>7s} {'%z>2':>6s}")
sub=d[(d["axis"]=="lowacc")&(d["level"]==20)&(d["algorithm"]=="maxent")]
rows=[]
for ent,grp in sub.groupby("entity"):
    if (ent,"maxent") not in E: 
        print(f"{ent:40s}   MISSING FROM ENVELOPE"); continue
    bm,bs=E[(ent,"maxent")]
    z=(bm-grp["auc"])/bs
    ex=100*np.mean(z>2)
    rows.append(ex)
    print(f"{ent:40s} {bs:7.4f} {z.mean():7.2f} {ex:6.1f}")
print(f"\nper-entity-averaged %z>2 = {np.mean(rows):.1f}  (published Maxent lowacc L20 = 58.8)")
print("envelope Maxent entity count:", sum(1 for k in E if k[1]=='maxent'))
