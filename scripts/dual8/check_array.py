import os, glob, pandas as pd
ROOT="results/task5c_benchmark_stability_array"
subs=sorted(os.listdir(ROOT))
print(f"total subdirs: {len(subs)}")
# look for the two missing maxent entities
for key in ["Faxonius_limosus_alien_maxent","Pacifastacus_leniusculus_alien_maxent",
            "faxonius","pacifastacus"]:
    hits=[s for s in subs if key.lower() in s.lower()]
    if hits: print(f"match '{key}':", hits)
print("\n--- all maxent+combined subdirs ---")
mx=[s for s in subs if "maxent" in s.lower() and "combined" in s.lower()]
print(f"count={len(mx)}"); [print(" ",s) for s in mx]
# schema of one benchmark_stability file
print("\n--- schema of one benchmark_stability.parquet ---")
f=glob.glob(os.path.join(ROOT,"*maxent_combined","benchmark_stability.parquet"))
if f:
    df=pd.read_parquet(f[0])
    print("file:",f[0]); print("shape:",df.shape); print("cols:",list(df.columns))
    print(df.head(3).to_string())
