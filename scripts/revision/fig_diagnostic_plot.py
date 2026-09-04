"""Diagnostic figure: largest-basin share vs fold balance, 13 entities."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.8,
                     "svg.fonttype":"none","pdf.fonttype":42})
T = pd.read_csv("results/revision/fig_partition_diagnostics.csv")
MK = {"DUAL":"o","SNAP":"s"}
fig, ax = plt.subplots(figsize=(5.0, 3.8))
for t, g in T.groupby("entity_type"):
    ax.scatter(g.largest_basin_pct, g.fold_min_max, s=48, marker=MK.get(t,"o"),
               facecolor="#1f77b4" if t=="DUAL" else "white",
               edgecolor="#1f77b4", lw=1.3, label=f"{t} (n={len(g)})", zorder=3)
for r in T.itertuples():
    ax.annotate(r.entity.split(" (")[0].replace("Austropotamobius","A.")
                .replace("Pontastacus","P.").replace("Pacifastacus","P.")
                .replace("Procambarus","P.").replace("Lacunicambarus","L.")
                .replace("Creaserinus","C.").replace("Cambarus","C.")
                .replace("Faxonius","F.").replace("Astacus","A."),
                (r.largest_basin_pct, r.fold_min_max), fontsize=6.5,
                xytext=(4,3), textcoords="offset points", alpha=0.75)
rho = spearmanr(T.largest_basin_pct, T.fold_min_max)
ax.set_xlabel("Records in the largest basin (%)")
ax.set_ylabel("Fold balance (min / max)")
ax.set_title("(Diagnostic) Basin concentration vs fold balance",
             fontweight="bold", fontsize=11)
ax.text(0.97, 0.05, f"Spearman ρ = {rho.statistic:+.3f}  (p = {rho.pvalue:.3f}, n = {len(T)})",
        transform=ax.transAxes, ha="right", fontsize=8, alpha=0.8)
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.25)
fig.tight_layout()
for ext in ("pdf","svg"):
    fig.savefig(f"results/figures/figD1_partition_diagnostic.{ext}", bbox_inches="tight")
print(f"[figD1] rho={rho.statistic:+.3f} p={rho.pvalue:.3f} n={len(T)}")
