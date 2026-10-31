"""Journal figures (Fig. 2, Fig. 3) from the experiment CSVs, averaged over seeds.
Usage: python make_figs.py RESULTS_DIR OUT_DIR"""
import sys, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, O = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.8, "xtick.color": "#333333",
                     "ytick.color": "#333333", "legend.frameon": False})
STY = {  # colour (CVD-validated), marker, linestyle, label
    "Cloud-only (teacher)": ("#C0392B", "o", "-", "Cloud-only"),
    "Edge-only (student)": ("#0072B2", "s", "-", "Edge-only"),
    "Fixed-confidence offloading": ("#E69F00", "^", "--", "Fixed-confidence offloading"),
    "EdgeWarn": ("#009E73", "D", ":", "EdgeWarn"),
}

# Fig. 3 availability vs failure rate
f3 = pd.read_csv(f"{R}/fig3_availability.csv")
g = f3.groupby(["kind", "rho", "method"]).availability.agg(["mean", "std"]).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, kind, title in zip(axs, ("independent", "correlated"), ("(a) Independent failures", "(b) Rainfall-correlated failures")):
    for m, (c, mk, ls, lab) in STY.items():
        d = g[(g.kind == kind) & (g.method == m)].sort_values("rho")
        ax.errorbar(d.rho, d["mean"], yerr=d["std"].fillna(0), color=c, marker=mk, ls=ls, lw=1.6, ms=5,
                    capsize=2, label=lab, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("Mean link failure rate ρ")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axs[0].set_ylabel("Warning availability (% of onsets)")
axs[0].set_ylim(0, 100)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(O, "Fig3.png"), dpi=600)
plt.close(fig)

# Fig. 2 F1 vs share escalated
f2 = pd.read_csv(f"{R}/fig2_tradeoff.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for m in ("EdgeWarn", "Fixed-confidence offloading"):
    c, mk, ls, lab = STY[m]
    d = f2[f2.method == m].groupby("knob")[["escalated_pct", "f1"]].mean().sort_values("escalated_pct")
    ax.plot(d.escalated_pct, d.f1, color=c, marker=mk, ls="-", lw=1.6, ms=5, label=lab,
            markeredgecolor="white", markeredgewidth=0.6)
t2 = pd.read_csv(f"{R}/table2_raw.csv")
for m, lab, ls in (("Cloud-only (teacher)", "Cloud-only F1", "--"), ("Edge GRU", "Edge GRU F1", ":")):
    v = t2[t2.method == m].f1.mean()
    ax.axhline(v, color="#555555", ls=ls, lw=1)
    ax.text(101, v, lab, va="center", ha="left", fontsize=7.5, color="#333333")
ax.set_xlabel("Windows escalated (%)")
ax.set_ylabel("F1-score")
ax.set_xlim(-3, 100)
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.55), fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(O, "Fig2.png"), dpi=600, bbox_inches="tight")
plt.close(fig)
print("figures written")
