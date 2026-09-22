"""
COMPUTATIONAL COUPLING --- The headline figure: which rewiring you compare against.

Design notes
------------
This figure has to carry the paper's central claim to a reader who will not read the
methods, so it is built around one visual fact: the bars point in opposite directions.
Everything else on the page is in service of that.

Panel A is the result. For each network, the connectome's measured advantage over each of
the two null ensembles, paired by instance. The two ensembles answer to the same phrase in
the literature. If they agreed, the bars in each pair would point the same way.

Panel B is the cause. Rank-reassignment does not merely randomise: it installs an
association between an edge's weight and its endpoints' degrees, which the empirical
graphs do not have and which weight-carrying rewiring does not create.

Panel C is the part that makes B an explanation rather than a story. If the difference
comes from permuting weights, then on a network with no weight variation the two
constructions are the same ensemble and must agree exactly. One network is binary. The
panel shows the prediction and the outcome side by side.

Colour: the two nulls are the two most separable slots of the reference palette, held
consistent across all three panels, with the empirical graph in ink. Direction is encoded
redundantly (sign of the bar, and position relative to the zero line) so the figure does
not rely on hue alone.
"""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

BCT = "#2a78d6"        # the ensemble the published toolbox builds
RANK = "#eb6834"       # the one we wrote from its description
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e2de"

NULLS = ("bct_randmio", "rank_reassigned")
NULL_LABEL = {"bct_randmio": "weights carried with edges\n(BCT randmio — what the\npublished toolbox calls)",
              "rank_reassigned": "weights re-assigned by degree\n(what we wrote from the\npublished description)"}
PRETTY = {"human": "Human", "macaque_modha": "Macaque", "mouse": "Mouse",
          "rat": "Rat", "drosophila": "Drosophila"}
ORDER = ["drosophila", "rat", "mouse", "human", "macaque_modha"]


def _style(ax) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.set_axisbelow(True)


def _contrasts(df: pd.DataFrame, rho: float = 1.0, unit: str = "rate") -> pd.DataFrame:
    """Connectome minus each null, paired by instance, with a 95% CI."""
    rows = []
    sub = df[(df.unit_model == unit) & np.isclose(df.spectral_radius, rho)]
    for net, g in sub.groupby("network"):
        emp = g[g.kind == "empirical"].sort_values("instance")[
            "memory_capacity"].to_numpy()
        for null in NULLS:
            nul = g[g.kind == null].sort_values("instance")[
                "memory_capacity"].to_numpy()
            k = min(len(emp), len(nul))
            if k < 3:
                continue
            d = emp[:k] - nul[:k]
            p = float(stats.ttest_rel(emp[:k], nul[:k]).pvalue) if d.std() > 0 else 1.0
            rows.append(dict(network=net, null=null, diff=float(d.mean()),
                             ci=float(1.96 * d.std(ddof=1) / np.sqrt(k)), p=p))
    return pd.DataFrame(rows)


def figure_null_choice(comparison_csv: str, mechanism_csv: str, out_path: str,
                       rho: float = 1.0) -> None:
    df = pd.read_csv(comparison_csv, keep_default_na=False, na_values=[""])
    for c in ("memory_capacity", "spectral_radius"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    con = _contrasts(df, rho=rho)
    mech = pd.read_csv(mechanism_csv)

    nets = [n for n in ORDER if n in set(con.network)]
    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.85], wspace=0.42,
                          left=0.095, right=0.985, top=0.78, bottom=0.30)

    # ---- Panel A: the result ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    y = np.arange(len(nets))
    h = 0.34
    for j, null in enumerate(NULLS):
        vals, errs = [], []
        for n in nets:
            r = con[(con.network == n) & (con.null == null)]
            vals.append(float(r["diff"].iloc[0]) if len(r) else np.nan)
            errs.append(float(r["ci"].iloc[0]) if len(r) else 0.0)
        off = (0.5 - j) * h
        ax.barh(y + off, vals, height=h, xerr=errs,
                color=(BCT if null == "bct_randmio" else RANK),
                edgecolor="white", linewidth=0.6, label=NULL_LABEL[null],
                error_kw=dict(ecolor=INK2, lw=0.9, capsize=2))
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[n] for n in nets], fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(nets) - 0.4)
    ax.set_xlabel("memory capacity: connectome minus rewiring", fontsize=8.5,
                  color=INK2)
    ax.set_title("A   Same graphs, same measure, two rewirings",
                 fontsize=10, color=INK, loc="left", pad=10, weight="bold")
    xl = ax.get_xlim()
    ax.text(xl[0] + 0.02 * (xl[1] - xl[0]), len(nets) - 0.42,
            "← rewiring wins", fontsize=7.5, color=MUTED, ha="left")
    ax.text(xl[1] - 0.02 * (xl[1] - xl[0]), len(nets) - 0.42,
            "connectome wins →", fontsize=7.5, color=MUTED, ha="right")
    ax.legend(fontsize=7, frameon=False, loc="lower center", labelcolor=INK2,
              bbox_to_anchor=(0.5, -0.52), ncol=2, handlelength=1.3,
              columnspacing=1.4)

    # ---- Panel B: the cause -------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2)
    ax2.yaxis.grid(True, color=GRID, lw=0.8)
    wt = [n for n in nets if n != "macaque_modha"]
    x = np.arange(len(wt))
    w = 0.27
    series = [("empirical", INK, "real connectome"),
              ("bct_randmio", BCT, "weights carried"),
              ("rank_reassigned", RANK, "weights re-assigned")]
    for j, (arm, colr, lab) in enumerate(series):
        vals = [float(mech[(mech.network == n) & (mech.arm == arm)]["w_deg_r"].mean())
                for n in wt]
        ax2.bar(x + (j - 1) * w, vals, width=w, color=colr, edgecolor="white",
                linewidth=0.6, label=lab)
    ax2.axhline(0, color=INK, lw=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([PRETTY[n] for n in wt], fontsize=8, color=INK, rotation=20)
    ax2.set_ylabel("weight–degree correlation", fontsize=8.5, color=INK2)
    ax2.set_title("B   Why: one rewiring installs\n     structure the data lacks",
                  fontsize=10, color=INK, loc="left", pad=10, weight="bold")
    ax2.legend(fontsize=7, frameon=False, loc="upper left", labelcolor=INK2)

    # ---- Panel C: the falsifiable prediction --------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    _style(ax3)
    ax3.yaxis.grid(True, color=GRID, lw=0.8)
    mac = con[con.network == "macaque_modha"]
    if len(mac) == 2:
        vals = [float(mac[mac.null == n]["diff"].iloc[0]) for n in NULLS]
        errs = [float(mac[mac.null == n]["ci"].iloc[0]) for n in NULLS]
        ax3.bar([0, 1], vals, yerr=errs, width=0.52, color=[BCT, RANK],
                edgecolor="white", linewidth=0.6,
                error_kw=dict(ecolor=INK2, lw=0.9, capsize=3))
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["weights\ncarried", "weights\nre-assigned"],
                            fontsize=8, color=INK)
        ax3.axhline(0, color=INK, lw=1.0)
        gap = abs(vals[0] - vals[1])
        ax3.annotate(f"they differ by\n{gap:.3f}", xy=(0.5, max(vals) * 1.02),
                     ha="center", fontsize=8, color=INK,
                     xytext=(0.5, max(vals) * 1.28), textcoords="data",
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax3.set_ylim(0, max(vals) * 1.55)
    ax3.set_ylabel("connectome minus rewiring", fontsize=8.5, color=INK2)
    ax3.set_title("C   The test that could\n     have refuted B",
                  fontsize=10, color=INK, loc="left", pad=10, weight="bold")
    ax3.text(0.5, -0.34,
             "Macaque is the one unweighted network.\n"
             "With no weights to permute, the two\n"
             "methods are the same ensemble — and\n"
             "the prediction is that they must agree.",
             transform=ax3.transAxes, ha="center", va="top",
             fontsize=7.2, color=INK2)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
