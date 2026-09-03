"""
Parity plot: held-out surrogate prediction versus finite-element truth.

Every point comes from a model that was in the TEST fold when it was predicted,
so no point is in-sample. Data: kfold_parity_nodes.csv (validation_kfold_dump.py).

With ~159k points a scatter would be a solid blob, so each panel is a density
hexbin on a log count scale; the identity line is the reference, and the axes are
forced equal so a departure from y=x is read as a slope, not as an artefact of
different scalings.

Run:  python plot_parity.py [out.png]
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CSV = "kfold_parity_nodes.csv"
UNITS = {"U1": "mm", "U2": "mm", "U3": "mm", "S_Mises": "MPa"}
REFERENCE_RED = "#d62728"


def panel(ax, true, pred, name, permodel_median):
    # Clip the display window to the bulk of the data: a handful of extreme
    # compliant models would otherwise set the limits and squeeze everything
    # else into a corner. The fit statistics below are computed on ALL points.
    lo = float(np.percentile(np.concatenate([true, pred]), 0.2))
    hi = float(np.percentile(np.concatenate([true, pred]), 99.8))
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    hb = ax.hexbin(true, pred, gridsize=70, extent=(lo, hi, lo, hi),
                   cmap="viridis", norm=LogNorm(), mincnt=1, linewidths=0)
    # Red for the y=x reference: it is the one hue viridis does not contain, so
    # the line stays readable over both the dark and the bright end of the ramp.
    # A white casing underneath separates it from the hexbins in the dense core,
    # where a bare line of any colour gets lost among the fill.
    ax.plot([lo, hi], [lo, hi], color="white", lw=5.0, solid_capstyle="butt",
            zorder=4, alpha=0.85)
    ax.plot([lo, hi], [lo, hi], color=REFERENCE_RED, lw=2.6, ls=(0, (6, 3)),
            zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    err = pred - true
    ss = 1 - float((err ** 2).sum()) / (float(((true - true.mean()) ** 2).sum()) or 1e-30)
    unit = UNITS.get(name, "")
    ax.set_xlabel("FEM %s [%s]" % (name, unit), fontsize=9)
    ax.set_ylabel("Surrogate %s [%s]" % (name, unit), fontsize=9)
    ax.set_title("%s   pooled $R^2$=%.3f   median/model=%.3f"
                 % (name, ss, permodel_median), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    return hb


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "parity.png"
    d = pd.read_csv(CSV)
    pm = pd.read_csv("kfold_permodel_r2.csv")
    targets = [c[:-5] for c in d.columns if c.endswith("_true")]

    fig, axes = plt.subplots(1, len(targets), figsize=(4.1 * len(targets), 4.3))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        hb = panel(ax, d[t + "_true"].to_numpy(float), d[t + "_pred"].to_numpy(float),
                   t, pm["r2_" + t].median())
    cb = fig.colorbar(hb, ax=list(axes), orientation="horizontal",
                      fraction=0.05, pad=0.16, aspect=60)
    cb.set_label("nodes per bin (log scale)", fontsize=9)
    fig.suptitle("Held-out parity: %s nodes, %d models, each held out exactly once"
                 % (f"{len(d):,}", d.model.nunique()), fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote %s  (%d points, %d models)" % (out, len(d), d.model.nunique()))


if __name__ == "__main__":
    main()
