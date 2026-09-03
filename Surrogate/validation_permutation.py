"""
Surrogate validation, part 2 - permutation test (overfitting check).

Question addressed: is the held-out R^2 real signal, or just a flexible model
exploiting chance structure? The permutation test answers this with a p-value.

Procedure (Ojala & Garriga, 2010; sklearn's permutation_test_score logic):
  1. Fit + score the model on the REAL data with a held-out-model split
     -> observed R^2 per target.
  2. Randomly SHUFFLE the target column across all rows, breaking the
     features->target relationship, and re-fit + re-score. Repeat n_perm times
     -> a NULL distribution of R^2 achievable by chance with the same model,
        same capacity, same split.
  3. p-value = (1 + #{null R^2 >= observed R^2}) / (1 + n_perm).

If the observed R^2 sits far above the entire null cloud (null centres on ~0,
observed ~0.6-0.9), then p < 1/(n_perm+1): the model learned real structure.
A model that were "just overfitting" would score ~0 here too, because
overfitting does not survive a held-out split.

Cost control: the null R^2 is ~0 regardless of fine details, so we subsample the
node cloud and use a lighter HGB. The conclusion is model-agnostic (it is a
property of the DATA, not of HGB), so one representative learner suffices; MLP and
the ensemble fit the same data and inherit the same p-value logic.

Outputs:
    validation_permutation.csv   observed + full null distribution per target
    validation_permutation.png   null histograms with the observed R^2 marked

Run (detached, ~1 h):  python -u validation_permutation.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from surrogate_common import load_dataset

N_PERM = 50
NODES_PER_MODEL = 500      # subsample for speed; null is robust to this
HGB_ITERS = 120            # lighter HGB; null ~0 regardless
SEED = 0
RAW_CSV = "validation_permutation.csv"
OUT_PNG = "validation_permutation.png"


def subsample_by_group(X, Y, groups, per_model, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > per_model:
            idx = rng.choice(idx, per_model, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    return X[keep], Y[keep], groups[keep]


def fit_score(Xtr, ytr, Xte, yte):
    est = HistGradientBoostingRegressor(max_iter=HGB_ITERS, learning_rate=0.05,
                                        random_state=0).fit(Xtr, ytr)
    return r2_score(yte, est.predict(Xte))


if __name__ == "__main__":
    X, Y, groups, targets = load_dataset()
    X, Y, groups = subsample_by_group(X, Y, groups, NODES_PER_MODEL, seed=SEED)
    print(f"permutation test on {X.shape[0]} subsampled rows, "
          f"{len(np.unique(groups))} models, targets={targets}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, Y, groups))
    Xtr, Xte = X[tr], X[te]

    # 1. observed R^2 on real labels
    observed = {t: fit_score(Xtr, Y[tr, j], Xte, Y[te, j])
                for j, t in enumerate(targets)}
    print("observed (real-label) held-out R2:",
          {t: round(v, 4) for t, v in observed.items()}, flush=True)

    # 2. null distribution: shuffle each target column globally, re-fit
    rng = np.random.default_rng(SEED + 1)
    null = {t: [] for t in targets}
    t00 = time.time()
    for p in range(N_PERM):
        perm = rng.permutation(len(Y))
        Yp = Y[perm]                       # break X->Y, keep each target's marginal
        for j, t in enumerate(targets):
            null[t].append(fit_score(Xtr, Yp[tr, j], Xte, Yp[te, j]))
        if (p + 1) % 5 == 0:
            print(f"  perm {p+1}/{N_PERM}  ({time.time()-t00:.0f}s)  "
                  + "  ".join(f"{t}~{np.mean(null[t]):+.3f}" for t in targets),
                  flush=True)

    # 3. p-values and save
    rows = []
    for t in targets:
        nd = np.array(null[t])
        pval = (1 + np.sum(nd >= observed[t])) / (1 + N_PERM)
        rows.append({"target": t, "observed_r2": observed[t],
                     "null_mean": nd.mean(), "null_std": nd.std(),
                     "null_max": nd.max(), "p_value": pval, "n_perm": N_PERM})
        print(f"{t}: observed {observed[t]:.4f}  null {nd.mean():+.4f}+/-{nd.std():.4f} "
              f"(max {nd.max():+.4f})  p={pval:.4f}")
    summary = pd.DataFrame(rows)

    # long form with the full null too, for the plot / appendix
    long = [{"target": t, "kind": "null", "r2": v} for t in targets for v in null[t]]
    long += [{"target": t, "kind": "observed", "r2": observed[t]} for t in targets]
    pd.DataFrame(long).to_csv(RAW_CSV, index=False)
    summary.to_csv("validation_permutation_summary.csv", index=False)
    print(f"\nsaved -> {RAW_CSV}, validation_permutation_summary.csv")

    # ---- plot: null histogram + observed marker per target ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
        TCOL = {"U1": "#2a78d6", "U2": "#008300", "U3": "#e87ba4", "S_Mises": "#eda100"}
        ncol = 2
        nrow = (len(targets) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(9, 3.3 * nrow), squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        axes = axes.ravel()
        for i, t in enumerate(targets):
            ax = axes[i]
            ax.set_facecolor(SURFACE)
            nd = np.array(null[t])
            ax.hist(nd, bins=20, color=MUTED, alpha=0.55, label="null (shuffled)")
            ax.axvline(observed[t], color=TCOL.get(t, "#d55181"), lw=2.5,
                       label=f"observed {observed[t]:.2f}")
            ax.axvline(0.0, color=MUTED, lw=0.8, ls="--")
            pval = (1 + np.sum(nd >= observed[t])) / (1 + N_PERM)
            ax.set_title(f"{t}   (p = {pval:.3f})", fontsize=11, color=INK)
            ax.set_xlabel("held-out R²", fontsize=9, color=INK)
            ax.set_ylabel("count", fontsize=9, color=INK)
            ax.tick_params(colors=INK, labelsize=8.5)
            ax.grid(True, color=MUTED, alpha=0.15)
            for sp in ax.spines.values():
                sp.set_color(MUTED); sp.set_alpha(0.4)
            ax.legend(fontsize=8, frameon=False)
        for j in range(len(targets), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Permutation test - observed R² vs null ({N_PERM} label shuffles)",
                     fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUT_PNG, dpi=140, facecolor=SURFACE)
        print(f"saved -> {OUT_PNG}")
    except Exception as e:
        print(f"(plot skipped: {e})")
