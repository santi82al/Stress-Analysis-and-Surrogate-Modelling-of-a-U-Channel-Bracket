"""
Surrogate validation, part 3 - nested cross-validation for the ensemble weight.

Concern: the ensemble weight (and, more loosely, the choice "an MLP+HGB ensemble
is best") were informed by looking at held-out scores. If a hyperparameter is
tuned on the same data used to report the score, the score is optimistically
biased. Nested CV removes that bias:

    OUTER GroupKFold  -> provides the test models (touched once).
      INNER GroupKFold on the outer-train models only -> out-of-fold MLP & HGB
        predictions, on which the best convex weight w* is selected PER TARGET.
      Then refit MLP+HGB on the full outer-train and score on the outer-test,
        using the inner-selected w* (which never saw the outer-test).

We report, per target:
  * nested test R^2 with the SELECTED weight  (bias-free estimate of a tuned ensemble)
  * test R^2 with the FIXED 0.5 weight          (what is actually deployed)
  * the selected w* per outer fold             (is it stable, and near 0.5?)

If nested-selected ~= fixed-0.5, then tuning the weight buys nothing and the
deployed untuned ensemble carries no selection bias.

Budget: light MLP (few epochs) and node subsampling; this study is about the
STABILITY of the weight choice, not squeezing the last R^2. Cross-refs the fuller
scores in validation_kfold.py.

Run (detached, ~2 h):  python -u validation_nested.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from surrogate_common import load_dataset

OUTER_K = 3
INNER_K = 3
MLP_MAX_ITER = 25
NODES_PER_MODEL = 2500
WEIGHTS = np.linspace(0.0, 1.0, 11)   # MLP fraction grid
OUT_CSV = "validation_nested.csv"


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


def hgb_predict(Xtr, Ytr, Xte):
    return np.column_stack([
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      random_state=0).fit(Xtr, Ytr[:, j]).predict(Xte)
        for j in range(Ytr.shape[1])])


def mlp_predict(Xtr, Ytr, Xte):
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    net = MLPRegressor(hidden_layer_sizes=(256, 256, 128), activation="relu",
                       solver="adam", learning_rate_init=1e-3, batch_size=4096,
                       max_iter=MLP_MAX_ITER, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=6, tol=1e-5,
                       random_state=0)
    net.fit(xs.transform(Xtr), ys.transform(Ytr))
    return ys.inverse_transform(net.predict(xs.transform(Xte)))


def select_weight(y_true, mlp_oof, hgb_oof):
    """Per column, the convex weight on MLP maximising OOF R^2."""
    best_w = np.empty(y_true.shape[1])
    for j in range(y_true.shape[1]):
        r2s = [r2_score(y_true[:, j], w * mlp_oof[:, j] + (1 - w) * hgb_oof[:, j])
               for w in WEIGHTS]
        best_w[j] = WEIGHTS[int(np.argmax(r2s))]
    return best_w


if __name__ == "__main__":
    X, Y, groups, targets = load_dataset()
    X, Y, groups = subsample_by_group(X, Y, groups, NODES_PER_MODEL)
    print(f"nested CV on {X.shape[0]} rows, {len(np.unique(groups))} models, "
          f"targets={targets}", flush=True)

    outer = GroupKFold(n_splits=OUTER_K)
    rows = []
    t00 = time.time()
    for ofold, (otr, ote) in enumerate(outer.split(X, Y, groups)):
        Xo, Yo, go = X[otr], Y[otr], groups[otr]

        # --- inner: out-of-fold predictions on the outer-train models ---
        mlp_oof = np.zeros_like(Yo)
        hgb_oof = np.zeros_like(Yo)
        inner = GroupKFold(n_splits=INNER_K)
        for itr, ite in inner.split(Xo, Yo, go):
            mlp_oof[ite] = mlp_predict(Xo[itr], Yo[itr], Xo[ite])
            hgb_oof[ite] = hgb_predict(Xo[itr], Yo[itr], Xo[ite])
        w_star = select_weight(Yo, mlp_oof, hgb_oof)
        print(f"outer {ofold}: selected w* = "
              + ", ".join(f"{t}={w:.2f}" for t, w in zip(targets, w_star))
              + f"  ({time.time()-t00:.0f}s)", flush=True)

        # --- refit on full outer-train, score on outer-test ---
        mlp_te = mlp_predict(Xo, Yo, X[ote])
        hgb_te = hgb_predict(Xo, Yo, X[ote])
        for j, t in enumerate(targets):
            sel = w_star[j] * mlp_te[:, j] + (1 - w_star[j]) * hgb_te[:, j]
            fix = 0.5 * mlp_te[:, j] + 0.5 * hgb_te[:, j]
            rows.append({
                "outer_fold": ofold, "target": t, "w_star": w_star[j],
                "test_r2_selected": r2_score(Y[ote, j], sel),
                "test_r2_fixed0.5": r2_score(Y[ote, j], fix),
                "test_r2_mlp": r2_score(Y[ote, j], mlp_te[:, j]),
                "test_r2_hgb": r2_score(Y[ote, j], hgb_te[:, j]),
            })
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"outer {ofold} scored ({time.time()-t00:.0f}s)\n", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n===============  NESTED CV SUMMARY  ===============")
    for t in targets:
        g = df[df.target == t]
        print(f"{t}:  nested-selected {g.test_r2_selected.mean():.3f}   "
              f"fixed-0.5 {g['test_r2_fixed0.5'].mean():.3f}   "
              f"(w* = {', '.join(f'{w:.2f}' for w in g.w_star)})   "
              f"MLP {g.test_r2_mlp.mean():.3f}  HGB {g.test_r2_hgb.mean():.3f}")
    print(f"\nsaved -> {OUT_CSV}\ntotal {time.time()-t00:.0f}s")
