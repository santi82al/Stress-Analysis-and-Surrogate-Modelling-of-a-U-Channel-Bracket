"""
Surrogate validation, part 1 - group k-fold cross-validation.

Replaces the single 80/20 held-out-model split with a proper K-fold partition of
the *models*: every model is in the test fold exactly once, so the reported score
is a distribution over K independent held-out estimates, not one lucky split.

For each fold and target we record BOTH:
    * test  R^2  (on models never seen during that fold's fit)
    * train R^2  (on the fit models themselves)
The gap train-minus-test is the direct, numeric generalisation-gap diagnostic.

Models compared: HGB, MLP (reduced-epoch budget for affordability), and their
0.5/0.5 ensemble. The ensemble reuses the per-fold HGB and MLP predictions, so it
costs nothing extra.

Why this is the right test for a surrogate: the held-out unit is a whole geometry
(all its nodes), so no node from a test model ever informs the fit. The MLP's
internal early-stopping uses a row-level 10% split of the *training* models only;
it never sees test models, so the group-held-out test R^2 below is uncontaminated
by it (the leak affects only which epoch is chosen, an efficiency issue, not the
test estimate).

Outputs:
    validation_kfold_raw.csv      one row per (fold, target, model_estimator)
    validation_kfold_summary.csv  mean / std / 95% CI per (target, estimator)

Run (detached, ~2-3 h):  python -u validation_kfold.py
"""

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from surrogate_common import load_dataset

K = 5                  # 5 folds = train 424 / test 106 models, mirrors the 80/20
MLP_MAX_ITER = 30      # reduced budget; multiseed study showed 30 ep tracks 60 ep
                       # within ~0.02, and the deployed 60-ep model is marginally
                       # better -> these CI bounds are conservative (a lower bound)
RAW_CSV = "validation_kfold_raw.csv"
SUMMARY_CSV = "validation_kfold_summary.csv"


def fit_hgb(Xtr, ytr):
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                         random_state=0).fit(Xtr, ytr)


def fit_mlp(Xtr, Ytr):
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    net = MLPRegressor(hidden_layer_sizes=(256, 256, 128), activation="relu",
                       solver="adam", learning_rate_init=1e-3, batch_size=4096,
                       max_iter=MLP_MAX_ITER, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=8, tol=1e-5,
                       random_state=0)
    net.fit(xs.transform(Xtr), ys.transform(Ytr))
    return net, xs, ys


def mlp_predict(bundle, X):
    net, xs, ys = bundle
    return ys.inverse_transform(net.predict(xs.transform(X)))


def ci95(vals):
    """Two-sided 95% t confidence interval half-width for the mean of K samples."""
    vals = np.asarray(vals, float)
    k = len(vals)
    if k < 2:
        return 0.0
    return float(stats.t.ppf(0.975, k - 1) * vals.std(ddof=1) / np.sqrt(k))


if __name__ == "__main__":
    X, Y, groups, targets = load_dataset()
    n_models = len(np.unique(groups))
    print(f"dataset: {X.shape[0]} rows, {n_models} models, targets={targets}")
    print(f"GroupKFold K={K}: each model tested exactly once\n", flush=True)

    gkf = GroupKFold(n_splits=K)
    rows = []
    t00 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(X, Y, groups)):
        n_tr = len(np.unique(groups[tr]))
        n_te = len(np.unique(groups[te]))
        t0 = time.time()

        # --- HGB: one regressor per target ---
        hgb_tr = np.empty((len(tr), len(targets)))
        hgb_te = np.empty((len(te), len(targets)))
        for j in range(len(targets)):
            est = fit_hgb(X[tr], Y[tr, j])
            hgb_tr[:, j] = est.predict(X[tr])
            hgb_te[:, j] = est.predict(X[te])
        print(f"fold {fold}: HGB done ({time.time()-t0:.0f}s)", flush=True)

        # --- MLP: one multi-output net ---
        bundle = fit_mlp(X[tr], Y[tr])
        mlp_tr = mlp_predict(bundle, X[tr])
        mlp_te = mlp_predict(bundle, X[te])
        print(f"fold {fold}: MLP done ({time.time()-t0:.0f}s)", flush=True)

        # --- ensemble: 0.5/0.5 of the above ---
        ens_tr = 0.5 * (hgb_tr + mlp_tr)
        ens_te = 0.5 * (hgb_te + mlp_te)

        preds = {"HGB": (hgb_tr, hgb_te), "MLP": (mlp_tr, mlp_te),
                 "Ensemble": (ens_tr, ens_te)}
        for est_name, (ptr, pte) in preds.items():
            for j, t in enumerate(targets):
                rows.append({
                    "fold": fold, "estimator": est_name, "target": t,
                    "train_r2": r2_score(Y[tr, j], ptr[:, j]),
                    "test_r2":  r2_score(Y[te, j], pte[:, j]),
                    "n_train_models": n_tr, "n_test_models": n_te,
                })
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)   # checkpoint each fold
        print(f"fold {fold}: {n_tr} train / {n_te} test models  "
              f"(cum {time.time()-t00:.0f}s)\n", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_CSV, index=False)

    # --- summary: mean / std / 95% CI over folds, plus generalisation gap ---
    summ = []
    for (est_name, t), g in raw.groupby(["estimator", "target"]):
        te = g["test_r2"].to_numpy()
        trn = g["train_r2"].to_numpy()
        summ.append({
            "estimator": est_name, "target": t,
            "test_r2_mean": te.mean(), "test_r2_std": te.std(ddof=1),
            "test_r2_ci95": ci95(te),
            "test_r2_min": te.min(), "test_r2_max": te.max(),
            "train_r2_mean": trn.mean(),
            "gap_mean": (trn - te).mean(),
        })
    summary = pd.DataFrame(summ).sort_values(["target", "estimator"])
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n================  GROUP K-FOLD SUMMARY  ================")
    for t in targets:
        print(f"\n{t}:")
        sub = summary[summary.target == t].sort_values("test_r2_mean", ascending=False)
        for _, r in sub.iterrows():
            print(f"  {r['estimator']:9s}  test R2 = {r['test_r2_mean']:.3f} "
                  f"+/- {r['test_r2_ci95']:.3f} (95% CI)   "
                  f"[{r['test_r2_min']:.3f}, {r['test_r2_max']:.3f}]   "
                  f"train {r['train_r2_mean']:.3f}  gap {r['gap_mean']:.3f}")
    print(f"\nsaved -> {RAW_CSV}, {SUMMARY_CSV}")
    print(f"total {time.time()-t00:.0f}s")
