"""
Retrain every deployable surrogate artifact on the FULL current DOE
(the two-stage HGB_2stage_* variant is retired and excluded).

Artifacts written (all trained on every model in the cache):
    pooled_surrogate_normalized.joblib   HGB per-target (baseline format)
    NN_sklearn_surrogate.joblib          sklearn MLP
    NN_torch_surrogate.pt (+ meta)       PyTorch MLP
    LGBM_pooled_surrogate.joblib         LightGBM per-target
    KNN_pooled_surrogate.joblib          kNN
    RIDGE_poly_surrogate.joblib          degree-3 polynomial ridge
    ENSEMBLE_mlp_hgb.joblib              0.5*MLP + 0.5*HGB (recommended)

Delete pooled_dataset_cache.npz first (or set SURROGATE_REFRESH_CACHE=1) to
pick up new models; the runner does that.

Run:  python -u retrain_deployables.py
"""

import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from surrogate_common import load_dataset, FEATURE_NAMES
from NN_sklearn_surrogate import fit_scaled
from NN_torch_surrogate import train_net, split_val_models, HIDDEN
from LGBM_pooled_surrogate import make_est as make_lgbm
from RIDGE_poly_surrogate import make_model as make_ridge


def stamp(msg, t0):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    X, Y, groups, targets = load_dataset()
    n_models = len(np.unique(groups))
    stamp(f"dataset: {X.shape[0]} rows, {n_models} models, targets={targets}", t0)

    # --- HGB (baseline format, reused by the ensemble) ---
    hgb = {}
    for j, t in enumerate(targets):
        hgb[t] = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                               random_state=0).fit(X, Y[:, j])
        stamp(f"HGB {t} trained", t0)
    joblib.dump({"models": hgb, "targets": targets, "normalized": True,
                 "n_models": n_models}, "pooled_surrogate_normalized.joblib")
    stamp("saved pooled_surrogate_normalized.joblib", t0)

    # --- LightGBM ---
    lgbm = {t: make_lgbm().fit(X, Y[:, j]) for j, t in enumerate(targets)}
    joblib.dump({"models": lgbm, "targets": targets, "features": FEATURE_NAMES,
                 "normalized": True, "n_models": n_models},
                "LGBM_pooled_surrogate.joblib")
    stamp("saved LGBM_pooled_surrogate.joblib", t0)

    # --- kNN ---
    xs = StandardScaler().fit(X)
    knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
    knn.fit(xs.transform(X), Y)
    joblib.dump({"knn": knn, "xscaler": xs, "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True,
                 "n_models": n_models}, "KNN_pooled_surrogate.joblib")
    stamp("saved KNN_pooled_surrogate.joblib", t0)

    # --- polynomial ridge (deg 3) ---
    ridge = {t: make_ridge(3).fit(X, Y[:, j]) for j, t in enumerate(targets)}
    joblib.dump({"models": ridge, "targets": targets, "features": FEATURE_NAMES,
                 "normalized": True, "n_models": n_models},
                "RIDGE_poly_surrogate.joblib")
    stamp("saved RIDGE_poly_surrogate.joblib", t0)

    # --- sklearn MLP (60-epoch budget) ---
    net, mxs, mys = fit_scaled(X, Y)
    joblib.dump({"net": net, "xscaler": mxs, "yscaler": mys, "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True,
                 "n_models": n_models}, "NN_sklearn_surrogate.joblib")
    stamp("saved NN_sklearn_surrogate.joblib", t0)

    # --- ensemble bundle (reuses the two fits above) ---
    joblib.dump({"hgb": hgb, "nn_bundle_file": "NN_sklearn_surrogate.joblib",
                 "weight_mlp": 0.5, "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True,
                 "n_models": n_models}, "ENSEMBLE_mlp_hgb.joblib")
    stamp("saved ENSEMBLE_mlp_hgb.joblib", t0)

    # --- PyTorch MLP (needs val models for early stopping; trains on the rest) ---
    all_idx = np.arange(len(X))
    fit_idx, val_idx = split_val_models(all_idx, groups, frac=0.1)
    tnet, txs, tys, best_epoch = train_net(X[fit_idx], Y[fit_idx],
                                           X[val_idx], Y[val_idx], verbose=False)
    torch.save(tnet.state_dict(), "NN_torch_surrogate.pt")
    joblib.dump({"state_dict_file": "NN_torch_surrogate.pt", "hidden": HIDDEN,
                 "xscaler": txs, "yscaler": tys, "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True,
                 "n_models": n_models}, "NN_torch_surrogate_meta.joblib")
    stamp(f"saved NN_torch_surrogate.pt (best epoch {best_epoch})", t0)
    stamp("ALL DEPLOYABLES RETRAINED", t0)
