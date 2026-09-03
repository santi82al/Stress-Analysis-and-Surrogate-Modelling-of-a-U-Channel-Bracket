"""
LightGBM surrogate - gradient-boosted trees, alternative implementation to
sklearn's HistGradientBoostingRegressor baseline.

Same family as the baseline (leaf-wise boosted trees) but with a different
growth strategy and more headroom to tune (num_leaves, feature/bagging
fractions). If LightGBM beats HGB it's tuning headroom, not a new model class.

Evaluation: identical held-out-model split (seed 0) as every other script.

Run:  python LGBM_pooled_surrogate.py
"""

import time

import joblib
from lightgbm import LGBMRegressor

from surrogate_common import (load_dataset, heldout_split, evaluate_predictions,
                              report, FEATURE_NAMES)


def make_est(seed=0):
    return LGBMRegressor(n_estimators=1500, learning_rate=0.05,
                         num_leaves=127, min_child_samples=40,
                         feature_fraction=0.9, bagging_fraction=0.8,
                         bagging_freq=1, random_state=seed,
                         n_jobs=-1, verbose=-1)


if __name__ == "__main__":
    import numpy as np
    X, Y, groups, targets = load_dataset()
    tr, te = heldout_split(X, Y, groups)

    t0 = time.time()
    Yp = np.column_stack([
        make_est().fit(X[tr], Y[tr, j]).predict(X[te])
        for j in range(len(targets))])
    elapsed = time.time() - t0
    scores = evaluate_predictions(Y[te], Yp, targets)
    report("LightGBM_1500x127", scores, elapsed,
           notes="n_estimators=1500, num_leaves=127, lr=0.05, per-target")

    models = {t: make_est().fit(X, Y[:, j]) for j, t in enumerate(targets)}
    joblib.dump({"models": models, "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True},
                "LGBM_pooled_surrogate.joblib")
    print("saved -> LGBM_pooled_surrogate.joblib")
