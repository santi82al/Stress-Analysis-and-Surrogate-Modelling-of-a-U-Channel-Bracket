"""
Ensemble surrogate: average of the sklearn MLP and the HGB baseline predictions.

The two best models err differently (smooth global net vs piecewise-constant
trees), so their mean often beats both. Scored per-target for the simple 50/50
average AND for the best weight found by grid search on the test set (an upper
bound kept for reference only; the deployed choice is the fixed 50/50).

Same canonical seed-0 held-out-model split as every other script.

Run:  python -u ENSEMBLE_mlp_hgb.py
"""

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from NN_sklearn_surrogate import fit_scaled, predict_scaled
from surrogate_common import (load_dataset, heldout_split, evaluate_predictions,
                              report)

if __name__ == "__main__":
    X, Y, groups, targets = load_dataset()
    tr, te = heldout_split(X, Y, groups)

    t0 = time.time()
    print("fitting HGB...", flush=True)
    Yp_hgb = np.column_stack([
        HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                      random_state=0)
        .fit(X[tr], Y[tr, j]).predict(X[te])
        for j in range(len(targets))])
    print("fitting MLP...", flush=True)
    bundle = fit_scaled(X[tr], Y[tr])
    Yp_mlp = predict_scaled(bundle, X[te])
    elapsed = time.time() - t0

    scores = evaluate_predictions(Y[te], 0.5 * (Yp_mlp + Yp_hgb), targets)
    report("ENSEMBLE_0.5MLP_0.5HGB", scores, elapsed,
           notes="mean of MLP(256x256x128) and HGB(500) predictions")

    # best weight per target, searched on the test set (reference only)
    for j, t in enumerate(targets):
        ws = np.linspace(0, 1, 21)
        r2s = [evaluate_predictions(Y[te], np.column_stack(
            [w * Yp_mlp[:, k] + (1 - w) * Yp_hgb[:, k] for k in range(len(targets))]),
            targets)[t] for w in ws]
        best = int(np.argmax(r2s))
        print(f"{t}: best w_MLP={ws[best]:.2f} -> R2={r2s[best]:.4f} "
              f"(MLP-only {r2s[-1]:.4f}, HGB-only {r2s[0]:.4f})")
