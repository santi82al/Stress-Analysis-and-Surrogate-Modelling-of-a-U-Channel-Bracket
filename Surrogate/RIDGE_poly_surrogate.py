"""
Polynomial ridge-regression surrogate - smooth global baseline.

Degree-2 (optionally degree-3) polynomial features over the 12 standardized
inputs, ridge-regularized linear fit per target. Unlike trees, this model:
  - reads coordinates continuously (smooth fields, no leaf clamping),
  - EXTRAPOLATES smoothly outside the training hull,
  - can represent the sign change of U1 across the mid-plane via cross-terms
    (e.g. params x (2Xn-1)).

It will underfit sharp local features (stress concentrations), so expect it to
trail the trees on S_Mises - the point is to see how much of the field is
explained by a smooth low-order response surface.

Evaluation: identical held-out-model split (seed 0) as every other script.

Run:  python RIDGE_poly_surrogate.py
"""

import time

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from surrogate_common import (load_dataset, heldout_split, evaluate_predictions,
                              report, FEATURE_NAMES)


def make_model(degree, alpha=1.0):
    return make_pipeline(StandardScaler(),
                         PolynomialFeatures(degree=degree, include_bias=False),
                         Ridge(alpha=alpha))


if __name__ == "__main__":
    X, Y, groups, targets = load_dataset()
    tr, te = heldout_split(X, Y, groups)

    for degree in (2, 3):
        t0 = time.time()
        Yp = np.column_stack([
            make_model(degree).fit(X[tr], Y[tr, j]).predict(X[te])
            for j in range(len(targets))])
        elapsed = time.time() - t0
        scores = evaluate_predictions(Y[te], Yp, targets)
        report(f"Ridge_poly_deg{degree}", scores, elapsed,
               notes=f"degree={degree}, alpha=1.0, per-target")

    # deployable: degree-3 on all data
    models = {t: make_model(3).fit(X, Y[:, j]) for j, t in enumerate(targets)}
    joblib.dump({"models": models, "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True},
                "RIDGE_poly_surrogate.joblib")
    print("saved -> RIDGE_poly_surrogate.joblib")
