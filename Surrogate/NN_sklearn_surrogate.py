"""
Neural-network surrogate (sklearn MLPRegressor) - alternative to the HGB baseline.

Same pooled point-wise formulation as pooled_surrogate_normalized.py:
    input  = [b_geo..P_load, wall_thk, Xn, Yn, Zn]   (12 features)
    output = U1, U2, U3, S_Mises

Differences vs. the tree baseline that matter for a net:
  - Inputs are standardized (trees are scale-invariant; nets are not).
  - Targets are standardized per component (S_Mises is ~1e2, U* ~1e-1; without
    this the shared loss is dominated by stress).
  - One multi-output net (shared hidden representation across the 4 fields)
    instead of one estimator per target - the fields are physically coupled, so
    sharing features is a potential advantage over independent trees.
  - A net reads coordinates continuously, so it can represent the U1 field
    smoothly instead of as piecewise constants.

Evaluation: identical held-out-model split (GroupShuffleSplit, seed 0, 20% of
whole models) as every other script -> numbers directly comparable.

Run:  python NN_sklearn_surrogate.py
"""

import time

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from surrogate_common import (load_dataset, heldout_split, evaluate_predictions,
                              report, FEATURE_NAMES)

HIDDEN = (256, 256, 128)


def make_net(seed=0, max_iter=60):
    # early_stopping uses a random 10% row split (models leak between halves);
    # that only picks the stopping epoch; the reported score is still the
    # held-out-MODEL evaluation done outside.
    return MLPRegressor(hidden_layer_sizes=HIDDEN, activation="relu",
                        solver="adam", learning_rate_init=1e-3,
                        batch_size=4096, max_iter=max_iter,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=8, tol=1e-5,
                        random_state=seed, verbose=True)


def fit_scaled(Xtr, Ytr, seed=0):
    """Standardize inputs and per-target outputs, fit one multi-output net.
    Returns (net, xscaler, yscaler)."""
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    net = make_net(seed)
    net.fit(xs.transform(Xtr), ys.transform(Ytr))
    return net, xs, ys


def predict_scaled(bundle, X):
    net, xs, ys = bundle
    return ys.inverse_transform(net.predict(xs.transform(X)))


if __name__ == "__main__":
    X, Y, groups, targets = load_dataset()
    print(f"dataset: {X.shape[0]} rows x {X.shape[1]} features "
          f"({len(np.unique(groups))} models), targets={targets}")

    # --- held-out-model evaluation (canonical split) ---
    tr, te = heldout_split(X, Y, groups)
    t0 = time.time()
    bundle = fit_scaled(X[tr], Y[tr])
    elapsed = time.time() - t0
    scores = evaluate_predictions(Y[te], predict_scaled(bundle, X[te]), targets)
    report("MLP_sklearn_256x256x128", scores, elapsed,
           notes=f"hidden={HIDDEN}, multi-output, std-scaled X and Y")

    # --- final model on ALL data, for deployment ---
    print("refitting on all models for the deployable surrogate...")
    bundle_full = fit_scaled(X, Y)
    joblib.dump({"net": bundle_full[0], "xscaler": bundle_full[1],
                 "yscaler": bundle_full[2], "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True},
                "NN_sklearn_surrogate.joblib")
    print("saved -> NN_sklearn_surrogate.joblib")
