"""Path bootstrap shared by the predictor scripts.

The predictors live in Predictor/, but the trained bundles (*.joblib) and the
training-side modules they import (pooled_surrogate_normalized, surrogate_common)
stay in Surrogate/ next to the code that fits them.
Importing this module first puts Surrogate/ on sys.path, so the imports below it
resolve, and resolve_bundle() finds the .joblib files whatever the CWD is.
"""

import sys
from pathlib import Path

SURROGATE_DIR = Path(__file__).resolve().parent.parent / "Surrogate"

if str(SURROGATE_DIR) not in sys.path:
    sys.path.insert(0, str(SURROGATE_DIR))


def resolve_bundle(name):
    """Locate a trained-model file: as given if it exists, else in Surrogate/."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return str(p)
    return str(SURROGATE_DIR / p)


# ---------------------------------------------------------------------------
# POST-PROCESSING FILTER - APPLIED TO EVERY PREDICTION THIS PACKAGE PRODUCES.
#
# Von Mises stress and displacement/reaction magnitudes are norms, so they are
# non-negative by definition. The surrogates regress them as unconstrained real
# values, so in low-stress regions they undershoot slightly below zero (observed
# on model_200: 85 of 9227 nodes, min -1.02 MPa against a 222 MPa range). Those
# values are non-physical, so they are clamped at zero on the way out.
#
# This is a physical-admissibility correction, NOT an accuracy improvement: on
# model_200 it moved S_Mises RMSE 3.756 -> 3.755 and left R2 = 0.974 unchanged.
# Displacement COMPONENTS (U1, U2, U3) are signed and are never clipped.
#
# This filter is a deliberate post-processing step, documented in the thesis
# (deployment chapter): the predicted fields are not raw regressor output.
# ---------------------------------------------------------------------------
NONNEGATIVE_TARGETS = ("S_Mises", "U_mag", "RF_mag")


def clip_nonnegative(df, verbose=True):
    """Clamp non-negative targets at 0, reporting what was clipped.

    Called by every predictor in this folder before the CSV is written.
    """
    for t in NONNEGATIVE_TARGETS:
        if t in df.columns:
            neg = df[t] < 0.0
            n = int(neg.sum())
            if n:
                if verbose:
                    print("clipped %d negative %s value(s) to 0 (min was %+.4e)"
                          % (n, t, df[t].min()))
                df.loc[neg, t] = 0.0
    return df
