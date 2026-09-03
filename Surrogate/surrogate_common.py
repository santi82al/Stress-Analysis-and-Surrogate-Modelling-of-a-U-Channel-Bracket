"""
Shared utilities for the alternative-algorithm surrogate scripts.

Provides:
  - load_dataset(): the pooled, per-model-normalized dataset (same construction as
    pooled_surrogate_normalized.py) with an .npz cache so every algorithm script
    doesn't re-read 400+ CSVs.
  - heldout_split(): the SAME GroupShuffleSplit (seed 0, 20% of whole models) used
    by the HGB baseline, so every algorithm's held-out-model R2 is directly
    comparable across scripts.
  - evaluate(): fit-per-target + held-out-model R2 report, returned as a dict.
  - append_findings_row(): tiny helper to log a result line to results_comparison.csv.

Cache invalidation: the cache stores the number of result files and the params-csv
mtime; if either changes the dataset is rebuilt.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from pooled_surrogate_normalized import build_pooled_dataset, PARAM_NAMES, NORM_COORD_NAMES

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "output")
PARAMS_CSV = os.path.join(_HERE, "..", "model_params.csv")
CACHE_NPZ = os.path.join(_HERE, "pooled_dataset_cache.npz")
COMPARISON_CSV = os.path.join(_HERE, "results_comparison.csv")

FEATURE_NAMES = PARAM_NAMES + NORM_COORD_NAMES

# Set by load_dataset(); report() stamps it on every result row so the
# comparison CSV records which dataset snapshot produced each score.
LAST_N_MODELS = None


def _n_result_files():
    return len(glob.glob(os.path.join(RESULTS_DIR, "model_*", "model_*_results.csv")))


def load_dataset(nodes_per_model=4000, seed=0, verbose=True):
    """Return X, Y, groups, targets - cached in pooled_dataset_cache.npz."""
    global LAST_N_MODELS
    n_files = _n_result_files()
    params_mtime = os.path.getmtime(PARAMS_CSV)
    if os.path.exists(CACHE_NPZ):
        d = np.load(CACHE_NPZ, allow_pickle=True)
        # FROZEN by default: the DOE generator may still be producing new
        # model_<i> folders while a comparison study runs; rebuilding
        # mid-comparison would put different algorithms on different datasets.
        # Set SURROGATE_REFRESH_CACHE=1 to force a rebuild on new data.
        frozen = os.environ.get("SURROGATE_REFRESH_CACHE", "0") != "1"
        if frozen and int(d["n_files"]) != n_files:
            print(f"NOTE: cache frozen at {int(d['n_files'])} result files; "
                  f"{n_files} now on disk (set SURROGATE_REFRESH_CACHE=1 to rebuild)")
        if ((frozen or int(d["n_files"]) == n_files)
                and float(d["params_mtime"]) == params_mtime
                and int(d["nodes_per_model"]) == nodes_per_model
                and int(d["seed"]) == seed):
            LAST_N_MODELS = len(np.unique(d["groups"]))
            if verbose:
                print(f"loaded cached dataset: {d['X'].shape[0]} rows, "
                      f"{LAST_N_MODELS} models")
            return d["X"], d["Y"], d["groups"], list(d["targets"])
    X, Y, groups, targets = build_pooled_dataset(RESULTS_DIR, PARAMS_CSV,
                                                 nodes_per_model=nodes_per_model,
                                                 seed=seed)
    LAST_N_MODELS = len(np.unique(groups))
    np.savez_compressed(CACHE_NPZ, X=X, Y=Y, groups=groups,
                        targets=np.array(targets, dtype=object),
                        n_files=n_files, params_mtime=params_mtime,
                        nodes_per_model=nodes_per_model, seed=seed)
    if verbose:
        print(f"built + cached dataset: {X.shape[0]} rows, "
              f"{len(np.unique(groups))} models -> {CACHE_NPZ}")
    return X, Y, groups, targets


def heldout_split(X, Y, groups, seed=0, test_size=0.2):
    """The canonical held-out-model split (identical to the HGB baseline)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, Y, groups))
    return tr, te


def evaluate_predictions(Y_true, Y_pred, targets):
    return {t: r2_score(Y_true[:, j], Y_pred[:, j]) for j, t in enumerate(targets)}


def report(name, scores, elapsed=None, notes=""):
    """Print a one-line summary and append it to results_comparison.csv."""
    line = "  ".join(f"{t}={v:+.4f}" for t, v in scores.items())
    extra = f"  ({elapsed:.0f}s)" if elapsed else ""
    print(f"[{name}] held-out-model R2:  {line}{extra}")
    row = {"algorithm": name, "timestamp": time.strftime("%Y-%m-%d %H:%M"),
           "notes": notes, **{t: round(v, 4) for t, v in scores.items()}}
    if elapsed is not None:
        row["train_seconds"] = round(elapsed, 1)
    if LAST_N_MODELS is not None:
        row["n_models_snapshot"] = LAST_N_MODELS
    df = pd.DataFrame([row])
    header = not os.path.exists(COMPARISON_CSV)
    df.to_csv(COMPARISON_CSV, mode="a", header=header, index=False)
    print(f"appended -> {COMPARISON_CSV}")
