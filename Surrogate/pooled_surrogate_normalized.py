"""
Pooled point-wise surrogate - NORMALIZED coordinate variant.

Node coordinates are normalized PER MODEL to a unit bounding box before pooling
(vs. using raw coordinates):

    X_norm = (X - X.min) / (X.max - X.min)     (per model, per axis)

Rationale: the DOE varies the geometry size (b_geo, m_geo, n_geo, ...), so every
model has a different physical bounding box. With raw XYZ a held-out model lands
in coordinate regions the trees never split on -> pure extrapolation -> the tree
ensemble clamps to the nearest trained leaf and generalizes poorly (U1 went
negative on held-out-model R2). Absolute size is still available to the model
through the geometry parameters; the normalized coordinate only encodes
position-within-part, which is comparable across models.

Also adds `learning_curve` to answer "will more DOE samples help?" empirically:
train on an increasing number of models against a fixed held-out set of models
and watch held-out R2 vs #train-models.

Each node of each FEM model is one training sample:
    input  = [b_geo..P_load, wall_thk, Xn, Yn, Zn]
    output = U1, U2, U3, S_Mises (S_Mises dropped automatically if all-zero)

Expects files:
    model_params.csv                          (Parameter column + one column per model_i)
    <results_dir>/model_<i>/model_<i>_results.csv
"""

import glob
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

# Reuse the repo-root, stdlib-only geometry check (shared with run_models.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geom_check import bbox_matches  # noqa: E402

PARAM_NAMES = ["b_geo", "m_geo", "n_geo", "t_geo",
               "f_geo", "e_geo", "i_geo", "P_load", "wall_thk"]
COORD_NAMES = ["X", "Y", "Z"]
NORM_COORD_NAMES = ["Xn", "Yn", "Zn"]
TARGET_NAMES = ["U1", "U2", "U3", "S_Mises"]   # S_Mises dropped automatically if all-zero


def load_params(params_csv):
    """Return {model_name: {param: float}} from model_params.csv."""
    df = pd.read_csv(params_csv, index_col="Parameter")
    df = df.drop(index="Description", errors="ignore")
    params = {}
    for col in df.columns:
        vals = df[col].astype(str).str.strip().astype(float)
        params[col] = {p: vals[p] for p in PARAM_NAMES}
    return params


def check_geometry_consistency(name, p, res, tol=0.02):
    """Compare a results DataFrame's node coordinate bounding box against the box
    implied by its model_params.csv row. Catches mismatched param/output pairings.
    """
    actual = {axis: (float(res[axis].min()), float(res[axis].max())) for axis in COORD_NAMES}
    ok, mismatches = bbox_matches(p, actual, tol=tol)
    for axis, (exp_lo, exp_hi), (act_lo, act_hi) in mismatches:
        print(f"  MISMATCH {name} axis {axis}: expected [{exp_lo:.2f}, {exp_hi:.2f}], "
              f"got [{act_lo:.2f}, {act_hi:.2f}]")
    return ok


def normalize_coords_inplace(res):
    """Add Xn, Yn, Zn columns: each raw axis scaled to [0, 1] by this model's own
    bounding box. A degenerate (zero-span) axis maps to 0.0."""
    for axis, naxis in zip(COORD_NAMES, NORM_COORD_NAMES):
        lo = float(res[axis].min())
        hi = float(res[axis].max())
        span = hi - lo
        res[naxis] = 0.0 if span == 0.0 else (res[axis] - lo) / span
    return res


def build_pooled_dataset(results_dir, params_csv,
                         nodes_per_model=4000, seed=0):
    """Stack sampled + per-model-normalized nodes from every model into one table.

    Returns X (n, 12) with normalized coordinates, Y (n, n_targets),
    groups (n,) model index per row, and the list of target names actually used.
    """
    params = load_params(params_csv)
    rng = np.random.default_rng(seed)
    frames, groups, bad_geometry = [], [], []

    files = sorted(glob.glob(os.path.join(results_dir, "model_*", "model_*_results.csv")))
    n_files = len(files)
    for i, path in enumerate(files, start=1):
        name = re.match(r"(model_\d+)_results", os.path.basename(path)).group(1)
        pct = 100 * i / n_files
        if name not in params:
            print(f"[{pct:5.1f}%] ({i}/{n_files}) skipping {path}: no column '{name}' in params csv")
            continue
        res = pd.read_csv(path)
        if not check_geometry_consistency(name, params[name], res):
            bad_geometry.append(name)
        # Normalize on the FULL node cloud (true per-model bbox) before subsampling.
        normalize_coords_inplace(res)
        if nodes_per_model and len(res) > nodes_per_model:
            idx = rng.choice(len(res), nodes_per_model, replace=False)
            res = res.iloc[idx]
        for p, v in params[name].items():
            res[p] = v
        frames.append(res)
        groups.append(np.full(len(res), int(name.split("_")[1])))
        print(f"[{pct:5.1f}%] ({i}/{n_files}) loaded {name}: {len(res)} rows")

    if bad_geometry:
        print(f"WARNING: geometry/params mismatch in {len(bad_geometry)} model(s): {bad_geometry}")

    df = pd.concat(frames, ignore_index=True)
    groups = np.concatenate(groups)

    # Drop targets that are entirely zero (known issue: NODAL stress export)
    targets = []
    for t in TARGET_NAMES:
        if t in df.columns and not np.allclose(df[t].values, 0.0):
            targets.append(t)
        else:
            print(f"target '{t}' missing or all-zero -> dropped")

    X = df[PARAM_NAMES + NORM_COORD_NAMES].to_numpy(float)
    Y = df[targets].to_numpy(float)
    return X, Y, groups, targets


def make_estimator(method="hgb", seed=0):
    if method == "rf":
        return RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                     min_samples_leaf=2, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                         random_state=seed)


def train(X, Y, targets, method="hgb", seed=0):
    """One regressor per output component. method: 'hgb' or 'rf'."""
    models = {}
    n_targets = len(targets)
    for j, t in enumerate(targets, start=1):
        print(f"[{100 * j / n_targets:5.1f}%] ({j}/{n_targets}) training {t}...")
        est = make_estimator(method, seed)
        est.fit(X, Y[:, j - 1])
        models[t] = est
        print(f"[{100 * j / n_targets:5.1f}%] ({j}/{n_targets}) trained {t}  "
              f"(train R2 = {est.score(X, Y[:, j - 1]):.4f})")
    return models


def evaluate_by_model(models, X, Y, groups, targets, method="hgb", seed=0):
    """Hold out entire models (not random rows) - the right test for a surrogate."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X, Y, groups))
    for j, t in enumerate(targets):
        est = make_estimator(method, seed)
        est.fit(X[tr], Y[tr, j])
        print(f"{t}: held-out-model R2 = {est.score(X[te], Y[te, j]):.4f}")


def learning_curve(X, Y, groups, targets, method="hgb", seed=0,
                   train_sizes=(10, 20, 30, 40, 50, 60, 70, 80), n_repeats=5,
                   out_csv="learning_curve_normalized.csv",
                   out_png="learning_curve_normalized.png"):
    """Grow the number of TRAIN models and record held-out-model R2 per target.

    Averaged over `n_repeats` independent draws per size: each repeat draws a fresh
    ~20% held-out test set and fresh random train subsets, so a single unlucky draw
    no longer dominates a point (the earlier single-draw curve was very noisy, e.g.
    U1 swinging to -1.4 at 40 models). Reports mean +/- std across repeats.

    Answers "will more DOE samples help?":
      - mean held-out R2 still rising at the largest size -> generate more models.
      - mean held-out R2 plateaued                        -> representation-limited.
    """
    n_models = len(np.unique(groups))

    # Per-repeat: fresh test split, then grow train subsets drawn from the remainder.
    # acc[size][target] collects the R2 from every repeat for later mean/std.
    acc = {}
    max_train_seen = 0
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + r)
        tr_idx, te_idx = next(gss.split(X, Y, groups))
        train_models = np.unique(groups[tr_idx])
        test_mask = np.isin(groups, np.unique(groups[te_idx]))
        Xte, Yte = X[test_mask], Y[test_mask]
        max_train = len(train_models)
        max_train_seen = max(max_train_seen, max_train)

        sizes = sorted({s for s in train_sizes if s <= max_train} | {max_train})
        for k in sizes:
            chosen = rng.choice(train_models, size=k, replace=False)
            train_mask = np.isin(groups, chosen)
            Xtr, Ytr = X[train_mask], Y[train_mask]
            for j, t in enumerate(targets):
                est = make_estimator(method, seed + r)
                est.fit(Xtr, Ytr[:, j])
                score = r2_score(Yte[:, j], est.predict(Xte))
                acc.setdefault(k, {}).setdefault(t, []).append(score)
        print(f"  repeat {r + 1}/{n_repeats} done (train pool {max_train}, "
              f"sizes {sizes[0]}..{sizes[-1]})")

    print(f"learning curve: {n_models} models total, {n_repeats} repeats per size")
    rows = []
    for k in sorted(acc):
        row = {"n_train_models": k, "n_repeats": len(acc[k][targets[0]])}
        for t in targets:
            vals = np.array(acc[k][t], float)
            row[f"{t}_mean"] = vals.mean()
            row[f"{t}_std"] = vals.std()
        rows.append(row)
        summ = "  ".join(f"{t}={row[f'{t}_mean']:+.3f}+/-{row[f'{t}_std']:.3f}" for t in targets)
        print(f"  train={k:3d} models: {summ}")

    lc = pd.DataFrame(rows)
    lc.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")

    plot_learning_curve(lc, targets, out_png, n_repeats=n_repeats)
    return lc


def plot_learning_curve(lc, targets, out_png, n_repeats=None):
    """Small-multiples learning-curve plot: one panel per target.

    Four series on one shared axis are hard to tell apart and get vertically
    crammed (a single catastrophic small-N point drags the y-range for everyone).
    Faceting gives each target its own vertical scale, and each panel's y-limits
    are driven by the MEAN values (not the huge small-N uncertainty bands), so the
    slope -- the thing a learning curve is read for -- is legible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional, never fail the run over it
        print(f"(skipped plot: {e})")
        return

    # Okabe-Ito colorblind-safe hues, assigned in fixed order (one per panel).
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    x = lc["n_train_models"].to_numpy()
    ncol = 2
    nrow = (len(targets) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 3.1 * nrow),
                             sharex=True, squeeze=False)
    axes = axes.ravel()

    for i, t in enumerate(targets):
        ax = axes[i]
        m = lc[f"{t}_mean"].to_numpy()
        s = lc[f"{t}_std"].to_numpy()
        c = colors[i % len(colors)]
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, linewidth=0)
        ax.plot(x, m, marker="o", ms=6, lw=2, color=c)
        ax.annotate(f"{m[-1]:.2f}", (x[-1], m[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color=c)
        # Scale to the means (+small pad); clip so a huge small-N band can't crush it.
        lo = max(-0.6, float(np.nanmin(m)) - 0.10)
        hi = min(1.0, float(np.nanmax(m)) + 0.12)
        ax.set_ylim(lo, hi)
        if lo < 0.0 < hi:
            ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
        ax.set_title(t, fontsize=11)
        ax.set_ylabel("held-out R²")
        ax.grid(True, alpha=0.25)

    for j in range(len(targets), len(axes)):
        axes[j].set_visible(False)
    for ax in axes[len(targets) - ncol:len(targets)]:
        ax.set_xlabel("number of training models")

    sup = "Learning curve (normalized coords"
    if n_repeats:
        sup += f", {n_repeats} repeats, mean ± std"
    fig.suptitle(sup + ")", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    print(f"saved -> {out_png}")


def predict_field(models, param_values, coords, targets):
    """param_values: dict {param: value}; coords: (n, 3) array of query XYZ (RAW).

    Coordinates are normalized here to the query cloud's own bounding box, matching
    the per-model normalization used in training. Returns a DataFrame with the raw
    X, Y, Z plus one column per target.
    """
    coords = np.asarray(coords, float)
    coords_n = np.empty_like(coords)
    for a in range(3):
        lo, hi = coords[:, a].min(), coords[:, a].max()
        span = hi - lo
        coords_n[:, a] = 0.0 if span == 0.0 else (coords[:, a] - lo) / span
    p = np.array([param_values[k] for k in PARAM_NAMES], float)
    Xq = np.hstack([np.tile(p, (len(coords), 1)), coords_n])
    out = {t: models[t].predict(Xq) for t in targets}
    return pd.DataFrame({**{c: coords[:, i] for i, c in enumerate(COORD_NAMES)}, **out})


if __name__ == "__main__":
    RESULTS_DIR = "../output"  # contains model_<i>/model_<i>_results.csv subfolders
    PARAMS_CSV = "../model_params.csv"

    X, Y, groups, targets = build_pooled_dataset(RESULTS_DIR, PARAMS_CSV)
    print(f"pooled dataset: {X.shape[0]} rows, {X.shape[1]} features, targets={targets}")

    models = train(X, Y, targets, method="hgb")
    if len(np.unique(groups)) >= 5:
        print("--- held-out-model evaluation (normalized) ---")
        evaluate_by_model(models, X, Y, groups, targets)
        print("--- learning curve (normalized) ---")
        learning_curve(X, Y, groups, targets)

    joblib.dump({"models": models, "targets": targets, "normalized": True},
                "pooled_surrogate_normalized.joblib")
    print("saved -> pooled_surrogate_normalized.joblib")
