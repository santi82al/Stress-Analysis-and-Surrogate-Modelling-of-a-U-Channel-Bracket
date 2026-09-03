"""
geom_check.py
-----------------------------------------------------------------------------
Single source of truth for checking that a FEM results CSV's node-coordinate
bounding box matches the geometry implied by its parameter set.

Stdlib-only (csv) on purpose, so it is importable from BOTH:
  - Abaqus Python  (run_models.py, where numpy/pandas are unavailable)
  - CPython        (Surrogate/pooled_surrogate.py)

Bounding-box formulas mirror geometry/CAE_models_builder.build_model_4walls:
    base box  X[0, W_geo]  Y[0, Y_geo]  Z[0, f_geo]
    arm extends in Z to Z_geo
  where
    W_geo = b_geo + 2*t_geo
    Y_geo = 4*m_geo - 2*n_geo + 4*e_geo
    Z_geo = 2*m_geo

NOTE: the bounding box depends only on b, m, n, t, e. P_load and wall_thk leave
no footprint in the bracket node coordinates, so a match here verifies geometry
only -- not the load magnitude or the clamping-wall thickness.
-----------------------------------------------------------------------------
"""

import csv

AXES = ("X", "Y", "Z")


def expected_bbox(params):
    """Bounding box implied by the parametric geometry.

    params: dict with at least b_geo, m_geo, n_geo, t_geo, e_geo (str or float).
    Returns {axis: (lo, hi)}.
    """
    b = float(params["b_geo"])
    m = float(params["m_geo"])
    n = float(params["n_geo"])
    t = float(params["t_geo"])
    e = float(params["e_geo"])
    w_geo = b + 2.0 * t
    y_geo = 4.0 * m - 2.0 * n + 4.0 * e
    z_geo = 2.0 * m
    return {"X": (0.0, w_geo), "Y": (0.0, y_geo), "Z": (0.0, z_geo)}


def read_csv_bounds(csv_path):
    """Return {axis: (min, max)} of the X/Y/Z columns in a results CSV.

    Returns None if the file is missing, empty, or lacks X/Y/Z columns.
    """
    try:
        f = open(csv_path, "r")
    except (IOError, OSError):
        return None
    try:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not all(a in reader.fieldnames for a in AXES):
            return None
        lo = {a: None for a in AXES}
        hi = {a: None for a in AXES}
        n_rows = 0
        for row in reader:
            for a in AXES:
                try:
                    v = float(row[a])
                except (TypeError, ValueError):
                    continue
                if lo[a] is None or v < lo[a]:
                    lo[a] = v
                if hi[a] is None or v > hi[a]:
                    hi[a] = v
            n_rows += 1
        if n_rows == 0 or any(lo[a] is None for a in AXES):
            return None
        return dict((a, (lo[a], hi[a])) for a in AXES)
    finally:
        f.close()


def bbox_matches(params, actual_bounds, tol=0.02):
    """Compare expected bbox (from params) against actual node bounds.

    tol is a relative tolerance per axis (fraction of the expected span).
    Returns (ok, mismatches) where mismatches is a list of
    (axis, (exp_lo, exp_hi), (act_lo, act_hi)) for each failing axis.
    """
    expected = expected_bbox(params)
    mismatches = []
    for axis in AXES:
        exp_lo, exp_hi = expected[axis]
        act_lo, act_hi = actual_bounds[axis]
        span = max(exp_hi - exp_lo, 1e-9)
        if abs(act_lo - exp_lo) / span > tol or abs(act_hi - exp_hi) / span > tol:
            mismatches.append((axis, (exp_lo, exp_hi), (act_lo, act_hi)))
    return (len(mismatches) == 0, mismatches)


def check_results_file(params, csv_path, tol=0.02):
    """Read a results CSV and compare its coordinate bounds to a parameter set.

    Returns (status, mismatches):
      status = "match"     -> geometry matches params
             = "mismatch"  -> geometry present but differs (mismatches populated)
             = "missing"   -> file absent / unreadable / no coordinate rows
    """
    bounds = read_csv_bounds(csv_path)
    if bounds is None:
        return ("missing", [])
    ok, mismatches = bbox_matches(params, bounds, tol=tol)
    return ("match" if ok else "mismatch", mismatches)
