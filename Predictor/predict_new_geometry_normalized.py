"""
Query the NORMALIZED pooled surrogate with a NEW geometry.

Loads pooled_surrogate_normalized.joblib and uses the normalized predict_field
(coordinates are scaled to the query cloud's own bounding box, matching the
per-model normalization used in training). This module also hosts the shared
prediction helpers (param parsing, query-point generation) used by the two-stage
predictor.

Usage examples
--------------
1) Reuse an existing results CSV's mesh nodes:

    python predict_new_geometry_normalized.py --coords-from ../output/model_0/model_0_results.csv \
        --params "b_geo=55,m_geo=48,n_geo=52,t_geo=6,f_geo=5,e_geo=4,i_geo=6,P_load=3500,wall_thk=9"

2) Auto-generated point cloud from parameters alone (no Abaqus):

    python predict_new_geometry_normalized.py \
        --params "b_geo=55,m_geo=48,n_geo=52,t_geo=6,f_geo=5,e_geo=4,i_geo=6,P_load=3500,wall_thk=9"

3) Compare against a solved FEM model (surrogate vs truth):

    python predict_new_geometry_normalized.py --coords-from ../output/model_7/model_7_results.csv \
        --params-from ../model_params.csv --model-col model_7 --compare

Output: predicted_field_normalized.csv with X, Y, Z + one column per predicted target.

POST-PROCESSING: S_Mises is clipped at 0 before the CSV is written (see
predictor_common.clip_nonnegative) - the output is not raw regressor output.
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from predictor_common import resolve_bundle, clip_nonnegative  # puts Surrogate/ on sys.path
from pooled_surrogate_normalized import PARAM_NAMES, COORD_NAMES, predict_field

SURROGATE_FILE = "pooled_surrogate_normalized.joblib"


# ---------------------------------------------------------------------------
# Shared prediction helpers (moved here from the retired raw predictor so the
# normalized + two-stage predictors carry no dependency on non-normalized code).
# ---------------------------------------------------------------------------
def parse_params_string(s):
    """'b_geo=55,m_geo=48,...' -> dict of floats. Validates completeness."""
    vals = {}
    for item in s.split(","):
        k, v = item.split("=")
        vals[k.strip()] = float(v)
    missing = [p for p in PARAM_NAMES if p not in vals]
    if missing:
        raise SystemExit("missing parameters: %s" % ", ".join(missing))
    return vals


def params_from_csv(params_csv, model_col):
    """Read one model's column from model_params.csv."""
    df = pd.read_csv(params_csv, index_col="Parameter")
    df = df.drop(index="Description", errors="ignore")
    col = df[model_col].astype(str).str.strip().astype(float)
    return {p: col[p] for p in PARAM_NAMES}


def coords_from_csv(path):
    """Reuse the X,Y,Z node cloud of an existing results CSV."""
    df = pd.read_csv(path)
    return df[COORD_NAMES].to_numpy(float), df


def generate_query_points(p, spacing=3.0):
    """Structured point cloud on the U-channel solid, derived the same way
    the CAE builder derives its dimensions (see CAE_models_builder.py):

        Y_geo = 4*m - 2*n + 4*e     (back wall height)
        W_geo = b + 2*t             (outer width)
        Z_geo = 2*m                 (arm depth -- NOT n_geo)

        back wall : X:[0,W]            Y:[0,Y]  Z:[0,f]
        side walls: X:[0,t], [W-t,W]   Y:[0,Y]  Z:[f,Z]
        bottom pad: X:[0,W]            Y:[0,e]  Z:[f,Z]

    A triangular pocket is then removed from the arm, matching the builder's
    CutExtrude: vertices (Z=2f, Y=Y_geo), (Z=Z_geo, Y=Y_geo), (Z=Z_geo, Y=2e).

    Verified against solved FEM meshes: 98.8-99.4% of real nodes fall inside
    this solid (the remainder sit exactly on the cut face).

    Points are volumetric (a few layers through each thickness), which is
    what the pooled model was trained on. Bolt bores are NOT carved out.
    """
    b, m, n, t, f, e = (p["b_geo"], p["m_geo"], p["n_geo"],
                        p["t_geo"], p["f_geo"], p["e_geo"])
    Y = 4*m - 2*n + 4*e
    W = b + 2*t
    Z = 2*m
    if Y <= 0:
        raise SystemExit("invalid geometry: Y_geo = %.1f <= 0" % Y)
    if Z <= 2*f:
        raise SystemExit("invalid geometry: Z_geo = %.1f <= 2*f_geo = %.1f" % (Z, 2*f))

    def grid(x0, x1, y0, y1, z0, z1):
        nx = max(2, int(round((x1-x0)/spacing)) + 1)
        ny = max(2, int(round((y1-y0)/spacing)) + 1)
        nz = max(2, int(round((z1-z0)/spacing)) + 1)
        xs, ys, zs = np.meshgrid(np.linspace(x0, x1, nx),
                                 np.linspace(y0, y1, ny),
                                 np.linspace(z0, z1, nz), indexing="ij")
        return np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    parts = [
        grid(0.0,   W,   0.0, Y, 0.0, f),   # back wall
        grid(0.0,   t,   0.0, Y, f,   Z),   # left side wall
        grid(W - t, W,   0.0, Y, f,   Z),   # right side wall
        grid(0.0,   W,   0.0, e, f,   Z),   # bottom pad (closes the U)
    ]
    pts = np.vstack(parts)
    # remove duplicated points on the shared planes (Z = f, and pad/wall joins)
    pts = np.unique(np.round(pts, 6), axis=0)

    # Triangular pocket cut from the arm: everything above the hypotenuse
    # running from (Z=2f, Y=Y_geo) down to (Z=Z_geo, Y=2e) is removed.
    s = np.clip((pts[:, 2] - 2*f) / (Z - 2*f), 0.0, 1.0)
    cut = (pts[:, 2] > 2*f) & (pts[:, 1] > Y + (2*e - Y) * s)
    return pts[~cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default=SURROGATE_FILE)
    ap.add_argument("--params", help='inline: "b_geo=55,m_geo=48,..."')
    ap.add_argument("--params-from", help="model_params.csv path")
    ap.add_argument("--model-col", help="column name in params csv, e.g. model_7")
    ap.add_argument("--coords-from", help="results CSV whose X,Y,Z to reuse")
    ap.add_argument("--spacing", type=float, default=3.0,
                    help="grid spacing [mm] for generated point cloud")
    ap.add_argument("--compare", action="store_true",
                    help="if --coords-from has true fields, print error metrics")
    ap.add_argument("--out", default="predicted_field_normalized.csv")
    args = ap.parse_args()

    if args.params:
        params = parse_params_string(args.params)
    elif args.params_from and args.model_col:
        params = params_from_csv(args.params_from, args.model_col)
    else:
        raise SystemExit("give --params or (--params-from + --model-col)")

    truth = None
    if args.coords_from:
        coords, truth = coords_from_csv(args.coords_from)
        print("query points: %d (from %s)" % (len(coords), args.coords_from))
    else:
        coords = generate_query_points(params, spacing=args.spacing)
        print("query points: %d (generated, spacing=%.1f mm)"
              % (len(coords), args.spacing))

    saved = joblib.load(resolve_bundle(args.surrogate))
    if not saved.get("normalized"):
        print("WARNING: loaded surrogate is not flagged normalized; "
              "predictions may be inconsistent.")
    models, targets = saved["models"], saved["targets"]
    df = clip_nonnegative(predict_field(models, params, coords, targets))
    df.to_csv(args.out, index=False)
    print("wrote %s  (columns: %s)" % (args.out, ", ".join(df.columns)))

    for tcol in targets:
        line = "%-8s min=%+.4e  max=%+.4e" % (tcol, df[tcol].min(), df[tcol].max())
        if args.compare and truth is not None and tcol in truth.columns:
            err = df[tcol].to_numpy() - truth[tcol].to_numpy()
            denom = max(abs(truth[tcol]).max(), 1e-30)
            line += "   | vs FEM: RMSE=%.3e  max|err|=%.3e  rel=%.2f%%" % (
                np.sqrt((err**2).mean()), abs(err).max(),
                100.0 * abs(err).max() / denom)
        print(line)


if __name__ == "__main__":
    main()
