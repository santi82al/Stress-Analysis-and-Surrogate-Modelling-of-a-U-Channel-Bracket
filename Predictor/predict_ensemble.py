"""
Predict a full field for a NEW geometry with the ENSEMBLE surrogate
(0.5*MLP + 0.5*HGB) - the best model of the algorithm comparison.

Same CLI as predict_new_geometry_normalized.py. The geometry is read from a
params CSV in model_params.csv layout (rows = parameters, one column per model):

1) New geometry, point cloud generated from the parameters alone (no Abaqus):

    python predict_ensemble.py --params-from test_params.csv --model-col test_1 \
        --out test_predicted_field.csv

2) Reuse an existing results CSV's mesh nodes:

    python predict_ensemble.py --params-from test_params.csv --model-col test_1 \
        --coords-from ../output/model_1/model_1_results.csv

3) Compare against a solved FEM model (surrogate vs truth):

    python predict_ensemble.py --params-from ../model_params.csv --model-col model_7 \
        --coords-from ../output/model_7/model_7_results.csv --compare

Output: X, Y, Z + one column per predicted target (U1, U2, U3, S_Mises).

POST-PROCESSING: S_Mises is clipped at 0 before the CSV is written (see
predictor_common.clip_nonnegative) - the output is not raw regressor output.
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from predictor_common import resolve_bundle, clip_nonnegative  # puts Surrogate/ on sys.path
from pooled_surrogate_normalized import PARAM_NAMES, COORD_NAMES
from predict_new_geometry_normalized import (parse_params_string, params_from_csv,
                                             coords_from_csv, generate_query_points)

BUNDLE = "ENSEMBLE_mlp_hgb.joblib"


def predict_field(bundle_path, param_values, coords):
    """coords: (n, 3) raw XYZ. Returns DataFrame with XYZ + one column/target."""
    b = joblib.load(resolve_bundle(bundle_path))
    nn = joblib.load(resolve_bundle(b["nn_bundle_file"]))
    targets, w = b["targets"], b["weight_mlp"]
    coords = np.asarray(coords, float)
    coords_n = np.empty_like(coords)
    for a in range(3):
        lo, hi = coords[:, a].min(), coords[:, a].max()
        span = hi - lo
        coords_n[:, a] = 0.0 if span == 0.0 else (coords[:, a] - lo) / span
    p = np.array([param_values[k] for k in PARAM_NAMES], float)
    Xq = np.hstack([np.tile(p, (len(coords), 1)), coords_n])
    Y_mlp = nn["yscaler"].inverse_transform(
        nn["net"].predict(nn["xscaler"].transform(Xq)))
    Y_hgb = np.column_stack([b["hgb"][t].predict(Xq) for t in targets])
    Yq = w * Y_mlp + (1 - w) * Y_hgb
    out = pd.DataFrame({c: coords[:, i] for i, c in enumerate(COORD_NAMES)})
    for j, t in enumerate(targets):
        out[t] = Yq[:, j]
    return clip_nonnegative(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--params", help='inline: "b_geo=55,m_geo=45,..."')
    ap.add_argument("--params-from", help="params CSV in model_params.csv layout")
    ap.add_argument("--model-col", help="column name in the params csv, e.g. test_1")
    ap.add_argument("--coords-from", help="results CSV whose X,Y,Z to reuse")
    ap.add_argument("--spacing", type=float, default=3.0,
                    help="grid spacing [mm] for generated point cloud")
    ap.add_argument("--compare", action="store_true",
                    help="if --coords-from has true fields, print error metrics")
    ap.add_argument("--out", default="ENSEMBLE_predicted_field.csv")
    args = ap.parse_args()

    if args.params:
        params = parse_params_string(args.params)
    elif args.params_from and args.model_col:
        params = params_from_csv(args.params_from, args.model_col)
    else:
        raise SystemExit("give --params or (--params-from + --model-col)")

    print("geometry: " + "  ".join("%s=%g" % (k, params[k]) for k in PARAM_NAMES))

    truth = None
    if args.coords_from:
        coords, truth = coords_from_csv(args.coords_from)
        print("query points: %d (from %s)" % (len(coords), args.coords_from))
    else:
        coords = generate_query_points(params, spacing=args.spacing)
        print("query points: %d (generated, spacing=%.1f mm)"
              % (len(coords), args.spacing))

    df = predict_field(args.bundle, params, coords)
    df.to_csv(args.out, index=False)
    print("wrote %s  (columns: %s)" % (args.out, ", ".join(df.columns)))

    for t in [c for c in df.columns if c not in COORD_NAMES]:
        line = "%-8s min=%+.4e  max=%+.4e" % (t, df[t].min(), df[t].max())
        if args.compare and truth is not None and t in truth.columns:
            yt = truth[t].to_numpy(float)
            err = df[t].to_numpy() - yt
            ss_res = float((err ** 2).sum())
            ss_tot = float(((yt - yt.mean()) ** 2).sum()) or 1e-30
            line += "   | vs FEM: RMSE=%.3e  R2=%.3f" % (
                np.sqrt((err ** 2).mean()), 1.0 - ss_res / ss_tot)
        print(line)


if __name__ == "__main__":
    main()
