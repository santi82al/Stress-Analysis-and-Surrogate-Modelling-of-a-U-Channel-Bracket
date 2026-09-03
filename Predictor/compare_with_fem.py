"""
Side-by-side comparison of the surrogate against a dedicated FE solve.

Unlike every earlier check in this project, the FE model here is solved for a
geometry that is NOT part of the DOE, so it is a genuinely independent
confirmation point rather than an in-sample illustration.

The surrogate is queried on the FE model's own node cloud (--coords-from), so the
two fields are compared node for node with no interpolation.

Usage:
    python compare_with_fem.py <fem_results.csv> <params_csv> <model_col> [out_prefix]

Writes:
    <prefix>_pred.csv     surrogate prediction on the FE nodes
    <prefix>_table.csv    the summary table (also printed, and as LaTeX)

Stress is summarised by the 95th percentile as well as the peak: the peak sits on
a singular node at the bore edge and is mesh-dependent, so it is reported but not
used as the headline agreement metric (see the convergence study).
"""

import sys

import numpy as np
import pandas as pd

from predictor_common import resolve_bundle  # puts Surrogate/ on sys.path
from predict_ensemble import predict_field, BUNDLE
from predict_new_geometry_normalized import params_from_csv, coords_from_csv
from pooled_surrogate_normalized import COORD_NAMES


def metrics(true, pred):
    err = pred - true
    ss_tot = float(((true - true.mean()) ** 2).sum()) or 1e-30
    return {
        "fem_min": true.min(), "sur_min": pred.min(),
        "fem_max": true.max(), "sur_max": pred.max(),
        "fem_p95": np.percentile(true, 95), "sur_p95": np.percentile(pred, 95),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": 1.0 - float((err ** 2).sum()) / ss_tot,
    }


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    fem_csv, params_csv, model_col = sys.argv[1:4]
    prefix = sys.argv[4] if len(sys.argv) > 4 else "fem_vs_surrogate"

    params = params_from_csv(params_csv, model_col)
    coords, fem = coords_from_csv(fem_csv)
    print("FE model: %s   %d nodes" % (model_col, len(coords)))
    print("geometry: " + "  ".join("%s=%g" % (k, v) for k, v in params.items()))

    pred = predict_field(BUNDLE, params, coords)
    pred.to_csv(prefix + "_pred.csv", index=False)

    targets = [c for c in pred.columns if c not in COORD_NAMES and c in fem.columns]
    rows = []
    for t in targets:
        m = metrics(fem[t].to_numpy(float), pred[t].to_numpy(float))
        m["target"] = t
        rows.append(m)
    table = pd.DataFrame(rows)[
        ["target", "fem_min", "sur_min", "fem_max", "sur_max",
         "fem_p95", "sur_p95", "rmse", "mae", "r2"]]
    table.to_csv(prefix + "_table.csv", index=False)

    print("\n%-9s %11s %11s   %11s %11s   %9s %7s"
          % ("target", "FEM max", "surrogate", "FEM p95", "surrogate", "RMSE", "R2"))
    for _, r in table.iterrows():
        print("%-9s %11.4g %11.4g   %11.4g %11.4g   %9.4g %7.3f"
              % (r.target, r.fem_max, r.sur_max, r.fem_p95, r.sur_p95, r.rmse, r.r2))

    print("\n%% LaTeX body for the comparison table")
    for _, r in table.iterrows():
        name = r.target.replace("_", r"\_")
        print(r"  %s & %.4g & %.4g & %.4g & %.4g & %.4g & %.3f \\"
              % (name, r.fem_p95, r.sur_p95, r.fem_max, r.sur_max, r.rmse, r.r2))
    print("\nwrote %s_pred.csv, %s_table.csv" % (prefix, prefix))


if __name__ == "__main__":
    main()
