"""
run_models.py
-----------------------------------------------------------------------------
For each model column in 'model_params.csv':
    0. skip check              - if output/model_N/model_N_results.csv already
                                  exists AND its geometry matches the current
                                  params (geom_check.check_results_file), skip
                                  the model. A mismatch regenerates it. This
                                  makes extending the DOE additive: only new /
                                  changed models are re-simulated.
    1. build_model_4walls()    - builds geometry, meshes, submits job,
                                  writes <model>_results.csv
    2. _move_model_files()     - always runs (finally), moves all Abaqus
                                  output files to output/model_N/
    3. analytical_validation() - computes analytical stress, reads that CSV,
                                  compares, appends to validation_summary.csv
    4. _cleanup_session_files()- deletes leftover Abaqus session files
-----------------------------------------------------------------------------
"""

import csv
import os
import shutil
import sys
import traceback
from pathlib import Path

# Abaqus' Python block-buffers stdout, so progress/status prints don't reach the
# terminal until the very end (and can be lost entirely if the job aborts).
# Force line-buffered output so messages stream live as each model runs.
try:
    sys.stdout.reconfigure(line_buffering=True)   # Python 3.7+ (Abaqus 2024 = 3.10)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

from geometry.CAE_models_builder import build_model_4walls
from geometry.validation import analytical_validation
from geom_check import check_results_file

PARAMS = ["b_geo", "m_geo", "n_geo", "t_geo", "f_geo", "e_geo", "i_geo", "P_load", "wall_thk"]

# Always work from the folder that contains this script
SCRIPT_DIR = Path(os.getcwd())
os.chdir(SCRIPT_DIR)

# Output base is overridable so a convergence run can keep its FEM results in a
# separate tree (e.g. Convergence/output) without touching the DOE's output/.
OUTPUT_BASE = SCRIPT_DIR / os.environ.get("MODEL_OUTPUT_DIR", "output")

print("Working directory: %s" % SCRIPT_DIR)
print("Output base: %s" % OUTPUT_BASE)
print("Validation summary: %s" % (SCRIPT_DIR / "validation_summary.csv"))


# -- Abaqus session-level files written regardless of model name ---------------
ABAQUS_SESSION_FILES = [
    "abaqus.rpy", "abaqus.rec", "abaqus.log", "abaqus_acis.log",
    "abaqus.odb", "odb_error.txt",
]


def _first_word(s):
    return s.strip().split()[0]


def _move_model_files(model_name):
    """
    Move every file in SCRIPT_DIR whose name starts with model_name
    into output/model_name/. Always called (via finally) even on failure,
    so partial Abaqus files don't accumulate in the root folder.
    Returns the destination folder path.
    """
    dest = OUTPUT_BASE / model_name
    dest.mkdir(parents=True, exist_ok=True)

    moved = []
    for f in SCRIPT_DIR.iterdir():
        if f.is_file() and f.name.startswith(model_name):
            shutil.move(str(f), str(dest / f.name))
            moved.append(f.name)

    if moved:
        print("[%s] Moved %d file(s) -> output/%s/" % (model_name, len(moved), model_name))
    return dest


def _cleanup_session_files():
    """
    Try to delete Abaqus session-level files left in the root directory.
    Skip any files that are still open (e.g., abaqus.rpy held by Abaqus).
    These are not model-specific so they aren't caught by _move_model_files.
    """
    deleted = []
    skipped = []
    for name in ABAQUS_SESSION_FILES:
        f = SCRIPT_DIR / name
        if f.exists():
            try:
                f.unlink()
                deleted.append(name)
            except (PermissionError, OSError):
                skipped.append(name)
    if deleted:
        print("Deleted session files: %s" % ", ".join(deleted))
    if skipped:
        print("Skipped (in use): %s" % ", ".join(skipped))


def load_and_run(csv_path="model_params.csv"):
    csv_path = SCRIPT_DIR / csv_path
    print("Looking for CSV at: %s" % csv_path)

    if not csv_path.exists():
        sys.exit("ERROR: %s not found." % csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    rows = {}
    for row in raw_rows:
        key = _first_word(row["Parameter"])
        rows[key] = {k.strip(): v.strip() for k, v in row.items()}

    missing = [p for p in PARAMS if p not in rows]
    if missing:
        print("Available parameter keys: %s" % sorted(rows.keys()))
        sys.exit("ERROR: missing rows in CSV: %s" % missing)

    model_cols = [k for k in rows[PARAMS[0]].keys() if k.startswith("model_")]
    if not model_cols:
        sys.exit("ERROR: no model_* columns found.")

    total = len(model_cols)
    results = {}
    for idx, col in enumerate(model_cols, start=1):
        try:
            model_id = int(col.split("_", 1)[1])
        except (IndexError, ValueError):
            model_id = col

        model_name = "model_%s" % model_id
        pct = 100.0 * (idx - 1) / total
        print("\n%s" % ("=" * 60))
        print("[%d/%d] (%.0f%%) Building and submitting %s ..." % (idx, total, pct, model_name))
        print("%s" % ("=" * 60))

        # Parse this model's parameters up front (needed for the skip check).
        try:
            kwargs = {p: float(rows[p][col]) for p in PARAMS}
        except (KeyError, ValueError) as e:
            print("[%s] PARAM PARSE FAILED - %s" % (model_name, e))
            results[model_id] = False
            continue

        # Optional per-model mesh divisor for convergence studies. Absent row or
        # blank cell -> builder default (24). Kept in a SEPARATE dict (mesh_kwargs)
        # rather than kwargs: kwargs is the geometry/load param set shared with the
        # skip check and analytical_validation(), neither of which accepts mesh_div.
        # Only build_model_4walls() takes it.
        mesh_kwargs = {}
        if "mesh_div" in rows:
            cell = rows["mesh_div"].get(col, "").strip()
            if cell:
                try:
                    mesh_kwargs["mesh_div"] = float(cell)
                except ValueError:
                    print("[%s] bad mesh_div '%s' - using default" % (model_name, cell))

        # Skip models whose existing output already matches these params.
        # The match function must confirm the existing results correspond to the
        # current params before we trust (and skip) them; a mismatch regenerates.
        results_csv = OUTPUT_BASE / model_name / ("%s_results.csv" % model_name)
        status, mismatches = check_results_file(kwargs, str(results_csv))
        if status == "match":
            print("[%s] SKIP - existing results match current params." % model_name)
            results[model_id] = True
            print("[%d/%d] (%.0f%%) %s finished (skipped)."
                  % (idx, total, 100.0 * idx / total, model_name))
            continue
        if status == "mismatch":
            print("[%s] Existing results do NOT match current params - regenerating:" % model_name)
            for axis, (elo, ehi), (alo, ahi) in mismatches:
                print("    axis %s: expected [%.2f, %.2f], got [%.2f, %.2f]"
                      % (axis, elo, ehi, alo, ahi))

        fem_ok = False
        model_dir = None
        try:
            # 1. FEM - build, solve, write <model>_results.csv
            build_model_4walls(MODEL_NAME=model_name, **kwargs, **mesh_kwargs)
            fem_ok = True

        except Exception as e:
            print("[%s] FEM FAILED - %s" % (model_name, e))
            traceback.print_exc()

        finally:
            # 2. Always move files - keeps root clean even after failures
            model_dir = _move_model_files(model_name)

        if fem_ok and model_dir:
            try:
                # 3. Analytical validation
                print("[%s] Running analytical validation ..." % model_name)
                analytical_validation(
                    MODEL_NAME=model_name,
                    results_dir=str(model_dir),
                    summary_csv=str(SCRIPT_DIR / "validation_summary.csv"),
                    **kwargs
                )
                results[model_id] = True
                print("[%s] done" % model_name)
            except Exception as e:
                print("[%s] Validation FAILED - %s" % (model_name, e))
                traceback.print_exc()
                results[model_id] = False
        else:
            results[model_id] = False

        print("[%d/%d] (%.0f%%) %s finished." % (idx, total, 100.0 * idx / total, model_name))

    # 4. Clean up session-level Abaqus files from root
    _cleanup_session_files()

    print("\n%s" % ("-" * 60))
    ok = len([v for v in results.values() if v])
    val_file = SCRIPT_DIR / "validation_summary.csv"
    print("Finished: %s/%s models built." % (ok, len(results)))
    if val_file.exists():
        print("Validation summary: %s" % val_file)
    else:
        print("WARNING: validation_summary.csv was not created.")


# call directly - sys.argv is polluted by Abaqus command-line args.
# CSV is selectable via env var so a convergence run can use a separate file
# without touching the DOE:  $env:MODEL_PARAMS_CSV="convergence_params.csv"
if True:
    load_and_run(os.environ.get("MODEL_PARAMS_CSV", "model_params.csv"))