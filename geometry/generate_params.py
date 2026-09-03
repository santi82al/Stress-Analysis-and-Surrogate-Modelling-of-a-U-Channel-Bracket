"""
generate_model_params.py
-----------------------------------------------------------------------------
Generates or EXTENDS model_params.csv with parameter sets sampled via Latin
Hypercube Sampling (LHS) between per-parameter bounds.

model_0 is always kept fixed as the base/reference case; model_1..model_N
are the sampled DOE points.

Two modes:
    # Overwrite from scratch: model_0 + N fresh samples (edit N_SAMPLES below).
    python geometry/generate_params.py
    python geometry/generate_params.py --fresh 10

    # Extend (non-destructive): keep every existing column, APPEND N new
    # feasible samples as model_<max+1> ... so already-run simulations stay
    # valid and are treated as extras, not replacements.
    python geometry/generate_params.py --extend 5

After extending, `abaqus cae noGUI=run_models.py` only builds the new models:
it skips every model whose existing output already matches its params.
-----------------------------------------------------------------------------
"""

import argparse
import csv
import random
from pathlib import Path

# -----------------------------------------------------------------------
# Settings - edit these
# -----------------------------------------------------------------------

# Number of sampled models to generate (model_1 .. model_N).
N_SAMPLES = 10

# (min, max) bounds for each parameter's LHS sampling range.
BOUNDS = {
    "b_geo":    (30.0,  70.0),
    "m_geo":    (30.0,  70.0),
    "n_geo":    (20.0,  60.0),
    "t_geo":    (3.0,   8.0),
    "f_geo":    (3.0,   8.0),
    "e_geo":    (3.0,   8.0),
    "i_geo":    (4.0,   9.0),
    "P_load":   (1000.0, 6000.0),
    "wall_thk": (6.0,   15.0),
}

# Reference / base case, kept fixed as model_0.
BASE_CASE = {
    "b_geo": 50, "m_geo": 50, "n_geo": 50, "t_geo": 5, "f_geo": 5,
    "e_geo": 5, "i_geo": 6, "P_load": 3000, "wall_thk": 10,
}
BASE_DESCRIPTION = "Base (Reference)"

PARAMS = ["b_geo", "m_geo", "n_geo", "t_geo", "f_geo", "e_geo", "i_geo", "P_load", "wall_thk"]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "model_params.csv"

MAX_RESAMPLE_ATTEMPTS = 200


# -----------------------------------------------------------------------
# Feasibility constraints (mirrors CAE_models_builder.build_model_4walls,
# plus bolt-bore clearance constraints)
# -----------------------------------------------------------------------

def is_feasible(sample):
    b = sample["b_geo"]
    m = sample["m_geo"]
    n = sample["n_geo"]
    f = sample["f_geo"]
    e = sample["e_geo"]
    i = sample["i_geo"]

    y_geo = 4 * m - 2 * n + 4 * e
    if y_geo <= 0:
        return False

    # Bottom & back-wall bores don't exit the side flanges.
    if not (i + 2 < b / 2):
        return False

    # Bottom bore stays clear of the back wall in Z.
    if not (m > i + f + 2):
        return False

    # Back-wall bore stays above the bottom-wall region.
    if not (2 * m - n + e > i + 2):
        return False

    return True


# -----------------------------------------------------------------------
# Latin Hypercube Sampling
# -----------------------------------------------------------------------

def lhs_unit_samples(n_samples, n_dims, rng):
    """Return an (n_samples, n_dims) array of LHS samples in [0, 1)."""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_dims, seed=rng.randint(0, 2**32 - 1))
        return sampler.random(n=n_samples)
    except ImportError:
        # Pure-Python LHS fallback: stratify each dimension into n_samples
        # bins, draw one random point per bin, then shuffle the bin order.
        samples = [[0.0] * n_dims for _ in range(n_samples)]
        for dim in range(n_dims):
            perm = list(range(n_samples))
            rng.shuffle(perm)
            for row, bin_idx in enumerate(perm):
                samples[row][dim] = (bin_idx + rng.random()) / n_samples
        return samples


def scale_to_bounds(unit_value, bounds):
    lo, hi = bounds
    return lo + unit_value * (hi - lo)


def generate_samples(n_samples, params, bounds, seed=None):
    rng = random.Random(seed)
    accepted = []
    attempts = 0

    while len(accepted) < n_samples and attempts < MAX_RESAMPLE_ATTEMPTS:
        needed = n_samples - len(accepted)
        unit_samples = lhs_unit_samples(needed, len(params), rng)

        for row in unit_samples:
            candidate = {p: scale_to_bounds(row[i], bounds[p]) for i, p in enumerate(params)}
            if is_feasible(candidate):
                accepted.append(candidate)
        attempts += 1

    if len(accepted) < n_samples:
        raise RuntimeError(
            "Could not find %d feasible parameter sets after %d resampling "
            "rounds (only found %d). Widen BOUNDS or check the m_geo/n_geo/"
            "e_geo constraint." % (n_samples, MAX_RESAMPLE_ATTEMPTS, len(accepted))
        )

    return accepted[:n_samples]


# -----------------------------------------------------------------------
# CSV writing
# -----------------------------------------------------------------------

def write_model_params_csv(path, base_case, base_description, samples, params):
    model_names = ["model_0"] + ["model_%d" % (i + 1) for i in range(len(samples))]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter"] + model_names)
        writer.writerow(["Description", base_description] + ["DOE sample %d" % (i + 1) for i in range(len(samples))])

        for p in params:
            row = [p, base_case[p]] + [round(sample[p], 4) for sample in samples]
            writer.writerow(row)


# -----------------------------------------------------------------------
# Extend (append new models without touching existing ones)
# -----------------------------------------------------------------------

def read_existing_csv(path):
    """Parse an existing model_params.csv.

    Returns (model_names, descriptions, data):
        model_names  : ['model_0', 'model_1', ...] in file order
        descriptions : {model_name: description_str}
        data         : {param: {model_name: value_str}}
    Returns None if the file does not exist.
    """
    if not path.exists():
        return None

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        model_names = [h.strip() for h in header[1:]]
        descriptions = {}
        data = {}
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            values = row[1:]
            if key == "Description":
                descriptions = dict(zip(model_names, [v.strip() for v in values]))
            else:
                data[key] = dict(zip(model_names, [v.strip() for v in values]))
    return model_names, descriptions, data


def next_model_index(model_names):
    """Smallest unused model_<i> index (max existing + 1)."""
    max_idx = -1
    for name in model_names:
        if name.startswith("model_"):
            try:
                max_idx = max(max_idx, int(name.split("_", 1)[1]))
            except ValueError:
                pass
    return max_idx + 1


def extend_model_params_csv(path, n_new, params, bounds, seed=None):
    """Append n_new feasible LHS samples to an existing CSV as new columns.

    Every existing column (values and descriptions) is preserved verbatim, so
    simulations already run against them stay valid. New models are numbered
    model_<max+1> ... and continue the 'DOE sample K' description count.
    Returns the list of newly added model names.
    """
    existing = read_existing_csv(path)
    if existing is None:
        raise RuntimeError(
            "%s does not exist -- run a fresh generate first "
            "(no --extend, or --fresh N)." % path)
    model_names, descriptions, data = existing

    missing = [p for p in params if p not in data]
    if missing:
        raise RuntimeError("existing CSV is missing parameter rows: %s" % missing)

    new_samples = generate_samples(n_new, params, bounds, seed=seed)
    start = next_model_index(model_names)
    existing_doe = sum(1 for m in model_names if m != "model_0")

    new_names = []
    for j, sample in enumerate(new_samples):
        name = "model_%d" % (start + j)
        new_names.append(name)
        descriptions[name] = "DOE sample %d" % (existing_doe + j + 1)
        for p in params:
            data[p][name] = round(sample[p], 4)

    all_names = model_names + new_names
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter"] + all_names)
        writer.writerow(["Description"] + [descriptions.get(n, "") for n in all_names])
        for p in params:
            writer.writerow([p] + [data[p][n] for n in all_names])

    return new_names


def main():
    parser = argparse.ArgumentParser(
        description="Generate or extend model_params.csv (LHS DOE).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--extend", type=int, metavar="N",
                       help="append N new feasible LHS models to the existing CSV "
                            "(non-destructive; keeps already-run simulations valid).")
    group.add_argument("--fresh", type=int, metavar="N",
                       help="overwrite the CSV with model_0 + N fresh LHS models.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible sampling.")
    args = parser.parse_args()

    if args.extend is not None:
        new_names = extend_model_params_csv(OUTPUT_PATH, args.extend, PARAMS, BOUNDS, seed=args.seed)
        print("Appended %d model(s) to %s: %s"
              % (len(new_names), OUTPUT_PATH, ", ".join(new_names)))
    else:
        n = args.fresh if args.fresh is not None else N_SAMPLES
        samples = generate_samples(n, PARAMS, BOUNDS, seed=args.seed)
        write_model_params_csv(OUTPUT_PATH, BASE_CASE, BASE_DESCRIPTION, samples, PARAMS)
        print("Wrote %d models (model_0 + %d sampled) to %s"
              % (1 + len(samples), len(samples), OUTPUT_PATH))


if __name__ == "__main__":
    main()
