"""
extract_peak.py  --  run with:  abaqus python Convergence\\extract_peak.py
-----------------------------------------------------------------------------
Standalone ODB reader for the convergence study. Opens each already-solved
Convergence/output/<model>/<model>.odb and extracts, from the last frame of
'Load_Step', TWO parallel convergence metric families so the study does not
depend on the singular global peak Von Mises:

  GLOBAL (stiffness / load path)
    u_load   |U| at the node nearest the load point (bore centre)  -> compliance
    rf_total magnitude of the summed reaction force  -> equilibrium check
             (should stay ~= P_load regardless of mesh; a validation, not a
              convergence quantity)

  LOCAL (stress)
    vm_probe Von Mises at the node nearest a fixed bulk probe point, placed a
             few bolt radii off the bore edge so it is OFF the singularity
    vm_p95   95th-percentile nodal Von Mises  -> singularity-robust stress level
    vm_max   global peak Von Mises  -> kept only for reference (does NOT converge)

Probe coordinates are derived from the (constant) convergence geometry read from
convergence_params.csv, so they are identical across every mesh; the nearest-node
distance is reported so you can see each probe is landing consistently.

Writes Convergence/peak_mises.csv with all columns.
-----------------------------------------------------------------------------
"""
from __future__ import print_function
import os
import csv
import glob
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
PARAMS_CSV = os.path.join(HERE, "convergence_params.csv")


def _geometry():
    """Read the (shared) convergence geometry from convergence_params.csv.

    All convergence columns use the same geometry, so the first model_* column
    defines the probe locations. Returns a dict of derived dimensions + probes.
    """
    rows = {}
    f = open(PARAMS_CSV, "r")
    try:
        reader = csv.DictReader(f)
        model_cols = [c for c in reader.fieldnames if c.startswith("model_")]
        col = model_cols[0]
        for row in reader:
            key = row["Parameter"].strip().split()[0]
            rows[key] = row[col]
    finally:
        f.close()
    g = dict((k, float(rows[k])) for k in
             ("b_geo", "m_geo", "n_geo", "t_geo", "f_geo", "e_geo", "i_geo"))
    W = g["b_geo"] + 2.0 * g["t_geo"]
    m, e, i = g["m_geo"], g["e_geo"], g["i_geo"]
    # Load point = bore centre (coupling control point, force applied here).
    load_pt = (W / 2.0, e / 2.0, m)
    # Stress probe = same bore mid-plane, shifted ~3 bolt radii outboard in X so
    # it sits in bulk material well clear of the singular bore/contact edge.
    stress_probe = (W / 2.0 + 3.0 * i, e / 2.0, m)
    return {"W": W, "load_pt": load_pt, "stress_probe": stress_probe}


def _mises(d):
    s11, s22, s33, s12, s13, s23 = [float(x) for x in d]
    return (((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) / 2.0
            + 3.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)) ** 0.5


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    k = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def _nearest_node(coords, target):
    """Return (label, distance) of the node closest to target (x,y,z)."""
    tx, ty, tz = target
    best_lbl, best_d2 = None, None
    for lbl, (x, y, z) in coords.items():
        d2 = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
        if best_d2 is None or d2 < best_d2:
            best_lbl, best_d2 = lbl, d2
    return best_lbl, (best_d2 ** 0.5 if best_d2 is not None else None)


def extract(odb_path, geo):
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step = odb.steps['Load_Step']
        nf = len(step.frames)
        if nf == 0:
            return {"n_frames": 0}
        frame = step.frames[nf - 1]
        inst = odb.rootAssembly.instances['BASE-1']
        coords = dict((n.label, n.coordinates) for n in inst.nodes)

        # --- displacement: global max + at load point --------------------
        U = frame.fieldOutputs['U'].getSubset(region=inst)
        u_by_node = dict((v.nodeLabel, v.magnitude) for v in U.values)
        u_max = max(u_by_node.values()) if u_by_node else None
        load_lbl, load_dist = _nearest_node(coords, geo["load_pt"])
        u_load = u_by_node.get(load_lbl)

        # --- reaction force: vector sum over instance --------------------
        RF = frame.fieldOutputs['RF'].getSubset(region=inst)
        rx = ry = rz = 0.0
        for v in RF.values:
            rx += v.data[0]; ry += v.data[1]; rz += v.data[2]
        rf_total = (rx * rx + ry * ry + rz * rz) ** 0.5

        # --- stress: nodal-averaged Von Mises ---------------------------
        S = frame.fieldOutputs['S'].getSubset(region=inst,
                                               position=ELEMENT_NODAL)
        sums, counts = {}, {}
        for v in S.values:
            lbl = v.nodeLabel
            acc = sums.setdefault(lbl, [0.0] * 6)
            for k in range(6):
                acc[k] += v.data[k]
            counts[lbl] = counts.get(lbl, 0) + 1
        vm_by_node = {}
        for lbl, acc in sums.items():
            vm_by_node[lbl] = _mises([c / counts[lbl] for c in acc])
        vm_vals = sorted(vm_by_node.values())
        vm_max = vm_vals[-1] if vm_vals else None
        vm_p95 = _percentile(vm_vals, 95.0)
        probe_lbl, probe_dist = _nearest_node(coords, geo["stress_probe"])
        vm_probe = vm_by_node.get(probe_lbl)

        return {"n_frames": nf, "u_max": u_max, "u_load": u_load,
                "load_dist": load_dist, "rf_total": rf_total,
                "vm_max": vm_max, "vm_p95": vm_p95, "vm_probe": vm_probe,
                "probe_dist": probe_dist}
    finally:
        odb.close()


def _fmt(x):
    return "" if x is None else "%.6f" % x


def main():
    geo = _geometry()
    print("Load point   : (%.2f, %.2f, %.2f)" % geo["load_pt"])
    print("Stress probe : (%.2f, %.2f, %.2f)" % geo["stress_probe"])
    print("")
    cols = ["model", "n_frames", "u_load", "u_max", "rf_total",
            "vm_probe", "vm_p95", "vm_max", "load_dist", "probe_dist"]
    rows = []
    for odb_path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "*.odb"))):
        model = os.path.splitext(os.path.basename(odb_path))[0]
        try:
            d = extract(odb_path, geo)
            d["model"] = model
            rows.append(d)
            print("%-10s frames=%-3d u_load=%s vm_probe=%s vm_p95=%s rf=%s"
                  % (model, d.get("n_frames", 0), _fmt(d.get("u_load")),
                     _fmt(d.get("vm_probe")), _fmt(d.get("vm_p95")),
                     _fmt(d.get("rf_total"))))
        except Exception as e:
            print("%-10s EXTRACT FAILED: %s" % (model, e))

    out = os.path.join(HERE, "peak_mises.csv")
    f = open(out, "w")
    try:
        f.write(",".join(cols) + "\n")
        for d in rows:
            f.write(",".join(
                str(d["model"]) if c == "model"
                else str(d.get("n_frames", 0)) if c == "n_frames"
                else _fmt(d.get(c)) for c in cols) + "\n")
    finally:
        f.close()
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
