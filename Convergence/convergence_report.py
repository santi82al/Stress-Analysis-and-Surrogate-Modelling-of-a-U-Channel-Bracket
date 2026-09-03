"""
convergence_report.py
-----------------------------------------------------------------------------
Join the per-mesh element count (Convergence/output/<model>/<model>_mesh.csv,
written by CAE_models_builder) with the two parallel convergence metric families
extracted from the ODBs by extract_peak.py (Convergence/peak_mises.csv), and
print a ranked convergence table.

Two metric families are tracked in parallel because the global peak Von Mises is
singularity-dominated and does not converge:

  GLOBAL  u_load : |U| at the load point (compliance / stiffness)
  LOCAL   vm_p95 : 95th-percentile nodal Von Mises (singularity-robust)
                   vm_probe : Von Mises at a fixed bulk probe point
                   vm_max   : global peak (reference only - does NOT converge)

Models with 0 frames (failed solve) are skipped.

Plain CPython + stdlib only (run OUTSIDE Abaqus, from the repo root):
    abaqus python Convergence/extract_peak.py     # once, to (re)build peaks
    python Convergence/convergence_report.py
Writes Convergence/convergence.csv and, if matplotlib is present,
Convergence/convergence.png.
-----------------------------------------------------------------------------
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
PEAKS_CSV = HERE / "peak_mises.csv"

# Columns carried through from peak_mises.csv (besides model / n_frames).
METRICS = ["u_load", "u_max", "vm_probe", "vm_p95", "vm_max"]
# Headline convergent metric from each family, reported as % change vs finest.
HEADLINE = ["u_load", "vm_p95"]


def _load_peaks():
    """Return {model: {n_frames, u_load, vm_p95, ...}} from peak_mises.csv."""
    peaks = {}
    if not PEAKS_CSV.exists():
        return peaks
    with open(PEAKS_CSV, "r") as f:
        for row in csv.DictReader(f):
            rec = {"n_frames": int(row.get("n_frames", "0") or 0)}
            for k in METRICS:
                v = row.get(k, "")
                rec[k] = float(v) if v not in ("", None) else None
            peaks[row["model"]] = rec
    return peaks


def collect():
    rows = []
    if not OUTPUT_DIR.is_dir():
        return rows
    peaks = _load_peaks()
    for model_dir in sorted(OUTPUT_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        mesh_csv = model_dir / ("%s_mesh.csv" % name)
        if not mesh_csv.exists() or name not in peaks:
            continue
        rec = peaks[name]
        if rec.get("vm_p95") is None:       # failed solve (0 frames) -> exclude
            print("Skipping %s (no converged frame, n_frames=%d)"
                  % (name, rec.get("n_frames", 0)))
            continue
        with open(mesh_csv, "r") as f:
            m = next(csv.DictReader(f))
        row = {"model": name, "mesh_div": float(m["mesh_div"]),
               "n_elements": int(m["n_elements"])}
        row.update((k, rec.get(k)) for k in METRICS)
        rows.append(row)
    rows.sort(key=lambda r: r["n_elements"])
    return rows


def collect_failed():
    """Meshes that ran but did not converge (0 frames) -> no metrics.

    Returns [{model, mesh_div, n_elements}] for the validity-floor plot.
    """
    failed = []
    if not OUTPUT_DIR.is_dir():
        return failed
    peaks = _load_peaks()
    for model_dir in sorted(OUTPUT_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        mesh_csv = model_dir / ("%s_mesh.csv" % name)
        rec = peaks.get(name)
        if not mesh_csv.exists() or rec is None:
            continue
        if rec.get("vm_p95") is None:       # solved-but-0-frames = diverged
            with open(mesh_csv, "r") as f:
                m = next(csv.DictReader(f))
            failed.append({"model": name, "mesh_div": float(m["mesh_div"]),
                           "n_elements": int(m["n_elements"])})
    failed.sort(key=lambda r: r["n_elements"])
    return failed


def _pct(val, ref):
    if val is None or not ref:
        return 0.0
    return 100.0 * (val - ref) / ref


def main():
    rows = collect()
    if not rows:
        print("No convergence data found under %s" % OUTPUT_DIR)
        print("Run a convergence job + extract_peak.py first.")
        return

    ref = {k: rows[-1][k] for k in HEADLINE}   # finest mesh = reference

    hdr = ("%-10s %6s %9s %10s %9s %10s %9s %10s" %
           ("model", "div", "elements",
            "u_load", "d%", "vm_p95", "d%", "vm_probe"))
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-10s %6.1f %9d %10.5f %8.2f%% %10.3f %8.2f%% %10.3f" %
              (r["model"], r["mesh_div"], r["n_elements"],
               r["u_load"], _pct(r["u_load"], ref["u_load"]),
               r["vm_p95"], _pct(r["vm_p95"], ref["vm_p95"]),
               r["vm_probe"] if r["vm_probe"] is not None else float("nan")))

    out_csv = HERE / "convergence.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mesh_div", "n_elements"] + METRICS +
                   ["u_load_pct_vs_fine", "vm_p95_pct_vs_fine"])
        for r in rows:
            w.writerow([r["model"], r["mesh_div"], r["n_elements"]] +
                       [r[k] for k in METRICS] +
                       [_pct(r["u_load"], ref["u_load"]),
                        _pct(r["vm_p95"], ref["vm_p95"])])
    print("\nWrote %s" % out_csv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Drop off-scale outliers from the PLOT (broken coarse-mesh contact
        # solves, e.g. div<=17 give u_load ~6.3 vs ~0.3 physical): they compress
        # the converged plateau to an unreadable flat line. They stay in the
        # table / CSV as evidence of the validity floor (see findings doc).
        pdata = rows
        if len(rows) > 3:
            med = sorted(r["u_load"] for r in rows)[len(rows) // 2]
            dropped = [r["model"] for r in rows if r["u_load"] > 5.0 * med]
            if dropped:
                print("Plot: excluding %s (broken/off-scale u_load) for readability"
                      % ", ".join(dropped))
                pdata = [r for r in rows if r["u_load"] <= 5.0 * med]
        n = [r["n_elements"] for r in pdata]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        # Log element-count axis: the valid meshes span ~6k-81k, so a linear axis
        # crams the coarse points together and hides the approach to the plateau.
        a1.plot(n, [r["u_load"] for r in pdata], "o-")
        a1.set_xscale("log")
        a1.set_xlabel("Number of elements (log)"); a1.set_ylabel("|U| at load point")
        a1.set_title("Global: compliance"); a1.grid(True, which="both", alpha=0.3)
        a2.plot(n, [r["vm_p95"] for r in pdata], "s-", label="P95")
        a2.plot(n, [r["vm_probe"] for r in pdata], "^--", label="probe")
        a2.set_xscale("log")
        a2.set_xlabel("Number of elements (log)"); a2.set_ylabel("Von Mises")
        a2.set_title("Local: stress"); a2.grid(True, which="both", alpha=0.3); a2.legend()
        fig.tight_layout()
        png = HERE / "convergence.png"
        fig.savefig(png, dpi=150)
        print("Wrote %s" % png)

        # ---- Second figure: validity floor (invalid + valid meshes) ----------
        # Two panels (global compliance + local stress). Shows every SOLVED mesh
        # (broken coarse contact solves = invalid, plus valid) and a shaded
        # "invalid" band up to the first physically valid mesh. Meshes that failed
        # to converge (no result) are deliberately NOT plotted.
        # valid/broken split is by u_load (the metric that collapses hardest);
        # the same split is applied to the stress panel for consistency.
        med = sorted(r["u_load"] for r in rows)[len(rows) // 2]
        valid = [r for r in rows if r["u_load"] <= 5.0 * med]
        broken = [r for r in rows if r["u_load"] > 5.0 * med]

        def _validity_panel(ax, ykey, ylabel, title):
            ax.set_xscale("log")
            if valid and broken:
                first_valid = min(r["n_elements"] for r in valid)
                last_invalid = max(r["n_elements"] for r in broken)
                xfloor = (last_invalid * first_valid) ** 0.5   # log midpoint
                xlo = min(r["n_elements"] for r in rows) * 0.85
                ax.axvspan(xlo, xfloor, color="tab:red", alpha=0.08)
                ax.axvline(xfloor, color="tab:red", ls="--", lw=1.2)
                ax.text(xfloor, 0.95, "  validity floor\n  (div=18, ~5.9k elem)",
                        transform=ax.get_xaxis_transform(), va="top", ha="left",
                        color="tab:red", fontsize=8)
            if broken:
                ax.plot([r["n_elements"] for r in broken],
                        [r[ykey] for r in broken], "X", color="tab:red", ms=10,
                        label="invalid (contact collapse)")
            ax.plot([r["n_elements"] for r in valid],
                    [r[ykey] for r in valid], "o-", color="tab:blue",
                    label="valid")
            ax.set_xlabel("Number of elements (log)"); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="center right", fontsize=8)

        fig2, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.5))
        _validity_panel(b1, "u_load", "|U| at load point",
                        "Global: compliance (all meshes)")
        _validity_panel(b2, "vm_p95", "Von Mises (P95)",
                        "Local: stress (all meshes)")
        fig2.tight_layout()
        png2 = HERE / "convergence_including_invalid.png"
        fig2.savefig(png2, dpi=150)
        print("Wrote %s" % png2)
    except ImportError:
        print("(matplotlib not available - skipped plot)")


if __name__ == "__main__":
    main()
