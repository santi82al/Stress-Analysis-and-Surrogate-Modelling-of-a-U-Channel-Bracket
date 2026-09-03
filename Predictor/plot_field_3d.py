"""
3D scatter view of a predicted (or FEM) field: node coordinates + color scale.

Interactive mode opens a window with two sliders - yaw (azimuth) and pitch
(elevation) - that rotate every panel together. The window title bar shows the
current angles, and on close the script prints the exact --elev/--azim flags
that reproduce the view, so report figures can be regenerated identically.

Usage
-----
1) One field, interactive:

    python plot_field_3d.py test_predicted_field.csv --field S_Mises

2) Same view, saved straight to a figure (no window, works headless):

    python plot_field_3d.py test_predicted_field.csv --field S_Mises \
        --elev 22 --azim -55 --save ../Thesis_LaTex/figs/field_smises.png

3) Surrogate vs FEM vs error, three linked panels on a shared color scale:

    python plot_field_3d.py pred.csv --compare-with ../output/model_7/model_7_results.csv \
        --field S_Mises

Color encoding follows the physics of the field:
  - non-negative magnitudes (S_Mises, U_mag) -> a jet-like rainbow, 0..p95,
    matching Abaqus's default contour spectrum and direction (blue = low,
    red = high), but with the green band darkened -- plain jet's pure green
    plateau is nearly as light as a white page and washes out in print
  - signed components (U1, U2, U3)           -> the same rainbow, symmetric
    about 0, for the same reason: Abaqus renders every field on one ramp, not
    a special diverging one for signed quantities
  - the error (surrogate - FEM) panel is the one exception: it is not a field
    Abaqus ever renders, so it keeps a diverging RdBu_r scale, gray = zero
    error, for its own readability
Color limits are clipped at the 95th percentile by default: the peak Von Mises
sits on a singular node at the bolt edge and would otherwise consume the whole
ramp and flatten every other feature. Use --clip 100 to disable.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

COORD_NAMES = ["X", "Y", "Z"]

# Fields that are norms: sequential ramp from 0. Everything else is signed and
# gets a diverging ramp centered on zero.
NONNEG_FIELDS = {"S_Mises", "U_mag", "RF_mag"}

UNITS = {"U1": "mm", "U2": "mm", "U3": "mm", "U_mag": "mm",
         "S_Mises": "MPa", "S_MaxPrincipal": "MPa", "S_MinPrincipal": "MPa",
         "RF1": "N", "RF2": "N", "RF3": "N", "RF_mag": "N"}

# Same blue-to-red rainbow direction as Abaqus's default spectrum, but with the
# green band pulled darker: plain jet's pure green (0,1,0) plateau is nearly as
# light as the white page background, so the mid-scale reads as washed out in
# print. Everything else follows jet's usual stops.
SEQUENTIAL = LinearSegmentedColormap.from_list("abaqus_jet", [
    (0.00, (0.00, 0.00, 0.60)),   # dark blue
    (0.15, (0.00, 0.00, 1.00)),   # blue
    (0.35, (0.00, 0.85, 1.00)),   # cyan
    (0.50, (0.05, 0.70, 0.15)),   # green, moderately darkened (was ~(0,1,0))
    (0.65, (0.80, 0.80, 0.00)),   # olive-yellow
    (0.80, (1.00, 0.55, 0.00)),   # orange
    (1.00, (0.60, 0.00, 0.00)),   # dark red
], N=256)
ERROR_DIVERGING = "RdBu_r"  # error panel only: diverging, gray at zero

# Which data axis points up, as a column order fed to matplotlib's (x, y, z).
#
# NOTE: do NOT implement this with view_init(vertical_axis=...). That option
# applies set_box_aspect's components to the WRONG data axes: a 66x128x90 box
# viewed straight down (no foreshortening) renders an X/Z length ratio of 1.939
# instead of the true 0.733 -- the part comes out roughly twice as wide as it
# should be. Reordering the columns keeps matplotlib in its native vertical-z
# configuration, where the aspect is exact (verified 0.516 vs 0.516).
#
# The permutation MUST be even (determinant +1), i.e. a cyclic rotation of the
# axes. An odd permutation such as (0,2,1) swaps two axes, which flips handedness
# and renders a MIRROR IMAGE of the part -- on this bracket, with its asymmetric
# relief cut, that is wrong, not merely unconventional. (0,2,1) was used here
# initially and drew X pointing left where Abaqus draws it right.
#
# With (2,0,1) at azim=45 the axes project as X(+0.86,-0.51), Y(0,+1),
# Z(-0.86,-0.51): X right-down, Y up, Z left-down, matching the Abaqus triad, so
# a predicted field and an ODB contour can be read side by side.
AXIS_ORDER = {
    "y": (2, 0, 1),   # XZ floor, Y up -- matches the part and the Abaqus view
    "z": (0, 1, 2),   # XY floor, Z up -- matplotlib's native orientation
    "x": (1, 2, 0),   # YZ floor, X up
}
AXIS_LABELS = ["X [mm]", "Y [mm]", "Z [mm]"]


def _assert_right_handed(order):
    """Guard against reintroducing a handedness-flipping (odd) permutation."""
    m = np.zeros((3, 3))
    for row, col in enumerate(order):
        m[row, col] = 1.0
    if round(float(np.linalg.det(m))) != 1:
        raise SystemExit("axis order %s flips handedness (det=-1): the part "
                         "would render mirrored" % (order,))


def label_above(cax):
    """Put a horizontal colorbar's ticks and label above its bar.

    The colorbars sit at the top of the figure (the bottom strip is reserved
    for the sliders), so their text has to go up, clear of the panel titles.
    """
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")


def color_spec(field, vals, clip_pct):
    """(cmap, vmin, vmax) chosen by what the field means, not by taste."""
    if field in NONNEG_FIELDS:
        return SEQUENTIAL, 0.0, float(np.percentile(vals, clip_pct))
    a = float(np.percentile(np.abs(vals), clip_pct)) or 1.0
    return SEQUENTIAL, -a, a


def scatter_panel(ax, xyz, vals, cmap, vmin, vmax, title, size, zoom=1.0,
                  order=(0, 1, 2)):
    """One 3D scatter panel with true geometric proportions.

    `order` permutes the data columns into matplotlib's (x, y, z) slots so the
    chosen axis lands in the native vertical slot -- see AXIS_ORDER.
    """
    q = xyz[:, list(order)]
    p = ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=vals, cmap=cmap,
                   vmin=vmin, vmax=vmax, s=size, linewidths=0,
                   depthshade=False)   # depth shading would corrupt the colors
    span = np.ptp(q, axis=0)
    # True proportions: the bracket must not look stretched.
    #
    # zoom MUST stay at 1.0. Values >1 scale the 3D box beyond the axes
    # rectangle and matplotlib silently clips the scatter to that rectangle --
    # measured 20 of 13064 points vanishing at elev=45/azim=45 with zoom=1.15.
    # Losing data points from a field plot is a correctness bug, so fill the
    # panel by sizing the axes rect instead.
    ax.set_box_aspect(np.maximum(span, span.max() * 1e-3), zoom=zoom)
    ax.set_xlabel(AXIS_LABELS[order[0]], fontsize=8)
    ax.set_ylabel(AXIS_LABELS[order[1]], fontsize=8)
    ax.set_zlabel(AXIS_LABELS[order[2]], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):   # recessive frame
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.9")
    ax.grid(True, alpha=0.25)
    return p


def add_view_sliders(fig, axes3d, elev0, azim0):
    """Two sliders driving view_init on every 3D panel at once."""
    ax_yaw = fig.add_axes([0.25, 0.085, 0.50, 0.025])
    ax_pit = fig.add_axes([0.25, 0.035, 0.50, 0.025])
    s_yaw = Slider(ax_yaw, "yaw (azim)", -180.0, 180.0, valinit=azim0, valstep=1.0)
    s_pit = Slider(ax_pit, "pitch (elev)", -90.0, 90.0, valinit=elev0, valstep=1.0)

    def update(_):
        for ax in axes3d:
            ax.view_init(elev=s_pit.val, azim=s_yaw.val)
        fig.canvas.manager.set_window_title(
            "3D field  |  elev=%.0f  azim=%.0f" % (s_pit.val, s_yaw.val))
        fig.canvas.draw_idle()

    def on_close(_event):
        # Read the axes, NOT the sliders: dragging the plot with the mouse
        # rotates the view without moving the sliders, so the slider values can
        # disagree with what is actually on screen.
        ax0 = axes3d[0]
        print("reproduce this view with:  --elev %.0f --azim %.0f"
              % (ax0.elev, ax0.azim))

    s_yaw.on_changed(update)
    s_pit.on_changed(update)
    fig.canvas.mpl_connect("close_event", on_close)
    update(None)
    return s_yaw, s_pit   # caller must keep these alive or they stop responding


def load_field(path, field):
    df = pd.read_csv(path)
    missing = [c for c in COORD_NAMES + [field] if c not in df.columns]
    if missing:
        raise SystemExit("%s is missing column(s): %s" % (path, ", ".join(missing)))
    return df[COORD_NAMES].to_numpy(float), df[field].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="predicted (or FEM results) CSV with X,Y,Z + fields")
    ap.add_argument("--field", default="S_Mises")
    ap.add_argument("--compare-with", help="FEM results CSV on the SAME node cloud")
    ap.add_argument("--clip", type=float, default=95.0,
                    help="percentile for color limits (100 = full range)")
    ap.add_argument("--size", type=float, default=14.0, help="marker size [pt^2]")
    ap.add_argument("--max-points", type=int, default=0,
                    help="decimate to N points for smoother rotation (0 = all)")
    ap.add_argument("--elev", type=float, default=45.0, help="initial pitch")
    ap.add_argument("--azim", type=float, default=45.0, help="initial yaw")
    ap.add_argument("--vertical-axis", choices=["x", "y", "z"], default="y",
                    help="which data axis points up. Default 'y' puts XZ on the "
                         "floor, matching the part (Y=0 is the ground wall); "
                         "'z' gives the XY floor. Implemented by reordering "
                         "columns, not view_init(vertical_axis=) -- see AXIS_ORDER.")
    ap.add_argument("--save", help="write a figure here instead of opening a window")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    order = AXIS_ORDER[args.vertical_axis]
    _assert_right_handed(order)
    unit = UNITS.get(args.field, "")
    label = "%s [%s]" % (args.field, unit) if unit else args.field
    xyz, vals = load_field(args.csv, args.field)

    truth = None
    if args.compare_with:
        xyz_t, truth = load_field(args.compare_with, args.field)
        if len(xyz_t) != len(xyz) or not np.allclose(xyz_t, xyz, atol=1e-6):
            raise SystemExit(
                "node clouds differ - predict with --coords-from %s so both files "
                "share the same nodes." % args.compare_with)

    if args.max_points and args.max_points < len(xyz):
        idx = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, vals = xyz[idx], vals[idx]
        if truth is not None:
            truth = truth[idx]
        print("decimated to %d points for display" % len(xyz))

    if truth is None:
        # Generous bottom margin and a modest zoom: 3D tick labels swing well
        # outside the axes rect as the view rotates, and at low elevations they
        # would otherwise land on top of the colorbar.
        # Colorbar goes ABOVE the plot: the bottom strip belongs to the sliders,
        # and a bottom colorbar collides with them in interactive mode.
        fig = plt.figure(figsize=(7.5, 7.2))
        fig.subplots_adjust(left=0.02, right=0.94, bottom=0.20, top=0.86)
        ax = fig.add_subplot(projection="3d")
        cmap, vmin, vmax = color_spec(args.field, vals, args.clip)
        p = scatter_panel(ax, xyz, vals, cmap, vmin, vmax, "", args.size,
                          order=order)
        cax = fig.add_axes([0.25, 0.92, 0.50, 0.025])
        fig.colorbar(p, cax=cax, orientation="horizontal", label=label)
        label_above(cax)
        panels = [ax]
    else:
        err = vals - truth
        # Both field panels share ONE scale (set by the FEM panel) or the
        # comparison is meaningless; the error panel gets its own diverging one.
        cmap, vmin, vmax = color_spec(args.field, truth, args.clip)
        emax = float(np.percentile(np.abs(err), args.clip)) or 1.0

        fig = plt.figure(figsize=(14.0, 7.6))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.24, top=0.82, wspace=0.02)
        panels = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        p0 = scatter_panel(panels[0], xyz, truth, cmap, vmin, vmax,
                           "FEM (Abaqus)", args.size, order=order)
        scatter_panel(panels[1], xyz, vals, cmap, vmin, vmax,
                      "Surrogate", args.size, order=order)
        pe = scatter_panel(panels[2], xyz, err, ERROR_DIVERGING, -emax, emax,
                           "Error (surrogate - FEM)", args.size, order=order)
        cax0 = fig.add_axes([0.09, 0.88, 0.42, 0.022])
        cax1 = fig.add_axes([0.70, 0.88, 0.22, 0.022])
        fig.colorbar(p0, cax=cax0, orientation="horizontal", label=label)
        fig.colorbar(pe, cax=cax1, orientation="horizontal",
                     label="error [%s]" % unit if unit else "error")
        label_above(cax0)
        label_above(cax1)

        rmse = float(np.sqrt((err ** 2).mean()))
        ss_tot = float(((truth - truth.mean()) ** 2).sum()) or 1e-30
        r2 = 1.0 - float((err ** 2).sum()) / ss_tot
        fig.suptitle("%s - RMSE = %.4g %s   R² = %.3f   (%d nodes)"
                     % (args.field, rmse, unit, r2, len(xyz)), fontsize=11, y=0.99)

    if args.save:
        for ax in panels:
            ax.view_init(elev=args.elev, azim=args.azim)
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print("saved -> %s  (elev=%.0f azim=%.0f)" % (args.save, args.elev, args.azim))
    else:
        _sliders = add_view_sliders(fig, panels, args.elev, args.azim)
        plt.show()


if __name__ == "__main__":
    main()
