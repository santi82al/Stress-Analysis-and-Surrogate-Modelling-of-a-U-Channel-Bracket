"""
geometry/validation.py
-----------------------------------------------------------------------------
Analytical stress model + comparison against the FEM results CSV.

Pure stdlib (math, csv) - runs in Abaqus Python OR regular Python, so it can
be called straight after build_model_4walls() inside the loop.
-----------------------------------------------------------------------------
"""

import csv
import os
from math import pi

# Abaqus API (only available when running inside Abaqus)
try:
    from abaqus import *
    from abaqusConstants import *
    HAS_ABAQUS = True
except ImportError:
    HAS_ABAQUS = False


def _linearize(samples, thickness):
    """
    Decompose a through-thickness stress distribution into membrane + bending.

    samples : list of (t_coord, sigma) pairs, t_coord in [0, thickness]
              sigma is one stress component (e.g. bending normal S33)
    thickness : section thickness in the through-thickness direction

    Returns (membrane, bending_surface):
      membrane       = average stress through the thickness (constant part)
      bending_surface = surface value of the linear (bending) part = |slope|*t/2

    The peak (nonlinear) part is discarded - this is exactly the linearized
    stress that beam/plate theory predicts, free of local concentrations.
    Uses manual loops (Abaqus shadows the built-in sum()).
    """
    n = len(samples)
    if n < 2 or thickness <= 0:
        return None, None

    # Least-squares linear fit: sigma = a + b*t
    St = St2 = Ss = Sts = 0.0
    for (t, s) in samples:
        St  += t
        St2 += t * t
        Ss  += s
        Sts += t * s
    denom = n * St2 - St * St
    if abs(denom) < 1e-20:
        # all samples at same t - cannot fit a slope; return mean as membrane
        return Ss / n, 0.0
    b = (n * Sts - St * Ss) / denom          # slope
    a = (Ss - b * St) / n                    # intercept
    t_mid = thickness / 2.0
    membrane = a + b * t_mid                  # value at mid-thickness
    bending_surface = abs(b) * (thickness / 2.0)
    return membrane, bending_surface


def analytical_validation(
    MODEL_NAME, b_geo, m_geo, n_geo, t_geo, f_geo, e_geo, i_geo, P_load, wall_thk,
    results_dir=".", summary_csv="validation_summary.csv",
    hole_exclusion_factor=2.0,
    wall_slab_factor=1.5,           # slab half-depth in Y about Y_geo/2, units of t_geo
):
    """
    1. Compute analytical stresses (wall section AA + end pad).
    2. Read {MODEL_NAME}_results.csv for the FEM peak von Mises.
    3. Compute % error and append one row to summary_csv.

    Returns a dict of all values.
    """

    # -------------------------------------------------------------------------
    # ANALYTICAL MODEL
    # -------------------------------------------------------------------------
    k_geo = 1.5 * i_geo
    D_geo = n_geo**2 * t_geo + b_geo * f_geo**2 / 2

    # --- Stress in the wall (Section AA) ---
    A_sect   = b_geo*f_geo + 2*n_geo*t_geo
    I_mom    = (2*t_geo*n_geo**3)/3 + (b_geo*f_geo**3)/3 - D_geo**2/A_sect
    v_geo    = n_geo - D_geo/A_sect
    Sigma_dw = P_load / A_sect
    Sigma_bw = P_load * m_geo * v_geo / I_mom
    Sigma_t  = Sigma_dw + Sigma_bw

    # --- Stress in the end pad ---
    xi  = b_geo / n_geo
    rho = i_geo / n_geo

    # Bending efficiency factor Q - interpolated from 4 curves at fixed rho.
    # NOTE: placeholder coefficients - replace with verified Bruhn/ESDU values.
    Q_01 = -0.0567*xi**3 + 0.3133*xi**2 - 0.1067*xi + 0.450  # rho = 0.1
    Q_02 = -0.0567*xi**3 + 0.2933*xi**2 - 0.1567*xi + 0.380  # rho = 0.2
    Q_03 = -0.0867*xi**3 + 0.3933*xi**2 - 0.2333*xi + 0.310  # rho = 0.3
    Q_04 = -0.0567*xi**3 + 0.2818*xi**2 - 0.1951*xi + 0.250  # rho = 0.4

    rho_nodes = [0.1, 0.2, 0.3, 0.4]
    Q_nodes   = [Q_01, Q_02, Q_03, Q_04]
    rho_c = max(rho_nodes[0], min(rho, rho_nodes[-1]))   # clip to [0.1, 0.4]
    Q = Q_nodes[-1]
    for j in range(len(rho_nodes) - 1):
        if rho_nodes[j] <= rho_c <= rho_nodes[j+1]:
            tt = (rho_c - rho_nodes[j]) / (rho_nodes[j+1] - rho_nodes[j])
            Q = Q_nodes[j] + tt * (Q_nodes[j+1] - Q_nodes[j])
            break

    l_eff    = m_geo - f_geo / 2
    Sigma_bp = 2 * P_load * Q * l_eff / (b_geo * e_geo**2)
    Tau_p    = P_load / (2 * k_geo * pi * e_geo)
    Sigma_vM = (Sigma_bp**2 + 3 * Tau_p**2)**0.5

    analytical_peak = max(Sigma_t, Sigma_vM)
    governing = "wall (Sigma_t)" if Sigma_t >= Sigma_vM else "pad (Sigma_vM)"

    # -------------------------------------------------------------------------
    # READ FEM RESULTS FROM ODB
    # Extracts: global peak von Mises, plus region-filtered peaks in the
    # pad (bottom wall, Y in [0, e_geo]) and wall (back wall, Z in [0, f_geo])
    # sections - the same regions the analytical model represents.
    # -------------------------------------------------------------------------
    odb_path = os.path.join(results_dir, MODEL_NAME + ".odb")
    fem_max_mises = None              # global peak (any location)
    fem_loc = (None, None, None)
    fem_pad_mises = None              # peak within end-pad region
    fem_wall_mises = None             # peak within wall (section AA) region
    fem_pad_nominal = None            # pad peak excluding bolt-hole concentration
    fem_wall_nominal = None           # wall peak excluding bolt-hole concentration
    fem_wall_membrane = None          # linearized membrane (direct) stress at section AA
    fem_wall_bending = None           # linearized bending stress at extreme fiber
    fem_wall_linear = None            # linearized total (direct+bending) at extreme fiber

    # Region boundaries (geometry-derived)
    W_geo = b_geo + 2*t_geo
    Y_geo = 4*m_geo - 2*n_geo + 4*e_geo
    Z_geo = 2*m_geo
    tol = 1e-6
    pad_y_max  = e_geo + tol          # pad = bottom ground wall, thickness e_geo
    wall_z_max = f_geo + tol          # wall = back wall, thickness f_geo

    # Bolt-hole axes (for excluding stress-concentration zones)
    #   Pad hole : axis // Y, centred at (X=W_geo/2, Z=m_geo)
    #   Wall hole: axis // Z, centred at (X=W_geo/2, Y=Y_geo/2)
    r_exclude   = hole_exclusion_factor * i_geo   # keep elements beyond this radius
    pad_hole_x  = W_geo / 2.0
    pad_hole_z  = m_geo
    wall_hole_x = W_geo / 2.0
    wall_hole_y = Y_geo / 2.0

    # Pad-root linearization section (for nominal bending stress).
    # Section AA is a HORIZONTAL cut at Y = Y_geo/2 (height of the back-wall bore).
    # It slices the back wall (b_geo x f_geo) + two side walls (n_geo x t_geo).
    # Bending is about the X-axis -> normal stress S22 varies linearly with Z,
    # neutral axis at Z_na = D_geo/A_sect, extreme fiber at Z = n_geo.
    # We linearize S22 vs Z within a thin Y-slab, excluding the back-wall bore.
    sec_y      = Y_geo / 2.0
    slab_half  = wall_slab_factor * t_geo
    sec_y0     = sec_y - slab_half
    sec_y1     = sec_y + slab_half
    bore_x     = W_geo / 2.0          # back-wall bore: axis // Z at (W/2, Y/2)
    D_geo_loc  = n_geo**2 * t_geo + b_geo * f_geo**2 / 2.0
    A_sect_loc = b_geo*f_geo + 2*n_geo*t_geo
    Z_na       = D_geo_loc / A_sect_loc
    wall_samples = []     # (cz, S22) pairs for linearization

    if os.path.exists(odb_path):
        try:
            if not HAS_ABAQUS:
                raise RuntimeError("Abaqus API required to read ODB file")
            odb = session.openOdb(name=odb_path, readOnly=True)
            step = odb.steps['Load_Step']
            nf   = len(step.frames)
            if nf == 0:
                raise ValueError('Load_Step has 0 frames')
            frame = step.frames[nf - 1]
            instance = odb.rootAssembly.instances['BASE-1']

            # Node-coordinate lookup
            node_coords = {n.label: n.coordinates for n in instance.nodes}

            # Element-centroid lookup (built once; Abaqus sum() shadow-safe)
            elem_centroid = {}
            for elem in instance.elements:
                cx = cy = cz = 0.0
                cnt = 0
                for lbl in elem.connectivity:
                    if lbl in node_coords:
                        c = node_coords[lbl]
                        cx += c[0]; cy += c[1]; cz += c[2]
                        cnt += 1
                if cnt > 0:
                    elem_centroid[elem.label] = (cx/cnt, cy/cnt, cz/cnt)

            # Scan stresses, track global + per-region peaks
            max_elem_label = None
            for v in frame.fieldOutputs['S'].values:
                s11, s22, s33, s12, s13, s23 = [float(x) for x in v.data]
                mises = (((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) / 2.0
                         + 3.0 * (s12**2 + s13**2 + s23**2)) ** 0.5

                # Global peak
                if fem_max_mises is None or mises > fem_max_mises:
                    fem_max_mises = mises
                    max_elem_label = v.elementLabel

                # Region classification by element centroid
                cen = elem_centroid.get(v.elementLabel)
                if cen is not None:
                    cx, cy, cz = cen
                    if cy <= pad_y_max:    # end-pad (bottom wall)
                        if fem_pad_mises is None or mises > fem_pad_mises:
                            fem_pad_mises = mises
                        d_pad = ((cx - pad_hole_x)**2 + (cz - pad_hole_z)**2) ** 0.5
                        if d_pad >= r_exclude:
                            if fem_pad_nominal is None or mises > fem_pad_nominal:
                                fem_pad_nominal = mises
                    if cz <= wall_z_max:   # wall section AA (back wall)
                        if fem_wall_mises is None or mises > fem_wall_mises:
                            fem_wall_mises = mises
                        d_wall = ((cx - wall_hole_x)**2 + (cy - wall_hole_y)**2) ** 0.5
                        if d_wall >= r_exclude:
                            if fem_wall_nominal is None or mises > fem_wall_nominal:
                                fem_wall_nominal = mises
                    # Section AA linearization samples: thin Y-slab at Y_geo/2,
                    # excluding the back-wall bore concentration zone.
                    if sec_y0 <= cy <= sec_y1:
                        in_bore_zone = (abs(cx - bore_x) < r_exclude and cz <= f_geo + r_exclude)
                        if not in_bore_zone:
                            wall_samples.append((cz, s22))

            print("DEBUG: FEM global=%s | pad=%s (nom %s) | wall=%s (nom %s)"
                  % (fem_max_mises, fem_pad_mises, fem_pad_nominal,
                     fem_wall_mises, fem_wall_nominal))

            # Location of global peak element
            if max_elem_label is not None:
                cen = elem_centroid.get(max_elem_label)
                if cen is not None:
                    fem_loc = cen

            # ----- Linearization at Section AA (Y = Y_geo/2), S22 vs Z -----
            # Fit S22 = a + b*Z. Extreme-fiber stress (side-wall tip, Z=n_geo)
            # = direct + bending, comparable to analytical Sigma_t.
            n_s = len(wall_samples)
            if n_s >= 2:
                Sz = Sz2 = Ss = Szs = 0.0
                for (z, s) in wall_samples:
                    Sz += z; Sz2 += z*z; Ss += s; Szs += z*s
                denom = n_s*Sz2 - Sz*Sz
                if abs(denom) > 1e-20:
                    b_fit = (n_s*Szs - Sz*Ss) / denom
                    a_fit = (Ss - b_fit*Sz) / n_s
                    mean_z = Sz / n_s
                    fem_wall_membrane = a_fit + b_fit*mean_z          # section-average (direct)
                    fem_wall_bending  = abs(b_fit) * (n_geo - Z_na)   # bending at extreme fiber
                    fem_wall_linear   = abs(a_fit + b_fit*n_geo)      # total at side-wall tip
                    print("DEBUG: wall AA linearization - membrane=%.2f bending=%.2f -> extreme=%.2f (from %d samples, slope=%.4f)"
                          % (fem_wall_membrane, fem_wall_bending, fem_wall_linear, n_s, b_fit))
                else:
                    print("DEBUG: wall AA - degenerate fit (%d samples all same Z)" % n_s)
            else:
                print("DEBUG: wall AA - insufficient samples (%d)" % n_s)

            odb.close()
        except Exception as e:
            print("WARNING: Failed to read stress from ODB: %s" % str(e))
            import traceback
            traceback.print_exc()
    else:
        print("WARNING: ODB not found -> %s" % odb_path)

    # -------------------------------------------------------------------------
    # ERRORS
    # Global error uses the unfiltered FEM peak (includes stress concentration).
    # Filtered error compares the analytical peak against the FEM peak in the
    # matching region (pad vs wall), a more meaningful nominal-to-nominal check.
    # -------------------------------------------------------------------------
    if fem_max_mises is not None:
        error_pct = (analytical_peak - fem_max_mises) / fem_max_mises * 100.0
    else:
        error_pct = None

    # Pick the FEM region matching the governing analytical mode
    if governing.startswith("wall"):
        fem_region_mises   = fem_wall_mises
        fem_region_nominal = fem_wall_nominal
    else:
        fem_region_mises   = fem_pad_mises
        fem_region_nominal = fem_pad_nominal

    if fem_region_mises:
        error_filtered = (analytical_peak - fem_region_mises) / fem_region_mises * 100.0
    else:
        error_filtered = None

    # Nominal error: analytical vs FEM peak in region with hole zone excluded
    if fem_region_nominal:
        error_nominal = (analytical_peak - fem_region_nominal) / fem_region_nominal * 100.0
    else:
        error_nominal = None

    # Linearized error: analytical Sigma_t vs FEM linearized extreme-fiber
    # stress at Section AA (Y_geo/2) - the rigorous nominal-to-nominal check.
    if fem_wall_linear:
        error_linear = (Sigma_t - fem_wall_linear) / fem_wall_linear * 100.0
    else:
        error_linear = None

    result = {
        "MODEL_NAME":         MODEL_NAME,
        "Sigma_t":            round(Sigma_t, 3),
        "Sigma_vM":           round(Sigma_vM, 3),
        "Governing":          governing,
        "Analytical_peak":    round(analytical_peak, 3),
        "FEM_max_Mises":      round(fem_max_mises, 3) if fem_max_mises is not None else None,
        "FEM_max_X":          round(fem_loc[0], 2) if fem_loc[0] is not None else None,
        "FEM_max_Y":          round(fem_loc[1], 2) if fem_loc[1] is not None else None,
        "FEM_max_Z":          round(fem_loc[2], 2) if fem_loc[2] is not None else None,
        "Error_pct":          round(error_pct, 2) if error_pct is not None else None,
        "FEM_wall_Mises":     round(fem_wall_mises, 3) if fem_wall_mises is not None else None,
        "FEM_wall_membrane":  round(fem_wall_membrane, 3) if fem_wall_membrane is not None else None,
        "FEM_wall_bending":   round(fem_wall_bending, 3) if fem_wall_bending is not None else None,
        "FEM_wall_linear":    round(fem_wall_linear, 3) if fem_wall_linear is not None else None,
        "Error_wall_pct":     round(error_linear, 2) if error_linear is not None else None,
    }

    # -------------------------------------------------------------------------
    # WRITE TO SUMMARY CSV
    # Overwrites any existing row for this model; rewrites the whole file.
    # This prevents duplicate rows from accumulating across runs.
    # -------------------------------------------------------------------------
    FIELDS = list(result.keys())

    # Read existing rows (excluding any previous entry for this model)
    existing = []
    if os.path.exists(summary_csv):
        with open(summary_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("MODEL_NAME") != MODEL_NAME:
                    existing.append(row)

    # Rewrite entire file with updated row
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in existing:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
        writer.writerow(result)

    # -- Console report -------------------------------------------------------
    mem_str = "%.1f" % fem_wall_membrane if fem_wall_membrane is not None else "N/A"
    lin_str = "%.1f" % fem_wall_linear   if fem_wall_linear   else "N/A"
    errl_str = "%+.1f%%" % error_linear  if error_linear is not None else "N/A"
    print("[%s] Sigma_t=%.1f (dw=%.1f) | FEM AA membrane=%s extreme=%s | wall error=%s"
          % (MODEL_NAME, Sigma_t, Sigma_dw, mem_str, lin_str, errl_str))

    return result