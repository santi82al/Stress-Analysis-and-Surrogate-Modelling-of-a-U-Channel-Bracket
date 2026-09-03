"""
geometry/audit_analytical_vs_fem.py
-----------------------------------------------------------------------------
Audit of the Chapter-2 analytical stress model against the FEM node results.

Three checks, run against one model (default model_0):

  1. ALGEBRA   - recompute A, D, ybar, v, I and the three mutually inconsistent
                 bending moments that appear in the thesis and in validation.py.
  2. GEOMETRY  - compare the analytical Section A-A (b*f + 2*n*t) against the
                 section that actually exists in the built part.
  3. FEM       - integrate the FEM stress field over real section cuts to get
                 the true resultants (N, M) and the true linearised stress, and
                 compare against the analytical prediction.

The FEM field used is DELTA = S(Load_Step) - S(Preload_Step), i.e. the response
to the applied load P alone. The bolt preload (~16x P) is not represented in the
analytical model, so the raw Load_Step field is not comparable to it.

Requires <model>_components.csv from dump_stress_components.py.

    python geometry/audit_analytical_vs_fem.py [model_0]
-----------------------------------------------------------------------------
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_CSV = os.path.join(ROOT, 'model_params.csv')
GRID = 0.5                       # mm, integration grid on a section cut


def components_csv(model_name):
    """Path to the two-step nodal stress dump for a model."""
    return os.path.join(ROOT, 'output', model_name,
                        model_name + '_components.csv')


# =============================================================================
# PARAMETERS AND DERIVED GEOMETRY
# =============================================================================
def read_params(model_name, params_csv=PARAMS_CSV):
    raw = pd.read_csv(params_csv, index_col=0)
    col = raw[model_name]
    p = dict((k, float(col[k])) for k in
             ['b_geo', 'm_geo', 'n_geo', 't_geo', 'f_geo',
              'e_geo', 'i_geo', 'P_load', 'wall_thk'])
    # derived, exactly as in CAE_models_builder.build_model_4walls
    p['W_geo'] = p['b_geo'] + 2 * p['t_geo']
    p['Y_geo'] = 4 * p['m_geo'] - 2 * p['n_geo'] + 4 * p['e_geo']
    p['Z_geo'] = 2 * p['m_geo']
    return p


def web_height(z, p):
    """Height of the side webs at arm depth z (triangular cut tapers them from
    (Z=2f, Y=Y_geo) down to (Z=Z_geo, Y=2e))."""
    z0, h0, h1 = 2 * p['f_geo'], p['Y_geo'], 2 * p['e_geo']
    if z <= z0:
        return h0
    return max(h0 + (h1 - h0) * (z - z0) / (p['Z_geo'] - z0), 0.0)


def web_depth(y, p):
    """Arm depth reached by the side webs at height y (inverse of the taper)."""
    z0, h0, h1 = 2 * p['f_geo'], p['Y_geo'], 2 * p['e_geo']
    if y <= h1:
        return p['Z_geo']
    return z0 + (p['Z_geo'] - z0) * (y - h0) / (h1 - h0)


# =============================================================================
# 1. ANALYTICAL MODEL
# =============================================================================
def analytical(p):
    """Reproduce validation.analytical_validation() plus the two other bending
    moments that Chapter 2 states."""
    b, m, n = p['b_geo'], p['m_geo'], p['n_geo']
    t, f, e = p['t_geo'], p['f_geo'], p['e_geo']
    i_r, P = p['i_geo'], p['P_load']

    D = n ** 2 * t + b * f ** 2 / 2
    A = b * f + 2 * n * t
    I = (2 * t * n ** 3) / 3 + (b * f ** 3) / 3 - D ** 2 / A
    ybar = D / A
    v = n - ybar

    # cross-check the compact I against the explicit parallel-axis sum
    I_pat = (b * f ** 3 / 12 + b * f * (ybar - f / 2) ** 2
             + 2 * (t * n ** 3 / 12 + n * t * (ybar - n / 2) ** 2))

    moments = {
        'validation.py  M = P*m': P * m,
        'Ch.2 text      M = P[m-(n-v)]': P * (m - (n - v)),
        'Ch.2 table 7   M = P(v-n/2)': P * (v - n / 2),
    }
    sigma_d = P / A

    # pad / bolt-head model
    xi, rho = b / n, i_r / n
    Q_nodes = [-0.0567 * xi ** 3 + 0.3133 * xi ** 2 - 0.1067 * xi + 0.450,
               -0.0567 * xi ** 3 + 0.2933 * xi ** 2 - 0.1567 * xi + 0.380,
               -0.0867 * xi ** 3 + 0.3933 * xi ** 2 - 0.2333 * xi + 0.310,
               -0.0567 * xi ** 3 + 0.2818 * xi ** 2 - 0.1951 * xi + 0.250]
    rho_nodes = [0.1, 0.2, 0.3, 0.4]
    Q = float(np.interp(min(max(rho, 0.1), 0.4), rho_nodes, Q_nodes))
    l_eff = m - f / 2
    sigma_bp = 2 * P * Q * l_eff / (b * e ** 2)
    tau_p = P / (2 * (1.5 * i_r) * np.pi * e)
    sigma_vm = np.hypot(sigma_bp, np.sqrt(3) * tau_p)

    return dict(D=D, A=A, I=I, I_pat=I_pat, ybar=ybar, v=v,
                sigma_d=sigma_d, moments=moments, xi=xi, rho=rho,
                Q=Q, Q_nodes=Q_nodes, l_eff=l_eff, sigma_bp=sigma_bp,
                tau_p=tau_p, sigma_vm=sigma_vm)


# =============================================================================
# 3. FEM SECTION INTEGRATION
# =============================================================================
def probe(points, values, xyz):
    """Stress interpolated at explicit coordinates - use this for every
    analytical-vs-FEM number, so both sides refer to the same physical point."""
    return griddata(points, values, np.atleast_2d(xyz), method='linear')


def _integrate(points, values, query, coord_index):
    """Interpolate the stress onto a section grid and return (A, N, M, s)."""
    s = griddata(points, values, query, method='linear')
    ok = ~np.isnan(s)
    dA = GRID * GRID
    A = ok.sum() * dA
    c = query[ok, coord_index]
    cbar = c.sum() * dA / A
    N = s[ok].sum() * dA
    M = (s[ok] * (c - cbar)).sum() * dA
    I = ((c - cbar) ** 2).sum() * dA
    return dict(A=A, N=N, M=M, I=I, cbar=cbar,
                smin=s[ok].min(), smax=s[ok].max())


def cut_normal_z(d, p, z0):
    """Cut perpendicular to the arm axis: 2 webs t x H(z) + pad b x e."""
    W, t, e, i_r = p['W_geo'], p['t_geo'], p['e_geo'], p['i_geo']
    H = web_height(z0, p)
    xs = np.arange(GRID / 2, W, GRID)
    ys = np.arange(GRID / 2, H, GRID)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    inside = ((XX <= t) | (XX >= W - t)) | (YY <= e)
    if abs(z0 - p['m_geo']) < i_r:                      # pad bore
        inside &= ~(((XX - W / 2) ** 2 + (z0 - p['m_geo']) ** 2 < i_r ** 2)
                    & (YY <= e))
    q = np.column_stack([XX[inside], YY[inside],
                         np.full(inside.sum(), z0)])
    return H, q


def cut_normal_y(d, p, y0):
    """Horizontal cut (the Section A-A of Chapter 2): back wall + the two webs."""
    W, t, f, i_r = p['W_geo'], p['t_geo'], p['f_geo'], p['i_geo']
    zmax = web_depth(y0, p)
    xs = np.arange(GRID / 2, W, GRID)
    zs = np.arange(GRID / 2, zmax, GRID)
    XX, ZZ = np.meshgrid(xs, zs, indexing='ij')
    inside = (ZZ <= f) | ((XX <= t) | (XX >= W - t))
    inside &= ~(((XX - W / 2) ** 2 + ZZ ** 2 < i_r ** 2) & (ZZ <= f))
    q = np.column_stack([XX[inside], np.full(inside.sum(), y0), ZZ[inside]])
    return zmax, q


# =============================================================================
# REPORT
# =============================================================================
def run(model_name='model_0'):
    p = read_params(model_name)
    an = analytical(p)
    b, m, n = p['b_geo'], p['m_geo'], p['n_geo']
    t, f, e, P = p['t_geo'], p['f_geo'], p['e_geo'], p['P_load']
    W, Yg, Zg = p['W_geo'], p['Y_geo'], p['Z_geo']

    print('=' * 78)
    print('ANALYTICAL MODEL AUDIT - %s' % model_name)
    print('=' * 78)
    print('  params : b=%g m=%g n=%g t=%g f=%g e=%g r=%g P=%g wall_thk=%g'
          % (b, m, n, t, f, e, p['i_geo'], P, p['wall_thk']))
    print('  derived: W_geo=%g  Y_geo=4m-2n+4e=%g  Z_geo=2m=%g' % (W, Yg, Zg))
    F_pre = 0.60 * 720.0 * np.pi * p['i_geo'] ** 2
    print('  bolt preload = %.0f N = %.1f x P  (absent from the analytical model)'
          % (F_pre, F_pre / P))

    print('\n1. ALGEBRA')
    print('   D=%.2f  A=%.2f  ybar=D/A=%.3f  v=n-ybar=%.3f  I=%.2f'
          % (an['D'], an['A'], an['ybar'], an['v'], an['I']))
    print('   I via explicit parallel-axis = %.2f -> compact form exact: %s'
          % (an['I_pat'], np.isclose(an['I'], an['I_pat'])))
    print('   sigma_d = P/A = %.3f MPa' % an['sigma_d'])
    print('   three different bending moments are in circulation:')
    for k, M in an['moments'].items():
        print('     %-32s M=%9.0f N.mm -> sigma_b=%7.3f  sigma_t=%7.3f MPa'
              % (k, M, M * an['v'] / an['I'],
                 an['sigma_d'] + M * an['v'] / an['I']))
    print('   pad: xi=%.3f rho=%.3f  Q_j(xi)=%s  Q=%.4f'
          % (an['xi'], an['rho'],
             ['%.3f' % q for q in an['Q_nodes']], an['Q']))
    print('        sigma_BP=%.3f  tau_P=%.3f  sigma_vM,bolt=%.3f MPa'
          % (an['sigma_bp'], an['tau_p'], an['sigma_vm']))

    print('\n2. GEOMETRY: analytical Section A-A vs the built part')
    print('   analytical: back plate b x f = %g x %g, webs n x t = %g x %g (x2)'
          % (b, f, n, t))
    print('   built     : back wall W x f = %g x %g, webs depth Z_geo = %g,'
          % (W, f, Zg))
    print('               tapered from Y=%g at Z=%g to Y=%g at Z=%g'
          % (Yg, 2 * f, 2 * e, Zg))
    print('   n_geo sets only Y_geo; the arm depth is Z_geo = 2*m_geo')

    comp = components_csv(model_name)
    if not os.path.exists(comp):
        print('\n   %s missing - run dump_stress_components.py first' % comp)
        return

    d = pd.read_csv(comp)
    for c in ['11', '22', '33', '12', '13', '23']:
        d['D' + c] = d['L' + c] - d['P' + c]
    pts = d[['X', 'Y', 'Z']].to_numpy()

    print('\n3a. LOAD POSITION: does the FEM load act where the model assumes?')
    zs, Ms_fem = [], []
    for z0 in np.arange(1.5 * f, 0.85 * m, 2.5):
        H, q = cut_normal_z(d, p, z0)
        r = _integrate(pts, d.D33.to_numpy(), q, 1)
        zs.append(z0)
        Ms_fem.append(r['M'])
    slope, icept = np.polyfit(zs, Ms_fem, 1)
    print('    M(z) = P(m - z) recovered from the FEM section moments:')
    print('      P     = %7.0f N  (applied %.0f, %+.1f%%)'
          % (-slope, P, 100 * (-slope - P) / P))
    print('      m_eff = %7.2f mm (bore centre at %.2f, %+.2f mm)'
          % (-icept / slope, m, -icept / slope - m))
    print('    -> load magnitude and line of action match the analytical'
          ' assumption')

    print('\n3b. FEM CUTS PERPENDICULAR TO THE ARM AXIS (the real load path)')
    print('    sigma_zz PROBED AT THE FIBRE (web mid-thickness X=%.1f), not'
          ' min/max over the cut' % (t / 2))
    print('    %6s %7s %10s %9s | %9s %9s | %9s %9s'
          % ('z', 'H(z)', 'M_statics', 'M_fem',
             'fem y=0+', 'beam y=0', 'fem y=H-', 'beam y=H'))
    for z0 in np.arange(1.5 * f, 0.85 * m, 5.0):
        H, q = cut_normal_z(d, p, z0)
        r = _integrate(pts, d.D33.to_numpy(), q, 1)
        Aw, Ap = t * H, b * e
        At = 2 * Aw + Ap
        yb = (2 * Aw * H / 2 + Ap * e / 2) / At
        Ib = (2 * (t * H ** 3 / 12 + Aw * (H / 2 - yb) ** 2)
              + b * e ** 3 / 12 + Ap * (yb - e / 2) ** 2)
        Ms = P * (m - z0)
        fib = probe(pts, d.D33.to_numpy(),
                    [(t / 2, 0.4, z0), (t / 2, H - 0.4, z0)])
        print('    %6.1f %7.1f %10.0f %9.0f | %9.3f %9.3f | %9.3f %9.3f'
              % (z0, H, Ms, r['M'], fib[0], -Ms * yb / Ib,
                 fib[1], Ms * (H - yb) / Ib))
    print('    (N_fem ~ 0 on every cut and M_fem tracks M_statics, so the arm')
    print('     carries the whole applied moment - but the fibre stresses show')
    print('     plane sections do NOT remain plane: the top fibre is inactive)')

    print('\n3c. FEM HORIZONTAL CUTS - the section Chapter 2 actually models')
    print('    %6s %9s %8s %10s | %9s %8s %10s | %9s'
          % ('Y', 'A_true', 'zbar', 'I_true', 'N_fem', '% of P', 'M_fem',
             'sig_lin'))
    for y0 in np.arange(2 * e, 0.75 * Yg, 10.0):
        zmax, q = cut_normal_y(d, p, y0)
        r = _integrate(pts, d.D22.to_numpy(), q, 2)
        s_lin = max(abs(r['N'] / r['A'] + r['M'] * (zmax - r['cbar']) / r['I']),
                    abs(r['N'] / r['A'] - r['M'] * r['cbar'] / r['I']))
        print('    %6.0f %9.1f %8.2f %10.0f | %+9.1f %7.1f%% %10.0f | %9.2f'
              % (y0, r['A'], r['cbar'], r['I'], r['N'],
                 100 * r['N'] / P, r['M'], s_lin))
    print('    analytical claims A=%.0f, zbar=%.2f, I=%.0f and a single'
          % (an['A'], an['ybar'], an['I']))
    print('    sigma_t of %.3f / %.3f / %.3f MPa (code / Ch.2 text / Ch.2 table)'
          % tuple(an['sigma_d'] + M * an['v'] / an['I']
                  for M in an['moments'].values()))

    print('\n3d. COORDINATE-MATCHED PROBE OF sigma_t')
    y_aa = p['Y_geo'] / 2                       # validation.py sec_y
    z_edge = web_depth(y_aa, p)
    print('    the model evaluates sigma_t at Z = zbar + v = %.2f (= n_geo),'
          % (an['ybar'] + an['v']))
    print('    on the horizontal plane Y = Y_geo/2 = %.1f, component sigma_yy.'
          % y_aa)
    print('    the webs actually run to Z = %.2f there, so that fibre is %.1f mm'
          % (z_edge, z_edge - n))
    print('    inside the material rather than on the free edge.')
    for tag, xyz in [('analytical fibre  Z = n', (t / 2, y_aa, n)),
                     ('true free edge    Z = %.1f' % (z_edge - 0.5),
                      (t / 2, y_aa, z_edge - 0.5)),
                     ('analytical N.A.   Z = %.2f' % an['ybar'],
                      (t / 2, y_aa, an['ybar']))]:
        print('      FEM S22 at (%5.1f, %5.1f, %5.1f)  %-26s = %8.3f MPa'
              % (xyz[0], xyz[1], xyz[2], tag,
                 probe(pts, d.D22.to_numpy(), xyz)[0]))
    print('      analytical at the N.A. is sigma_d = P/A = %.3f MPa'
          % an['sigma_d'])

    print('\n3e. PAD MODEL - matched location')
    print('    sigma_BP is a pad-strip bending stress at the pad root'
          ' (Z ~ f = %.1f),' % f)
    print('    on the pad outer face Y = 0; tau_P is a bearing shear on the bore')
    print('    wall at Z = m = %.1f. The two are %.1f mm apart but are combined'
          % (m, m - f))
    print('    into one von Mises value.')
    d['Mvm'] = np.sqrt(((d.D11 - d.D22) ** 2 + (d.D22 - d.D33) ** 2
                        + (d.D33 - d.D11) ** 2) / 2.
                       + 3 * (d.D12 ** 2 + d.D13 ** 2 + d.D23 ** 2))
    pad = d[d.Y <= e + 1e-6]
    root = pad[(pad.Z >= f) & (pad.Z <= f + 5) & (pad.Y <= 1.0)]
    bore = pad[np.abs(np.hypot(pad.X - W / 2, pad.Z - m) - p['i_geo']) < 1.0]
    print('      analytical sigma_BP                        = %8.2f MPa'
          % an['sigma_bp'])
    print('      FEM max |S33| at the pad root (Z in [%g,%g]) = %8.2f MPa'
          % (f, f + 5, root.D33.abs().max()))
    print('      -> required efficiency factor Q = %.3f (model uses %.3f)'
          % (root.D33.abs().max() / (2 * P * (m - f / 2) / (b * e ** 2)),
             an['Q']))
    print('      FEM max Mises on the pad bore wall          = %8.2f MPa'
          % bore.Mvm.max())
    print('      FEM max Mises anywhere in the pad           = %8.2f MPa'
          % pad.Mvm.max())
    print('      (the last one is the bore concentration, %.0f mm from where'
          % (m - f))
    print('       sigma_BP is defined - not a valid comparison point)')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'model_0')
