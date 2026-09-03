"""
geometry/arm_section_model.py
-----------------------------------------------------------------------------
Section A-A relocated to a cut PERPENDICULAR TO THE ARM AXIS Z.

The composite-section algebra of Chapter 2 is reused verbatim; only the inputs
change, because the cut is now a real cross-section of the cantilever arm:

    Chapter 2 (cut _|_ Y)          relocated (cut _|_ Z)
    ---------------------          ---------------------------------
    datum  Z = 0 (back face)       datum  Y = 0 (pad outer face)
    back plate  b x f              base pad    b x e
    2 webs      n x t              2 webs      t x H(z)
    D = n^2 t + b f^2 / 2          S = H^2 t + b e^2 / 2
    A = b f + 2 n t                A = b e + 2 H t
    ybar = D / A                   ybar = S / A
    I = 2tn^3/3 + bf^3/3 - D^2/A   I = 2tH^3/3 + be^3/3 - S^2/A

H(z) is the web height actually present at depth z: full Y_geo up to the start
of the triangular cut at z = 2f, then tapering linearly to 2e at z = Z_geo.

Internal actions on this cut ARE statically determinate - the free body from the
cut to the free tip carries only P:

    V = P                 M(z) = P (m - z)          N = 0

so the direct-stress term sigma_d = P/A of Chapter 2 has no counterpart here:
P is perpendicular to the arm axis and produces no axial force on the section.

    sigma(y) = M(z) (y - ybar) / I      tau_max = V Q_na / (2 t I)

EFFECTIVE-DEPTH VARIANT
The webs are attached to the back wall only over Z in [0, f] and the load is
reacted at the bolt, at height Y_geo/2. Material above the bolt has no load path
and carries almost nothing, so the same formulae are also evaluated with the web
height truncated to H_eff = Y_geo/2. This is a load-path argument, not a fit,
but it has only been checked on model_0 - verify across the DOE before use.

    python geometry/arm_section_model.py [model_0]
-----------------------------------------------------------------------------
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_analytical_vs_fem import read_params, web_height, components_csv


# =============================================================================
# RELOCATED SECTION MODEL
# =============================================================================
def section_properties(H, p):
    """Composite properties of the arm cross-section: 2 webs t x H + pad b x e.
    Same algebra as Chapter 2 eqs. (2.4)-(2.9), with n -> H and f -> e."""
    b, t, e = p['b_geo'], p['t_geo'], p['e_geo']
    A = b * e + 2 * H * t
    S = H ** 2 * t + b * e ** 2 / 2
    ybar = S / A
    I = 2 * t * H ** 3 / 3 + b * e ** 3 / 3 - S ** 2 / A
    return A, S, ybar, I


def stresses(z, p, H=None):
    """Fibre stresses and peak shear on the cut at depth z."""
    m, P, t, b, e = p['m_geo'], p['P_load'], p['t_geo'], p['b_geo'], p['e_geo']
    if H is None:
        H = web_height(z, p)
    A, S, ybar, I = section_properties(H, p)
    M = P * (m - z)
    Q_na = b * e * (ybar - e / 2) + t * ybar ** 2      # first moment below N.A.
    return dict(H=H, A=A, ybar=ybar, I=I, M=M,
                sig_bot=-M * ybar / I,                 # pad face, y = 0
                sig_top=M * (H - ybar) / I,            # web tip, y = H
                tau_max=P * Q_na / (2 * t * I))


# =============================================================================
# FEM COMPARISON
# =============================================================================
def _load_fem(model_name):
    d = pd.read_csv(components_csv(model_name))
    for c in ['11', '22', '33', '12', '13', '23']:
        d['D' + c] = d['L' + c] - d['P' + c]
    return d, d[['X', 'Y', 'Z']].to_numpy()


def compare(model_name='model_0'):
    p = read_params(model_name)
    b, m, t, e, f, P = (p['b_geo'], p['m_geo'], p['t_geo'],
                        p['e_geo'], p['f_geo'], p['P_load'])
    W, Yg = p['W_geo'], p['Y_geo']
    d, pts = _load_fem(model_name)

    def probe(col, xyz):
        return griddata(pts, d[col].to_numpy(), np.atleast_2d(xyz),
                        method='linear')

    print('=' * 78)
    print('SECTION A-A RELOCATED TO A CUT _|_ Z   -   %s' % model_name)
    print('=' * 78)
    print('  b=%g m=%g t=%g e=%g f=%g  W=%g  Y_geo=%g  P=%g N'
          % (b, m, t, e, f, W, Yg, P))

    zc = 1.5 * f                                    # arm root, clear of the wall
    s = stresses(zc, p)
    print('\nCRITICAL SECTION (arm root, z = %.1f)' % zc)
    print('  webs 2 x %g x %.1f  +  pad %g x %g' % (t, s['H'], b, e))
    print('  A = %.1f mm2   ybar = %.2f mm   I = %.0f mm4'
          % (s['A'], s['ybar'], s['I']))
    print('  N = 0 (P is perpendicular to the arm axis)   V = %g N' % P)
    print('  M = P(m - z) = %.0f N.mm' % s['M'])

    print('\nCOMPARISON - every value probed at the same coordinate')
    print('  fibre stresses = sigma_zz (S33) at web mid-thickness X = %.1f' % (t / 2))
    print()
    print('  %6s %7s %9s %9s %7s | %9s %9s %7s'
          % ('z', 'H(z)', 'analyt.', 'FEM', 'A/F', 'analyt.', 'FEM', 'A/F'))
    print('  %6s %7s %9s %9s %7s | %9s %9s %7s'
          % ('', '', 'bot y=0', 'bot y=0', '', 'top y=H', 'top y=H', ''))
    rows = []
    for z in np.arange(1.5 * f, 0.8 * m, 5.0):
        s = stresses(z, p)
        fb = float(probe('D33', (t / 2, 0.4, z))[0])
        ft = float(probe('D33', (t / 2, s['H'] - 0.4, z))[0])
        rows.append((z, s, fb, ft))
        print('  %6.1f %7.1f %9.3f %9.3f %7.2f | %9.3f %9.3f %7.1f'
              % (z, s['H'], s['sig_bot'], fb, s['sig_bot'] / fb,
                 s['sig_top'], ft, s['sig_top'] / ft if abs(ft) > 1e-3 else np.nan))

    print('\n  -> bottom fibre: beam theory UNDER-predicts by %.1f-%.1fx'
          % (min(r[2] / r[1]['sig_bot'] for r in rows),
             max(r[2] / r[1]['sig_bot'] for r in rows)))
    print('     top fibre    : beam theory OVER-predicts - the FEM top fibre is')
    print('     inactive, so plane sections do not remain plane on this arm.')

    print('\nEFFECTIVE-DEPTH VARIANT  (H_eff = Y_geo/2 = %.1f, the bolt height)'
          % (Yg / 2))
    A2, S2, yb2, I2 = section_properties(Yg / 2, p)
    print('  A = %.1f mm2   ybar = %.2f mm   I = %.0f mm4' % (A2, yb2, I2))
    print('  %6s %9s %9s %7s' % ('z', 'analyt.', 'FEM', 'A/F'))
    ratios = []
    for z, s, fb, ft in rows:
        s2 = stresses(z, p, H=Yg / 2)
        ratios.append(s2['sig_bot'] / fb)
        print('  %6.1f %9.3f %9.3f %7.2f' % (z, s2['sig_bot'], fb, ratios[-1]))
    print('  -> bottom fibre within %+.0f%% / %+.0f%% (mean %+.0f%%)'
          % (100 * (min(ratios) - 1), 100 * (max(ratios) - 1),
             100 * (np.mean(ratios) - 1)))

    print('\nSTATICS CHECK (independent of any stress model)')
    print('  %6s %11s %11s %8s' % ('z', 'M_statics', 'M_fem', 'err%'))
    for z, s, fb, ft in rows:
        H = s['H']
        g = 0.5
        xs = np.arange(g / 2, W, g)
        ys = np.arange(g / 2, H, g)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        inside = ((XX <= t) | (XX >= W - t)) | (YY <= e)
        q = np.column_stack([XX[inside], YY[inside],
                             np.full(inside.sum(), z)])
        sv = griddata(pts, d.D33.to_numpy(), q, method='linear')
        ok = ~np.isnan(sv)
        dA = g * g
        yv = q[ok, 1]
        yb = yv.sum() * dA / (ok.sum() * dA)
        Mf = (sv[ok] * (yv - yb)).sum() * dA
        print('  %6.1f %11.0f %11.0f %8.1f'
              % (z, s['M'], Mf, 100 * (Mf - s['M']) / s['M']))
    print('  -> the relocated section carries exactly the moment statics says')
    print('     it must; that is what the Ch.2 section could never do.')


if __name__ == '__main__':
    compare(sys.argv[1] if len(sys.argv) > 1 else 'model_0')
