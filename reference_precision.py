"""Independent high-precision Lane--Emden reference on [0, 1] for n = 0.5, 1.5, 2, 3, 4.

For z = x*x write y = sum a[k] z**k and y**n = sum b[k] z**k.  The ODE gives
    a[k+1] = -b[k] / (2*(k+1)*(2*k+3)),
and the identity y*(y**n)' = n*y**n*y' gives
    b[k] = sum(((n+1)*j - k) * a[j] * b[k-j], j = 1..k) / k.
This origin-centred series satisfies y(0) = 1 and y'(0) = 0 exactly (no finite-x start-up error).
It is evaluated with two precision/order pairs, (40 digits, 80 terms in z, i.e. order 160 in x) and
(60 digits, 120 terms, order 240), whose difference and the 60-digit ODE residual are the
convergence checks (empirical checks, not a rigorous error bound).  The 60-digit series is compared
with independent SciPy DOP853 and Radau solves started at x0 = 1e-5 and with the original DOP853
reference of the manuscript (the _ref_sol of src/n*.py).

Usage:  python reference_precision.py            ->  reference_precision.npz / reference_precision.json
The archive is read by src/verified_reference.py (opt-in via BFLNN_REFERENCE_ARCHIVE) and by the
reference check of src/seeds30_mono_lr.py.
"""
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
SRC = HERE / 'src'
CASES = {0.5: ('n05', 'no.5'), 1.5: ('n15', 'n1.5'), 2: ('n2', 'n2'), 3: ('n3', 'n3'), 4: ('n4', 'n4')}
LOW, HIGH = (40, 80), (60, 120)           # (decimal digits, number of terms in z = x**2)
TOL_DIFF, TOL_RESIDUAL = 1e-30, 1e-50     # thresholds of the convergence checks


def coefficients(n, terms, dps):
    with mp.workdps(dps):
        nn = mp.mpf(n)
        a, b = [mp.mpf(1)], [mp.mpf(1)]
        for k in range(terms):
            a.append(-b[k] / (2 * (k + 1) * (2 * k + 3)))
            ell = k + 1
            b.append(mp.fsum(((nn + 1) * j - ell) * a[j] * b[ell - j] for j in range(1, ell + 1)) / ell)
        return a


def evaluate(coeff, xs, dps):
    """y, y', y'' of the series at the float64 points xs (converted exactly)."""
    with mp.workdps(dps):
        a = list(reversed(coeff))
        da = [2 * k * coeff[k] for k in range(len(coeff) - 1, 0, -1)]
        d2a = [2 * k * (2 * k - 1) * coeff[k] for k in range(len(coeff) - 1, 0, -1)]
        out = []
        for xx in xs:
            v = mp.mpf(float(xx)); z = v * v
            out.append((mp.polyval(a, z), v * mp.polyval(da, z), mp.polyval(d2a, z)))
        return out


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'), SRC / (fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRC))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def ode_errors(method, n, x, y_ref, dy_ref, y0, dy0):
    rhs = lambda t, s: [s[1], -2.0 / t * s[1] - max(s[0], 0.0) ** n]
    sol = solve_ivp(rhs, [x[0], 1.0], [y0, dy0], method=method, t_eval=x, rtol=2.3e-14, atol=1e-15)
    err, derr = np.abs(sol.y[0] - y_ref), np.abs(sol.y[1] - dy_ref)
    return sol.y[0], {'rtol': 2.3e-14, 'atol': 1e-15, 'nfev': int(sol.nfev),
                      'max_absolute_error': float(err.max()), 'mean_absolute_error': float(err.mean()),
                      'derivative': {'max_absolute_error': float(derr.max()), 'mean_absolute_error': float(derr.mean())}}


def main():
    t0 = time.time()
    x = np.linspace(1e-5, 1.0, 500)
    x11 = np.round(np.linspace(0.0, 1.0, 11), 1)
    arrays = {'x': x, 'x11': x11}
    cases, all_pass = [], True
    for n, (prefix, fname) in CASES.items():
        c_low, c_high = coefficients(n, LOW[1], LOW[0]), coefficients(n, HIGH[1], HIGH[0])
        high, low = evaluate(c_high, x, HIGH[0]), evaluate(c_low, x, HIGH[0])
        with mp.workdps(HIGH[0]):
            diff = max(abs(h[0] - l[0]) for h, l in zip(high, low))
            nn = mp.mpf(n)
            resid = max(abs(h[2] + 2 * h[1] / mp.mpf(float(xx)) + mp.power(h[0], nn)) for h, xx in zip(high, x))
            rounding = [abs(mp.mpf(float(h[0])) - h[0]) for h in high]
        passed = bool(diff < TOL_DIFF and resid < TOL_RESIDUAL)
        all_pass &= passed
        y, dy, d2y = (np.array([float(h[i]) for h in high]) for i in range(3))
        h11 = evaluate(c_high, x11, HIGH[0])
        arrays.update({
            f'{prefix}_y': y, f'{prefix}_dy': dy, f'{prefix}_d2y': d2y,
            f'{prefix}_y_decimal60': np.array([mp.nstr(h[0], 60) for h in high]),
            f'{prefix}_coeff_decimal60': np.array([mp.nstr(c, 60) for c in c_high]),
            f'{prefix}_y11': np.array([float(h[0]) for h in h11]), f'{prefix}_dy11': np.array([float(h[1]) for h in h11]),
            f'{prefix}_d2y11': np.array([float(h[2]) for h in h11]),
            f'{prefix}_y11_decimal60': np.array([mp.nstr(h[0], 60) for h in h11])})
        y0, dy0 = y[0], dy[0]
        arrays[f'{prefix}_dop853'], dop = ode_errors('DOP853', n, x, y, dy, y0, dy0)
        arrays[f'{prefix}_radau'], rad = ode_errors('Radau', n, x, y, dy, y0, dy0)
        mod = load_module(fname)
        mod._build_reference()
        paper = mod._ref_sol.sol(x)[0]
        arrays[f'{prefix}_paper_reference'] = paper
        pe = np.abs(paper - y)
        cases.append({'n': n, 'precision_low_dps': LOW[0], 'precision_high_dps': HIGH[0],
                      'order_low_in_x': 2 * LOW[1], 'order_high_in_x': 2 * HIGH[1],
                      'max_40dps80terms_vs_60dps120terms': mp.nstr(diff, 12), 'max_highprecision_ode_residual': mp.nstr(resid, 12),
                      'convergence_pass': passed, 'min_y': float(y.min()), 'DOP853': dop, 'Radau': rad,
                      'paper_reference': {'max_absolute_error': float(pe.max()), 'mean_absolute_error': float(pe.mean())},
                      'float64_rounding_floor': {'max_absolute_error': float(max(rounding)),
                                                 'mean_absolute_error': float(sum(rounding) / len(rounding))}})
        print(f'n = {n}: |40-vs-60 digits| = {mp.nstr(diff, 3)}, residual = {mp.nstr(resid, 3)}, pass = {passed}, '
              f'paper DOP853 reference error = {pe.max():.2e}', flush=True)
    np.savez(HERE / 'reference_precision.npz', **arrays)
    report = {'method': __doc__.strip(), 'grid_points': 500, 'x_min': 1e-5, 'x_max': 1.0,
              'convergence_thresholds': {'precision_pair_difference': TOL_DIFF, 'ode_residual': TOL_RESIDUAL},
              'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': {
                  'numpy': np.__version__, 'scipy': scipy.__version__, 'mpmath': mp.__version__}},
              'cases': cases, 'elapsed_seconds': time.time() - t0, 'all_convergence_checks_pass': bool(all_pass)}
    (HERE / 'reference_precision.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('all convergence checks pass:', all_pass)


if __name__ == '__main__':
    main()
