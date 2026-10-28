# effect of column scaling: none / sup-norm / derivative-aware (W2,inf)

import contextlib
import importlib.util
import io
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import numpy as np
from scipy.integrate import solve_ivp

HERE    = os.path.dirname(os.path.abspath(__file__))
FILES   = {0: 'n0', 0.5: 'no.5', 1: 'n1', 1.5: 'n1.5', 2: 'n2', 3: 'n3', 4: 'n4', 5: 'n5'}
BASES   = ['bernoulli', 'chebyshev', 'legendre']
LABEL   = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
SCHEMES = ['none', 'sup', 'sobolev']
DEGREE, X_MAX, N_SEEDS = 5, 1.0, 5
N_TRAIN, N_VAL, N_EVAL = 30, 105, 500


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'),
                                                  os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(n, x):
    if n == 0: return 1.0 - x**2 / 6.0
    if n == 1: return np.sin(x) / x
    if n == 5: return 1.0 / np.sqrt(1.0 + x**2 / 3.0)
    x0  = 1e-7
    rhs = lambda t, s: [s[1], -2.0 / t * s[1] - max(s[0], 0.0)**n]
    sol = solve_ivp(rhs, [x0, X_MAX + 0.05], [1 - x0**2 / 6, -x0 / 3],
                    method='DOP853', dense_output=True, rtol=1e-13, atol=1e-15)
    return sol.sol(x)[0]


def make_column_norms(mod, scheme):
    """Return a replacement for mod.get_column_norms implementing `scheme`."""
    cache = {}
    def get_column_norms(name, degree, x_max=1.0, n_ref=10_000):
        key = (name, degree, x_max)
        if key not in cache:
            x_ref = np.linspace(1e-6, x_max, n_ref)
            P, dP, d2P = mod._get_basis_raw(name, x_ref, degree, x_max)
            if scheme == 'none':
                c = np.ones(P.shape[1])
            elif scheme == 'sup':
                c = np.abs(P).max(axis=0)
            else:
                c = np.max([np.abs(P).max(axis=0), np.abs(dP).max(axis=0),
                            np.abs(d2P).max(axis=0)], axis=0)
            cache[key] = np.maximum(c, 1e-12)
        return cache[key]
    return get_column_norms


def run_one_n(item):
    n, fname = item
    mod   = load_module(fname)
    x_tr  = np.linspace(1e-5, X_MAX, N_TRAIN)
    half  = X_MAX / (2.0 * N_VAL)
    x_val = np.linspace(1e-5 + half, X_MAX - half / 2.0, N_VAL)
    x_ev  = np.linspace(1e-5, X_MAX, N_EVAL)
    y_ev  = reference(n, x_ev)
    rows  = []
    for scheme in SCHEMES:
        mod.get_column_norms = make_column_norms(mod, scheme)
        for b in BASES:
            Pe, dPe, d2Pe = mod.get_basis(b, x_ev, DEGREE, X_MAX)
            maes = []
            for seed in range(N_SEEDS):
                with contextlib.redirect_stdout(io.StringIO()):
                    W = mod.train(b, DEGREE, x_tr, x_val=x_val, x_max=X_MAX, seed=seed)[0]
                y, _, _ = mod.forward(x_ev, W, Pe, dPe, d2Pe)
                maes.append(np.mean(np.abs(y - y_ev)))
            rows.append((n, scheme, b, np.mean(maes), np.std(maes)))
    return rows


if __name__ == '__main__':
    with Pool(min(len(FILES), os.cpu_count() or 1, WORKERS)) as pool:
        results = [r for rows in pool.map(run_one_n, FILES.items()) for r in rows]

    table = {(n, s, b): (m, sd) for n, s, b, m, sd in results}
    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    out('MAE (mean +/- std over 5 seeds, M=500 grid) for each column-scaling scheme')
    out(f"{'n':>4}  {'Net':6s}" + ''.join(f"  {s:>22s}" for s in SCHEMES))
    for n in FILES:
        for b in BASES:
            cells = ''.join(f"  {table[(n, s, b)][0]:10.2e} +/- {table[(n, s, b)][1]:8.1e}"
                            for s in SCHEMES)
            out(f"{n:>4}  {LABEL[b]:6s}{cells}")
        out()

    out('Best basis (lowest mean MAE) per n under each scheme:')
    for s in SCHEMES:
        winners = [LABEL[min(BASES, key=lambda b: table[(n, s, b)][0])] for n in FILES]
        out(f"  {s:8s}: " + '  '.join(f"n={n}:{w}" for n, w in zip(FILES, winners)))

    with open(os.path.join(HERE, 'norm_test_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
