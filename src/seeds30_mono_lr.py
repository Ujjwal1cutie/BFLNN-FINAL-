# reference check (DOP853 vs Radau vs mpmath), 30 seeds + Mann-Whitney,
# monomial basis, residual and y', y'' errors, learning rate sweep

import contextlib
import importlib.util
import io
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import mannwhitneyu

HERE  = os.path.dirname(os.path.abspath(__file__))
FILES = {0: ('n0', 'exact_n0'), 0.5: ('no.5', 'reference_n05'), 1: ('n1', 'exact_n1'),
         1.5: ('n1.5', 'reference_n15'), 2: ('n2', 'reference_n2'), 3: ('n3', 'reference_n3'),
         4: ('n4', 'reference_n4'), 5: ('n5', 'exact_n5')}
BASES  = ['bernoulli', 'chebyshev', 'legendre', 'monomial']
LABEL  = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN', 'monomial': 'MonoNN'}
N_SEEDS_B = 30
LR_LIST   = [1e-3, 3e-4, 1e-4, 3e-5]
N_SEEDS_C = 5


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'),
                                                  os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original = mod._get_basis_raw

    def basis_raw(name, x_phys, degree, x_max=1.0):
        if name != 'monomial':
            return original(name, x_phys, degree, x_max)
        t = x_phys / x_max
        k = np.arange(degree + 1)
        P   = t[:, None] ** k
        dP  = np.where(k >= 1, k * t[:, None] ** np.maximum(k - 1, 0), 0.0)
        d2P = np.where(k >= 2, k * (k - 1) * t[:, None] ** np.maximum(k - 2, 0), 0.0)
        return P, dP / x_max, d2P / x_max**2
    mod._get_basis_raw = basis_raw
    return mod


def grids(mod):
    X = mod.X_MAX
    half = X / (2.0 * mod.N_VAL)
    return (mod.uniform_nodes(mod.N_TRAIN, X),
            np.linspace(1e-5 + half, X - half / 2.0, mod.N_VAL),
            np.linspace(1e-5, X, 500))


# ── reference solution with derivatives (DOP853, tight tolerances) ────────────
def reference_with_derivatives(n, x):
    import verified_reference
    if verified_reference.enabled():
        return verified_reference.reference_values(n, x, derivatives=True)
    x0 = 1e-7
    y0 = [1 - x0**2 / 6 + n * x0**4 / 120, -x0 / 3 + n * x0**3 / 30]
    rhs = lambda t, s: [s[1], -2 / t * s[1] - np.sign(s[0]) * abs(s[0])**n]
    sol = solve_ivp(rhs, [x0, x.max() + 0.01], y0, method='DOP853', dense_output=True,
                    rtol=1e-14, atol=1e-16)
    y, dy = sol.sol(x)
    d2y = -2 / x * dy - (np.ones_like(y) if n == 0 else y**n)
    return y, dy, d2y


# ── A. reference verification ─────────────────────────────────────────────────
def verify_reference():
    import json
    from pathlib import Path
    # The independently generated series is checked at two precisions/orders;
    # unlike low-order finite-x Taylor initialization it has no startup error.
    archive_path = Path(os.environ.get('BFLNN_REFERENCE_ARCHIVE',
                                       str(Path(HERE).parent / 'reference_precision.npz')))
    report = json.loads(archive_path.with_suffix('.json').read_text(encoding='utf-8'))
    if not report.get('all_convergence_checks_pass', False):
        raise ValueError('Independent origin-series reference convergence checks failed')
    with np.load(archive_path, allow_pickle=False) as archive:
        x_chk = archive['x11'][1:].copy()
        prefix = {0.5: 'n05', 1.5: 'n15', 2: 'n2', 3: 'n3', 4: 'n4'}
        independent = {n: archive[key + '_y11'][1:].copy() for n, key in prefix.items()}
    lines = ['A. Reliability of the original DOP853 reference: max |difference| on x in [0.1, 1] (10 points)',
             '   Independent origin series: 60 digits/order 240, checked against 40 digits/order 160.',
             f'   Reference archive: {archive_path.name}',
             f"  {'n':>4}  {'DOP853 1e-14':>13}  {'Radau':>11}  {'series 60-digit':>16}"]
    for n in [0.5, 1.5, 2, 3, 4]:
        mod = load_module(FILES[n][0])
        # Bypass the optional accurate-reference wrapper for this diagnostic:
        # the subject being tested here is the paper's original DOP853 solve.
        mod._build_reference()
        y_paper = mod._ref_sol.sol(x_chk)[0]
        x0 = 1e-7
        y0 = [1 - x0**2 / 6 + n * x0**4 / 120, -x0 / 3 + n * x0**3 / 30]
        rhs = lambda t, s: [s[1], -2 / t * s[1] - np.sign(s[0]) * abs(s[0])**n]
        d14 = solve_ivp(rhs, [x0, 1.01], y0, method='DOP853', dense_output=True,
                        rtol=1e-14, atol=1e-14).sol(x_chk)[0]
        rad = solve_ivp(rhs, [x0, 1.01], y0, method='Radau', dense_output=True,
                        rtol=1e-12, atol=1e-14).sol(x_chk)[0]
        ymp = independent[n]
        lines.append(f"  {n:>4}  {np.abs(y_paper - d14).max():13.1e}  {np.abs(y_paper - rad).max():11.1e}"
                     f"  {np.abs(y_paper - ymp).max():16.1e}")
    return lines


# ── B. 30 seeds, monomial baseline, residual and derivative errors ────────────
def run_seeds(item):
    n, (fname, ref_name) = item
    mod = load_module(fname)
    x_tr, x_val, x_ev = grids(mod)
    y_ref = getattr(mod, ref_name)(x_ev)
    _, dy_ref, d2y_ref = reference_with_derivatives(n, x_ev)
    out = {}
    for b in BASES:
        Pe, dPe, d2Pe = mod.get_basis(b, x_ev, mod.DEGREE, mod.X_MAX)
        rows = []
        for seed in range(N_SEEDS_B):
            with contextlib.redirect_stdout(io.StringIO()):
                W = mod.train(b, mod.DEGREE, x_tr, x_val=x_val, x_max=mod.X_MAX, seed=seed)[0]
            y, dy, d2y = mod.forward(x_ev, W, Pe, dPe, d2Pe)
            yn = np.ones_like(y) if n == 0 else np.maximum(y, 1e-12)**n
            r = d2y + 2 / x_ev * dy + yn
            rows.append((np.mean(np.abs(y - y_ref)), np.abs(r).max(),
                         np.abs(dy - dy_ref).max(), np.abs(d2y - d2y_ref).max()))
        out[b] = np.array(rows)
    return n, out


# ── C. learning-rate sensitivity ──────────────────────────────────────────────
def run_lr(item):
    n, lr = item
    fname, ref_name = FILES[n]
    mod = load_module(fname)
    x_tr, x_val, x_ev = grids(mod)
    y_ref = getattr(mod, ref_name)(x_ev)
    res = {}
    for b in BASES:
        Pe, dPe, d2Pe = mod.get_basis(b, x_ev, mod.DEGREE, mod.X_MAX)
        maes = []
        for seed in range(N_SEEDS_C):
            with contextlib.redirect_stdout(io.StringIO()):
                W = mod.train(b, mod.DEGREE, x_tr, x_val=x_val, x_max=mod.X_MAX, lr=lr, seed=seed)[0]
            maes.append(np.mean(np.abs(mod.forward(x_ev, W, Pe, dPe, d2Pe)[0] - y_ref)))
        res[b] = (np.mean(maes), np.std(maes))
    return n, lr, res


if __name__ == '__main__':
    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    for l in verify_reference():
        out(l)
    out()

    with Pool(min(8, os.cpu_count() or 1, WORKERS)) as pool:
        seeds = dict(pool.map(run_seeds, FILES.items()))
        lrs = pool.map(run_lr, [(n, lr) for n in [1, 3] for lr in LR_LIST])

    out(f'B1. MAE over {N_SEEDS_B} seeds (M = 500 grid)')
    out(f"  {'n':>4} {'Net':7s} {'mean':>10} {'std':>9} {'median':>10} {'IQR (25%..75%)':>22}")
    for n in FILES:
        for b in BASES:
            m = seeds[n][b][:, 0]
            q1, q3 = np.percentile(m, [25, 75])
            out(f"  {n:>4} {LABEL[b]:7s} {m.mean():10.2e} {m.std():9.1e} {np.median(m):10.2e}"
                f"   {q1:9.2e}..{q3:9.2e}")
        out()

    out('B2. Two-sided Mann-Whitney U tests on the 30 MAE values (raw p-values)')
    out(f"  {'n':>4}  {'BFLNN vs LNN':>13}  {'BFLNN vs ChNN':>14}  {'BFLNN vs MonoNN':>16}  {'better median':>14}")
    OTHERS = ['legendre', 'chebyshev', 'monomial']
    raw_p = {}
    for n in FILES:
        mb = seeds[n]['bernoulli'][:, 0]
        ps = [mannwhitneyu(mb, seeds[n][o][:, 0]).pvalue for o in OTHERS]
        for o, p in zip(OTHERS, ps):
            raw_p[(n, o)] = float(p)
        best = LABEL[min(BASES, key=lambda b: np.median(seeds[n][b][:, 0]))]
        out(f"  {n:>4}  {ps[0]:13.2e}  {ps[1]:14.2e}  {ps[2]:16.2e}  {best:>14}")
    out()

    # Holm-Bonferroni correction over all 24 planned comparisons (8 indices x 3 opponents)
    order = sorted(raw_p, key=raw_p.get)
    holm_p, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * raw_p[key]))
        holm_p[key] = running
    out(f'B2b. Holm-adjusted p-values over all {len(order)} comparisons (alpha = 0.05)')
    out(f"  {'n':>4}  {'raw p (LNN)':>12}  {'Holm p (LNN)':>13}  {'significant':>12}  {'max Holm p (ChNN, MonoNN)':>26}")
    for n in FILES:
        sig = 'yes' if holm_p[(n, 'legendre')] < 0.05 else 'no'
        out(f"  {n:>4}  {raw_p[(n, 'legendre')]:12.3e}  {holm_p[(n, 'legendre')]:13.3e}  {sig:>12}  "
            f"{max(holm_p[(n, 'chebyshev')], holm_p[(n, 'monomial')]):26.2e}")
    out()
    import json
    with open(os.path.join(HERE, 'seeds30_mae_raw.json'), 'w', encoding='utf-8') as fh:
        json.dump({str(n): {b: seeds[n][b][:, 0].tolist() for b in BASES} for n in FILES}, fh)

    out(f'B3. Median over {N_SEEDS_B} seeds of: max |r(x)|, max |y\' - y_ref\'|, max |y\'\' - y_ref\'\'|')
    out(f"  {'n':>4} {'Net':7s} {'max|r|':>10} {'max|dy err|':>12} {'max|d2y err|':>13}")
    for n in FILES:
        for b in BASES:
            a = np.median(seeds[n][b][:, 1:], axis=0)
            out(f"  {n:>4} {LABEL[b]:7s} {a[0]:10.2e} {a[1]:12.2e} {a[2]:13.2e}")
        out()

    out(f'C. Learning-rate sensitivity: MAE mean +/- std over {N_SEEDS_C} seeds')
    out(f"  {'n':>4} {'lr':>7}" + ''.join(f"  {LABEL[b]:>20s}" for b in BASES))
    for n, lr, res in sorted(lrs, key=lambda t: (t[0], -t[1])):
        out(f"  {n:>4} {lr:7.0e}" + ''.join(f"  {res[b][0]:9.2e} +/- {res[b][1]:7.1e}" for b in BASES))

    with open(os.path.join(HERE, 'seeds30_mono_lr_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
