# degree m = 3..9 for all bases, best degree per basis, LM optimum per degree

import contextlib
import importlib.util
import io
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import numpy as np
from scipy.optimize import least_squares

HERE  = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('bca', os.path.join(HERE, 'conditioning.py'))
bca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bca)

FILES = {0: ('n0', 'exact_n0'), 0.5: ('no.5', 'reference_n05'), 1: ('n1', 'exact_n1'),
         1.5: ('n1.5', 'reference_n15'), 2: ('n2', 'reference_n2'), 3: ('n3', 'reference_n3'),
         4: ('n4', 'reference_n4'), 5: ('n5', 'exact_n5')}
BASES   = ['bernoulli', 'chebyshev', 'legendre']
LABEL   = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
DEGREES = [3, 4, 5, 6, 7, 8, 9]
N_SEEDS = 5


def run(item):
    (n, (fname, ref_name)), m = item
    mod = bca.load_module(fname)
    X = mod.X_MAX
    x  = mod.uniform_nodes(mod.N_TRAIN, X)
    half = X / (2.0 * mod.N_VAL)
    xv = np.linspace(1e-5 + half, X - half / 2.0, mod.N_VAL)
    xe = np.linspace(1e-5, X, 500)
    ye = getattr(mod, ref_name)(xe)
    out = {}
    for b in BASES:
        Pe, dPe, d2Pe = mod.get_basis(b, xe, m, X)
        maes = []
        for seed in range(N_SEEDS):
            with contextlib.redirect_stdout(io.StringIO()):
                W = mod.train(b, m, x, x_val=xv, x_max=X, seed=seed)[0]
            maes.append(np.mean(np.abs(mod.forward(xe, W, Pe, dPe, d2Pe)[0] - ye)))
        out[b] = (np.mean(maes), np.std(maes))
    # attainable error: best LM solution over the five initialisations
    P, dP, d2P = mod.get_basis('bernoulli', x, m, X)
    Pe, dPe, d2Pe = mod.get_basis('bernoulli', xe, m, X)
    best = np.inf
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        W = least_squares(bca.residual, np.random.randn(m + 1) * 0.1, jac=bca.residual_jacobian,
                          args=(x, P, dP, d2P, n), method='lm',
                          xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20_000).x
        best = min(best, np.mean(np.abs(bca.trial(W, xe, Pe, dPe, d2Pe)[4] - ye)))
    out['lm'] = best
    return n, m, out


if __name__ == '__main__':
    jobs = [(item, m) for item in FILES.items() for m in DEGREES]
    with Pool(min(8, os.cpu_count() or 1, WORKERS)) as pool:
        res = {(n, m): o for n, m, o in pool.map(run, jobs)}

    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)

    emit('A. MAE (mean over 5 seeds) versus degree m; last column: best observed LM residual-fit MAE')
    emit(f"  {'n':>4} {'m':>2}" + ''.join(f"  {LABEL[b]:>20s}" for b in BASES) + f"  {'LM':>9}")
    for n in FILES:
        for m in DEGREES:
            o = res[(n, m)]
            emit(f"  {n:>4} {m:>2}" + ''.join(f"  {o[b][0]:9.2e} +/- {o[b][1]:7.1e}" for b in BASES)
                 + f"  {o['lm']:9.2e}")
        emit()

    emit('B. Each basis at its own best degree (lowest mean MAE over m = 3..9)')
    emit(f"  {'n':>4}" + ''.join(f"  {LABEL[b] + ' (m*, MAE)':>22s}" for b in BASES) + f"  {'best LM (m)':>16}")
    for n in FILES:
        cells = []
        for b in BASES:
            m_star = min(DEGREES, key=lambda m: res[(n, m)][b][0])
            cells.append(f"  m={m_star}, {res[(n, m_star)][b][0]:9.2e}     ")
        m_lm = min(DEGREES, key=lambda m: res[(n, m)]['lm'])
        emit(f"  {n:>4}" + ''.join(cells) + f"  {res[(n, m_lm)]['lm']:9.2e} ({m_lm})")

    with open(os.path.join(HERE, 'degree_sweep_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
