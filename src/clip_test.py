# effect of gradient clipping: clip norm 1 (default), 10, and no clipping
# all other settings as in n*.py, 5 seeds

import contextlib
import importlib.util
import io
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
FILES = {0: ('n0', 'exact_n0'), 0.5: ('no.5', 'reference_n05'), 1: ('n1', 'exact_n1'),
         1.5: ('n1.5', 'reference_n15'), 2: ('n2', 'reference_n2'), 3: ('n3', 'reference_n3'),
         4: ('n4', 'reference_n4'), 5: ('n5', 'exact_n5')}
BASES = ['bernoulli', 'chebyshev', 'legendre']
LABEL = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
CLIPS = [1.0, 10.0, np.inf]
N_SEEDS = 5


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'), os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(item):
    n, (fname, ref) = item
    mod = load_module(fname)
    X = mod.X_MAX
    x  = mod.uniform_nodes(mod.N_TRAIN, X)
    half = X / (2.0 * mod.N_VAL)
    xv = np.linspace(1e-5 + half, X - half / 2.0, mod.N_VAL)
    xe = np.linspace(1e-5, X, 500)
    ye = getattr(mod, ref)(xe)
    out = {}
    for b in BASES:
        Pe, dPe, d2Pe = mod.get_basis(b, xe, mod.DEGREE, X)
        for c in CLIPS:
            maes, stops = [], []
            for seed in range(N_SEEDS):
                with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
                    W, th, vh, he, _ = mod.train(b, mod.DEGREE, x, x_val=xv, x_max=X, clip_norm=c, seed=seed)
                    y = mod.forward(xe, W, Pe, dPe, d2Pe)[0]
                err = np.mean(np.abs(y - ye))
                maes.append(err if np.isfinite(err) else np.nan)
                stops.append(he[-1])
            out[(b, c)] = (np.nanmean(maes), np.nanstd(maes), int(np.sum(np.isnan(maes))), np.median(stops))
    return n, out


if __name__ == '__main__':
    with Pool(min(8, os.cpu_count() or 1, WORKERS)) as pool:
        res = dict(pool.map(run, FILES.items()))
    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)
    emit('MAE mean +/- std over 5 seeds (median stopping epoch) for clip norm 1 / 10 / none')
    emit(f"  {'n':>4} {'Net':6s}" + ''.join(f"  {'clip=' + str(c):>30s}" for c in CLIPS))
    for n in FILES:
        for b in BASES:
            cells = ''
            for c in CLIPS:
                m, s, nf, st = res[n][(b, c)]
                cells += f"  {m:9.2e} +/- {s:7.1e} ({int(st):5d})" + (f" [{nf} diverged]" if nf else '')
            emit(f"  {n:>4} {LABEL[b]:6s}{cells}")
        emit()
    open(os.path.join(HERE, 'clip_test_out.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
