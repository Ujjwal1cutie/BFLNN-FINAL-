# practical degree selection: for every seed, choose m in 3..9 with the lowest
# validation residual loss (no reference solution used), then report the MAE of
# the selected model. Compared with the fixed degree m = 5.
# usage: python degree_select.py [clip_norm]   (default 1, 'inf' for no clipping)

import contextlib
import importlib.util
import io
import json
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import sys

import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
FILES = {0: ('n0', 'exact_n0'), 0.5: ('no.5', 'reference_n05'), 1: ('n1', 'exact_n1'),
         1.5: ('n1.5', 'reference_n15'), 2: ('n2', 'reference_n2'), 3: ('n3', 'reference_n3'),
         4: ('n4', 'reference_n4'), 5: ('n5', 'exact_n5')}
BASES   = ['bernoulli', 'chebyshev', 'legendre']
LABEL   = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
DEGREES = [3, 4, 5, 6, 7, 8, 9]
N_SEEDS = 5
CLIP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # gradient clipping norm (inf = none)


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'), os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(job):
    (n, (fname, ref)), m = job
    mod = load_module(fname)
    X = mod.X_MAX
    x  = mod.uniform_nodes(mod.N_TRAIN, X)
    half = X / (2.0 * mod.N_VAL)
    xv = np.linspace(1e-5 + half, X - half / 2.0, mod.N_VAL)
    xe = np.linspace(1e-5, X, 500)
    ye = getattr(mod, ref)(xe)
    out = {}
    records = []
    for b in BASES:
        Pe, dPe, d2Pe = mod.get_basis(b, xe, m, X)
        Pv, dPv, d2Pv = mod.get_basis(b, xv, m, X)
        for seed in range(N_SEEDS):
            with contextlib.redirect_stdout(io.StringIO()):
                W, th, vh, he, _ = mod.train(b, m, x, x_val=xv, x_max=X, clip_norm=CLIP, seed=seed)
            mae = np.mean(np.abs(mod.forward(xe, W, Pe, dPe, d2Pe)[0] - ye))
            # Early-stopping tolerance can make min(history) differ from the
            # loss of the returned weights. Select the actual returned model.
            yv, dyv, d2yv = mod.forward(xv, W, Pv, dPv, d2Pv)
            yn = np.ones_like(yv) if n == 0 else np.maximum(yv, 1e-12)**n
            validation_loss = float(np.mean((d2yv + 2 / xv * dyv + yn)**2))
            if not np.isfinite(validation_loss) or not np.isfinite(mae):
                raise RuntimeError(f'Non-finite degree-selection result: n={n}, m={m}, basis={b}, seed={seed}')
            out[(b, seed)] = (validation_loss, mae)
            records.append({
                'basis': b, 'seed': seed, 'degree': m, 'polytropic_index': n,
                'clipping_norm': None if np.isinf(CLIP) else CLIP,
                'returned_weights_validation_mse': validation_loss,
                'minimum_recorded_validation_mse': float(min(vh)) if vh else None,
                'mae': float(mae), 'weights': W.ravel().tolist(),
                'epochs': int(he[-1]) if he else 0,
            })
            path = os.path.join(HERE, f'degree_select_clip{CLIP:g}_n{n:g}_m{m}_raw.json')
            temporary = path + '.tmp'
            with open(temporary, 'w', encoding='utf-8') as fh:
                json.dump({'schema_version': 1, 'complete': False, 'records': records},
                          fh, indent=2, allow_nan=False)
                fh.write('\n')
            os.replace(temporary, path)
    with open(temporary, 'w', encoding='utf-8') as fh:
        json.dump({'schema_version': 1, 'complete': True, 'records': records},
                  fh, indent=2, allow_nan=False)
        fh.write('\n')
    os.replace(temporary, path)
    return n, m, out


if __name__ == '__main__':
    jobs = [(item, m) for item in FILES.items() for m in DEGREES]
    with Pool(min(8, os.cpu_count() or 1, WORKERS)) as pool:
        raw = {(n, m): o for n, m, o in pool.map(run, jobs)}
    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)
    emit(f'Degree chosen per seed by the lowest validation loss (m = 3..9), clip norm = {CLIP}; MAE of the chosen models')
    emit(f"  {'n':>4}" + ''.join(f"  {LABEL[b] + ': m chosen / MAE (m=5 MAE)':>44s}" for b in BASES))
    summary = {}
    for n in FILES:
        cells = ''
        for b in BASES:
            chosen, maes, fixed = [], [], []
            for seed in range(N_SEEDS):
                m_sel = min(DEGREES, key=lambda m: raw[(n, m)][(b, seed)][0])
                chosen.append(m_sel); maes.append(raw[(n, m_sel)][(b, seed)][1])
                fixed.append(raw[(n, 5)][(b, seed)][1])
            summary[(n, b)] = (chosen, np.mean(maes), np.mean(fixed))
            cells += f"  m={''.join(str(c) for c in chosen)} {np.mean(maes):9.2e} ({np.mean(fixed):9.2e})"
        emit(f"  {n:>4}{cells}")
    emit()
    emit('Ratio of selected-model MAE to BFLNN selected-model MAE:')
    for n in FILES:
        bb = summary[(n, 'bernoulli')][1]
        emit(f"  n={n:<4} LNN/BFLNN = {summary[(n, 'legendre')][1] / bb:8.1f}   ChNN/BFLNN = {summary[(n, 'chebyshev')][1] / bb:8.1f}")
    open(os.path.join(HERE, 'degree_select_out.txt' if CLIP == 1.0 else f'degree_select_clip{CLIP:g}_out.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
