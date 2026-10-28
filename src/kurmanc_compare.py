# setting of Kurmanc & Ozavsar (2025): degree 4, 20 / 10 points, n = 1 and 5

import contextlib, importlib.util, io, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def load(f):
    s = importlib.util.spec_from_file_location(f.replace('.', '_'), os.path.join(HERE, f + '.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

lines = []
for fname, ref, n, N in [('n1', 'exact_n1', 1, 20), ('n5', 'exact_n5', 5, 10)]:
    m = load(fname)
    x_tr = np.linspace(1e-5, 1.0, N)
    half = 1.0 / 210; x_val = np.linspace(1e-5 + half, 1 - half / 2, 105)
    x_ev = np.linspace(1e-5, 1.0, 500); y_ev = getattr(m, ref)(x_ev)
    for b in m.BASES:
        Pe, dPe, d2Pe = m.get_basis(b, x_ev, 4, 1.0)
        maes = []
        for seed in range(5):
            with contextlib.redirect_stdout(io.StringIO()):
                W = m.train(b, 4, x_tr, x_val=x_val, x_max=1.0, seed=seed)[0]
            maes.append(np.mean(np.abs(m.forward(x_ev, W, Pe, dPe, d2Pe)[0] - y_ev)))
        s = f"n={n} N={N} K=5  {b:10s} MAE mean={np.mean(maes):.2e}  std={np.std(maes):.1e}  min={np.min(maes):.2e}"
        print(s, flush=True); lines.append(s)
open(os.path.join(HERE, 'kurmanc_compare_out.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
