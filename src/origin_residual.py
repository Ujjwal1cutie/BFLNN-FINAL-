# residual at the origin of the trained BFLNN (Remark 3.4 of the paper)
# The trial solution y = 1 + x^2 tanh(W.Phi(x)) gives (2/x) y' = 4 s + 2 x s' z' (finite at x = 0), so
#     lim_{x->0+} r(x) = y''(0) + 4 s0 + y(0)^n = 6 s0 + 1,   s0 = tanh(W.Phi(0)).
# This script trains BFLNN with the settings of n*.py (degree 5, 30 points, seeds 0-4) and prints
# |6 s0 + 1| for each n, together with |r(x_1)| at the first collocation point x_1 = 1e-5 and the
# difference |r(x_1) - (6 s0 + 1)|, which is O(x_1).

import contextlib
import importlib.util
import io
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {0: 'n0', 0.5: 'no.5', 1: 'n1', 1.5: 'n1.5', 2: 'n2', 3: 'n3', 4: 'n4', 5: 'n5'}
N_SEEDS = 5


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'), os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(s)

    emit('BFLNN (m = 5, N_c = 30, seeds 0-4): residual limit at the origin, |6 s0 + 1| with s0 = tanh(W.Phi(0))')
    emit(f"  {'n':>4}  {'max_seed |6 s0+1|':>18}  {'min_seed':>10}  {'max_seed |r(x1)|':>17}  {'max_seed |r(x1)-(6 s0+1)|':>26}")
    for n, fname in FILES.items():
        mod = load_module(fname)
        X = mod.X_MAX
        x = mod.uniform_nodes(mod.N_TRAIN, X)
        half = X / (2.0 * mod.N_VAL)
        xv = np.linspace(1e-5 + half, X - half / 2.0, mod.N_VAL)
        x1 = np.array([x[0]])
        P0, dP0, d2P0 = mod.get_basis('bernoulli', np.array([0.0]), mod.DEGREE, X)
        P1, dP1, d2P1 = mod.get_basis('bernoulli', x1, mod.DEGREE, X)
        r0s, r1s, dif = [], [], []
        for seed in range(N_SEEDS):
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
                W = mod.train('bernoulli', mod.DEGREE, x, x_val=xv, x_max=X, seed=seed)[0]
            s0 = float(np.tanh(P0 @ W).ravel()[0])
            r0 = 6.0 * s0 + 1.0
            y, dy, d2y = mod.forward(x1, W, P1, dP1, d2P1)
            r1 = float(d2y[0] + 2.0 / x1[0] * dy[0] + max(y[0], 1e-12) ** n)
            r0s.append(abs(r0)); r1s.append(abs(r1)); dif.append(abs(r1 - r0))
        emit(f"  {n:>4}  {max(r0s):18.2e}  {min(r0s):10.2e}  {max(r1s):17.2e}  {max(dif):26.2e}")
    open(os.path.join(HERE, 'origin_residual_out.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
