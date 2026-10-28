# tanh vs identity vs sin activation, same training settings

import contextlib
import importlib.util
import io
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

HERE  = os.path.dirname(os.path.abspath(__file__))
FILES = {0: ('n0', 'exact_n0'), 0.5: ('no.5', 'reference_n05'), 1: ('n1', 'exact_n1'),
         1.5: ('n1.5', 'reference_n15'), 2: ('n2', 'reference_n2'), 3: ('n3', 'reference_n3'),
         4: ('n4', 'reference_n4'), 5: ('n5', 'exact_n5')}
BASES = ['bernoulli', 'chebyshev', 'legendre']
LABEL = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
ACTS  = ['tanh', 'identity', 'sin']
N_SEEDS = 5


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'),
                                                  os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def activation(name, z):
    """g, g', g'', g''' evaluated at z."""
    if name == 'tanh':
        g = np.tanh(z); g1 = 1 - g**2; g2 = -2 * g * g1; g3 = -2 * (g1**2 + g * g2)
    elif name == 'sin':
        g = np.sin(z); g1 = np.cos(z); g2 = -g; g3 = -g1
    else:
        g = z.copy(); g1 = np.ones_like(z); g2 = np.zeros_like(z); g3 = np.zeros_like(z)
    return g, g1, g2, g3


def residual_and_jacobian(W, x, P, dP, d2P, n, act, need_jac=True):
    z, zp, zpp = P @ W, dP @ W, d2P @ W
    g, g1, g2, g3 = activation(act, z)
    N0, N1, N2 = g, g1 * zp, g2 * zp**2 + g1 * zpp
    y   = 1 + x**2 * N0
    dy  = 2 * x * N0 + x**2 * N1
    d2y = 2 * N0 + 4 * x * N1 + x**2 * N2
    yc  = np.maximum(y, 1e-12)
    r = d2y + 2 / x * dy + (np.ones_like(y) if n == 0 else yc**n)
    if not need_jac:
        return r, y
    c = lambda v: v[:, None]
    dN0 = c(g1) * P
    dN1 = c(g2 * zp) * P + c(g1) * dP
    dN2 = c(g3 * zp**2 + g2 * zpp) * P + c(2 * g2 * zp) * dP + c(g1) * d2P
    x1 = c(x)
    dY, dDY = x1**2 * dN0, 2 * x1 * dN0 + x1**2 * dN1
    dD2Y = 2 * dN0 + 4 * x1 * dN1 + x1**2 * dN2
    J = dD2Y + 2 / x1 * dDY
    if n != 0:
        J = J + n * c(yc**(n - 1) * (y > 1e-12)) * dY
    return r, y, J


def train(act, P, dP, d2P, Pv, dPv, d2Pv, x, xv, n, seed, lr=1e-4, epochs=50_000,
          record_every=200, patience=50, es_tol=1e-4, clip=1.0, return_info=False):
    """Same procedure as train() in n*.py, for an arbitrary activation."""
    np.random.seed(seed)
    K = P.shape[1]
    W = np.random.randn(K, 1).ravel() * 0.1
    m = np.zeros(K); v = np.zeros(K)
    best_val, best_W, counter = np.inf, W.copy(), 0
    patience_val = np.inf
    best_epoch, t = 0, 0
    train_hist, val_hist, hist_epochs = [], [], []
    for t in range(1, epochs + 1):
        r, _, J = residual_and_jacobian(W, x, P, dP, d2P, n, act)
        g = 2.0 / len(x) * (r[:, None] * J).sum(axis=0)
        gn = np.linalg.norm(g)
        if gn > clip:
            g = g * clip / gn
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        W = W - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        if t % record_every == 0:
            train_hist.append(float(np.mean(residual_and_jacobian(W, x, P, dP, d2P, n, act, False)[0]**2)))
            vl = np.mean(residual_and_jacobian(W, xv, Pv, dPv, d2Pv, n, act, need_jac=False)[0]**2)
            val_hist.append(float(vl)); hist_epochs.append(t)
            if vl < best_val:
                best_val, best_W, best_epoch = vl, W.copy(), t
            if vl < patience_val * (1 - es_tol):
                patience_val, counter = vl, 0
            else:
                counter += 1
            if counter > patience:
                break
    if t and t % record_every != 0:
        train_hist.append(float(np.mean(residual_and_jacobian(W, x, P, dP, d2P, n, act, False)[0]**2)))
        vl = float(np.mean(residual_and_jacobian(W, xv, Pv, dPv, d2Pv, n, act, False)[0]**2))
        val_hist.append(vl); hist_epochs.append(t)
        if vl < best_val:
            best_val, best_W, best_epoch = vl, W.copy(), t
    info = {'checkpoint_epoch': best_epoch, 'stopped_epoch': t,
            'checkpoint_val_loss': float(np.mean(residual_and_jacobian(best_W, xv, Pv, dPv, d2Pv, n, act, False)[0]**2)),
            'checkpoint_train_loss': float(np.mean(residual_and_jacobian(best_W, x, P, dP, d2P, n, act, False)[0]**2)),
            'hist_epochs': hist_epochs, 'train_hist': train_hist, 'val_hist': val_hist,
            'lr': lr, 'max_epochs': epochs, 'record_every': record_every,
            'patience': patience, 'es_tol': es_tol, 'clip_norm': clip}
    return (best_W, info) if return_info else best_W


def run_one_n(item):
    n, (fname, ref_name) = item
    mod = load_module(fname)
    X = mod.X_MAX
    x  = mod.uniform_nodes(mod.N_TRAIN, X)
    half = X / (2.0 * mod.N_VAL)
    xv = np.linspace(1e-5 + half, X - half / 2.0, mod.N_VAL)
    xe = np.linspace(1e-5, X, 500)
    ye = getattr(mod, ref_name)(xe)
    res, floor = {}, {}
    details = {'n': n, 'degree': mod.DEGREE, 'x_train': x.tolist(),
               'x_val': xv.tolist(), 'x_eval': xe.tolist(), 'y_ref': ye.tolist(),
               'adam': [], 'lm': []}
    for b in BASES:
        P, dP, d2P = mod.get_basis(b, x, mod.DEGREE, X)
        Pv, dPv, d2Pv = mod.get_basis(b, xv, mod.DEGREE, X)
        Pe, dPe, d2Pe = mod.get_basis(b, xe, mod.DEGREE, X)
        for act in ACTS:
            maes = []
            for seed in range(N_SEEDS):
                W, info = train(act, P, dP, d2P, Pv, dPv, d2Pv, x, xv, n, seed, return_info=True)
                r_eval, y_eval = residual_and_jacobian(W, xe, Pe, dPe, d2Pe, n, act, False)
                err = y_eval - ye
                mae = float(np.mean(np.abs(err)))
                maes.append(mae)
                details['adam'].append(dict(info, basis=b, activation=act, seed=seed,
                                             W=W.tolist(), MAE=mae, RMSE=float(np.sqrt(np.mean(err**2))),
                                             Linf=float(np.abs(err).max()), max_residual=float(np.abs(r_eval).max())))
            res[(b, act)] = (np.mean(maes), np.std(maes))
            if b == 'bernoulli':           # empirical LM reference for the common function class
                best = np.inf
                for seed in range(N_SEEDS):
                    np.random.seed(seed)
                    lm = least_squares(lambda w: residual_and_jacobian(w, x, P, dP, d2P, n, act, False)[0],
                                      np.random.randn(mod.DEGREE + 1) * 0.1,
                                      jac=lambda w: residual_and_jacobian(w, x, P, dP, d2P, n, act)[2],
                                      method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20_000)
                    W = lm.x
                    r_eval, y_eval = residual_and_jacobian(W, xe, Pe, dPe, d2Pe, n, act, False)
                    err = y_eval - ye
                    mae = float(np.mean(np.abs(err)))
                    best = min(best, mae)
                    details['lm'].append({'basis': b, 'activation': act, 'seed': seed,
                                          'W': W.tolist(), 'MAE': mae, 'Linf': float(np.abs(err).max()),
                                          'max_residual': float(np.abs(r_eval).max()), 'success': bool(lm.success),
                                          'status': int(lm.status), 'message': str(lm.message),
                                          'nfev': int(lm.nfev), 'njev': None if lm.njev is None else int(lm.njev),
                                          'cost': float(lm.cost), 'optimality': float(lm.optimality)})
                floor[act] = best
    detail_dir = os.path.join(HERE, 'run_details')
    os.makedirs(detail_dir, exist_ok=True)
    detail_path = os.path.join(detail_dir, f'activation_n{n:g}.json')
    with open(detail_path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(details, fh, indent=2, allow_nan=False)
    os.replace(detail_path + '.tmp', detail_path)
    return n, res, floor


if __name__ == '__main__':
    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    # ── check: generic trainer with tanh == train() of the paper code ─────────
    mod = load_module('n1')
    x = mod.uniform_nodes(30, 1.0); xv = np.linspace(1e-5 + 1 / 210, 1 - 1 / 420, 105)
    for b in BASES:
        P, dP, d2P = mod.get_basis(b, x, 5, 1.0); Pv, dPv, d2Pv = mod.get_basis(b, xv, 5, 1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            W_ref = mod.train(b, 5, x, x_val=xv, seed=0)[0].ravel()
        W_new = train('tanh', P, dP, d2P, Pv, dPv, d2Pv, x, xv, 1, 0)
        out(f"check {LABEL[b]}: max|W_generic - W_paper| = {np.abs(W_new - W_ref).max():.1e}")
    out()

    workers = min(8, os.cpu_count() or 1, max(1, int(os.environ.get('BFLNN_WORKERS', '8'))))
    with Pool(workers) as pool:
        results = pool.map(run_one_n, FILES.items())

    out('MAE mean +/- std over 5 seeds (Adam, paper settings), and best observed LM residual-fit MAE')
    out(f"  {'n':>4} {'act':9s}" + ''.join(f"  {LABEL[b]:>20s}" for b in BASES) + f"  {'LM MAE':>10s}")
    for n, res, floor in results:
        for act in ACTS:
            out(f"  {n:>4} {act:9s}" + ''.join(f"  {res[(b, act)][0]:9.2e} +/- {res[(b, act)][1]:7.1e}" for b in BASES)
                + f"  {floor[act]:10.2e}")
        out()

    out('Best basis (lowest mean MAE) per n and activation:')
    for act in ACTS:
        out(f"  {act:9s}: " + '  '.join(
            f"n={n}:{LABEL[min(BASES, key=lambda b: res[(b, act)][0])]}" for n, res, _ in results))

    with open(os.path.join(HERE, 'activation_test_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
