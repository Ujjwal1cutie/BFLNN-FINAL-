# Frank-Kamenetskii problem y'' + (k/x) y' + delta e^y = 0, y'(0) = 0, y(1) = 0
# trial solution (1 - x^2)(c + x^2 tanh(W.Phi)); k = 1 exact, k = 2 shooting

import importlib.util
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

HERE  = os.path.dirname(os.path.abspath(__file__))
BASES = ['bernoulli', 'chebyshev', 'legendre']
LABEL = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
CASES = [(1, 1.0), (1, 1.5), (1, 1.9), (2, 1.0), (2, 2.0), (2, 3.0)]
DEGREE, N_SEEDS = 5, 5

_spec = importlib.util.spec_from_file_location('n0', os.path.join(HERE, 'n0.py'))
basis_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(basis_mod)          # only get_basis() is used


# ── reference solutions ───────────────────────────────────────────────────────
def reference(k, delta, x):
    if k == 1:
        B = ((8 - 2 * delta) - np.sqrt((8 - 2 * delta)**2 - 4 * delta**2)) / (2 * delta)
        return 2 * np.log((1 + B) / (1 + B * x**2))
    x0 = 1e-6
    def shoot(a, dense=False):
        y0 = [a - delta * np.exp(a) * x0**2 / (2 * (k + 1)), -delta * np.exp(a) * x0 / (k + 1)]   # series at x = 0
        rhs = lambda t, s: [s[1], -k / t * s[1] - delta * np.exp(s[0])]
        sol = solve_ivp(rhs, [x0, 1.0], y0, method='DOP853', rtol=1e-13, atol=1e-15,
                        dense_output=dense)
        return sol if dense else sol.y[0, -1]
    a_grid = np.linspace(0.0, 3.0, 301)          # lower branch: first sign change of y(1)
    vals = [shoot(a) for a in a_grid]
    i = next(i for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
    a = brentq(shoot, a_grid[i], a_grid[i + 1], xtol=1e-15)
    sol = shoot(a, dense=True)
    return np.where(x < x0, a, sol.sol(np.maximum(x, x0))[0])


# ── trial solution, residual and Jacobian w.r.t. theta = (c, W) ───────────────
def model(theta, x, P, dP, d2P, k, delta, need_jac=True):
    c, W = theta[0], theta[1:]
    z, zp, zpp = P @ W, dP @ W, d2P @ W
    s = np.tanh(z); s1 = 1 - s**2; s2 = -2 * s * s1; s3 = -2 * (s1**2 + s * s2)
    u   = c + x**2 * s
    u1  = 2 * x * s + x**2 * s1 * zp
    u1x = 2 * s + x * s1 * zp                              # u'/x, finite at x = 0
    u2  = 2 * s + 4 * x * s1 * zp + x**2 * (s2 * zp**2 + s1 * zpp)
    D, D1 = 1 - x**2, -2 * x
    y   = D * u
    d2y = -2 * u + 2 * D1 * u1 + D * u2
    kxy = k * (-2 * u + D * u1x)                           # (k/x) y'
    r = d2y + kxy + delta * np.exp(y)
    if not need_jac:
        return r, y
    C = lambda v: v[:, None]
    xs = C(x)
    du   = xs**2 * C(s1) * P
    du1  = 2 * xs * C(s1) * P + xs**2 * (C(s2 * zp) * P + C(s1) * dP)
    du1x = 2 * C(s1) * P + xs * (C(s2 * zp) * P + C(s1) * dP)
    du2  = (2 * C(s1) * P + 4 * xs * (C(s2 * zp) * P + C(s1) * dP)
            + xs**2 * (C(s3 * zp**2 + s2 * zpp) * P + C(2 * s2 * zp) * dP + C(s1) * d2P))
    Ds, D1s = C(D), C(D1)
    JW = (-2 * du + 2 * D1s * du1 + Ds * du2) + k * (-2 * du + Ds * du1x) + delta * C(np.exp(y)) * Ds * du
    Jc = (-2 + 0) + k * (-2) + delta * np.exp(y) * D           # du/dc = 1, du1/dc = du2/dc = 0
    return r, y, np.column_stack([Jc, JW])


def train(P, dP, d2P, Pv, dPv, d2Pv, x, xv, k, delta, seed, lr=1e-4, epochs=50_000,
          record_every=200, patience=50, es_tol=1e-4, clip=1.0, return_info=False):
    np.random.seed(seed)
    theta = np.random.randn(P.shape[1] + 1) * 0.1
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    best_val, best, counter = np.inf, theta.copy(), 0
    patience_val = np.inf
    best_epoch, t = 0, 0
    train_hist, val_hist, hist_epochs = [], [], []
    for t in range(1, epochs + 1):
        r, _, J = model(theta, x, P, dP, d2P, k, delta)
        g = 2.0 / len(x) * (r[:, None] * J).sum(axis=0)
        gn = np.linalg.norm(g)
        if gn > clip:
            g *= clip / gn
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g**2
        theta = theta - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        if t % record_every == 0:
            train_hist.append(float(np.mean(model(theta, x, P, dP, d2P, k, delta, False)[0]**2)))
            vl = np.mean(model(theta, xv, Pv, dPv, d2Pv, k, delta, False)[0]**2)
            val_hist.append(float(vl)); hist_epochs.append(t)
            if vl < best_val:
                best_val, best, best_epoch = vl, theta.copy(), t
            if vl < patience_val * (1 - es_tol):
                patience_val, counter = vl, 0
            else:
                counter += 1
            if counter > patience:
                break
    if t and t % record_every != 0:
        train_hist.append(float(np.mean(model(theta, x, P, dP, d2P, k, delta, False)[0]**2)))
        vl = float(np.mean(model(theta, xv, Pv, dPv, d2Pv, k, delta, False)[0]**2))
        val_hist.append(vl); hist_epochs.append(t)
        if vl < best_val:
            best_val, best, best_epoch = vl, theta.copy(), t
    info = {'checkpoint_epoch': best_epoch, 'stopped_epoch': t,
            'checkpoint_val_loss': float(np.mean(model(best, xv, Pv, dPv, d2Pv, k, delta, False)[0]**2)),
            'checkpoint_train_loss': float(np.mean(model(best, x, P, dP, d2P, k, delta, False)[0]**2)),
            'hist_epochs': hist_epochs, 'train_hist': train_hist, 'val_hist': val_hist,
            'lr': lr, 'max_epochs': epochs, 'record_every': record_every,
            'patience': patience, 'es_tol': es_tol, 'clip_norm': clip}
    return (best, info) if return_info else best


def run_case(case):
    k, delta = case
    x  = np.linspace(1e-5, 1.0, 30)
    xv = np.linspace(1e-5 + 1 / 210, 1 - 1 / 420, 105)
    xe = np.linspace(1e-5, 1.0, 500)
    ye = reference(k, delta, xe)
    y0 = float(reference(k, delta, np.array([0.0]))[0])
    out = {'y0': y0}
    details = {'k': k, 'delta': delta, 'degree': DEGREE, 'y0_ref': y0,
               'x_train': x.tolist(), 'x_val': xv.tolist(), 'x_eval': xe.tolist(),
               'y_ref': ye.tolist(), 'adam': [], 'lm': []}
    for b in BASES:
        P, dP, d2P = basis_mod.get_basis(b, x, DEGREE, 1.0)
        Pv, dPv, d2Pv = basis_mod.get_basis(b, xv, DEGREE, 1.0)
        Pe, dPe, d2Pe = basis_mod.get_basis(b, xe, DEGREE, 1.0)
        maes, c_err = [], []
        for seed in range(N_SEEDS):
            th, info = train(P, dP, d2P, Pv, dPv, d2Pv, x, xv, k, delta, seed, return_info=True)
            r_eval, y_eval = model(th, xe, Pe, dPe, d2Pe, k, delta, False)
            err = y_eval - ye
            mae, central_err = float(np.mean(np.abs(err))), float(abs(th[0] - y0))
            maes.append(mae)
            c_err.append(central_err)
            details['adam'].append(dict(info, basis=b, seed=seed, theta=th.tolist(),
                                         MAE=mae, RMSE=float(np.sqrt(np.mean(err**2))),
                                         Linf=float(np.abs(err).max()), central_error=central_err,
                                         max_residual=float(np.abs(r_eval).max())))
        out[b] = (np.mean(maes), np.std(maes), np.mean(c_err))
        if b == 'bernoulli':
            np.random.seed(0)
            lm = least_squares(lambda q: model(q, x, P, dP, d2P, k, delta, False)[0],
                               np.random.randn(DEGREE + 2) * 0.1,
                               jac=lambda q: model(q, x, P, dP, d2P, k, delta)[2],
                               method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15)
            th = lm.x
            r_eval, y_eval = model(th, xe, Pe, dPe, d2Pe, k, delta, False)
            err = y_eval - ye
            out['lm'] = float(np.mean(np.abs(err)))
            details['lm'].append({'basis': b, 'seed': 0, 'theta': th.tolist(),
                                  'MAE': out['lm'], 'Linf': float(np.abs(err).max()),
                                  'central_error': float(abs(th[0] - y0)),
                                  'max_residual': float(np.abs(r_eval).max()), 'success': bool(lm.success),
                                  'status': int(lm.status), 'message': str(lm.message),
                                  'nfev': int(lm.nfev), 'njev': None if lm.njev is None else int(lm.njev),
                                  'cost': float(lm.cost), 'optimality': float(lm.optimality)})
    detail_dir = os.path.join(HERE, 'run_details')
    os.makedirs(detail_dir, exist_ok=True)
    detail_path = os.path.join(detail_dir, f'thermal_k{k}_delta{delta:g}.json')
    with open(detail_path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(details, fh, indent=2, allow_nan=False)
    os.replace(detail_path + '.tmp', detail_path)
    return case, out


if __name__ == '__main__':
    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)

    # Jacobian check against finite differences
    x = np.linspace(1e-5, 1, 30); P, dP, d2P = basis_mod.get_basis('bernoulli', x, DEGREE, 1.0)
    th = np.random.RandomState(1).randn(DEGREE + 2) * 0.3
    _, _, J = model(th, x, P, dP, d2P, 2, 2.0)
    Jfd = np.column_stack([(model(th + 1e-6 * e, x, P, dP, d2P, 2, 2.0, False)[0]
                            - model(th - 1e-6 * e, x, P, dP, d2P, 2, 2.0, False)[0]) / 2e-6
                           for e in np.eye(DEGREE + 2)])
    emit(f"Jacobian check: max rel. diff = {np.abs(J - Jfd).max() / np.abs(J).max():.1e}")
    xr = np.linspace(1e-2, 0.99, 50); h = 1e-4
    for k, d in [(1, 1.9), (2, 3.0)]:      # reference check: boundary values and ODE residual (central differences)
        yy = reference(k, d, np.concatenate([xr - h, xr, xr + h, [0.0, 1.0]]))
        ym, y0_, yp = yy[:50], yy[50:100], yy[100:150]
        res = (yp - 2 * y0_ + ym) / h**2 + k / xr * (yp - ym) / (2 * h) + d * np.exp(y0_)
        emit(f"reference k={k} delta={d}: y(0)={yy[150]:.10f}, y(1)={yy[151]:.1e}, "
             f"max |ODE residual| (central differences, h={h:g}) = {np.max(np.abs(res)):.1e}")
    emit()

    workers = min(len(CASES), os.cpu_count() or 1, max(1, int(os.environ.get('BFLNN_WORKERS', str(len(CASES))))))
    with Pool(workers) as pool:
        results = pool.map(run_case, CASES)

    emit('MAE (mean +/- std over 5 seeds, M = 500) and mean |c - y(0)| (central temperature error)')
    emit('Last column is the best observed LM residual-fit MAE, not a proven error floor.')
    emit(f"  {'k':>2} {'delta':>5} {'y(0)':>9}" + ''.join(f"  {LABEL[b]:>32s}" for b in BASES) + f"  {'LM MAE':>9}")
    for (k, d), o in results:
        emit(f"  {k:>2} {d:5.2f} {o['y0']:9.6f}" + ''.join(
            f"  {o[b][0]:8.2e} +/- {o[b][1]:7.1e} |dy0| {o[b][2]:7.1e}" for b in BASES) + f"  {o['lm']:9.2e}")

    with open(os.path.join(HERE, 'thermal_bvp_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
