# solving on [0, xi_1] (first zero): LM error vs degree, Adam, predicted xi_1

import contextlib
import importlib.util
import io
import json
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

HERE    = os.path.dirname(os.path.abspath(__file__))
FILES   = {0: 'n0', 0.5: 'no.5', 1: 'n1', 1.5: 'n1.5', 2: 'n2', 3: 'n3', 4: 'n4', 5: 'n5'}
BASES   = ['bernoulli', 'chebyshev', 'legendre']
LABEL   = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
LM_DEGREES   = [5, 8, 11]
ADAM_DEGREES = [5, 8]
N_SEEDS, N_TRAIN, N_VAL, N_EVAL = 5, 30, 105, 500
R_N5 = 10.0
Y_FLOOR = 1e-12


def write_checkpoint(study, n, records, *, complete=False, metadata=None):
    """Atomically preserve each finished fit; non-finite numbers become null."""
    def json_value(value):
        if isinstance(value, dict):
            return {str(k): json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(v) for v in value]
        if isinstance(value, np.ndarray):
            return json_value(value.tolist())
        if isinstance(value, np.generic):
            return json_value(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    payload = {
        'schema_version': 1, 'study': study, 'polytropic_index': n,
        'complete': complete, 'metadata': metadata or {}, 'records': records,
    }
    path = os.path.join(HERE, f'{study}_n{n:g}_raw.json')
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as fh:
        json.dump(json_value(payload), fh, indent=2, allow_nan=False)
        fh.write('\n')
    os.replace(temporary, path)


def lm_diagnostics(result):
    return {
        'lm_success': bool(result.success), 'lm_status': int(result.status),
        'lm_message': str(result.message), 'lm_nfev': int(result.nfev),
        'lm_njev': None if result.njev is None else int(result.njev),
        'lm_cost': float(result.cost), 'lm_optimality': float(result.optimality),
    }


def adam_objective_diagnostics(mod, W, x_train, train_basis, x_val, val_basis):
    """Keep the base Adam objective separate from LM floor-style diagnostics."""
    loss = next(getattr(mod, name) for name in dir(mod) if name.startswith('compute_ode_loss_'))
    weights = np.asarray(W).reshape(-1, 1)
    return {
        'adam_objective_train_mse': float(loss(x_train, weights, *train_basis)),
        'adam_objective_validation_mse': float(loss(x_val, weights, *val_basis)),
    }


def solution_diagnostics(W, n, x_train, train_basis, x_eval, eval_basis,
                         x_val=None, val_basis=None):
    """Record the actual returned function, including positivity-floor use."""
    W = np.asarray(W).ravel()
    y_train = trial(W, x_train, *train_basis)[4]
    y_eval = trial(W, x_eval, *eval_basis)[4]
    r_train = residual(W, x_train, *train_basis, n)
    r_eval = residual(W, x_eval, *eval_basis, n)
    info = {
        'weights': W.tolist(), 'train_residual_mse': float(np.mean(r_train**2)),
        'evaluation_max_abs_residual': float(np.max(np.abs(r_eval))),
        'min_y_train': float(np.min(y_train)), 'min_y_eval': float(np.min(y_eval)),
        'floor_active_train_points': int(np.count_nonzero(y_train <= Y_FLOOR)) if n else 0,
        'floor_active_eval_points': int(np.count_nonzero(y_eval <= Y_FLOOR)) if n else 0,
    }
    if x_val is not None and val_basis is not None:
        r_val = residual(W, x_val, *val_basis, n)
        info['validation_residual_mse'] = float(np.mean(r_val**2))
    return info


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'),
                                                  os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_solution(n, x_end):
    """DOP853 solution on [x0, x_end] and first zero xi_1 (None if not reached)."""
    x0  = 1e-7
    y0  = [1 - x0**2 / 6 + n * x0**4 / 120, -x0 / 3 + n * x0**3 / 30]
    rhs = lambda t, s: [s[1], -2.0 / t * s[1] - np.sign(s[0]) * abs(s[0])**n]
    hit = lambda t, s: s[0]
    hit.terminal, hit.direction = False, -1
    sol = solve_ivp(rhs, [x0, x_end], y0, method='DOP853', dense_output=True,
                    events=hit, rtol=1e-13, atol=1e-15)
    if not sol.success:
        raise RuntimeError(f'Reference integration failed for n={n}: {sol.message}')
    xi1 = sol.t_events[0][0] if len(sol.t_events[0]) else None
    def y_ref(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < x0, 1 - x**2 / 6, sol.sol(np.maximum(x, x0))[0])
    return y_ref, xi1


# ── trial solution, residual and Jacobian (generic n, safe fractional power) ──
def trial(W, x, P, dP, d2P):
    z = P @ W; s = np.tanh(z); s2 = 1 - s**2; zp = dP @ W; zpp = d2P @ W
    y   = 1 + x**2 * s
    dy  = 2 * x * s + x**2 * s2 * zp
    d2y = 2 * s + 4 * x * s2 * zp + x**2 * (-2 * s * s2 * zp**2 + s2 * zpp)
    return s, s2, zp, zpp, y, dy, d2y


def residual(W, x, P, dP, d2P, n):
    *_, y, dy, d2y = trial(W, x, P, dP, d2P)
    yn = np.ones_like(y) if n == 0 else np.maximum(y, Y_FLOOR)**n
    return d2y + 2 / x * dy + yn


def jacobian(W, x, P, dP, d2P, n):
    s, s2, zp, zpp, y, _, _ = trial(W, x, P, dP, d2P)
    x1 = x[:, None]; s, s2, zp, zpp = s[:, None], s2[:, None], zp[:, None], zpp[:, None]
    dt = s2 * P; ds2 = -2 * s * s2 * P
    dDY  = 2 * x1 * dt + x1**2 * (ds2 * zp + s2 * dP)
    dd2N = -2 * (dt * s2 * zp**2 + s * ds2 * zp**2 + s * s2 * 2 * zp * dP) + ds2 * zpp + s2 * d2P
    J = 2 * dt + 4 * x1 * (ds2 * zp + s2 * dP) + x1**2 * dd2N + 2 / x1 * dDY
    if n != 0:
        # Preserve the original clipped residual, but differentiate the floor.
        # At y == Y_FLOOR it is nonsmooth; use the inactive-branch derivative.
        d_power = np.where(y > Y_FLOOR,
                           n * np.maximum(y, Y_FLOOR)**(n - 1), 0.0)
        J = J + d_power[:, None] * x1**2 * dt
    return J


def predicted_zero(W, mod, b, m, R):
    """First zero of the trained trial solution, searched on [0.5R, 1.2R]."""
    xs = np.linspace(0.5 * R, 1.2 * R, 2001)
    P, dP, d2P = mod.get_basis(b, xs, m, R)
    ys = trial(W, xs, P, dP, d2P)[4]
    idx = np.where(np.diff(np.sign(ys)) != 0)[0]
    if not len(idx):
        return np.nan
    f = lambda v: trial(W, np.array([v]), *mod.get_basis(b, np.array([v]), m, R))[4][0]
    return brentq(f, xs[idx[0]], xs[idx[0] + 1], xtol=1e-14)


def run_one_n(item):
    n, fname = item
    mod = load_module(fname)
    y_ref, xi1 = reference_solution(n, 1.3 * (R_N5 if n == 5 else 20.0))
    R = R_N5 if n == 5 else xi1
    if R is None:
        raise RuntimeError(f'Reference first zero was not found for n={n}')
    x_tr  = np.linspace(1e-5, R, N_TRAIN)
    half  = R / (2.0 * N_VAL)
    x_val = np.linspace(1e-5 + half, R - half / 2.0, N_VAL)
    x_ev  = np.linspace(1e-5, R, N_EVAL)
    y_ev  = y_ref(x_ev)
    lm, adam = {}, {}
    records = []
    metadata = {
        'R': R, 'reference_xi1': xi1, 'domain_uses_reference_radius': n != 5,
        'root_search_interval': [0.5 * R, 1.2 * R],
        'positivity_floor': Y_FLOOR,
        'nonlinearity': '1 if n=0; max(y, positivity_floor)**n otherwise',
        'nonlinearity_scope': 'LM fits and shared residual diagnostics only',
        'adam_nonlinearity': '1 if n=0; max(y,1e-12)**n for n=0.5,1.5; y**n for integer n',
        'objective_caveat': 'Adam and LM objectives differ if integer-index predictions enter the floor-active region.',
        'training_points': N_TRAIN, 'validation_points': N_VAL,
        'evaluation_points': N_EVAL,
    }

    for b in BASES:
        for m in LM_DEGREES:
            P, dP, d2P = mod.get_basis(b, x_tr, m, R)
            Pe, dPe, d2Pe = mod.get_basis(b, x_ev, m, R)
            val_basis = mod.get_basis(b, x_val, m, R)
            np.random.seed(0)
            W0 = np.random.randn(m + 1) * 0.1
            fitted = least_squares(residual, W0, jac=jacobian, args=(x_tr, P, dP, d2P, n),
                                   method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                   max_nfev=50_000)
            W = fitted.x
            err = np.abs(trial(W, x_ev, Pe, dPe, d2Pe)[4] - y_ev)
            lm[(b, m)] = (err.mean(), err.max())
            records.append({
                'stage': 'lm_from_initialization', 'basis': b, 'degree': m, 'seed': 0,
                'mae': float(err.mean()), 'linf': float(err.max()),
                **lm_diagnostics(fitted),
                **solution_diagnostics(W, n, x_tr, (P, dP, d2P), x_ev,
                                       (Pe, dPe, d2Pe), x_val, val_basis),
            })
            write_checkpoint('xi1_domain', n, records, metadata=metadata)

        for m in ADAM_DEGREES:
            Pe, dPe, d2Pe = mod.get_basis(b, x_ev, m, R)
            train_basis = mod.get_basis(b, x_tr, m, R)
            val_basis = mod.get_basis(b, x_val, m, R)
            maes, linfs, zeros = [], [], []
            for seed in range(N_SEEDS):
                with contextlib.redirect_stdout(io.StringIO()):
                    trained = mod.train(b, m, x_tr, x_val=x_val, x_max=R, seed=seed)
                    W = trained[0].ravel()
                err = np.abs(trial(W, x_ev, Pe, dPe, d2Pe)[4] - y_ev)
                maes.append(err.mean()); linfs.append(err.max())
                if n != 5:
                    zeros.append(predicted_zero(W, mod, b, m, R))
                records.append({
                    'stage': 'adam', 'basis': b, 'degree': m, 'seed': seed,
                    'mae': float(err.mean()), 'linf': float(err.max()),
                    'epochs': int(trained[3][-1]) if trained[3] else 0,
                    'elapsed_seconds': float(trained[4]),
                    'predicted_zero': zeros[-1] if zeros else None,
                    **adam_objective_diagnostics(mod, W, x_tr, train_basis, x_val, val_basis),
                    **solution_diagnostics(W, n, x_tr, train_basis, x_ev,
                                           (Pe, dPe, d2Pe), x_val, val_basis),
                })
                write_checkpoint('xi1_domain', n, records, metadata=metadata)
            adam[(b, m)] = (np.mean(maes), np.std(maes), np.mean(linfs),
                            np.nanmean(np.abs(np.array(zeros) - xi1)) if zeros else np.nan,
                            int(np.sum(np.isnan(zeros))) if zeros else 0)
    write_checkpoint('xi1_domain', n, records, complete=True, metadata=metadata)
    return n, R, xi1, lm, adam


if __name__ == '__main__':
    with Pool(min(len(FILES), os.cpu_count() or 1, WORKERS)) as pool:
        results = pool.map(run_one_n, FILES.items())

    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    out('=' * 96)
    out('Domain [0, R], R = xi_1(n)  (R = 10 for n = 5)')
    out('=' * 96)
    for n, R, xi1, _, _ in results:
        out(f"  n = {n:<4}  R = {R:.6f}" + (f"   (reference xi_1 = {xi1:.6f})" if n != 5 else ''))

    out()
    out('A. Observed Levenberg-Marquardt residual fits: MAE / Linf on [0,R]')
    out(f"  {'n':>4} {'Net':6s}" + ''.join(f"   {'m=' + str(m):>21s}" for m in LM_DEGREES))
    for n, R, xi1, lm, _ in results:
        for b in BASES:
            out(f"  {n:>4} {LABEL[b]:6s}" + ''.join(
                f"   {lm[(b, m)][0]:9.2e} / {lm[(b, m)][1]:9.2e}" for m in LM_DEGREES))
        out()

    out('B. Adam, paper settings, 5 seeds: MAE mean +/- std, mean Linf, mean |xi_1 error|')
    for n, R, xi1, _, adam in results:
        for m in ADAM_DEGREES:
            for b in BASES:
                a = adam[(b, m)]
                z = '' if n == 5 else f"   |dxi1| = {a[3]:9.2e}" + (f"  (no zero: {a[4]} seeds)" if a[4] else '')
                out(f"  n={n:<4} m={m:<2} {LABEL[b]:6s} MAE = {a[0]:9.2e} +/- {a[1]:8.1e}"
                    f"   Linf = {a[2]:9.2e}{z}")
        out()

    with open(os.path.join(HERE, 'xi1_domain_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
