# [0, xi_1]: effect of N, and Adam followed by LM refinement

import contextlib
import importlib.util
import io
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('xi1_domain',
                                               os.path.join(HERE, 'xi1_domain.py'))
des = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(des)

FILES, BASES, LABEL = des.FILES, des.BASES, des.LABEL
N_LIST       = [30, 60, 120]
LM_DEGREES   = [5, 8, 11]
PIPE_DEGREES = [8, 11]
N_SEEDS, N_VAL, N_EVAL = 5, 105, 500


def lm_fit(W0, x, P, dP, d2P, n):
    return least_squares(des.residual, W0, jac=des.jacobian, args=(x, P, dP, d2P, n),
                         method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15,
                         max_nfev=50_000)


def run_one_n(item):
    n, fname = item
    mod = des.load_module(fname)
    y_ref, xi1 = des.reference_solution(n, 1.3 * (des.R_N5 if n == 5 else 20.0))
    R    = des.R_N5 if n == 5 else xi1
    if R is None:
        raise RuntimeError(f'Reference first zero was not found for n={n}')
    x_ev = np.linspace(1e-5, R, N_EVAL)
    y_ev = y_ref(x_ev)
    half  = R / (2.0 * N_VAL)
    x_val = np.linspace(1e-5 + half, R - half / 2.0, N_VAL)
    records = []
    metadata = {
        'R': R, 'reference_xi1': xi1, 'domain_uses_reference_radius': n != 5,
        'root_search_interval': [0.5 * R, 1.2 * R],
        'positivity_floor': des.Y_FLOOR,
        'nonlinearity': '1 if n=0; max(y, positivity_floor)**n otherwise',
        'nonlinearity_scope': 'LM fits and shared residual diagnostics only',
        'adam_nonlinearity': '1 if n=0; max(y,1e-12)**n for n=0.5,1.5; y**n for integer n',
        'objective_caveat': 'Adam and LM objectives differ if integer-index predictions enter the floor-active region.',
        'collocation_counts': N_LIST, 'validation_points': N_VAL,
        'evaluation_points': N_EVAL,
    }

    def mae(W, b, m):
        Pe, dPe, d2Pe = mod.get_basis(b, x_ev, m, R)
        return np.mean(np.abs(des.trial(W, x_ev, Pe, dPe, d2Pe)[4] - y_ev))

    # ── A. collocation density, LM from the paper's initialisation (seed 0)
    floor = {}
    for N in N_LIST:
        x_tr = np.linspace(1e-5, R, N)
        for b in BASES:
            for m in LM_DEGREES:
                P, dP, d2P = mod.get_basis(b, x_tr, m, R)
                eval_basis = mod.get_basis(b, x_ev, m, R)
                val_basis = mod.get_basis(b, x_val, m, R)
                np.random.seed(0)
                fitted = lm_fit(np.random.randn(m + 1) * 0.1, x_tr, P, dP, d2P, n)
                W = fitted.x
                floor[(N, b, m)] = mae(W, b, m)
                records.append({
                    'stage': 'lm_from_initialization', 'training_points': N,
                    'basis': b, 'degree': m, 'seed': 0,
                    'mae': float(floor[(N, b, m)]),
                    **des.lm_diagnostics(fitted),
                    **des.solution_diagnostics(W, n, x_tr, (P, dP, d2P),
                                               x_ev, eval_basis, x_val, val_basis),
                })
                des.write_checkpoint('xi1_refine', n, records, metadata=metadata)

    # ── B. Adam (paper settings) -> LM refinement, N = 30
    x_tr = np.linspace(1e-5, R, 30)
    pipe = {}
    for b in BASES:
        for m in PIPE_DEGREES:
            P, dP, d2P = mod.get_basis(b, x_tr, m, R)
            eval_basis = mod.get_basis(b, x_ev, m, R)
            val_basis = mod.get_basis(b, x_val, m, R)
            a_mae, r_mae, r_xi = [], [], []
            for seed in range(N_SEEDS):
                with contextlib.redirect_stdout(io.StringIO()):
                    trained = mod.train(b, m, x_tr, x_val=x_val, x_max=R, seed=seed)
                    W_a = trained[0].ravel()
                fitted = lm_fit(W_a, x_tr, P, dP, d2P, n)
                W_r = fitted.x
                a_mae.append(mae(W_a, b, m)); r_mae.append(mae(W_r, b, m))
                if n != 5:
                    r_xi.append(abs(des.predicted_zero(W_r, mod, b, m, R) - xi1))
                records.append({
                    'stage': 'adam_then_lm', 'training_points': 30,
                    'basis': b, 'degree': m, 'seed': seed,
                    'adam_mae': float(a_mae[-1]), 'mae': float(r_mae[-1]),
                    'adam_weights': W_a.tolist(),
                    'adam_epochs': int(trained[3][-1]) if trained[3] else 0,
                    'adam_elapsed_seconds': float(trained[4]),
                    **des.adam_objective_diagnostics(mod, W_a, x_tr, (P, dP, d2P), x_val, val_basis),
                    'root_absolute_error': r_xi[-1] if r_xi else None,
                    **des.lm_diagnostics(fitted),
                    **des.solution_diagnostics(W_r, n, x_tr, (P, dP, d2P),
                                               x_ev, eval_basis, x_val, val_basis),
                })
                des.write_checkpoint('xi1_refine', n, records, metadata=metadata)
            pipe[(b, m)] = (np.mean(a_mae), np.mean(r_mae), np.std(r_mae),
                            np.nanmean(r_xi) if r_xi else np.nan,
                            int(np.sum(np.isnan(r_xi))) if r_xi else 0)
    des.write_checkpoint('xi1_refine', n, records, complete=True, metadata=metadata)
    return n, R, xi1, floor, pipe


if __name__ == '__main__':
    with Pool(min(len(FILES), os.cpu_count() or 1, WORKERS)) as pool:
        results = pool.map(run_one_n, FILES.items())

    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    out('A. Observed LM residual-fit MAE on [0,R] versus collocation count N')
    out(f"  {'n':>4} {'Net':6s} {'m':>3}" + ''.join(f"  {'N=' + str(N):>10s}" for N in N_LIST))
    for n, R, xi1, floor, _ in results:
        for b in BASES:
            for m in LM_DEGREES:
                out(f"  {n:>4} {LABEL[b]:6s} {m:>3}" + ''.join(
                    f"  {floor[(N, b, m)]:10.2e}" for N in N_LIST))
        out()

    out('B. Adam (paper settings, 5 seeds) -> LM refinement, N = 30')
    out(f"  {'n':>4} {'m':>3} {'Net':6s} {'MAE Adam':>10} {'MAE Adam+LM':>12} {'std':>9} {'|dxi1|':>10}")
    for n, R, xi1, _, pipe in results:
        for m in PIPE_DEGREES:
            for b in BASES:
                p = pipe[(b, m)]
                z = '' if n == 5 else f"{p[3]:10.2e}" + (f"  (no zero: {p[4]})" if p[4] else '')
                out(f"  {n:>4} {m:>3} {LABEL[b]:6s} {p[0]:10.2e} {p[1]:12.2e} {p[2]:9.1e} {z}")
        out()

    with open(os.path.join(HERE, 'xi1_refine_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
