# Export of the main-experiment summary (mean/std MAE over the seeds, best-seed MAE at the eleven
# points x = 0, 0.1, ..., 1, classical integrators) used by plot_summary.py.
# usage: python export_baseline_results.py   ->  ../results/baseline_results.json

import contextlib
import importlib.util
import io
import json
import os
import platform
import sys
from multiprocessing import Pool

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))
FILES = {0: ('n0', 'exact_n0'), 0.5: ('no.5', 'reference_n05'), 1: ('n1', 'exact_n1'),
         1.5: ('n1.5', 'reference_n15'), 2: ('n2', 'reference_n2'), 3: ('n3', 'reference_n3'),
         4: ('n4', 'reference_n4'), 5: ('n5', 'exact_n5')}
X_PTS = np.round(np.linspace(0.0, 1.0, 11), 1)


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'), os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics(err):
    err = np.abs(err)
    return {'MAE': float(np.mean(err)), 'RMSE': float(np.sqrt(np.mean(err ** 2))), 'Linf': float(np.max(err))}


def run_case(item):
    n, (fname, ref_name) = item
    mod = load_module(fname)
    ref = getattr(mod, ref_name)
    X = mod.X_MAX
    x_eval = np.linspace(1e-5, X, 500)
    half = X / (2.0 * mod.N_VAL)
    x_val = np.linspace(1e-5 + half, X - half / 2.0, mod.N_VAL)
    x_train = mod.uniform_nodes(mod.N_TRAIN, X)
    y_ref = ref(x_eval)
    x_pos = X_PTS[X_PTS > 0.0]
    y_pts = ref(X_PTS)
    case = {'n': n, 'bases': {}, 'classical11': {}}
    for b in mod.BASES:
        with contextlib.redirect_stdout(io.StringIO()):
            agg = mod.run_multiseed(b, mod.DEGREE, x_train, x_val, x_eval, y_ref)
        best = agg['best']
        Phi, dPhi, d2Phi = mod.get_basis(b, x_pos, mod.DEGREE, X)
        y_pos, _, _ = mod.forward(x_pos, best['W'], Phi, dPhi, d2Phi)
        champion = metrics(np.concatenate([[1.0], y_pos]) - y_pts)
        case['bases'][b] = {
            'dense_summary': {'MAE_mean': agg['MAE_mean'], 'MAE_std': agg['MAE_std'],
                              'RMSE_mean': agg['RMSE_mean'], 'RMSE_std': agg['RMSE_std'],
                              'Linf_mean': agg['Linf_mean'], 'MeanRelE_mean': agg['MeanRelE_mean']},
            'time_mean': agg['time_mean'],
            'champion_seed': int(best['seed']),
            'champion_selection': 'lowest validation loss of the returned weights',
            'champion_metrics_11': champion}
    for key, method in (('euler', mod.euler_method), ('heun', mod.heun_method), ('rk4', mod.rk4_method)):
        y = np.concatenate([[1.0], method(x_pos, h=0.01)])
        case['classical11'][key] = {'metrics_primary': metrics(y - y_pts)}
    y = np.concatenate([[1.0], mod.rk45_scipy(x_pos, rtol=1e-6, atol=1e-9)])
    case['classical11']['rk45'] = {'metrics_primary': metrics(y - y_pts)}
    return f'{n:g}', case


if __name__ == '__main__':
    with Pool(min(len(FILES), WORKERS)) as pool:
        cases = dict(pool.map(run_case, FILES.items()))
    payload = {
        'schema_version': 2,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'numpy': np.__version__, 'scipy': scipy.__version__},
        'metric_definitions': {
            'dense_grid': '500 equally spaced points from 1e-5 to X_MAX',
            'pointwise_grid': '0,0.1,...,1 (eleven points)',
            'reference_policy': 'exact solution for n = 0, 1, 5; DOP853 (rtol 1e-12, atol 1e-14) otherwise'},
        'npz_sha256': None,
        'cases': cases}
    out = os.path.join(HERE, '..', 'results', 'baseline_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1)
    print('written', os.path.abspath(out))
