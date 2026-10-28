"""Run all experiments of the paper in the order used for the reported results.

usage: python run_all.py            (from the repository root)

Wave 1 runs the eight main scripts one after the other (their training times are reported in the
paper), wave 2 runs the remaining analyses in three parallel groups, wave 3 runs the timing
benchmark alone.  Figures of the n*.py scripts are written to figs/, stdout logs to logs/.
Set BFLNN_WORKERS to limit the number of worker processes of each multiprocessing script.
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC, LOGS = os.path.join(ROOT, 'src'), os.path.join(ROOT, 'logs')
os.makedirs(LOGS, exist_ok=True)
env = dict(os.environ, MPLBACKEND='Agg', BFLNN_FIGURE_DIR=os.path.join(ROOT, 'figs'),
           BFLNN_WORKERS=os.environ.get('BFLNN_WORKERS', '6'), PYTHONIOENCODING='utf-8')
env.pop('BFLNN_REFERENCE_ARCHIVE', None)
env.pop('BFLNN_CACHE_DIR', None)
status = {}
status_lock = threading.Lock()   # wave 2 runs three threads that share run_status.json


def run(script, *args, cwd=SRC):
    t = time.time()
    name = script.replace('.py', '') + ('_' + '_'.join(args) if args else '')
    with open(os.path.join(LOGS, name + '.txt'), 'w', encoding='utf-8') as fh:
        code = subprocess.call([sys.executable, script, *args], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    with status_lock:
        status[name] = {'exit': code, 'seconds': round(time.time() - t, 1)}
        with open(os.path.join(ROOT, 'run_status.json'), 'w') as fh:
            json.dump(status, fh, indent=1)


if __name__ == '__main__':
    run('reference_precision.py', cwd=ROOT)
    for s in ['n0.py', 'no.5.py', 'n1.py', 'n1.5.py', 'n2.py', 'n3.py', 'n4.py', 'n5.py']:
        run(s)
    groups = [
        [('conditioning.py',), ('hessian_landscape.py',), ('kurmanc_compare.py',), ('thermal_bvp.py',), ('xi1_domain.py',), ('xi1_refine.py',)],
        [('degree_sweep.py',), ('degree_select.py',), ('degree_select.py', 'inf'), ('norm_test.py',), ('activation_test.py',)],
        [('seeds30_mono_lr.py',), ('seed_correlation.py',), ('clip_test.py',), ('hyper_grid.py',), ('plot_seeds.py',), ('export_baseline_results.py',), ('plot_summary.py',)],
    ]
    threads = [threading.Thread(target=lambda g=g: [run(*x) for x in g]) for g in groups]
    for t in threads: t.start()
    for t in threads: t.join()
    run('timing_ad.py')
    run('verify_numerics.py', cwd=ROOT)
