# sensitivity to the training hyperparameters: learning rate x clipping norm
# grid (4 x 3 settings), all n, 5 seeds; writes a table and the robustness figure

import contextlib
import importlib.util
import io
import os
from multiprocessing import Pool

WORKERS = max(1, int(os.environ.get('BFLNN_WORKERS', os.cpu_count() or 1)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
FILES = {0: ('n0', 'exact_n0'), 0.5: ('no.5', 'reference_n05'), 1: ('n1', 'exact_n1'),
         1.5: ('n1.5', 'reference_n15'), 2: ('n2', 'reference_n2'), 3: ('n3', 'reference_n3'),
         4: ('n4', 'reference_n4'), 5: ('n5', 'exact_n5')}
BASES = ['bernoulli', 'legendre', 'chebyshev']
LABEL = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
COLOR = {'bernoulli': '#6B3FA0', 'chebyshev': '#0D9E75', 'legendre': '#D84F2A'}
LRS   = [1e-3, 3e-4, 1e-4, 3e-5]
CLIPS = [1.0, 10.0, np.inf]
N_SEEDS = 5


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'), os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(job):
    (n, (fname, ref)), lr = job
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
            maes = []
            for seed in range(N_SEEDS):
                with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
                    W = mod.train(b, mod.DEGREE, x, x_val=xv, x_max=X, lr=lr, clip_norm=c, seed=seed)[0]
                    e = np.mean(np.abs(mod.forward(xe, W, Pe, dPe, d2Pe)[0] - ye))
                maes.append(e if np.isfinite(e) else np.nan)
            out[(b, c)] = np.nanmedian(maes)
    return n, lr, out


if __name__ == '__main__':
    jobs = [(item, lr) for item in FILES.items() for lr in LRS]
    with Pool(min(8, os.cpu_count() or 1, WORKERS)) as pool:
        raw = {}
        for n, lr, out in pool.imap_unordered(run, jobs):
            for (b, c), v in out.items():
                raw[(n, b, lr, c)] = v
            print('done', n, lr, flush=True)

    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)
    emit('Median MAE over 5 seeds, m = 5, for 4 learning rates x 3 clipping norms (12 settings)')
    emit(f"  {'n':>4} {'Net':6s} {'default':>9} {'best':>9} {'worst':>9} {'worst/best':>11} {'default/best':>13}")
    for n in FILES:
        for b in BASES:
            vals = np.array([raw[(n, b, lr, c)] for lr in LRS for c in CLIPS])
            d = raw[(n, b, 1e-4, 1.0)]
            emit(f"  {n:>4} {LABEL[b]:6s} {d:9.2e} {np.nanmin(vals):9.2e} {np.nanmax(vals):9.2e} "
                 f"{np.nanmax(vals) / np.nanmin(vals):11.1f} {d / np.nanmin(vals):13.1f}")
        emit()
    open(os.path.join(HERE, 'hyper_grid_out.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

    # ── figure: one panel per n, 12 settings per basis, default marked ──────────
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.8,
                         'ps.fonttype': 42, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.4), sharey=False)
    rng = np.random.RandomState(0)
    for ax, n in zip(axes.ravel(), FILES):
        for i, b in enumerate(BASES):
            vals = np.array([raw[(n, b, lr, c)] for lr in LRS for c in CLIPS])
            vals = np.maximum(vals, 1e-17)
            xs = i + rng.uniform(-0.18, 0.18, len(vals))
            ax.vlines(i, np.nanmin(vals), np.nanmax(vals), color=COLOR[b], linewidth=1.2, alpha=0.6)
            ax.scatter(xs, vals, s=16, color=COLOR[b], edgecolor='white', linewidth=0.5, zorder=3)
            ax.scatter([i], [max(raw[(n, b, 1e-4, 1.0)], 1e-17)], marker='*', s=110, color=COLOR[b],
                       edgecolor='black', linewidth=0.6, zorder=4)
        ax.set_yscale('log'); ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[b] for b in BASES])
        ax.set_xlim(-0.6, 2.6); ax.set_title(f'$n={n:g}$', fontsize=9.5)
        ax.grid(axis='y', color='#e3e3e3', linewidth=0.6); ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel('MAE (median, 5 seeds)')
    handles = [plt.Line2D([], [], marker='o', linestyle='none', color='#777777', markersize=5,
                          label='one of 12 settings (learning rate $\\times$ clipping norm)'),
               plt.Line2D([], [], marker='*', linestyle='none', color='#777777', markeredgecolor='black',
                          markersize=10, label='default setting ($\\eta=10^{-4}$, clipping norm 1)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, 'plots', f'Fig_hyper_robustness.{ext}'), dpi=300, bbox_inches='tight')
