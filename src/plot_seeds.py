# Figures 2-4 with median and min-max band over the 5 seeds

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
OUT   = os.path.join(HERE, 'plots_seeds')
FILES = {'n0': ('n0', 'exact_n0'), 'n0.5': ('no.5', 'reference_n05'), 'n1': ('n1', 'exact_n1'),
         'n1.5': ('n1.5', 'reference_n15'), 'n2': ('n2', 'reference_n2'), 'n3': ('n3', 'reference_n3'),
         'n4': ('n4', 'reference_n4'), 'n5': ('n5', 'exact_n5')}


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'),
                                                  os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_figures(item):
    folder, (fname, ref_name) = item
    mod = load_module(fname)
    X = mod.X_MAX
    x_eval  = np.linspace(1e-5, X, 500)
    half_dx = X / (2.0 * mod.N_VAL)
    x_val   = np.linspace(1e-5 + half_dx, X - half_dx / 2.0, mod.N_VAL)
    x_train = mod.uniform_nodes(mod.N_TRAIN, X)
    y_ref   = getattr(mod, ref_name)(x_eval)

    res = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for b in mod.BASES:
            res[b] = mod.run_multiseed(b, mod.DEGREE, x_train, x_val, x_eval, y_ref)['all']

    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'legend.fontsize': 8, 'lines.linewidth': 2.0,
                         'legend.framealpha': 0.92, 'axes.grid': False,
                         'ps.fonttype': 42, 'pdf.fonttype': 42})
    os.makedirs(os.path.join(OUT, folder), exist_ok=True)
    summary = []

    # ── solution curves, median and range over seeds (paper Figure 2) ─────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_eval, y_ref, '-', lw=7, color=mod.COLORS['exact'], alpha=0.85,
            solid_capstyle='round', label='Reference', zorder=1)
    for b in mod.BASES:
        Y = np.array([m['y_pred'] for m in res[b]])
        ax.fill_between(x_eval, Y.min(axis=0), Y.max(axis=0), color=mod.COLORS[b], alpha=0.25,
                        linewidth=0, zorder=4)
        ax.plot(x_eval, np.median(Y, axis=0), mod.LS[b], lw=2.2, color=mod.COLORS[b], label=mod.LABELS[b],
                marker=mod.MRK[b], ms=7, markevery=mod.MARK_EVERY[b], markerfacecolor='white',
                markeredgecolor=mod.COLORS[b], markeredgewidth=1.6, zorder=5)
        summary.append(f"  {folder:5s} {b:10s} max spread of y over seeds = {(Y.max(axis=0) - Y.min(axis=0)).max():.1e}")
    ax.set_xlabel('x'); ax.set_ylabel('y(x)  (median, min–max band, 5 seeds)')
    ax.set_xlim(0, X)
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, folder, f'Figure_1_seeds.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3: pointwise absolute error, median and range over seeds ───────
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in mod.BASES:
        # errors below 1e-17 (exact zeros in floating point) are shown at 1e-17
        E = np.maximum(np.array([np.abs(m['y_pred'] - y_ref) for m in res[b]]), 1e-17)
        med = np.median(E, axis=0)
        summary.append(f"  {folder:5s} {b:10s} median curve: 10%-90% over x = "
                       f"{np.percentile(med, 10):.1e}..{np.percentile(med, 90):.1e}, max = {med.max():.1e} | "
                       f"Linf per seed = {E.max(axis=1).min():.1e}..{E.max(axis=1).max():.1e}")
        ax.fill_between(x_eval, E.min(axis=0), E.max(axis=0), color=mod.COLORS[b],
                        alpha=0.18, linewidth=0)
        ax.semilogy(x_eval, np.median(E, axis=0), mod.LS[b], lw=2, color=mod.COLORS[b],
                    label=mod.LABELS[b], marker=mod.MRK[b], ms=6, markevery=mod.MARK_EVERY[b],
                    markerfacecolor='white', markeredgecolor=mod.COLORS[b], markeredgewidth=1.4)
    ax.set_xlabel('x'); ax.set_ylabel('|ŷ − y_ref|  (median, min–max band, 5 seeds)')
    ax.set_xlim(0, X); ax.set_ylim(bottom=5e-18)
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, folder, f'Figure_2_seeds.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 4: training / validation loss, median and range over seeds ─────
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in mod.BASES:
        runs = res[b]
        # running minimum (best value so far) of each seed; after early stopping the
        # final value is held, so that every epoch is summarised over all 5 seeds
        ep = np.array(sorted({e for m in runs for e in m['hist_epochs']}))
        tr = np.empty((len(runs), len(ep))); va = np.empty_like(tr)
        for s, m in enumerate(runs):
            he = np.asarray(m['hist_epochs'])
            t_run = np.minimum.accumulate(m['train_hist'])
            v_run = np.minimum.accumulate(m['val_hist'])
            k = np.clip(np.searchsorted(he, ep, side='right') - 1, 0, len(he) - 1)
            tr[s] = t_run[k]; va[s] = v_run[k]
        ax.fill_between(ep, tr.min(axis=0), tr.max(axis=0),
                        color=mod.COLORS[b], alpha=0.18, linewidth=0)
        ax.semilogy(ep, np.median(tr, axis=0), '-', lw=2.4, color=mod.COLORS[b],
                    alpha=0.95, label=f"{mod.LABELS[b]} train")
        ax.semilogy(ep, np.median(va, axis=0), '--', lw=1.3, color=mod.COLORS[b],
                    alpha=0.6, label=f"{mod.LABELS[b]} val")
        stops = [m['hist_epochs'][-1] for m in runs]
        ax.plot(stops, [0.0] * len(stops), linestyle='none', marker='|', ms=12, mew=1.6,
                color=mod.COLORS[b], transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Best MSE residual loss so far  (median, min–max, 5 seeds)')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, folder, f'Figure_3_seeds.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return '\n'.join(summary)


if __name__ == '__main__':
    with Pool(min(len(FILES), os.cpu_count() or 1, WORKERS)) as pool:
        for f in pool.imap_unordered(make_figures, FILES.items()):
            print(f, flush=True)
