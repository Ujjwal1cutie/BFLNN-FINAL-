# loss landscape near the optimum: Gauss-Newton Hessian eigenvalues,
# 1-D loss profiles, and convergence milestones of Adam

import contextlib
import importlib.util
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('bca', os.path.join(HERE, 'conditioning.py'))
bca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bca)

FILES, BASES, LABEL = bca.FILES, bca.BASES, bca.LABEL
COLORS = {'bernoulli': '#6B3FA0', 'chebyshev': '#0D9E75', 'legendre': '#D84F2A'}
MARKER = {'bernoulli': 's', 'chebyshev': '^', 'legendre': 'D'}
DEGREE, X_MAX, N_SEEDS = 5, 1.0, 5
THRESH = [1e-4, 1e-6, 1e-8]


def first_epoch_below(hist, epochs, thr):
    idx = np.where(np.asarray(hist) < thr)[0]
    return int(epochs[idx[0]]) if len(idx) else None


if __name__ == '__main__':
    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    x_tr  = np.linspace(1e-5, X_MAX, 30)
    half  = X_MAX / 210.0
    x_val = np.linspace(1e-5 + half, X_MAX - half / 2.0, 105)

    # ── A. convergence milestones ─────────────────────────────────────────────
    out('A. Adam convergence milestones (median over 5 seeds); "-" = never reached')
    out(f"  {'n':>4} {'Net':6s}" + ''.join(f"  {'L<' + format(t, '.0e'):>9s}" for t in THRESH)
        + f"  {'stop ep.':>9s}  {'final val. loss':>15s}")
    milestones = {}
    for n, fname in FILES.items():
        mod = bca.load_module(fname)
        for b in BASES:
            hits = {t: [] for t in THRESH}; stops, finals = [], []
            for seed in range(N_SEEDS):
                with contextlib.redirect_stdout(io.StringIO()):
                    W, th, vh, he, _ = mod.train(b, DEGREE, x_tr, x_val=x_val, x_max=X_MAX, seed=seed)
                for t in THRESH:
                    hits[t].append(first_epoch_below(th, he, t))
                stops.append(he[-1]); finals.append(min(vh))
            med = []
            for t in THRESH:
                v = [h for h in hits[t] if h is not None]
                med.append(f"{int(np.median(v)):9d}" if len(v) >= 3 else f"{'-':>9s}")
            milestones[(n, b)] = (med, np.median(stops), np.median(finals))
            out(f"  {n:>4} {LABEL[b]:6s}" + ''.join(f"  {m}" for m in med)
                + f"  {int(np.median(stops)):9d}  {np.median(finals):15.2e}")
        out()

    # ── B. Gauss-Newton Hessian spectrum at the common optimum ────────────────
    out('B. Eigenvalues of the Gauss-Newton Hessian (2/N) J^T J at the observed LM fit (n = 1)')
    mod = bca.load_module('n1')
    spectra, eigvec, opt = {}, {}, {}
    for b in BASES:
        P, dP, d2P = mod.get_basis(b, x_tr, DEGREE, X_MAX)
        np.random.seed(0)
        W0 = np.random.randn(DEGREE + 1) * 0.1
        Ws = least_squares(bca.residual, W0, jac=bca.residual_jacobian, args=(x_tr, P, dP, d2P, 1),
                           method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15).x
        J = bca.residual_jacobian(Ws, x_tr, P, dP, d2P, 1)
        H = 2.0 / len(x_tr) * J.T @ J
        lam, V = np.linalg.eigh(H)
        spectra[b], eigvec[b], opt[b] = lam[::-1], V[:, ::-1], (Ws, P, dP, d2P)
        out(f"  {LABEL[b]:6s} " + ' '.join(f"{v:9.2e}" for v in lam[::-1])
            + f"   lam_max/lam_min = {lam[-1] / lam[0]:8.1e}")
    out('  (n = 0 ... 5: lam_max differs by < 5% from the n = 1 values; see script output below)')
    for n, fname in FILES.items():
        m = bca.load_module(fname)
        row = []
        for b in BASES:
            P, dP, d2P = m.get_basis(b, x_tr, DEGREE, X_MAX)
            np.random.seed(0)
            Ws = least_squares(bca.residual, np.random.randn(DEGREE + 1) * 0.1, jac=bca.residual_jacobian,
                               args=(x_tr, P, dP, d2P, n), method='lm',
                               xtol=1e-15, ftol=1e-15, gtol=1e-15).x
            J = bca.residual_jacobian(Ws, x_tr, P, dP, d2P, n)
            lam = np.linalg.eigvalsh(2.0 / len(x_tr) * J.T @ J)
            row.append(f"{LABEL[b]}: lam_max={lam[-1]:.2e} lam_min={lam[0]:.2e}")
        out(f"  n={n:<4} " + ' | '.join(row))

    # ── C. figure: spectrum + 1-D profiles ────────────────────────────────────
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'legend.fontsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#555555', 'axes.linewidth': 0.8,
                         'ps.fonttype': 42, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1))
    ax = axes[0]
    for off, b in zip([-0.2, 0.0, 0.2], ['bernoulli', 'legendre', 'chebyshev']):
        ax.plot(np.arange(1, DEGREE + 2) + off, spectra[b], linestyle='none', marker=MARKER[b],
                color=COLORS[b], markersize=6, markeredgecolor='white', markeredgewidth=0.8,
                label=LABEL[b])
    ax.set_yscale('log'); ax.set_xlabel('Eigenvalue index $i$'); ax.set_ylabel(r'$\lambda_i$ of $\mathbf{H}$')
    ax.set_xticks(range(1, DEGREE + 2)); ax.set_title('(a) Gauss–Newton spectrum, $n=1$')
    ax.grid(axis='y', color='#dddddd', linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for ax, which, title in [(axes[1], 0, '(b) stiffest direction $v_1$'),
                             (axes[2], -1, '(c) flattest direction $v_6$')]:
        a = np.linspace(-2e-4, 2e-4, 401)  # common parameter-displacement range, not actual Adam update norms
        for b in ['bernoulli', 'legendre', 'chebyshev']:
            Ws, P, dP, d2P = opt[b]
            v = eigvec[b][:, which]
            L = [np.mean(bca.residual(Ws + t * v, x_tr, P, dP, d2P, 1)**2) for t in a]
            ax.plot(a * 1e4, L, color=COLORS[b], linewidth=1.6, label=LABEL[b])
        for t in (-1.0, 1.0):
            ax.axvline(t, color='#888888', linestyle=':', linewidth=0.9)
        ax.text(1.0, 0.97, r' $\eta=10^{-4}$', transform=ax.get_xaxis_transform(), va='top', fontsize=8, color='#555555')
        ax.set_yscale('log'); ax.set_xlabel(r'step $\alpha$ along unit eigenvector ($\times10^{-4}$)')
        ax.set_ylabel(r'$\mathcal{L}(\mathbf{W}^*+\alpha v)$')
        ax.set_title(title); ax.grid(axis='y', color='#dddddd', linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, 'plot_summary'), exist_ok=True)
    for ext in ('eps', 'pdf', 'png'):
        fig.savefig(os.path.join(HERE, 'plot_summary', f'Fig_loss_landscape.{ext}'), dpi=300, bbox_inches='tight')

    with open(os.path.join(HERE, 'hessian_landscape_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
