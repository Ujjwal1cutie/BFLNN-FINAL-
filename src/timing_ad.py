# Wall time per fixed-weight loss-and-gradient evaluation; no optimizer update.
# Analytical derivatives (NumPy / PyTorch) versus end-to-end autodiff (PyTorch).

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import importlib.util
import platform
import time
from math import comb

import numpy as np
import scipy
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
N_LIST = [30, 120, 480]
M_LIST = [5, 11, 23]                    # K = 6, 12, 24
EPOCHS = 2000
REPEATS = 3


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'),
                                                  os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── basis recurrences in torch (differentiable w.r.t. x) ──────────────────────
def bernoulli_torch(t, m):
    B = [torch.ones_like(t)]
    for k in range(1, m + 1):
        bk = t**k
        for j in range(k):
            bk = bk - comb(k, j) / (k - j + 1) * B[j]
        B.append(bk)
    return torch.stack(B, dim=1)


def chebyshev_torch(t, m):
    T = [torch.ones_like(t), t]
    for k in range(2, m + 1):
        T.append(2 * t * T[-1] - T[-2])
    return torch.stack(T[:m + 1], dim=1)


def best_time(fn, epochs=EPOCHS, repeats=REPEATS, budget=0.3):
    """Median wall time per call over `repeats` blocks; each block runs at most
    `epochs` calls and stops early once `budget` seconds have elapsed."""
    fn()                                          # warm-up
    times = []
    for _ in range(repeats):
        calls, t0 = 0, time.perf_counter()
        while calls < epochs and (calls < 20 or time.perf_counter() - t0 < budget):
            fn(); calls += 1
        times.append((time.perf_counter() - t0) / calls)
    return float(np.median(times))


if __name__ == '__main__':
    mod = load_module('n1')
    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    out(f"Platform : {platform.platform()} | {platform.processor()}")
    out(f"Software : Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}, PyTorch {torch.__version__} (CPU, float64; "
        "PyTorch intra-op threads=1; NumPy thread limits requested via environment)")
    out('Scope: fixed weights; loss and parameter gradient only. Excludes Adam, '
        'gradient clipping, validation, and early stopping.')
    out(f'Protocol: median of {REPEATS} timing blocks after one warm-up call; '
        f'at most {EPOCHS} calls per block, at least 20 calls, 0.3-second target budget.')
    out()

    # ── A. one-time basis construction (Phi, Phi', Phi'') ──────────────────────
    out('A. Raw construction of Phi, Phi\', Phi\'\' (N = 30), microseconds; '
        'excludes normalization-grid and validation-grid preparation')
    out(f"  {'K':>4}  {'Bernoulli':>10}  {'Chebyshev':>10}  {'Legendre':>10}")
    for m in M_LIST:
        x = np.linspace(1e-5, 1.0, 30)
        row = []
        for b in ['bernoulli', 'chebyshev', 'legendre']:
            row.append(1e6 * best_time(lambda: mod._get_basis_raw(b, x, m, 1.0), epochs=200))
        out(f"  {m + 1:>4}  " + '  '.join(f"{v:10.1f}" for v in row))
    out()

    # ── B. loss-and-gradient evaluation cost, with weights held fixed ─────────
    out('B. Time per loss-and-gradient evaluation, microseconds')
    out(f"  {'N':>4} {'K':>3}  {'(i) analytic':>13}  {'(ii) analytic':>13}  "
        f"{'(iii) full AD':>13}  {'(iii) full AD':>13}  {'(iii)/(ii)':>10}")
    out(f"  {'':>4} {'':>3}  {'NumPy, Bern.':>13}  {'torch, Bern.':>13}  "
        f"{'torch, Bern.':>13}  {'torch, Cheb.':>13}  {'Bern.':>10}")
    max_grad_diff = 0.0
    max_cheb_grad_diff = 0.0
    for N in N_LIST:
        for m in M_LIST:
            K = m + 1
            x = np.linspace(1e-5, 1.0, N)
            Phi, dPhi, d2Phi = mod.get_basis('bernoulli', x, m, 1.0)
            norms = mod.get_column_norms('bernoulli', m, 1.0)
            cheb_norms = mod.get_column_norms('chebyshev', m, 1.0)
            np.random.seed(0)
            W_np = np.random.randn(K, 1) * 0.1

            # (i) paper code
            t_i = best_time(lambda: mod.loss_and_grad_n1(x, W_np, Phi, dPhi, d2Phi))

            # (ii) analytic derivatives, reverse-mode for W
            xt = torch.tensor(x)
            P, dP, d2P = (torch.tensor(a) for a in (Phi, dPhi, d2Phi))
            W = torch.tensor(W_np.ravel(), requires_grad=True)
            def analytic_torch():
                W.grad = None
                z = P @ W; s = torch.tanh(z); s2 = 1 - s**2; zp = dP @ W; zpp = d2P @ W
                y = 1 + xt**2 * s
                dy = 2 * xt * s + xt**2 * s2 * zp
                d2y = 2 * s + 4 * xt * s2 * zp + xt**2 * (-2 * s * s2 * zp**2 + s2 * zpp)
                loss = torch.mean((d2y + 2 / xt * dy + y)**2)
                loss.backward()
                return loss, W.grad
            t_ii = best_time(analytic_torch)

            # (iii) full AD: basis rebuilt in the graph, nested derivatives in x
            nb = torch.tensor(norms); nc = torch.tensor(cheb_norms)
            def full_ad(basis_fn, scale, shift, nrm):
                def step():
                    W.grad = None
                    xa = xt.clone().requires_grad_(True)
                    Pa = basis_fn(scale * xa + shift, m) / nrm
                    y = 1 + xa**2 * torch.tanh(Pa @ W)
                    dy = torch.autograd.grad(y.sum(), xa, create_graph=True)[0]
                    d2y = torch.autograd.grad(dy.sum(), xa, create_graph=True)[0]
                    loss = torch.mean((d2y + 2 / xa * dy + y)**2)
                    loss.backward()
                    return loss, W.grad
                return step
            ad_b = full_ad(bernoulli_torch, 1.0, 0.0, nb)
            ad_c = full_ad(chebyshev_torch, 2.0, -1.0, nc)
            t_iii_b = best_time(ad_b)
            t_iii_c = best_time(ad_c)

            # gradient agreement (i) vs (ii) vs (iii)
            g_i = mod.loss_and_grad_n1(x, W_np, Phi, dPhi, d2Phi)[1].ravel()
            g_ii = analytic_torch()[1].detach().numpy().copy()
            g_iii = ad_b()[1].detach().numpy().copy()
            if not all(np.all(np.isfinite(g)) for g in (g_i, g_ii, g_iii)):
                raise FloatingPointError(f'Nonfinite benchmark gradient: N={N}, K={K}')
            rel = max(np.abs(g_ii - g_i).max(), np.abs(g_iii - g_i).max()) / np.abs(g_i).max()
            if not np.isfinite(rel) or rel > 1e-8:
                raise AssertionError(f'Bernoulli gradient disagreement: N={N}, K={K}, rel={rel}')
            max_grad_diff = max(max_grad_diff, rel)
            Pc, dPc, d2Pc = mod.get_basis('chebyshev', x, m, 1.0)
            gc = mod.loss_and_grad_n1(x, W_np, Pc, dPc, d2Pc)[1].ravel()
            gc_ad = ad_c()[1].detach().numpy().copy()
            if not np.all(np.isfinite(gc)) or not np.all(np.isfinite(gc_ad)):
                raise FloatingPointError(f'Nonfinite Chebyshev benchmark gradient: N={N}, K={K}')
            rel_c = np.max(np.abs(gc-gc_ad)) / max(float(np.max(np.abs(gc))), 1e-300)
            if not np.isfinite(rel_c) or rel_c > 1e-8:
                raise AssertionError(f'Chebyshev gradient disagreement: N={N}, K={K}, rel={rel_c}')
            max_cheb_grad_diff = max(max_cheb_grad_diff, rel_c)

            out(f"  {N:>4} {K:>3}  {1e6 * t_i:13.1f}  {1e6 * t_ii:13.1f}  "
                f"{1e6 * t_iii_b:13.1f}  {1e6 * t_iii_c:13.1f}  {t_iii_b / t_ii:10.1f}")
    out()
    out('C. Bernoulli gradient agreement, maximum relative infinity-norm '
        f'difference against (i), across (ii) and (iii): {max_grad_diff:.1e}')
    out('   Chebyshev full-AD versus analytical NumPy gradient, maximum relative '
        f'infinity-norm difference: {max_cheb_grad_diff:.1e}; all finite, tolerance 1e-8')

    with open(os.path.join(HERE, 'timing_ad_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
