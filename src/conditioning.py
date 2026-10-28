# conditioning of the three bases: LM optimum vs Adam, max |phi''|,
# residual Jacobian (cond, column norms, correlation) and clipping

import contextlib
import importlib.util
import io
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

HERE   = os.path.dirname(os.path.abspath(__file__))
FILES  = {0: 'n0', 0.5: 'no.5', 1: 'n1', 1.5: 'n1.5', 2: 'n2', 3: 'n3', 4: 'n4', 5: 'n5'}
BASES  = ['bernoulli', 'chebyshev', 'legendre']
LABEL  = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN'}
DEGREE = 5
X_MAX  = 1.0
N_TRAIN, N_VAL, N_EVAL = 30, 105, 500
CLIP_EPOCHS = 20_000


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname.replace('.', '_'),
                                                  os.path.join(HERE, fname + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(n, x):
    from verified_reference import enabled, reference_values
    if enabled():
        return reference_values(n, x)
    if n == 0: return 1.0 - x**2 / 6.0
    if n == 1: return np.sin(x) / x
    if n == 5: return 1.0 / np.sqrt(1.0 + x**2 / 3.0)
    x0  = 1e-7
    rhs = lambda t, s: [s[1], -2.0 / t * s[1] - max(s[0], 0.0)**n]
    sol = solve_ivp(rhs, [x0, X_MAX + 0.05], [1 - x0**2 / 6, -x0 / 3],
                    method='DOP853', dense_output=True, rtol=1e-13, atol=1e-15)
    return sol.sol(x)[0]


# ── trial solution y = 1 + x^2 tanh(Phi W) and its derivatives ────────────────
def trial(W, x, Phi, dPhi, d2Phi):
    z    = Phi @ W
    s    = np.tanh(z)
    s2   = 1.0 - s**2
    zp   = dPhi @ W
    zpp  = d2Phi @ W
    y    = 1.0 + x**2 * s
    dy   = 2.0 * x * s + x**2 * s2 * zp
    d2y  = 2.0 * s + 4.0 * x * s2 * zp + x**2 * (-2.0 * s * s2 * zp**2 + s2 * zpp)
    return s, s2, zp, zpp, y, dy, d2y


def residual(W, x, Phi, dPhi, d2Phi, n):
    # On [0, 1] the trial solution satisfies y > 1 - x^2 >= 0, so no positivity floor is needed here
    # (the extended-domain scripts xi1_domain.py / xi1_refine.py use max(y, 1e-12)**n instead).
    *_, y, dy, d2y = trial(W, x, Phi, dPhi, d2Phi)
    return d2y + (2.0 / x) * dy + (np.ones_like(y) if n == 0 else y**n)


def residual_jacobian(W, x, Phi, dPhi, d2Phi, n):
    s, s2, zp, zpp, y, _, _ = trial(W, x, Phi, dPhi, d2Phi)
    x1 = x[:, None]
    s, s2, zp, zpp = s[:, None], s2[:, None], zp[:, None], zpp[:, None]
    dtanh  = s2 * Phi
    dsech2 = -2.0 * s * s2 * Phi
    dY     = x1**2 * dtanh
    dDY    = 2.0 * x1 * dtanh + x1**2 * (dsech2 * zp + s2 * dPhi)
    dd2N   = (-2.0 * (dtanh * s2 * zp**2 + s * dsech2 * zp**2 + s * s2 * 2.0 * zp * dPhi)
              + dsech2 * zpp + s2 * d2Phi)
    dD2Y   = 2.0 * dtanh + 4.0 * x1 * (dsech2 * zp + s2 * dPhi) + x1**2 * dd2N
    J      = dD2Y + (2.0 / x1) * dDY
    if n != 0:
        J = J + n * (y[:, None] ** (n - 1)) * dY
    return J


def cond(A):
    sv = np.linalg.svd(A, compute_uv=False)
    return sv[0] / sv[-1]


def clip_fraction(mod, x, Phi, dPhi, d2Phi, epochs=CLIP_EPOCHS, lr=1e-4, seed=0):
    """Fraction of Adam epochs (paper settings) with ||grad|| > 1."""
    loss_and_grad = getattr(mod, [k for k in dir(mod) if k.startswith('loss_and_grad')][0])
    np.random.seed(seed)
    W = np.random.randn(DEGREE + 1, 1) * 0.1
    m = np.zeros_like(W); v = np.zeros_like(W); clipped = 0
    for t in range(1, epochs + 1):
        _, g = loss_and_grad(x, W, Phi, dPhi, d2Phi)
        gn = np.linalg.norm(g)
        if gn > 1.0:
            g = g / gn; clipped += 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        W = W - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return clipped / epochs


if __name__ == '__main__':
    lines = []
    def out(s=''):
        print(s, flush=True); lines.append(s)

    x_ref = np.linspace(1e-6, X_MAX, 10_000)
    x_tr  = np.linspace(1e-5, X_MAX, N_TRAIN)
    half  = X_MAX / (2.0 * N_VAL)
    x_val = np.linspace(1e-5 + half, X_MAX - half / 2.0, N_VAL)
    x_ev  = np.linspace(1e-5, X_MAX, N_EVAL)

    # ── Part A: Markov-type derivative magnitudes (basis property, n-independent)
    mod1 = load_module('n1')
    out('=' * 78)
    out('A. max_x |phi_k\'\'(x)| of the column-normalised bases on [0,1], k = 0..5')
    out('=' * 78)
    for b in BASES:
        _, _, d2P = mod1.get_basis(b, x_ref, DEGREE, X_MAX)
        out(f"  {LABEL[b]:6s} " + '  '.join(f"{v:7.1f}" for v in np.abs(d2P).max(axis=0)))
    out('  (Chebyshev: 4 k^2(k^2-1)/3 = Markov bound; Legendre: 4 (k-1)k(k+1)(k+2)/8;')
    out('   Bernoulli: k(k-1) nu_{k-2}/nu_k with B_k\'\' = k(k-1) B_{k-2})')

    # ── Part B: LM optimum vs Adam, conditioning, per n
    out()
    out('=' * 78)
    out('B. MAE on M=500 grid: observed Levenberg-Marquardt residual fit vs Adam (paper settings, seed 0)')
    out('   and residual-Jacobian conditioning at the observed fit')
    out('=' * 78)
    out(f"  {'n':>4} {'Net':6s} {'MAE_LM':>10} {'MAE_Adam':>10} {'Adam/LM':>8} "
        f"{'cond(Phi)':>9} {'cond(J)':>8} {'maxcorr':>8} {'Jcol5/Jcol0':>11}")
    summary = {}
    for n, fname in FILES.items():
        mod  = load_module(fname)
        y_ev = reference(n, x_ev)
        for b in BASES:
            P,  dP,  d2P  = mod.get_basis(b, x_tr, DEGREE, X_MAX)
            Pe, dPe, d2Pe = mod.get_basis(b, x_ev, DEGREE, X_MAX)

            np.random.seed(0)
            W0 = np.random.randn(DEGREE + 1) * 0.1
            W_lm = least_squares(residual, W0, jac=residual_jacobian,
                                 args=(x_tr, P, dP, d2P, n), method='lm',
                                 xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20_000).x
            mae_lm = np.mean(np.abs(trial(W_lm, x_ev, Pe, dPe, d2Pe)[4] - y_ev))

            with contextlib.redirect_stdout(io.StringIO()):
                W_adam = mod.train(b, DEGREE, x_tr, x_val=x_val, x_max=X_MAX, seed=0)[0].ravel()
            mae_adam = np.mean(np.abs(trial(W_adam, x_ev, Pe, dPe, d2Pe)[4] - y_ev))

            J     = residual_jacobian(W_lm, x_tr, P, dP, d2P, n)
            G     = J.T @ J
            corr  = np.abs(G / np.sqrt(np.outer(np.diag(G), np.diag(G))) - np.eye(DEGREE + 1)).max()
            jcol  = np.linalg.norm(J, axis=0)
            summary.setdefault(b, []).append((cond(J), corr, jcol))
            out(f"  {n:>4} {LABEL[b]:6s} {mae_lm:10.2e} {mae_adam:10.2e} {mae_adam / mae_lm:8.1f} "
                f"{cond(P):9.1f} {cond(J):8.1f} {corr:8.3f} {jcol[-1] / jcol[0]:11.1f}")

    # ── Part C: gradient clipping activity (n = 1)
    out()
    out('=' * 78)
    out(f'C. Residual-Jacobian column norms (n=1) and fraction of the first {CLIP_EPOCHS} Adam')
    out('   epochs in which gradient clipping (||g|| > 1) is active')
    out('=' * 78)
    for b in BASES:
        P, dP, d2P = mod1.get_basis(b, x_tr, DEGREE, X_MAX)
        jcol = summary[b][2][2]          # n = 1 entry
        frac = clip_fraction(mod1, x_tr, P, dP, d2P)
        out(f"  {LABEL[b]:6s} ||J_k|| = " + ' '.join(f"{v:7.1f}" for v in jcol)
            + f"   clipped: {100 * frac:5.1f}%")

    with open(os.path.join(HERE, 'conditioning_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
