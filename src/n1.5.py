import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.stats import ttest_ind
from numpy.polynomial.legendre import leggauss
from math import comb
import time
import warnings
warnings.filterwarnings('ignore')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── colours / styles ──────────────────────────────────────────────────────────
COLORS = {
    'bernoulli': '#6B3FA0',
    'chebyshev': '#0D9E75',
    'legendre':  '#D84F2A',
    'euler':     '#888888',
    'heun':      '#CC8800',
    'rk4':       '#0055CC',
    'rk45':      '#009999',
    'exact':     "#7EE8279A",
}
LABELS = {
    'bernoulli': 'BernoulliNN (proposed)',
    'chebyshev': 'ChebyshevNN',
    'legendre':  'LegendreNN',
    'euler':     'Euler  (h=0.01)',
    'heun':      'Heun/RK2 (h=0.01)',
    'rk4':       'RK4  (h=0.01)',
    'rk45':      'RK45 (rtol=1e-6)',
}
LS   = {'bernoulli': '--', 'chebyshev': '-.', 'legendre': ':'}
MRK  = {'bernoulli': 's',  'chebyshev': '^',  'legendre': 'D'}
BASES = ['bernoulli', 'chebyshev', 'legendre']

MARK_EVERY = {'bernoulli': (0, 45), 'chebyshev': (15, 45), 'legendre': (30, 45)}

# ── hyperparameters ───────────────────────────────────────────────────────────
N_POLY        = 1.5           # n = 1.5  →  no exact solution; DOP853 reference
N_TRAIN       = 30
N_VAL         = 105
DEGREE        = 5
X_MAX         = 1.0
LR            = 1e-4
EPOCHS        = 50_000
PAT_CHECKS    = 50
RECORD_EVERY  = 200
ES_TOL        = 1e-4
CLIP_NORM     = 1.0
N_SEEDS       = 5
DEGREE_STUDY  = [3, 5, 7, 9]
N_STUDY       = [10, 20, 30, 50, 80]

# ── reference solution for n=1.5 (no closed-form exact solution exists) ───────
# Reference: DOP853 with rtol=1e-12, atol=1e-14, dense output.
# Near x=0 the series expansion  y ≈ 1 − x²/6 + n·x⁴/120  is used.

_REF_X0 = 1e-8
_ref_sol = None

def _lane_emden_rhs(x, state, n=N_POLY):
    """RHS for n=1.5: y'' = −(2/x)y' − y^n.  Handles x≈0 via series limit."""
    u, du = state
    un = np.maximum(u, 0.0) ** n
    if x < 1e-10:
        return [du, -un / 3.0]
    return [du, -(2.0 / x) * du - un]


def _build_reference():
    global _ref_sol
    x0 = _REF_X0
    y0 = [1.0 - x0**2/6.0 + N_POLY*x0**4/120.0,
          -x0/3.0 + N_POLY*x0**3/30.0]
    _ref_sol = solve_ivp(_lane_emden_rhs, [x0, X_MAX + 0.05], y0,
                         method='DOP853', dense_output=True,
                         rtol=1e-12, atol=1e-14)


from reference_hook import verified_reference

@verified_reference(N_POLY)
def reference_n15(x):
    """High-accuracy reference y(x) for n=1.5."""
    if _ref_sol is None:
        _build_reference()
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.empty_like(x)
    small = x < _REF_X0
    y[small] = 1.0 - x[small]**2/6.0 + N_POLY*x[small]**4/120.0
    if np.any(~small):
        y[~small] = _ref_sol.sol(x[~small])[0]
    return y if y.size > 1 else float(y[0])


# ==============================================================================
#  SECTION 1 — Polynomial bases
# ==============================================================================

def bernoulli_basis(x, degree):
    K, N = degree + 1, len(x)
    B = np.zeros((N, K)); B[:, 0] = 1.0
    for k in range(1, K):
        B[:, k] = x ** k
        for j in range(k):
            B[:, k] -= comb(k, j) / (k - j + 1) * B[:, j]
    dB = np.zeros((N, K))
    for k in range(1, K):
        dB[:, k] = k * B[:, k - 1]
    d2B = np.zeros((N, K))
    for k in range(2, K):
        d2B[:, k] = k * (k - 1) * B[:, k - 2]
    return B, dB, d2B


def chebyshev_basis_exact(x, degree):
    K, N = degree + 1, len(x)
    T = np.zeros((N, K)); U = np.zeros((N, K))
    for k in range(K):
        if k == 0:   T[:, k] = 1.0;   U[:, k] = 1.0
        elif k == 1: T[:, k] = x;     U[:, k] = 2.0 * x
        else:
            T[:, k] = 2.0 * x * T[:, k-1] - T[:, k-2]
            U[:, k] = 2.0 * x * U[:, k-1] - U[:, k-2]
    dT = np.zeros((N, K))
    for k in range(1, K): dT[:, k] = k * U[:, k-1]
    dU = np.zeros((N, K)); dU[:, 0] = 0.0
    if K > 1: dU[:, 1] = 2.0
    for k in range(2, K):
        dU[:, k] = 2.0 * U[:, k-1] + 2.0 * x * dU[:, k-1] - dU[:, k-2]
    d2T = np.zeros((N, K))
    for k in range(1, K): d2T[:, k] = k * dU[:, k-1]
    return T, dT, d2T


def legendre_basis_exact(x, degree):
    K, N = degree + 1, len(x)
    P = np.zeros((N, K))
    for k in range(K):
        if k == 0:   P[:, k] = 1.0
        elif k == 1: P[:, k] = x
        else:
            P[:, k] = ((2*k-1)*x*P[:, k-1] - (k-1)*P[:, k-2]) / k
    dP = np.zeros((N, K)); dP[:, 0] = 0.0
    if K > 1: dP[:, 1] = 1.0
    for k in range(2, K): dP[:, k] = (2*k-1)*P[:, k-1] + dP[:, k-2]
    d2P = np.zeros((N, K)); d2P[:, 0] = 0.0
    if K > 1: d2P[:, 1] = 0.0
    if K > 2: d2P[:, 2] = 3.0
    for k in range(3, K): d2P[:, k] = (2*k-1)*dP[:, k-1] + d2P[:, k-2]
    return P, dP, d2P


# ==============================================================================
#  SECTION 2 — Basis with domain mapping and column normalisation
# ==============================================================================

def _get_basis_raw(name, x_phys, degree, x_max=1.0):
    if name == 'bernoulli':
        t, c = x_phys / x_max, 1.0 / x_max
        P, dP, d2P = bernoulli_basis(t, degree)
    elif name == 'chebyshev':
        t, c = 2.0 * x_phys / x_max - 1.0, 2.0 / x_max
        P, dP, d2P = chebyshev_basis_exact(t, degree)
    else:
        t, c = 2.0 * x_phys / x_max - 1.0, 2.0 / x_max
        P, dP, d2P = legendre_basis_exact(t, degree)
    return P, dP * c, d2P * (c ** 2)


_norm_cache: dict = {}

def get_column_norms(name, degree, x_max=1.0, n_ref=10_000):
    key = (name, degree, x_max)
    if key not in _norm_cache:
        x_ref = np.linspace(1e-6, x_max, n_ref)
        Phi_ref, _, _ = _get_basis_raw(name, x_ref, degree, x_max)
        _norm_cache[key] = np.maximum(np.abs(Phi_ref).max(axis=0), 1e-12)
    return _norm_cache[key]


def get_basis(name, x_phys, degree, x_max=1.0):
    Phi, dPhi, d2Phi = _get_basis_raw(name, x_phys, degree, x_max)
    norms = get_column_norms(name, degree, x_max)
    return Phi / norms, dPhi / norms, d2Phi / norms


# ==============================================================================
#  SECTION 3 — Collocation helpers
# ==============================================================================

def uniform_nodes(N, x_max=1.0):
    return np.linspace(1e-5, x_max, N)


def optimal_collocation(name, N, x_max=1.0):
    x_min = 1e-5
    if name == 'bernoulli':
        return np.linspace(x_min, x_max, N)
    elif name == 'chebyshev':
        j = np.arange(1, N+1, dtype=float)
        t = np.cos(j * np.pi / (N+1))
        return np.sort(np.clip(0.5*(x_max+x_min) + 0.5*(x_max-x_min)*t,
                               x_min, x_max))
    else:
        nodes, _ = leggauss(N)
        return np.sort(np.clip(0.5*(x_max+x_min) + 0.5*(x_max-x_min)*nodes,
                               x_min, x_max))


# ==============================================================================
#  SECTION 4 — Forward pass
# ==============================================================================

def forward(x, W, Phi, dPhi, d2Phi):
    z      = Phi @ W
    tanh_z = np.tanh(z)
    sech2  = 1.0 - tanh_z ** 2
    zp     = dPhi  @ W
    zpp    = d2Phi @ W
    x1 = x.reshape(-1, 1)
    x2 = x1 ** 2
    dN   = sech2 * zp
    d2N  = -2.0 * tanh_z * sech2 * (zp**2) + sech2 * zpp
    y    = (1.0 + x2 * tanh_z).ravel()
    dy   = (2.0 * x1 * tanh_z + x2 * dN).ravel()
    d2y  = (2.0 * tanh_z + 4.0 * x1 * dN + x2 * d2N).ravel()
    return y, dy, d2y


# ==============================================================================
#  SECTION 5 — Loss and analytical gradient  (n = 1.5)
# ==============================================================================
# Residual: r = y'' + (2/x)y' + y^1.5
# d(y^1.5)/dW = 1.5 * y^0.5 * dy/dW  where dy/dW = x^2 * sech^2(z) * Phi

def loss_and_grad_n15(x, W, Phi, dPhi, d2Phi, n=N_POLY):
    """Loss and gradient for n=1.5: residual = y'' + (2/x)y' + y^1.5."""
    z      = Phi @ W
    tanh_z = np.tanh(z)
    sech2  = 1.0 - tanh_z ** 2
    zp     = dPhi  @ W
    zpp    = d2Phi @ W
    x1 = x.reshape(-1, 1)
    x2 = x1 ** 2
    dN  = sech2 * zp
    d2N = -2.0 * tanh_z * sech2 * (zp**2) + sech2 * zpp
    y   = (1.0 + x2 * tanh_z).ravel()
    dy  = (2.0 * x1 * tanh_z + x2 * dN).ravel()
    d2y = (2.0 * tanh_z + 4.0 * x1 * dN + x2 * d2N).ravel()

    yc  = np.maximum(y, 1e-12)            # safe fractional power
    yn  = yc ** n                          # y^1.5
    res  = d2y + (2.0 / x) * dy + yn
    loss = float(np.mean(res ** 2))

    dtanh_dW  = sech2 * Phi
    dsech2_dW = -2.0 * tanh_z * sech2 * Phi
    dY_dW     = x2 * sech2 * Phi                        # dy/dW
    dDY_dW    = 2.0 * x1 * sech2 * Phi + x2 * (dsech2_dW * zp + sech2 * dPhi)
    dd2N_dW   = (-2.0 * (dtanh_dW * sech2 * (zp**2)
                         + tanh_z * dsech2_dW * (zp**2)
                         + tanh_z * sech2 * 2.0 * zp * dPhi)
                 + dsech2_dW * zpp + sech2 * d2Phi)
    dD2Y_dW   = (2.0 * sech2 * Phi
                 + 4.0 * x1 * (dsech2_dW * zp + sech2 * dPhi)
                 + x2 * dd2N_dW)
    # The floored power is constant wherever y <= 1e-12.
    power_slope = n * (yc ** (n - 1.0)) * (y > 1e-12)
    dYN_dW = power_slope[:, None] * dY_dW
    dR_dW = dD2Y_dW + (2.0 / x[:, None]) * dDY_dW + dYN_dW
    grad  = (2.0 / len(x)) * (res[:, None] * dR_dW).sum(axis=0, keepdims=True).T
    return loss, grad


def compute_ode_loss_n15(x, W, Phi, dPhi, d2Phi, n=N_POLY):
    y, dy, d2y = forward(x, W, Phi, dPhi, d2Phi)
    yn = np.maximum(y, 1e-12) ** n
    return float(np.mean((d2y + (2.0/x)*dy + yn)**2))


# ==============================================================================
#  SECTION 6 — Training loop
# ==============================================================================

from train_cache import cached_training

@cached_training
def train(name, degree, x_train, x_val=None, x_max=1.0,
          lr=LR, epochs=EPOCHS, patience_checks=PAT_CHECKS,
          record_every=RECORD_EVERY, es_tol=ES_TOL,
          clip_norm=CLIP_NORM, seed=0):
    np.random.seed(seed)
    K   = degree + 1
    W   = np.random.randn(K, 1) * 0.1
    m_a = np.zeros((K, 1)); v_a = np.zeros((K, 1))
    b1, b2, eps_a = 0.9, 0.999, 1e-8

    Phi_tr, dPhi_tr, d2Phi_tr = get_basis(name, x_train, degree, x_max)
    if x_val is not None:
        Phi_vl, dPhi_vl, d2Phi_vl = get_basis(name, x_val, degree, x_max)

    best_val = np.inf
    patience_val = np.inf
    best_W   = W.copy()
    counter  = 0
    th = []; vh = []; he = []
    t0 = time.time()

    epoch = 0
    for epoch in range(1, epochs + 1):
        loss, grad = loss_and_grad_n15(x_train, W, Phi_tr, dPhi_tr, d2Phi_tr)
        gnorm = np.linalg.norm(grad)
        if gnorm > clip_norm:
            grad *= clip_norm / gnorm
        m_a = b1 * m_a + (1 - b1) * grad
        v_a = b2 * v_a + (1 - b2) * (grad ** 2)
        mh  = m_a / (1 - b1 ** epoch)
        vhh = v_a / (1 - b2 ** epoch)
        W   = W - lr * mh / (np.sqrt(vhh) + eps_a)

        if epoch % record_every == 0:
            # Histories describe the updated weights at the same epoch.
            th.append(compute_ode_loss_n15(x_train, W, Phi_tr, dPhi_tr, d2Phi_tr))
            he.append(epoch)
            if x_val is not None:
                vl = compute_ode_loss_n15(x_val, W, Phi_vl, dPhi_vl, d2Phi_vl)
                vh.append(vl)
                # Keep every absolute improvement; patience uses its own anchor.
                if vl < best_val:
                    best_val = vl
                    best_W = W.copy()
                if vl < patience_val * (1.0 - es_tol):
                    patience_val = vl
                    counter = 0
                else:
                    counter += 1
                if counter > patience_checks:
                    break

        if epoch % 10000 == 0:
            msg = f"    [{name:10s}] Epoch {epoch:5d} | train={loss:.3e}"
            if vh: msg += f"  val={vh[-1]:.3e}"
            print(msg)

    # Include a final partial recording interval in short/custom runs.
    if epoch and epoch % record_every != 0:
        th.append(compute_ode_loss_n15(x_train, W, Phi_tr, dPhi_tr, d2Phi_tr))
        he.append(epoch)
        if x_val is not None:
            vl = compute_ode_loss_n15(x_val, W, Phi_vl, dPhi_vl, d2Phi_vl)
            vh.append(vl)
            if vl < best_val:
                best_val = vl
                best_W = W.copy()
    if x_val is None:
        best_W = W.copy()
    return best_W, th, vh, he, time.time() - t0


# ==============================================================================
#  SECTION 7 — Error metrics
# ==============================================================================

def compute_metrics(y_pred, y_ref):
    err   = y_pred - y_ref
    denom = np.abs(y_ref) + 1e-14
    return {
        'MAE':      float(np.mean(np.abs(err))),
        'RMSE':     float(np.sqrt(np.mean(err**2))),
        'Linf':     float(np.max(np.abs(err))),
        'L2':       float(np.sqrt(np.sum(err**2))),
        'MaxRelE':  float(np.max(np.abs(err) / denom)),
        'MeanRelE': float(np.mean(np.abs(err) / denom)),
    }


# ==============================================================================
#  SECTION 8 — Multi-seed runner
# ==============================================================================

def run_multiseed(name, degree, x_train, x_val, x_eval, y_ref,
                  x_max=X_MAX, n_seeds=N_SEEDS):
    Phi_ev, dPhi_ev, d2Phi_ev = get_basis(name, x_eval, degree, x_max)
    Phi_tr, dPhi_tr, d2Phi_tr = get_basis(name, x_train, degree, x_max)
    if x_val is not None:
        Phi_vl, dPhi_vl, d2Phi_vl = get_basis(name, x_val, degree, x_max)
    all_m = []
    for seed in range(n_seeds):
        W, th, vh, he, elapsed = train(name, degree, x_train,
                                       x_val=x_val, x_max=x_max, seed=seed)
        y_pred, _, _ = forward(x_eval, W, Phi_ev, dPhi_ev, d2Phi_ev)
        m = compute_metrics(y_pred, y_ref)
        checkpoint_train_loss = compute_ode_loss_n15(x_train, W, Phi_tr, dPhi_tr, d2Phi_tr)
        checkpoint_val_loss = (compute_ode_loss_n15(x_val, W, Phi_vl, dPhi_vl, d2Phi_vl)
                               if x_val is not None else None)
        checkpoint_epoch = (he[int(np.argmin(vh))] if vh else (he[-1] if he else 0))
        m.update({'W': W, 'y_pred': y_pred,
                  'train_hist': th, 'val_hist': vh,
                  'hist_epochs': he, 'time': elapsed, 'seed': seed,
                  'checkpoint_train_loss': checkpoint_train_loss,
                  'checkpoint_val_loss': checkpoint_val_loss,
                  'checkpoint_epoch': checkpoint_epoch,
                  'stopped_epoch': he[-1] if he else 0})
        all_m.append(m)

    agg: dict = {}
    for key in ['MAE', 'RMSE', 'Linf', 'L2', 'MaxRelE', 'MeanRelE', 'time']:
        v = [m[key] for m in all_m]
        agg[key + '_mean'] = float(np.mean(v))
        agg[key + '_std']  = float(np.std(v))

    def _score(m):
        return (m['checkpoint_val_loss'] if m['checkpoint_val_loss'] is not None
                else m['checkpoint_train_loss'])
    agg['best'] = all_m[int(np.argmin([_score(m) for m in all_m]))]
    agg['all']  = all_m
    return agg


# ==============================================================================
#  SECTION 9 — Classical numerical methods for comparison
# ==============================================================================

def _series_ic(x0, n=N_POLY):
    y0  = 1.0 - x0**2/6.0 + n*x0**4/120.0
    dy0 = -x0/3.0 + n*x0**3/30.0
    return y0, dy0


def euler_method(x_pts_pos, h=0.01):
    x0 = 1e-5
    y0, dy0 = _series_ic(x0)
    xc, yc, dyc = x0, y0, dy0
    results = []
    for xt in x_pts_pos:
        while xc < xt - 1e-14:
            step = min(h, xt - xc)
            rhs  = _lane_emden_rhs(xc, [yc, dyc])
            yc  += step * rhs[0]
            dyc += step * rhs[1]
            xc  += step
        results.append(yc)
    return np.array(results)


def heun_method(x_pts_pos, h=0.01):
    x0 = 1e-5
    y0, dy0 = _series_ic(x0)
    xc, yc, dyc = x0, y0, dy0
    results = []
    for xt in x_pts_pos:
        while xc < xt - 1e-14:
            step = min(h, xt - xc)
            r1    = _lane_emden_rhs(xc, [yc, dyc])
            ytmp  = yc  + step * r1[0]
            dytmp = dyc + step * r1[1]
            r2    = _lane_emden_rhs(xc + step, [ytmp, dytmp])
            yc   += step * (r1[0] + r2[0]) / 2.0
            dyc  += step * (r1[1] + r2[1]) / 2.0
            xc   += step
        results.append(yc)
    return np.array(results)


def rk4_method(x_pts_pos, h=0.01):
    x0 = 1e-5
    y0, dy0 = _series_ic(x0)
    xc, yc, dyc = x0, y0, dy0
    results = []
    for xt in x_pts_pos:
        while xc < xt - 1e-14:
            step = min(h, xt - xc)
            k1 = _lane_emden_rhs(xc,          [yc,               dyc])
            k2 = _lane_emden_rhs(xc+step/2,   [yc+step/2*k1[0],  dyc+step/2*k1[1]])
            k3 = _lane_emden_rhs(xc+step/2,   [yc+step/2*k2[0],  dyc+step/2*k2[1]])
            k4 = _lane_emden_rhs(xc+step,     [yc+step*k3[0],    dyc+step*k3[1]])
            yc  += step * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
            dyc += step * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
            xc  += step
        results.append(yc)
    return np.array(results)


def rk45_scipy(x_pts_pos, rtol=1e-6, atol=1e-9):
    x0 = 1e-7
    y0, dy0 = _series_ic(x0)
    sol = solve_ivp(_lane_emden_rhs,
                    [x0, x_pts_pos[-1] + 0.1],
                    [y0, dy0], method='RK45',
                    t_eval=x_pts_pos,
                    rtol=rtol, atol=atol)
    return sol.y[0]


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == '__main__':
    from run_outputs import configure
    configure(__file__)

    plt.rcParams.update({
        'figure.dpi':    130,
        'font.size':     10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'lines.linewidth': 2.0,
        'legend.framealpha': 0.92,
        'axes.grid': False,
    })

    print("=" * 72)
    print("  LANE-EMDEN  n = 1.5   |   REFERENCE: DOP853 (rtol=1e-12) — no exact")
    print(f"  DEGREE={DEGREE} (K={DEGREE+1})  N_TRAIN={N_TRAIN}  N_SEEDS={N_SEEDS}")
    print("=" * 72)

    # ── Build reference once ───────────────────────────────────────────────────
    print("\n  Building high-accuracy reference (DOP853, rtol=1e-12) …")
    _build_reference()
    print("    Done.")

    # ── Grids ─────────────────────────────────────────────────────────────────
    x_eval  = np.linspace(1e-5, X_MAX, 500)
    half_dx = X_MAX / (2.0 * N_VAL)
    x_val   = np.linspace(1e-5 + half_dx, X_MAX - half_dx/2.0, N_VAL)
    x_train = uniform_nodes(N_TRAIN, X_MAX)
    y_exact_eval = reference_n15(x_eval)

    # ── 11 specific x points ──────────────────────────────────────────────────
    x_pts     = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 1.0])
    x_pts_pos = x_pts[x_pts > 0.0]
    y_exact_pts = reference_n15(x_pts)

    # ── Train all three FLNN networks ─────────────────────────────────────────
    print("\n  Training FLNN networks (n=1.5) …")
    results: dict = {}
    for basis in BASES:
        print(f"\n  ── {LABELS[basis]} ──")
        agg = run_multiseed(basis, DEGREE, x_train, x_val,
                            x_eval, y_exact_eval)
        results[basis] = agg
        print(f"    MAE = {agg['MAE_mean']:.3e} ± {agg['MAE_std']:.1e}"
              f"   Linf = {agg['Linf_mean']:.3e}"
              f"   t = {agg['time_mean']:.1f}s")

    # ── Classical methods at 11 specific points ────────────────────────────────
    print("\n  Computing classical methods …")
    STEP = 0.01
    cm_euler = euler_method(x_pts_pos, h=STEP)
    cm_heun  = heun_method (x_pts_pos, h=STEP)
    cm_rk4   = rk4_method  (x_pts_pos, h=STEP)
    cm_rk45  = rk45_scipy  (x_pts_pos, rtol=1e-6, atol=1e-9)
    cm = {
        'euler': np.concatenate([[1.0], cm_euler]),
        'heun':  np.concatenate([[1.0], cm_heun]),
        'rk4':   np.concatenate([[1.0], cm_rk4]),
        'rk45':  np.concatenate([[1.0], cm_rk45]),
    }
    print("    Done.")

    # ── FLNN predictions at 11 specific points ─────────────────────────────────
    flnn_pts: dict = {}
    for basis in BASES:
        Phi_p, dPhi_p, d2Phi_p = get_basis(basis, x_pts_pos, DEGREE, X_MAX)
        W_best = results[basis]['best']['W']
        y_pos, _, _ = forward(x_pts_pos, W_best, Phi_p, dPhi_p, d2Phi_p)
        flnn_pts[basis] = np.concatenate([[1.0], y_pos])

    # ── Degree ablation (seed=0) ───────────────────────────────────────────────
    print("\n  Degree ablation …")
    deg_study: dict = {b: {} for b in BASES}
    for deg in DEGREE_STUDY:
        for basis in BASES:
            W, _, _, _, _ = train(basis, deg, x_train, x_val=x_val,
                                  x_max=X_MAX, seed=0)
            Phi_ev, dPhi_ev, d2Phi_ev = get_basis(basis, x_eval, deg, X_MAX)
            yp, _, _ = forward(x_eval, W, Phi_ev, dPhi_ev, d2Phi_ev)
            deg_study[basis][deg] = compute_metrics(yp, y_exact_eval)['MAE']
        print(f"    K={deg+1} done")

    # ── Collocation sensitivity (seed=0) ──────────────────────────────────────
    print("\n  Collocation sensitivity …")
    sens: dict = {b: {} for b in BASES}
    for Nv in N_STUDY:
        x_tr = uniform_nodes(Nv, X_MAX)
        for basis in BASES:
            W, _, _, _, _ = train(basis, DEGREE, x_tr, x_val=x_val,
                                  x_max=X_MAX, seed=0)
            Phi_ev, dPhi_ev, d2Phi_ev = get_basis(basis, x_eval, DEGREE, X_MAX)
            yp, _, _ = forward(x_eval, W, Phi_ev, dPhi_ev, d2Phi_ev)
            sens[basis][Nv] = compute_metrics(yp, y_exact_eval)['MAE']
        print(f"    N={Nv} done")

    # ── Node study (seed=0) ────────────────────────────────────────────────────
    print("\n  Node study (uniform vs optimal) …")
    node_study: dict = {b: {} for b in BASES}
    for basis in BASES:
        for nt, x_tr in [('uniform', uniform_nodes(N_TRAIN, X_MAX)),
                          ('optimal', optimal_collocation(basis, N_TRAIN, X_MAX))]:
            W, _, _, _, _ = train(basis, DEGREE, x_tr, x_val=x_val,
                                  x_max=X_MAX, seed=0)
            Phi_ev, dPhi_ev, d2Phi_ev = get_basis(basis, x_eval, DEGREE, X_MAX)
            yp, _, _ = forward(x_eval, W, Phi_ev, dPhi_ev, d2Phi_ev)
            node_study[basis][nt] = compute_metrics(yp, y_exact_eval)['MAE']
    print("    Done.")

    # ── Gram matrix conditioning ───────────────────────────────────────────────
   # degrees_cond = range(2, 13)
   # kappa = {b: [] for b in BASES}
   # for deg in degrees_cond:
       # x_nodes = uniform_nodes(N_TRAIN, X_MAX)
       # for basis in BASES:
           # Phi_g, _, _ = get_basis(basis, x_nodes, deg, X_MAX)
           ## G = Phi_g.T @ Phi_g
            #sv = np.linalg.svd(G, compute_uv=False)
           # kappa[basis].append(float(sv[0] / (sv[-1] + 1e-300)))


    # ==========================================================================
    #  FIGURES
    # ==========================================================================

    # ── Fig 1: Solution curves — dense grid ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_eval, y_exact_eval, '-', lw=7, color=COLORS['exact'],
            alpha=0.85, solid_capstyle='round',
            label='Reference (DOP853, rtol=1e-12)', zorder=1)
    for b in BASES:
        ax.plot(x_eval, results[b]['best']['y_pred'],
                LS[b], lw=2.2, color=COLORS[b], label=LABELS[b],
                marker=MRK[b], ms=7, markevery=MARK_EVERY[b],
                markerfacecolor='white', markeredgecolor=COLORS[b],
                markeredgewidth=1.6, zorder=5)
    ax.set_xlabel('x'); ax.set_ylabel('y(x)')
    #ax.set_title('Lane-Emden  n = 1.5 — Solution Comparison (best seed)',
                 #fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    plt.tight_layout(); plt.show()

    # ── Fig 2: Absolute error — dense grid ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in BASES:
        err = np.abs(results[b]['best']['y_pred'] - y_exact_eval)
        ax.semilogy(x_eval, err + 1e-20, LS[b], lw=2,
                    color=COLORS[b], label=LABELS[b],
                    marker=MRK[b], ms=6, markevery=MARK_EVERY[b],
                    markerfacecolor='white', markeredgecolor=COLORS[b],
                    markeredgewidth=1.4)
    ax.set_xlabel('x'); ax.set_ylabel('|ŷ − y_ref|')
    #ax.set_title('Lane-Emden  n = 1.5 — Absolute Error (best seed, log scale)',
                 #fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    # ── Fig 3: Training convergence ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in BASES:
        ld = results[b]['best']
        ep = ld['hist_epochs']
        ax.semilogy(ep, ld['train_hist'], '-', lw=2.4,
                    color=COLORS[b], alpha=0.95, label=f"{LABELS[b]} train")
        if ld['val_hist']:
            ax.semilogy(ep[:len(ld['val_hist'])], ld['val_hist'], '--',
                        lw=1.3, color=COLORS[b], alpha=0.45,
                        label=f"{LABELS[b]} val")
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Residual Loss')
    #ax.set_title('Lane-Emden  n = 1.5 — Training Convergence (best seed)',
                 #fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()

    # ── Fig 4: Seed variability — MAE bar chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(BASES))
    for bi, b in enumerate(BASES):
        vals = [results[b]['all'][s]['MAE'] for s in range(N_SEEDS)]
        ax.bar(bi, np.mean(vals), color=COLORS[b],
               label=LABELS[b], alpha=0.85, edgecolor='white', width=0.55)
        ax.errorbar(bi, np.mean(vals), yerr=np.std(vals),
                    fmt='none', color='#333', capsize=5, lw=1.8)
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[b] for b in BASES], fontsize=9)
    ax.set_ylabel(f'MAE  (mean ± std,  {N_SEEDS} seeds)')
    #ax.set_title('Lane-Emden  n = 1.5 — Seed Variability',
                 #fontweight='bold')
    for bi, b in enumerate(BASES):
        vals = [results[b]['all'][s]['MAE'] for s in range(N_SEEDS)]
        ax.text(bi, np.mean(vals)*1.5,
                f"{np.mean(vals):.1e}", ha='center', fontsize=8)
    plt.tight_layout(); plt.show()

    # ── Fig 5: ODE residuals at convergence ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for bi, b in enumerate(BASES):
        ax = axes[bi]
        W_best = results[b]['best']['W']
        Phi_e, dPhi_e, d2Phi_e = get_basis(b, x_eval, DEGREE, X_MAX)
        y_, dy_, d2y_ = forward(x_eval, W_best, Phi_e, dPhi_e, d2Phi_e)
        yn_ = np.maximum(y_, 1e-12) ** N_POLY
        residual = np.abs(d2y_ + (2.0 / x_eval) * dy_ + yn_)
        ax.semilogy(x_eval, residual + 1e-20, '-', lw=2, color=COLORS[b])
        #ax.set_title(LABELS[b], fontweight='bold', color=COLORS[b])
        ax.set_xlabel('x'); ax.set_ylabel('|r(x)|')
        ax.set_xlim(0, 1)
    plt.tight_layout(); plt.show()

    # ── Fig 6: Column sup-norms ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = np.arange(DEGREE + 1)
    offs = {'bernoulli': -0.06, 'chebyshev': 0.0, 'legendre': 0.06}
    for b in BASES:
        norms = get_column_norms(b, DEGREE, X_MAX)
        ax.semilogy(ks + offs[b], norms, '-', lw=2,
                    marker=MRK[b], ms=9,
                    markerfacecolor='white', markeredgecolor=COLORS[b],
                    markeredgewidth=1.8,
                    color=COLORS[b], label=LABELS[b])
        for k, v in zip(ks, norms):
            ax.annotate(f'{v:.2f}', (k + offs[b], v),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=7, color=COLORS[b])
    ax.set_xlabel('Column index  k')
    ax.set_ylabel(r'$\|\Phi_{:,k}\|_\infty$  (before normalisation)')
    ax.set_xticks(ks); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    # ── Fig 7: Gram matrix conditioning  [commented out as in your version] ───
   # degrees_cond = range(2, 13)
   # fig, ax = plt.subplots(figsize=(9, 5))
   # ...
   # plt.tight_layout(); plt.show()

    # ── Fig 8: Degree ablation ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    K_vals = [d + 1 for d in DEGREE_STUDY]
    for b in BASES:
        maes = [deg_study[b][d] for d in DEGREE_STUDY]
        ax.semilogy(K_vals, maes, '-', lw=2,
                    marker=MRK[b], ms=9,
                    markerfacecolor='white', markeredgecolor=COLORS[b],
                    markeredgewidth=1.6,
                    color=COLORS[b], label=LABELS[b])
        for kv, mae in zip(K_vals, maes):
            ax.annotate(f'{mae:.0e}', (kv, mae),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=7, color=COLORS[b])
    ax.set_xlabel('Number of basis functions  K  (= degree + 1)')
    ax.set_ylabel('MAE vs reference solution')
    ax.set_xticks(K_vals)
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    # ── Fig 9: Collocation sensitivity ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in BASES:
        maes = [sens[b][Nv] for Nv in N_STUDY]
        ax.semilogy(N_STUDY, maes, '-', lw=2,
                    marker=MRK[b], ms=9,
                    markerfacecolor='white', markeredgecolor=COLORS[b],
                    markeredgewidth=1.6,
                    color=COLORS[b], label=LABELS[b])
        for nv, mae in zip(N_STUDY, maes):
            ax.annotate(f'{mae:.0e}', (nv, mae),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=7, color=COLORS[b])
    ax.axvline(N_TRAIN, color='gray', ls='--', lw=1.5,
               label=f'Default N={N_TRAIN}')
    ax.set_xlabel('Number of training collocation points  N')
    ax.set_ylabel('MAE vs reference solution')
    ax.set_xticks(N_STUDY)
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    # ── Fig 10: Node study ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(BASES))
    bw = 0.35
    hatches = ['///', '']
    node_types = ['uniform', 'optimal']
    node_colors_alpha = [0.65, 0.95]
    for ni, (nt, ht, al) in enumerate(zip(node_types, hatches,
                                           node_colors_alpha)):
        maes = [node_study[b][nt] for b in BASES]
        ax.bar(x_pos + ni*bw, maes, bw,
               hatch=ht, alpha=al, edgecolor='black',
               color=[COLORS[b] for b in BASES],
               label=f'{nt} nodes')
    ax.set_yscale('log')
    ax.set_xticks(x_pos + bw/2)
    ax.set_xticklabels([LABELS[b] for b in BASES], fontsize=9)
    ax.set_ylabel('MAE vs reference solution')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    # ── Fig 11: Solution at 11 specific x points ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_pts, y_exact_pts, '-', lw=6, color=COLORS['exact'],
            alpha=0.8, solid_capstyle='round', zorder=1)
    ax.plot(x_pts, y_exact_pts, 'o', ms=13, markerfacecolor='none',
            markeredgecolor='black', markeredgewidth=1.8,
            label='Reference (rtol=1e-12)', zorder=2, linestyle='none')
    for b in BASES:
        ax.plot(x_pts, flnn_pts[b], LS[b], lw=1.6,
                marker=MRK[b], ms=6,
                color=COLORS[b], markerfacecolor=COLORS[b],
                markeredgecolor='white', markeredgewidth=0.8,
                alpha=0.95, label=LABELS[b], zorder=5)
    ax.set_xlabel('x'); ax.set_ylabel('y(x)')
    ax.set_xticks(x_pts)
    ax.set_xticklabels([f'{v:.1f}' for v in x_pts])
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    # ── Fig 12: Error at 11 specific x points ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for b in BASES:
        err = np.abs(flnn_pts[b] - y_exact_pts)
        ax.semilogy(x_pts_pos, err[1:] + 1e-20, LS[b], lw=1.8,
                    marker=MRK[b], ms=8,
                    color=COLORS[b], markerfacecolor='white',
                    markeredgecolor=COLORS[b], markeredgewidth=1.6,
                    label=LABELS[b])
    ax.set_xlabel('x')
    ax.set_ylabel('|ŷ − y_ref|  (log scale)')
    ax.set_xticks(x_pts_pos)
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()

    # ── Fig 13: Classical methods comparison ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: solution at 11 points
    ax1.plot(x_pts, y_exact_pts, '-', lw=6, color=COLORS['exact'],
             alpha=0.8, solid_capstyle='round', zorder=1)
    ax1.plot(x_pts, y_exact_pts, 'o', ms=12, markerfacecolor='none',
             markeredgecolor='black', markeredgewidth=1.6,
             label='Reference', zorder=2, linestyle='none')
    for b in BASES:
        ax1.plot(x_pts, flnn_pts[b], LS[b], lw=1.5,
                 marker=MRK[b], ms=5, color=COLORS[b],
                 markeredgecolor='white', markeredgewidth=0.6,
                 label=LABELS[b], zorder=5)
    cm_ls  = {'euler': ':', 'heun': '--', 'rk4': '-.', 'rk45': '-'}
    cm_mrk = {'euler': 'v', 'heun': 'P', 'rk4': '*',  'rk45': 'h'}
    for m in ['euler', 'heun', 'rk4', 'rk45']:
        ax1.plot(x_pts, cm[m], cm_ls[m], lw=1.3,
                 marker=cm_mrk[m], ms=5, color=COLORS[m],
                 markeredgecolor='white', markeredgewidth=0.6,
                 label=LABELS[m], zorder=4)
    ax1.set_xlabel('x'); ax1.set_ylabel('y(x)')
    ax1.set_title('Solutions at specific x points', fontweight='bold')
    #ax1.set_xticks(x_pts)
    ax1.legend(fontsize=7, ncol=2)
   # ax1.grid(alpha=0.3, linestyle='--')

    # Panel 2: absolute error (log scale)
    for b in BASES:
        err = np.abs(flnn_pts[b] - y_exact_pts)
        ax2.semilogy(x_pts_pos, err[1:] + 1e-20, LS[b], lw=2,
                     marker=MRK[b], ms=7, color=COLORS[b],
                     markerfacecolor='white', markeredgecolor=COLORS[b],
                     markeredgewidth=1.5, label=LABELS[b])
    for m in ['euler', 'heun', 'rk4', 'rk45']:
        err = np.abs(cm[m] - y_exact_pts)
        ax2.semilogy(x_pts_pos, err[1:] + 1e-20, cm_ls[m], lw=1.5,
                     marker=cm_mrk[m], ms=6, color=COLORS[m],
                     markeredgecolor='white', markeredgewidth=0.5,
                     label=LABELS[m])
    ax2.set_xlabel('x'); ax2.set_ylabel('|error|  (log scale)')
    ax2.set_title('Absolute Error  (x = 0.1, …, 1.0)', fontweight='bold')
    #ax2.set_xticks(x_pts_pos)
    ax2.legend(fontsize=7, ncol=2)
   # ax2.grid(alpha=0.3, which='both', linestyle='--')
    plt.tight_layout(); plt.show()


    # ==========================================================================
    #  TABLES
    # ==========================================================================

    SEP = "─" * 120

    # ── TABLE 1 ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*120}")
    print("  TABLE 1 — Point-wise: FLNN methods vs Reference Solution")
    print(f"  Lane-Emden  n=1.5  |  Reference: DOP853 (rtol=1e-12)  |  Best seed")
    print(f"{'═'*120}")
    hdr1 = (f"  {'x':>5}  {'Reference':>14}  "
            f"{'BernoulliNN':>14}  {'|Err B|':>12}  "
            f"{'ChebyshevNN':>14}  {'|Err C|':>12}  "
            f"{'LegendreNN':>14}  {'|Err L|':>12}")
    print(hdr1)
    print(f"  {SEP[:116]}")
    for i, x in enumerate(x_pts):
        ye = y_exact_pts[i]
        yb = flnn_pts['bernoulli'][i]
        yc = flnn_pts['chebyshev'][i]
        yl = flnn_pts['legendre'][i]
        print(f"  {x:5.1f}  {ye:14.10f}  "
              f"{yb:14.10f}  {abs(yb-ye):12.3e}  "
              f"{yc:14.10f}  {abs(yc-ye):12.3e}  "
              f"{yl:14.10f}  {abs(yl-ye):12.3e}")
    print(f"  {SEP[:116]}")
    for basis, lbl in [('bernoulli','BernoulliNN'),
                        ('chebyshev','ChebyshevNN'),
                        ('legendre', 'LegendreNN ')]:
        err_arr = np.abs(flnn_pts[basis] - y_exact_pts)
        print(f"  {lbl}  ►  MAE = {np.mean(err_arr):.3e}"
              f"   L∞ = {np.max(err_arr):.3e}")
    print()

    # ── TABLE 2 ────────────────────────────────────────────────────────────────
    print(f"{'═'*120}")
    print("  TABLE 2 — Error Metrics Summary (mean ± std over 5 seeds)")
    print(f"  Lane-Emden  n=1.5  |  Reference: DOP853 (rtol=1e-12)  |  M=500 eval pts")
    print(f"{'═'*120}")
    hdr2 = (f"  {'Method':22}  {'MAE (mean±std)':>26}  "
            f"{'RMSE (mean±std)':>26}  {'L∞':>12}  "
            f"{'MeanRelE':>12}  {'Time(s)':>8}")
    print(hdr2)
    print(f"  {SEP[:116]}")
    for basis in BASES:
        a = results[basis]
        mae_str  = f"{a['MAE_mean']:.3e} ± {a['MAE_std']:.1e}"
        rmse_str = f"{a['RMSE_mean']:.3e} ± {a['RMSE_std']:.1e}"
        print(f"  {LABELS[basis]:22}  {mae_str:>26}  {rmse_str:>26}"
              f"  {a['Linf_mean']:12.3e}  {a['MeanRelE_mean']:12.3e}"
              f"  {a['time_mean']:8.1f}")
    print()

    # ── TABLE 3 ────────────────────────────────────────────────────────────────
    print(f"{'═'*120}")
    print("  TABLE 3 — Comparison: FLNN Methods vs Classical Numerical Methods")
    print(f"  Lane-Emden  n=1.5  |  All errors vs reference DOP853 (rtol=1e-12)")
    print(f"  Classical step:  h = {STEP}   |   FLNN: best seed")
    print(f"{'═'*120}")

    all_methods_ordered = [
        ('exact',     'Reference (rtol=1e-12)',  y_exact_pts),
        ('bernoulli', 'BernoulliNN (proposed)',  flnn_pts['bernoulli']),
        ('legendre',  'LegendreNN',              flnn_pts['legendre']),
        ('chebyshev', 'ChebyshevNN',             flnn_pts['chebyshev']),
        ('rk45',      f'RK45 scipy (rtol=1e-6)', cm['rk45']),
        ('rk4',       f'RK4   (h={STEP})',       cm['rk4']),
        ('heun',      f'Heun  (h={STEP})',       cm['heun']),
        ('euler',     f'Euler (h={STEP})',       cm['euler']),
    ]

    x_cols = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    x_col_idx = [list(x_pts).index(xc) for xc in x_cols]
    col_hdr = "  " + f"{'Method':<28} " + "  ".join([f"y({xc:.1f})" + " " * 5
                                                       for xc in x_cols])
    print(col_hdr)
    print(f"  {SEP[:116]}")
    for key, name, y_arr in all_methods_ordered:
        row_vals = "  ".join([f"{y_arr[idx]:12.8f}" for idx in x_col_idx])
        print(f"  {name:<28} {row_vals}")

    print(f"\n  {SEP[:116]}")
    print(f"  {'Method':<28}  {'MAE':>12}  {'L∞':>12}  {'RMSE':>12}")
    print(f"  {SEP[:60]}")
    for key, name, y_arr in all_methods_ordered:
        if key == 'exact':
            print(f"  {name:<28}  {'—':>12}  {'—':>12}  {'—':>12}")
            continue
        err = np.abs(y_arr - y_exact_pts)
        mae  = np.mean(err)
        linf = np.max(err)
        rmse = np.sqrt(np.mean(err**2))
        print(f"  {name:<28}  {mae:12.3e}  {linf:12.3e}  {rmse:12.3e}")
    print()

    # ── Quick ranking ──────────────────────────────────────────────────────────
    print(f"{'═'*72}")
    print("  RANKING  (by MAE at 11 specific x points, best seed)")
    print(f"{'═'*72}")
    ranking = []
    for key, name, y_arr in all_methods_ordered:
        if key == 'exact': continue
        mae = np.mean(np.abs(y_arr - y_exact_pts))
        ranking.append((mae, name))
    ranking.sort()
    for rank, (mae, name) in enumerate(ranking, 1):
        arrow = "  ← proposed" if "BernoulliNN" in name else ""
        print(f"  {rank}.  {name:<30}  MAE = {mae:.3e}{arrow}")
    print()

    print("=" * 72)
    print("  ALL FIGURES AND 3 TABLES COMPLETE  (n=1.5 only)")
    print("=" * 72)
