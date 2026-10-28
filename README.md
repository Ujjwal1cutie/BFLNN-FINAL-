# bflnn-lane-emden

Code for the paper "A Bernoulli Functional Link Neural Network for Solving the
Lane-Emden Equation: A Comparative Study with Chebyshev and Legendre Neural
Networks" (U. K. Mishra, P. Shukla, ABV-IIITM Gwalior).

Single layer FLNN with Bernoulli polynomials as the functional expansion
(BFLNN), compared with Chebyshev (ChNN) and Legendre (LNN) networks.
Trial solution y = 1 + x^2 tanh(W.Phi(x)). Derivatives are computed
analytically (B_k' = k B_(k-1) for the Bernoulli basis, recurrences for the
other two), no autodiff.

## Setup

```
pip install -r requirements.txt
```

All numbers in the paper were regenerated with this code in one environment:
Python 3.13.14, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1, mpmath 1.3.0,
PyTorch 2.14.1 (CPU), Windows 11, Intel Core Ultra 5 125H. torch is only needed
for `timing_ad.py` and mpmath only for `seeds30_mono_lr.py` and
`reference_precision.py`. The training runs are deterministic (fixed seeds);
other library versions can change the last digits of a result (training of the
Chebyshev network in particular is sensitive to round-off).

## Files

Run everything from inside `src/` (the scripts load the `n*.py` files from
the same folder), or run `python run_all.py` from the repository root.

Main experiments, one file per polytropic index n:

- `n0.py`, `no.5.py` (n = 0.5), `n1.py`, `n1.5.py`, `n2.py`, `n3.py`, `n4.py`, `n5.py`

Each one trains BFLNN, ChNN and LNN (degree 5, 30 points, 5 seeds), prints the
error tables, compares with Euler / Heun / RK4 / RK45 and shows the plots.
Reference: exact solution for n = 0, 1, 5, DOP853 otherwise. The "best seed" is
the seed with the lowest validation loss of the returned weights.

Other scripts:

- `conditioning.py` - LM optimum vs Adam, derivative bounds, Jacobian conditioning
- `hessian_landscape.py` - Hessian spectrum, loss profiles, convergence milestones
- `norm_test.py` - column normalization test
- `activation_test.py` - tanh / identity / sin
- `degree_sweep.py` - degree 3 to 9, attainable error of each degree
- `degree_select.py` - degree chosen by the validation loss of the returned weights (run with `inf` for no clipping)
- `clip_test.py` - gradient clipping norm 1 / 10 / none
- `hyper_grid.py` - learning rate x clipping grid, robustness figure
- `seeds30_mono_lr.py` - 30 seeds, monomial basis, Mann-Whitney tests with Holm correction, derivative errors, learning rate, reference check
- `seed_correlation.py` - Spearman correlation of the 30-seed MAE values between the bases (justifies the independent-sample tests)
- `xi1_domain.py`, `xi1_refine.py` - solution up to the first zero xi_1 (Levenberg-Marquardt, Adam, Adam followed by LM)
- `timing_ad.py` - time of one fixed-weight loss-and-gradient evaluation, analytic vs autodiff
- `kurmanc_compare.py` - comparison in the setting of Kurmanc & Ozavsar (2025)
- `thermal_bvp.py` - thermal explosion (Frank-Kamenetskii) problem
- `plot_seeds.py` - median / range figures over the seeds
- `export_baseline_results.py`, `plot_summary.py` - summary figures (the first writes `results/baseline_results.json`)
- `reference_precision.py` (repository root) - independent 60-digit power-series reference for n = 0.5, 1.5, 2, 3, 4 and its convergence checks; writes `reference_precision.npz/json`
- `verify_numerics.py` (repository root) - finite-difference check of the analytic gradients, basis derivatives and Jacobians (`verification_numerics.json`)

## Script for each table and figure of the paper

| Paper | Script |
|---|---|
| Table 2 | `timing_ad.py` |
| Tables 6 and 20 | `degree_select.py` (also with `inf`), `degree_sweep.py` |
| Table 7, Table 16, Fig. 5, Suppl. Tables S1-S16, Suppl. Figs. S1-S2 | `n0.py` ... `n5.py` |
| Tables 8 and 9, Fig. 8 | `conditioning.py`, `hessian_landscape.py` |
| Tables 10, 11 and 18 | `seeds30_mono_lr.py`, `seed_correlation.py` |
| Fig. 9 | `hyper_grid.py` |
| Table 12 | `kurmanc_compare.py` |
| Tables 13 and 14 | `xi1_domain.py`, `xi1_refine.py` |
| Table 15 | `thermal_bvp.py` |
| Table 17 | `norm_test.py` |
| Table 19 | `activation_test.py` |
| Table 21 | `clip_test.py` |
| Figs. 2-4 | `plot_seeds.py` |
| Figs. 6 and 7 | `export_baseline_results.py`, `plot_summary.py` |

The scripts write their output to `*_out.txt` in `src/`. The outputs used in
the paper are in `results/` (with the stdout logs of the `n*.py` scripts in
`results/logs/` and the raw per-run records in `results/raw/` and
`results/run_details/`), the figures in `figures/`.
Most scripts take 1-10 min (they use multiprocessing, `BFLNN_WORKERS` sets the
number of workers); `seeds30_mono_lr.py` and `hyper_grid.py` take about 20-40 min.

## Settings

Adam (lr 1e-4), gradient clipping at 1, validation every 200 epochs, early
stopping after more than 50 consecutive checks without improvement (51 checks,
i.e. 10,200 epochs; relative tolerance 1e-4), max 50000 epochs, the weights with the
lowest validation loss are returned, W0 ~ N(0, 0.1^2), seeds 0-4 (0-29 in
`seeds30_mono_lr.py`).

For non-integer n the power y^n is evaluated as max(y, 1e-12)^n; on [0, 1] the
trial solution is positive, so the floor is only relevant on the extended
domains of `xi1_domain.py` / `xi1_refine.py`.

## License

MIT, see `LICENSE`.
