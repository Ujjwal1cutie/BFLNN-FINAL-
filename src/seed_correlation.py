# Spearman correlation of the 30-seed MAE values between BFLNN and the other bases
# (reads seeds30_mae_raw.json written by seeds30_mono_lr.py, or the copy in results/; Section 4.11)

import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'seeds30_mae_raw.json')            # fresh run of seeds30_mono_lr.py
if not os.path.exists(RAW):
    RAW = os.path.join(HERE, '..', 'results', 'seeds30_mae_raw.json')   # archived output

if __name__ == '__main__':
    data = json.load(open(RAW, encoding='utf-8'))
    lines = ['Spearman rank correlation over the 30 seeds: MAE of BFLNN vs MAE of the other basis',
             f"  {'n':>4}  {'opponent':>10}  {'rho':>6}  {'p':>6}"]
    rhos, n_sig = [], 0
    for n, d in data.items():
        b = np.asarray(d['bernoulli'], float)
        for o in ['legendre', 'chebyshev', 'monomial']:
            rho, p = spearmanr(b, np.asarray(d[o], float))
            rhos.append(abs(rho)); n_sig += p < 0.05
            lines.append(f"  {n:>4}  {o:>10}  {rho:6.2f}  {p:6.2f}")
    lines.append(f'max |rho| = {max(rhos):.2f}; pairs with p < 0.05: {n_sig} of {len(rhos)}')
    text = '\n'.join(lines)
    print(text)
    open(os.path.join(HERE, 'seed_correlation_out.txt'), 'w', encoding='utf-8').write(text + '\n')
