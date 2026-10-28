"""Summary figures from the freshly verified baseline export, with no stored numbers."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('BFLNN_BASELINE_RESULTS', str(HERE.parent / 'results' / 'baseline_results.json')))
OUT = HERE / 'plot_summary'
OUT.mkdir(exist_ok=True)
with SOURCE.open(encoding='utf-8') as stream:
    data = json.load(stream)
N_VALUES = [0, .5, 1, 1.5, 2, 3, 4, 5]
BASES = ['bernoulli', 'legendre', 'chebyshev']
cases = [data['cases'][f'{n:g}'] for n in N_VALUES]
COLORS = {'bernoulli': '#6B3FA0', 'chebyshev': '#0D9E75', 'legendre': '#D84F2A',
          'euler': '#888888', 'heun': '#CC8800', 'rk4': '#0055CC', 'rk45': '#009999'}
MARKERS = dict(zip(['bernoulli', 'chebyshev', 'legendre', 'euler', 'heun', 'rk4', 'rk45'],
                   ['s', '^', 'D', 'o', 'v', 'P', 'X']))
LABEL = {'bernoulli': 'BFLNN', 'chebyshev': 'ChNN', 'legendre': 'LNN',
         'euler': 'Euler (h=0.01)', 'heun': 'Heun (h=0.01)',
         'rk4': 'RK4 (h=0.01)', 'rk45': 'RK45 (rtol=1e-6)'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def style(ax, ylabel):
    ax.set_yscale('log')
    ax.set_xticks(range(8), [f'{n:g}' for n in N_VALUES])
    ax.set_xlabel('Polytropic index n')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color='#dddddd', linewidth=.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for extension in ('pdf', 'png', 'eps'):
        fig.savefig(OUT / (name + '.' + extension), dpi=220, bbox_inches='tight')
    plt.close(fig)


x = np.arange(8)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for offset, basis in zip([-.18, 0, .18], BASES):
    mean = np.array([c['bases'][basis]['dense_summary']['MAE_mean'] for c in cases])
    std = np.array([c['bases'][basis]['dense_summary']['MAE_std'] for c in cases])
    ax.errorbar(x + offset, np.maximum(mean, 1e-17), yerr=[np.minimum(std, .9*mean), std],
                fmt=MARKERS[basis], color=COLORS[basis], capsize=2.5, label=LABEL[basis])
style(ax, 'MAE (mean +/- std, 5 seeds, 500 points)')
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.14), ncol=3, frameon=False)
fig.tight_layout()
save(fig, 'Fig_summary_mae')

fig, ax = plt.subplots(figsize=(7.2, 4))
for method in ['euler', 'heun', 'rk45', 'rk4'] + BASES:
    values = ([c['bases'][method]['champion_metrics_11']['MAE'] for c in cases] if method in BASES else
              [c['classical11'][method]['metrics_primary']['MAE'] for c in cases])
    ax.plot(x, np.maximum(values, 1e-17), marker=MARKERS[method],
            linestyle='-' if method in BASES else '--', color=COLORS[method],
            label=LABEL[method], linewidth=1.3)
style(ax, 'MAE on eleven points (validation-selected FLNN seed)')
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.23), ncol=4, frameon=False, fontsize=8)
fig.tight_layout()
save(fig, 'Fig_summary_classical')
(OUT / 'provenance.json').write_text(json.dumps({
    'baseline_source': str(SOURCE.resolve()), 'npz_sha256': data['npz_sha256'],
    'reference_policy': data['metric_definitions']['reference_policy'],
    'error_floor_for_log_display_only': 1e-17,
}, indent=2), encoding='utf-8')
print('Saved freshly data-driven summary plots to', OUT)
