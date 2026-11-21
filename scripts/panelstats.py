# -*- coding: utf-8 -*-
"""Pairwise statistics drawn onto the definition figures.

Shared by fig_defs.py and stats_general_ari_30d.py so both draw the comparisons
the same way and report the same numbers.
"""
import numpy as np
from scipy import stats

DARK = '#14181c'
XMAX = 100          # data region
XPAD = 166          # axis limit, the extra room holds the comparison brackets


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return p*100, max(0, c-h)*100, min(1, c+h)*100


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, float); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def fmt_p(p):
    if p < 0.001: return 'p<0.001'
    if p < 0.01:  return f'p={p:.3f}'
    return f'p={p:.2f}'


def pairwise(counts, pairs):
    """counts: {label: (k, n)}.  pairs: [(labelA, labelB), ...].
    Returns list of dicts with Fisher p and Holm-adjusted p."""
    out = []
    for a, b in pairs:
        ka, na = counts[a]; kb, nb = counts[b]
        tab = np.array([[ka, na-ka], [kb, nb-kb]])
        out.append(dict(group_a=a, group_b=b, k_a=ka, n_a=na, k_b=kb, n_b=nb,
                        pct_a=round(ka/na*100, 1), pct_b=round(kb/nb*100, 1),
                        p=float(stats.fisher_exact(tab)[1])))
    adj = holm([o['p'] for o in out])
    for o, q in zip(out, adj):
        o['p_holm'] = float(q)
    return out


def overall_p(counts):
    """Chi-square across all strata."""
    tab = np.array([[k, n-k] for k, n in counts.values()])
    try:
        return float(stats.chi2_contingency(tab)[1])
    except ValueError:
        return float('nan')


def draw_brackets(ax, labels, comps, use_holm=False, x0=120, step=13, fs=6.5):
    """Draw comparison brackets to the right of a horizontal dot plot.

    labels: the panel's labels, top to bottom (same order passed to the plot).
    comps:  output of pairwise(), in the order they should be drawn.
    """
    ypos = {lab: y for lab, y in zip(labels, np.arange(len(labels))[::-1])}
    for j, c in enumerate(comps):
        x = x0 + j*step
        ya, yb = ypos[c['group_a']], ypos[c['group_b']]
        ax.plot([x, x], [ya, yb], color=DARK, lw=0.9, clip_on=False, zorder=5)
        for yy in (ya, yb):
            ax.plot([x-3.4, x], [yy, yy], color=DARK, lw=0.9, clip_on=False, zorder=5)
        p = c['p_holm'] if use_holm else c['p']
        ax.text(x + 2.2, (ya+yb)/2, fmt_p(p), rotation=90, va='center', ha='left',
                fontsize=fs, color=DARK if p < 0.05 else '#666', clip_on=False,
                fontweight='bold' if p < 0.05 else 'normal')
