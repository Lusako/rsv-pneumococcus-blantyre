# -*- coding: utf-8 -*-
"""Figure 3 — pneumococcal carriage density by virus group, by consensus ARI
status, by individual virus, and adjusted density differences.

Panels C and D share the same row order: Any virus first, then the seven
individual viruses ordered by number of carriers.

Usage:  python3 figure3.py [Merged_Data_2.xlsx] [out.png]
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else 'Merged_Data_2.xlsx'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figure3.png'

GREY, BLUE, AQUA, ORANGE, DARK = '#9aa3ab', '#2a78d6', '#1baf7a', '#eb6834', '#14181c'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'savefig.dpi': 600, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
rng = np.random.default_rng(20240101)


# ---------------------------------------------------------------- models ----
def logit(X, y, firth=False, tol=1e-10, it=400):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        I = np.linalg.pinv(X.T @ (X * W[:, None])); g = X.T @ (y - p)
        if firth:
            H = (X * W[:, None]) @ I @ X.T; g = g + X.T @ (np.diag(H) * (0.5 - p))
        s = I @ g; b += s
        if np.max(np.abs(s)) < tol: break
    return b


def ols(X, y, names):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (r @ r / dof)); t = stats.t.ppf(.975, dof)
    return {n: dict(beta=float(b[i]), lo=float(b[i] - t * se[i]), hi=float(b[i] + t * se[i]),
                    p=float(2 * (1 - stats.t.cdf(abs(b[i] / se[i]), dof))))
            for i, n in enumerate(names) if n != '_c'}


def design(df, terms):
    cols, names = [np.ones(len(df))], ['_c']
    for t in terms:
        if isinstance(t, tuple):
            v, ref = t
            for lv in sorted(z for z in df[v].unique() if z != ref):
                cols.append((df[v] == lv).astype(float).values); names.append(f'{v}={lv}')
        else:
            cols.append(df[t].astype(float).values); names.append(t)
    return np.column_stack(cols), names


# ----------------------------------------------------------------- cohort ---
d0 = pd.read_excel(SRC)
F = 'A history of self-reported feverishness or measured fever of ≥ 38°C'
SUS = 'Clinical suspicion of Acute Respiratory Infection'
VIR = ['SARS-CoV-2', 'FLUA', 'FLUB', 'HRV', 'RSV', 'HPIV', 'HPeV', 'HMPV', 'EV', 'HAdV', 'HBoV', 'FLUC']
lo, hi = d0.columns.get_loc(F), d0.columns.get_loc('Other symptom')
SYMS = [c for c in d0.columns[lo:hi + 1] if c != SUS]
d0['anysym'] = pd.concat([(d0[c] == 'Yes') for c in SYMS], axis=1).any(axis=1)
d = d0[d0.anysym].copy().reset_index(drop=True)                      # 335 symptomatic
Y = lambda c: (d[c] == 'Yes')

d['spn'] = ((d.lytA_Ct <= 35) & (d.piaB_Ct <= 35) & (d.SP2020_Ct <= 35)).fillna(False).astype(int)
d['log10dens'] = np.log10(pd.to_numeric(d.lytA_quantity, errors='coerce'))
d['agegrp'] = pd.cut(d.age, [-1, 0, 1, 2, 9], labels=['<1 y', '1 y', '2 y', '3–5 y']).astype(str)
d['male'] = (d.Gender.astype(str).str.lower().str[0] == 'm').astype(int)
d['season'] = np.where(pd.to_datetime(d['sample date']).dt.month.isin([11, 12, 1, 2, 3, 4]),
                       'Rainy (Nov–Apr)', 'Dry (May–Oct)')
d['anyvir'] = (d[VIR] == 'Positive').any(axis=1).astype(int)
nv = (d[VIR] == 'Positive').sum(axis=1)
d['virgrp'] = np.where(nv == 0, 'No virus', np.where(nv == 1, 'Single virus', 'Multiple viruses'))

onset = pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str) + ' ' +
                       d['Month symptoms started'].astype(str) + ' ' +
                       d['Year symptoms started'].astype(str), format='%d %b %Y', errors='coerce')
gap = (pd.to_datetime(d['sample date']) - onset).dt.days
RESP = ['Cough', 'Sore Throat', 'Shortness of breath OR Tachypnoea', 'Runny nose (rhinorrhea)']
gari = (pd.concat([Y(c) for c in RESP], axis=1).any(axis=1) & gap.between(0, 14)).astype(int)
d['ari'] = (gari.astype(bool) & Y(SUS)).astype(int)
d['agree'] = (gari == Y(SUS).astype(int))

COV = [('agegrp', '<1 y'), 'male', ('season', 'Dry (May–Oct)')]
carr = d[d.spn == 1].dropna(subset=['log10dens']).copy()             # 222 carriers
cS = carr[carr.agree].copy()                                         # carriers in the 274

# ------------------------------------------------- panel C / D row order ----
NM = {'HRV': 'Rhinovirus', 'FLUA': 'Influenza A', 'HBoV': 'Bocavirus', 'RSV': 'RSV',
      'HMPV': 'Metapneumovirus', 'HAdV': 'Adenovirus', 'SARS-CoV-2': 'SARS-CoV-2'}
order = sorted(NM, key=lambda v: -(carr[v] == 'Positive').sum())
ROWS = [('anyvir', 'Any virus')] + [(v, NM[v]) for v in order]       # shared by C and D

EST = []
for col, lab in ROWS:
    mask = (carr['anyvir'] == 1) if col == 'anyvir' else (carr[col] == 'Positive')
    sub = carr[mask | (carr.virgrp == 'No virus')].copy()
    sub['_x'] = mask[mask.index.isin(sub.index)].reindex(sub.index).fillna(False).astype(float)
    X, nm = design(sub, ['_x'] + COV)
    e = ols(X, sub.log10dens.values, nm)['_x']
    EST.append(dict(label=lab, n=int(mask.sum()), vals=carr.loc[mask, 'log10dens'].dropna().values, **e))

# ------------------------------------------------------------------ plot ----
fig = plt.figure(figsize=(9.8, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.55, height_ratios=[1, 1.30])
panel = lambda ax, L: fig.text(ax.get_position().x0 - 0.052, ax.get_position().y1 + 0.055,
                               L, fontsize=14, fontweight='bold', va='top', ha='left')
fmtp = lambda p: ('p<0.001' if p < 0.001 else
                  f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}')

# ---- A : density by virus group ----
ax = fig.add_subplot(gs[0, 0])
grp = ['No virus', 'Single virus', 'Multiple viruses']
vals = {g: carr.loc[carr.virgrp == g, 'log10dens'].values for g in grp}
for i, (g, c) in enumerate(zip(grp, [GREY, BLUE, AQUA])):
    v = vals[g]
    ax.plot(i + rng.normal(0, 0.075, len(v)), v, 'o', ms=3.4, color=c, alpha=.55, mec='none')
    ax.plot([i - .26, i + .26], [np.median(v)] * 2, color=DARK, lw=2.4, zorder=4)
    ax.plot([i, i], [np.percentile(v, 25), np.percentile(v, 75)], color=DARK, lw=1.1, zorder=4)
    ax.text(i, 0.25, f'n={len(v)}', ha='center', fontsize=7.2, color='#666')
sig = [(a, b) for a, b in [(0, 1), (0, 2), (1, 2)]
       if stats.mannwhitneyu(vals[grp[a]], vals[grp[b]]).pvalue < 0.05]
for j, (a, b) in enumerate(sig):
    yh = 6.8 + j * .5; p = float(stats.mannwhitneyu(vals[grp[a]], vals[grp[b]]).pvalue)
    ax.plot([a, a, b, b], [yh - .12, yh, yh, yh - .12], color=DARK, lw=0.9)
    ax.text((a + b) / 2, yh + .05, fmtp(p), ha='center', fontsize=6.8, fontweight='bold', color=DARK)
ax.set_xticks(range(3)); ax.set_xticklabels(['No\nvirus', 'Single\nvirus', 'Multiple\nviruses'])
ax.set_ylabel('log$_{10}$ lytA copies/mL'); ax.set_ylim(0, 7.6); ax.set_xlim(-.6, 2.6)
ax.grid(axis='y', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.set_title('Density by virus group', fontsize=9); panel(ax, 'A.')

# ---- B : density by consensus ARI status ----
ax = fig.add_subplot(gs[0, 1])
lab2 = ['Consensus\nnon-ARI', 'Consensus\nARI']
v2 = [cS.loc[cS.ari == 0, 'log10dens'].values, cS.loc[cS.ari == 1, 'log10dens'].values]
for i, (v, c) in enumerate(zip(v2, [GREY, AQUA])):
    ax.plot(i + rng.normal(0, 0.075, len(v)), v, 'o', ms=3.4, color=c, alpha=.55, mec='none')
    ax.plot([i - .26, i + .26], [np.median(v)] * 2, color=DARK, lw=2.4, zorder=4)
    ax.plot([i, i], [np.percentile(v, 25), np.percentile(v, 75)], color=DARK, lw=1.1, zorder=4)
    ax.text(i, 0.25, f'n={len(v)}', ha='center', fontsize=7.2, color='#666')
pb = float(stats.mannwhitneyu(v2[0], v2[1]).pvalue)
if pb < 0.05:
    ax.plot([0, 0, 1, 1], [6.9, 7.0, 7.0, 6.9], color=DARK, lw=0.9)
    ax.text(.5, 7.05, fmtp(pb), ha='center', fontsize=6.8, fontweight='bold', color=DARK)
ax.set_xticks([0, 1]); ax.set_xticklabels(lab2)
ax.set_ylabel('log$_{10}$ lytA copies/mL'); ax.set_ylim(0, 7.6); ax.set_xlim(-.6, 1.6)
ax.grid(axis='y', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.set_title('Density by consensus ARI status', fontsize=9); panel(ax, 'B.')

# ---- C : raw carrier densities, Any virus first (same order as D) ----
ax = fig.add_subplot(gs[1, 0])
novir = carr.loc[carr.virgrp == 'No virus', 'log10dens'].values
yy = np.arange(len(EST))[::-1]
for i, e in enumerate(EST):
    anyv = e['label'] == 'Any virus'
    col = BLUE if anyv else ORANGE
    jit, ms, al = (0.17, 3.0, .45) if anyv else (0.085, 3.2, .55)
    ax.plot(e['vals'], yy[i] + rng.normal(0, jit, len(e['vals'])), 'o', ms=ms,
            color=col, alpha=al, mec='none', zorder=2)
    ax.plot([np.median(e['vals'])] * 2, [yy[i] - .26, yy[i] + .26], color=DARK, lw=2.2, zorder=4)
    ax.text(7.15, yy[i], str(len(e['vals'])), va='center', ha='right', fontsize=7.0, color='#666')
ax.axvline(np.median(novir), color=DARK, lw=0.9, ls='--', alpha=.55)
ax.set_yticks(yy); ax.set_yticklabels([e['label'] for e in EST])
ax.set_xlabel('log$_{10}$ lytA copies/mL'); ax.set_xlim(1.6, 7.2); ax.set_xticks([2, 3, 4, 5, 6])
ax.set_ylim(-.7, len(EST) - .3)
ax.grid(axis='x', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.set_title('Density in carriers, by virus', fontsize=9); panel(ax, 'C.')

# ---- D : adjusted density differences, same rows ----
ax = fig.add_subplot(gs[1, 1])
for i, e in enumerate(EST):
    sig_ = e['p'] < 0.05; col = BLUE if sig_ else GREY
    ax.plot([e['lo'], e['hi']], [yy[i]] * 2, color=col, lw=2.6, solid_capstyle='round')
    ax.plot([e['beta']], [yy[i]], 'o', color=col, ms=6.5, mec='white', mew=1.0, zorder=3)
    ax.text(1.88, yy[i], str(len(e['vals'])), va='center', ha='right', fontsize=7, color='#555')
    ax.text(2.02, yy[i], fmtp(e['p']), va='center', ha='left', fontsize=6.8,
            color=DARK if sig_ else '#777', fontweight='bold' if sig_ else 'normal')
ax.axvline(0, color=DARK, lw=1.2)
ax.set_yticks(yy); ax.set_yticklabels([e['label'] for e in EST])
ax.set_xlim(-1.3, 3.0); ax.set_xticks([-1, 0, 1, 1.5]); ax.set_ylim(-.7, len(EST) - .3)
ax.set_xlabel('Adjusted difference in log$_{10}$ density (95% CI)')
ax.grid(axis='x', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.spines['left'].set_visible(True)
ax.set_title('Adjusted density difference by virus', fontsize=9); panel(ax, 'D.')

fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.085)
fig.savefig(OUT, facecolor='white'); plt.close(fig)
print(f'wrote {OUT}   carriers={len(carr)}  agreement-subset carriers={len(cS)}')
for e in EST:
    print(f"  {e['label']:17s} n={len(e['vals']):3d}  beta {e['beta']:+.2f} "
          f"({e['lo']:+.2f} to {e['hi']:+.2f})  p={e['p']:.4f}")
