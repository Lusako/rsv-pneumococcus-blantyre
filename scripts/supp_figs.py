# -*- coding: utf-8 -*-
"""Supplementary Figures 1-3 and Supplementary Table 1.

Pneumococcal carriage and carriage density under four symptom definitions,
all applied to the symptomatic cohort (n = 335).  The Consensus ARI definition
is analysed on the agreement subset (n = 275), as in the main analysis.

Usage:  python3 supp_figs.py <Merged_Data_2.xlsx> <figures_dir> <source_data_dir>
"""
import sys, os, csv
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else '../data/Merged_Data_2.xlsx'
FIGDIR = sys.argv[2] if len(sys.argv) > 2 else '../figures'
SDDIR = sys.argv[3] if len(sys.argv) > 3 else '../source_data'
os.makedirs(FIGDIR, exist_ok=True); os.makedirs(SDDIR, exist_ok=True)

BLUE, GREY, DARK = '#2a78d6', '#9aa3ab', '#14181c'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'savefig.dpi': 600, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})


# ---------- model helpers (identical to recompute.py) ------------------------
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan,) * 3
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p * 100, max(0, c - h) * 100, min(1, c + h) * 100


def logit(X, y, names, firth=False, tol=1e-10, it=400):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        I = np.linalg.pinv(X.T @ (X * W[:, None])); g = X.T @ (y - p)
        if firth:
            H = (X * W[:, None]) @ I @ X.T; g = g + X.T @ (np.diag(H) * (0.5 - p))
        s = I @ g; b += s
        if np.max(np.abs(s)) < tol: break
    p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ (X * W[:, None]))))
    return {n_: dict(OR=float(np.exp(b[i])), lo=float(np.exp(b[i] - 1.96 * se[i])),
                     hi=float(np.exp(b[i] + 1.96 * se[i])),
                     p=float(2 * (1 - stats.norm.cdf(abs(b[i] / se[i])))))
            for i, n_ in enumerate(names) if n_ != '_c'}


def ols(X, y, names):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (r @ r / dof)); t = stats.t.ppf(.975, dof)
    return {n_: dict(beta=float(b[i]), lo=float(b[i] - t * se[i]), hi=float(b[i] + t * se[i]),
                     p=float(2 * (1 - stats.t.cdf(abs(b[i] / se[i]), dof))))
            for i, n_ in enumerate(names) if n_ != '_c'}


def design(df, terms):
    cols, names = [np.ones(len(df))], ['_c']
    for t in terms:
        if isinstance(t, tuple):
            v, ref = t
            for lv in sorted(x for x in df[v].unique() if x != ref):
                cols.append((df[v] == lv).astype(float).values); names.append(f'{v}={lv}')
        else:
            cols.append(df[t].astype(float).values); names.append(t)
    return np.column_stack(cols), names


# ---------- cohort -----------------------------------------------------------
d0 = pd.read_excel(SRC)
Y0 = lambda f, c: (f[c] == 'Yes')
F = 'A history of self-reported feverishness or measured fever of ≥ 38°C'
SUS = 'Clinical suspicion of Acute Respiratory Infection'
VIR = ['SARS-CoV-2', 'FLUA', 'FLUB', 'HRV', 'RSV', 'HPIV', 'HPeV', 'HMPV', 'EV', 'HAdV', 'HBoV', 'FLUC']
lo = d0.columns.get_loc(F); hi = d0.columns.get_loc('Other symptom')
SYMS = [c for c in d0.columns[lo:hi + 1] if c != SUS]
d0['anysym'] = pd.concat([Y0(d0, c) for c in SYMS], axis=1).any(axis=1)
d = d0[d0.anysym].copy().reset_index(drop=True)          # symptomatic cohort, n = 335
Y = lambda c: (d[c] == 'Yes')

d['spn'] = ((d.lytA_Ct <= 35) & (d.piaB_Ct <= 35) & (d.SP2020_Ct <= 35)).fillna(False).astype(int)
d['log10dens'] = np.log10(pd.to_numeric(d.lytA_quantity, errors='coerce'))
d['agegrp'] = pd.cut(d.age, [-1, 0, 1, 2, 9], labels=['<1 y', '1 y', '2 y', '3–5 y']).astype(str)
d['male'] = (d.Gender.astype(str).str.lower().str[0] == 'm').astype(int)
d['season'] = np.where(pd.to_datetime(d['sample date']).dt.month.isin([11, 12, 1, 2, 3, 4]),
                       'Rainy (Nov–Apr)', 'Dry (May–Oct)')
onset = pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str) + ' ' +
                       d['Month symptoms started'].astype(str) + ' ' +
                       d['Year symptoms started'].astype(str), format='%d %b %Y', errors='coerce')
gap = (pd.to_datetime(d['sample date']) - onset).dt.days

RESP = ['Cough', 'Sore Throat', 'Shortness of breath OR Tachypnoea', 'Runny nose (rhinorrhea)']

d['clinsus'] = Y(SUS).astype(int)
d['ili'] = (Y(F) & Y('Cough') & gap.between(0, 10)).astype(int)
d['genari30'] = (pd.concat([Y(c) for c in RESP], axis=1).any(axis=1) & gap.between(0, 14)).astype(int)
d['consensus'] = (d['genari30'].astype(bool) & Y(SUS)).astype(int)
d['agree'] = (d['genari30'] == Y(SUS).astype(int))

COV = [('agegrp', '<1 y'), 'male', ('season', 'Dry (May–Oct)')]

DEFS = [('clinsus', 'Clinical suspicion of ARI', False),
        ('ili', 'WHO influenza-like illness', False),
        ('genari30', 'General ARI (0–14 d)', False),
        ('consensus', 'Consensus ARI (primary)', True)]

d['anyvir'] = (d[VIR] == 'Positive').any(axis=1).astype(int)

rows = []
print(f'symptomatic cohort n = {len(d)}  (excluded {int((~d0.anysym).sum())} with no symptom)')
for col, label, agree_only in DEFS:
    f = d[d.agree].copy() if agree_only else d
    fc = f[f.spn == 1].dropna(subset=['log10dens'])
    k, n = int(f[col].sum()), len(f)
    X, nm = design(f, [col] + COV); o = logit(X, f.spn.values, nm)[col]
    X, nm = design(fc, [col] + COV); b = ols(X, fc.log10dens.values, nm)[col]
    cp, cl, ch = wilson(int(f.loc[f[col] == 1, 'spn'].sum()), int((f[col] == 1).sum()))
    np_, nl, nh = wilson(int(f.loc[f[col] == 0, 'spn'].sum()), int((f[col] == 0).sum()))
    med1 = fc.loc[fc[col] == 1, 'log10dens'].median()
    med0 = fc.loc[fc[col] == 0, 'log10dens'].median()
    mw = float(stats.mannwhitneyu(fc.loc[fc[col] == 1, 'log10dens'],
                                  fc.loc[fc[col] == 0, 'log10dens']).pvalue)
    # virus -> density effect with this symptom definition carried as a covariate
    X, nm = design(fc, ['anyvir', col] + COV); vb = ols(X, fc.log10dens.values, nm)['anyvir']
    rows.append(dict(definition=label, n_analysed=n, n_meeting=k,
                     carriage_meeting_pct=round(cp, 1), carriage_meeting_lo=round(cl, 1),
                     carriage_meeting_hi=round(ch, 1),
                     carriage_not_meeting_pct=round(np_, 1), carriage_not_meeting_lo=round(nl, 1),
                     carriage_not_meeting_hi=round(nh, 1),
                     aOR=round(o['OR'], 2), aOR_lo=round(o['lo'], 2), aOR_hi=round(o['hi'], 2),
                     aOR_p=round(o['p'], 4),
                     n_carriers=len(fc), median_density_meeting=round(med1, 2),
                     median_density_not_meeting=round(med0, 2), density_mannwhitney_p=round(mw, 4),
                     density_beta=round(b['beta'], 3), density_lo=round(b['lo'], 3),
                     density_hi=round(b['hi'], 3), density_p=round(b['p'], 4),
                     virus_density_beta=round(vb['beta'], 3), virus_density_lo=round(vb['lo'], 3),
                     virus_density_hi=round(vb['hi'], 3), virus_density_p=round(vb['p'], 4)))
    print(f"  {label:28s} n={n:3d} meets {k:3d}  aOR {o['OR']:.2f} ({o['lo']:.2f}-{o['hi']:.2f}) "
          f"p={o['p']:.3f} | beta {b['beta']:+.2f} ({b['lo']:+.2f},{b['hi']:+.2f}) p={b['p']:.3f}"
          f" | virus beta {vb['beta']:+.2f} ({vb['lo']:+.2f},{vb['hi']:+.2f})")

with open(f'{SDDIR}/supplementary_table1_definitions.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'wrote {SDDIR}/supplementary_table1_definitions.csv')

# ---------- Supplementary Figure 1 -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
yy = np.arange(len(rows))[::-1]
cols = [BLUE if r['definition'].startswith('Consensus') else GREY for r in rows]
lws = [3.0 if r['definition'].startswith('Consensus') else 2.2 for r in rows]
mss = [8 if r['definition'].startswith('Consensus') else 6 for r in rows]

ax = axes[0]
for i, r in enumerate(rows):
    ax.plot([r['aOR_lo'], r['aOR_hi']], [yy[i]] * 2, color=cols[i], lw=lws[i], solid_capstyle='round')
    ax.plot([r['aOR']], [yy[i]], 'o', color=cols[i], ms=mss[i], mec='white', mew=1.1, zorder=3)
ax.axvline(1, color=DARK, lw=1.2)
ax.set_xscale('log'); ax.minorticks_off()
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(['0.5', '1', '2', '4', '8'])
ax.set_xlim(0.45, 9.0)
ax.set_yticks(yy); ax.set_yticklabels([r['definition'] for r in rows])
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel('Adjusted OR for pneumococcal carriage')
ax.set_title('Carriage prevalence')
ax.grid(axis='x', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.spines['left'].set_visible(True)

ax = axes[1]
for i, r in enumerate(rows):
    ax.plot([r['density_lo'], r['density_hi']], [yy[i]] * 2, color=cols[i], lw=lws[i], solid_capstyle='round')
    ax.plot([r['density_beta']], [yy[i]], 'o', color=cols[i], ms=mss[i], mec='white', mew=1.1, zorder=3)
ax.axvline(0, color=DARK, lw=1.2)
ax.set_yticks(yy); ax.set_yticklabels([str(r['n_meeting']) for r in rows], color='#5b646c')
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel('Adjusted difference in log$_{10}$ density')
ax.set_title('Carriage density in carriers')
ax.grid(axis='x', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.spines['left'].set_visible(True)

for ax, L in zip(axes, ['A.', 'B.']):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.075, 0.97, L, fontsize=14, fontweight='bold', va='top', ha='left')
fig.subplots_adjust(left=0.19, right=0.985, top=0.86, bottom=0.14, wspace=0.42)
fig.savefig(f'{FIGDIR}/supplementary_figure3_definitions.png', facecolor='white')
plt.close(fig)
print(f'wrote {FIGDIR}/supplementary_figure3_definitions.png')


# ============================================================================
# Supplementary Figure 2 -- viral load (Ct) vs pneumococcal carriage density
# ============================================================================
ORANGE, AQUA = '#eb6834', '#1baf7a'
CTCOL = {'RSV': 'Respiratory syncytial virus', 'HRV': 'Rhinovirus', 'FLUA': 'Influenza A',
         'HMPV': 'Human metapneumovirus', 'HAdV': 'Adenovirus', 'HBoV': 'Bocavirus',
         'SARS-CoV-2': 'Covid', 'EV': 'Enteroviruses'}
CTCOL = {k: v for k, v in CTCOL.items() if v in d.columns}
NM = {'HRV': 'Rhinovirus', 'HMPV': 'hMPV', 'RSV': 'RSV', 'HBoV': 'Bocavirus',
      'HAdV': 'Adenovirus', 'SARS-CoV-2': 'SARS-CoV-2', 'FLUA': 'Influenza A', 'EV': 'EV'}
PAL = {'HRV': '#2a78d6', 'HMPV': '#1baf7a', 'RSV': '#eb6834', 'HBoV': '#f0a500',
       'HAdV': '#7b5cd6', 'SARS-CoV-2': '#e0569c', 'FLUA': '#1f7a3d', 'EV': '#9aa3ab'}

dd = d.copy()
for k, col in CTCOL.items():
    dd.loc[dd[k] != 'Positive', col] = np.nan
    dd.loc[dd[col] <= 5, col] = np.nan            # placeholder values, not real Ct
sub = dd[list(CTCOL.values())]
dd['minCt'] = sub.min(axis=1)
okc = sub.notna().any(axis=1)
inv = {v: k for k, v in CTCOL.items()}
wv = pd.Series(pd.NA, index=dd.index, dtype=object); wv[okc] = sub[okc].idxmin(axis=1).map(inv)
dd['whichV'] = wv
cc = dd[(dd.spn == 1) & dd.minCt.notna()].dropna(subset=['log10dens']).copy()
sp = stats.spearmanr(cc.minCt, cc.log10dens)
SPEC = [('Crude', ['minCt']), ('+ age, sex, season', ['minCt'] + COV),
        ('+ virus identity', ['minCt', ('whichV', 'HRV')]),
        ('Fully adjusted', ['minCt', ('whichV', 'HRV')] + COV)]
est = []
for lab, terms in SPEC:
    X, nm = design(cc, terms); est.append((lab, ols(X, cc.log10dens.values, nm)['minCt']))
cc['ctc'] = cc.minCt - cc.groupby('whichV').minCt.transform('mean')
X, nm = design(cc, ['ctc']); est.append(('Within-virus Ct', ols(X, cc.log10dens.values, nm)['ctc']))

with open(f'{SDDIR}/supplementary_figure2_viral_ct_vs_density.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['metric', 'value'])
    w.writerow(['n_carriers_with_virus', len(cc)])
    w.writerow(['spearman_rho', round(float(sp.statistic), 4)])
    w.writerow(['spearman_p', round(float(sp.pvalue), 5)])
    for lab, e in est:
        w.writerow([f'{lab} beta', f"{e['beta']:+.4f} ({e['lo']:+.4f} to {e['hi']:+.4f}), p={e['p']:.4f}"])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
ax = axes[0]
counts = cc.whichV.value_counts()
for v in counts.index:
    m = cc.whichV == v
    ax.plot(cc.loc[m, 'minCt'], cc.loc[m, 'log10dens'], 'o', ms=5.5, mec='white', mew=0.8,
            color=PAL.get(v, GREY), label=f'{NM.get(v, v)} ({int(counts[v])})', zorder=3)
sl, ic, *_ = stats.linregress(cc.minCt, cc.log10dens)
xs = np.linspace(cc.minCt.min(), cc.minCt.max(), 50)
ax.plot(xs, ic + sl * xs, '-', color='#555c63', lw=2,
        label=f'Crude fit (p = {est[0][1]["p"]:.3f})', zorder=2)
ax.invert_xaxis()
ax.set_xlabel('Viral cycle threshold (lower = higher viral load)')
ax.set_ylabel('log$_{10}$ lytA copies/mL')
ax.set_title(f'Viral load vs pneumococcal density (n = {len(cc)})')
ax.grid(color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
h_, l_ = ax.get_legend_handles_labels()
ax.legend(h_, l_, ncol=3, fontsize=7.0, loc='upper center', bbox_to_anchor=(0.5, -0.22),
          frameon=False, handletextpad=0.4, columnspacing=1.3, borderaxespad=0)

ax = axes[1]
yy = np.arange(len(est))[::-1]
for i, (lab, e) in enumerate(est):
    ax.plot([e['lo'], e['hi']], [yy[i]] * 2, color=BLUE, lw=2.6, solid_capstyle='round')
    ax.plot([e['beta']], [yy[i]], 's', color=BLUE, ms=6.5, mec='white', mew=1.0, zorder=3)
ax.axvline(0, color=DARK, lw=1.2)
ax.set_yticks(yy); ax.set_yticklabels([l for l, _ in est])
ax.set_ylim(-0.6, len(est) - 0.4)
ax.set_xlabel('Change in log$_{10}$ density per Ct unit')
ax.set_title('Attenuation on adjustment')
ax.grid(axis='x', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.spines['left'].set_visible(True)

for ax, L in zip(axes, ['A.', 'B.']):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.075, 0.985, L, fontsize=14, fontweight='bold', va='top', ha='left')
fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.30, wspace=0.52)
OUT = f'{FIGDIR}/supplementary_figure2_viral_load.png'
fig.savefig(OUT, facecolor='white'); plt.close(fig)
print(f"wrote {OUT}  (n = {len(cc)}, rho = {sp.statistic:+.3f}, p = {sp.pvalue:.4f})")


# ============================================================================
# Supplementary Figure 1 -- seasonality
# ============================================================================
from matplotlib.colors import LinearSegmentedColormap
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#eef5fe', '#cde2fb', '#9ec5f4', '#6da7ec',
                                                     '#3987e5', '#256abf', '#184f95', '#0d366b'])
MN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
VLAB = {'HRV': 'Rhinovirus', 'FLUA': 'Influenza A', 'HBoV': 'Bocavirus', 'RSV': 'RSV',
        'HMPV': 'Metapneumovirus', 'HAdV': 'Adenovirus', 'SARS-CoV-2': 'SARS-CoV-2',
        'HPIV': 'Parainfluenza', 'FLUB': 'Influenza B', 'EV': 'Enterovirus', 'HPeV': 'Parechovirus'}
d['month'] = pd.to_datetime(d['sample date']).dt.month
x = np.arange(12)
nmo = [int((d.month == m + 1).sum()) for m in range(12)]
spn_p = [d.loc[d.month == m + 1, 'spn'].mean() * 100 if nmo[m] else np.nan for m in range(12)]
vir_p = [d.loc[d.month == m + 1, 'anyvir'].mean() * 100 if nmo[m] else np.nan for m in range(12)]
vlist = [v for v in VIR if v in VLAB]
M = np.array([[int(((d.month == m + 1) & (d[v] == 'Positive')).sum()) for m in range(12)]
              for v in vlist], float)
order = np.argsort(-M.sum(axis=1)); vlist = [vlist[i] for i in order]; M = M[order]
keep = M.sum(axis=1) > 0; vlist = [v for v, k_ in zip(vlist, keep) if k_]; M = M[keep]

with open(f'{SDDIR}/supplementary_figure1_seasonality.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['month', 'n_children', 'carriage_pct', 'any_virus_pct'] + [VLAB[v] for v in vlist])
    for m in range(12):
        w.writerow([MN[m], nmo[m], round(spn_p[m], 1) if nmo[m] else '',
                    round(vir_p[m], 1) if nmo[m] else ''] + [int(M[i, m]) for i in range(len(vlist))])

fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), height_ratios=[1, 1.65])
ax = axes[0]
ax.plot(x, spn_p, '-o', color=BLUE, lw=1.8, ms=4.5, mec='white', mew=.9, label='Pneumococcal carriage')
ax.plot(x, vir_p, '-o', color=ORANGE, lw=1.8, ms=4.5, mec='white', mew=.9, label='Any respiratory virus')
ax.set_ylim(0, 100); ax.set_ylabel('Prevalence (%)')
ax.set_xticks(x); ax.set_xticklabels(MN); ax.set_xlim(-0.5, 11.5)
ax.grid(axis='y', color='#e6e9ec', lw=0.7); ax.set_axisbelow(True)
ax.legend(ncol=2, loc='upper right', fontsize=7.5)
ax.set_title('Monthly prevalence, 2024', fontsize=9)

ax = axes[1]
vmax = M.max(); SEQ.set_bad('#fbfcfd')
im = ax.imshow(np.ma.masked_where(M == 0, M), aspect='auto', cmap=SEQ, vmin=0, vmax=vmax)
ax.set_xticks(x); ax.set_xticklabels([f'{MN[i]}\nn={nmo[i]}' for i in range(12)], fontsize=7.5)
ax.set_yticks(np.arange(len(vlist)))
ax.set_yticklabels([f'{VLAB[v]} ({int(M[i].sum())})' for i, v in enumerate(vlist)], fontsize=8)
for i in range(len(vlist)):
    for j in range(12):
        c = int(M[i, j])
        if c == 0: continue
        ax.text(j, i, str(c), ha='center', va='center', fontsize=7.5,
                color='white' if c > vmax * .55 else DARK, fontweight='bold' if c >= 4 else 'normal')
ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(vlist), 1), minor=True)
ax.grid(which='minor', color='white', lw=1.4); ax.tick_params(which='minor', length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title('Children positive per month, by virus (blank = none detected)', fontsize=9)
cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.028, ticks=[0, vmax / 2, vmax])
cb.ax.set_yticklabels(['0', f'{vmax/2:.0f}', f'{vmax:.0f}'], fontsize=7.5)
cb.outline.set_visible(False); cb.set_label('children', fontsize=7.5)
fig.tight_layout(h_pad=1.6, rect=[0.02, 0, 1, 0.96])
for ax, L in zip(axes, ['A.', 'B.']):
    bb = ax.get_position()
    fig.text(0.012, bb.y1 + 0.035, L, fontsize=14, fontweight='bold', va='top', ha='left')
fig.savefig(f'{FIGDIR}/supplementary_figure1_seasonality.png', facecolor='white'); plt.close(fig)
print(f'wrote {FIGDIR}/supplementary_figure1_seasonality.png')
