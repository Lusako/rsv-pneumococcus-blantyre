#!/usr/bin/env python3
"""Carriage prevalence and density compared, for one symptom definition.

Definition
----------
General ARI (0-14 d) = at least one of cough, sore throat, shortness of breath or
runny nose, AND symptom onset 0-14 days before the sample was taken.

Reports the statistics for figures/definitions/figure_def_general_ari_30d.png,
which fig_defs.py draws:
  1  pneumococcal carriage PREVALENCE, meets vs does not meet (Wilson CI, Fisher)
  2  carriage DENSITY among carriers, meets vs does not meet (median, Mann-Whitney)
  3  whether adjusting the primary virus models for this definition changes them
  4  the pairwise comparisons drawn as brackets on panels B and C

Usage: stats_definition.py <Merged_Data_2.xlsx> <out_dir> [gari30|consensus|consensus_strict]
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
SRC = sys.argv[1]; OUT = sys.argv[2]
DEF = sys.argv[3] if len(sys.argv) > 3 else 'gari30'   # gari30 | consensus | consensus_strict
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panelstats import pairwise, overall_p


# ---------------------------------------------------------------- statistics
def logit(X, y, names, tol=1e-9, it=200):
    """Newton-Raphson logistic regression; returns per-term OR, CI, p."""
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-X@b)); W = p*(1-p)
        XtWX = X.T@(X*W[:, None])
        step = np.linalg.solve(XtWX + np.eye(X.shape[1])*1e-10, X.T@(y-p))
        b += step
        if np.max(np.abs(step)) < tol: break
    p = 1/(1+np.exp(-X@b)); W = p*(1-p)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T@(X*W[:, None]))))
    return {n: dict(OR=float(np.exp(b[i])), lo=float(np.exp(b[i]-1.96*se[i])),
                    hi=float(np.exp(b[i]+1.96*se[i])),
                    p=float(2*(1-stats.norm.cdf(abs(b[i]/se[i])))))
            for i, n in enumerate(names) if n != '_const'}

def ols(X, y, names):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X@b; dof = len(y) - X.shape[1]
    s2 = r@r/dof
    se = np.sqrt(np.diag(np.linalg.pinv(X.T@X))*s2)
    t = stats.t.ppf(0.975, dof)
    return {n: dict(beta=float(b[i]), lo=float(b[i]-t*se[i]), hi=float(b[i]+t*se[i]),
                    p=float(2*(1-stats.t.cdf(abs(b[i]/se[i]), dof))))
            for i, n in enumerate(names) if n != '_const'}

def design(df, terms):
    cols, names = [np.ones(len(df))], ['_const']
    for t in terms:
        if isinstance(t, tuple):                      # (categorical, reference level)
            var, ref = t
            for lv in sorted(x for x in df[var].unique() if x != ref):
                cols.append((df[var] == lv).astype(float).values); names.append(f'{var}={lv}')
        else:
            cols.append(df[t].astype(float).values); names.append(t)
    return np.column_stack(cols), names

def wilson(k, n, z=1.96):
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return p*100, max(0, c-h)*100, min(1, c+h)*100

def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'

# ---------------------------------------------------------------- data
d = pd.read_excel(SRC)
Y = lambda c: (d[c] == 'Yes')
VIR = ['SARS-CoV-2','FLUA','FLUB','HRV','RSV','HPIV','HPeV','HMPV','EV','HAdV','HBoV','FLUC']
d['anyvir'] = (d[VIR] == 'Positive').any(axis=1).astype(int)
d['spn'] = ((d.lytA_Ct <= 35) & (d.piaB_Ct <= 35) & (d.SP2020_Ct <= 35)).fillna(False).astype(int)
d['log10dens'] = np.log10(pd.to_numeric(d.lytA_quantity, errors='coerce'))
# age bins must match analyse.py: '<1 y' is age 0, not age 0-1
d['agegrp'] = pd.cut(d.age, [-1, 0, 1, 2, 9], labels=['<1 y','1 y','2 y','3-5 y']).astype(str)
d['male'] = (d.Gender.astype(str).str.lower().str[0] == 'm').astype(int)
d['season'] = np.where(pd.to_datetime(d['sample date']).dt.month.isin([11,12,1,2,3,4]),
                       'Rainy', 'Dry')
onset = pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str) + ' ' +
                       d['Month symptoms started'].astype(str) + ' ' +
                       d['Year symptoms started'].astype(str), format='%d %b %Y', errors='coerce')
gap = (pd.to_datetime(d['sample date']) - onset).dt.days
RESP = ['Cough', 'Sore Throat', 'Shortness of breath OR Tachypnoea', 'Runny nose (rhinorrhea)']
SUS = 'Clinical suspicion of Acute Respiratory Infection'
d['gari30'] = (pd.concat([Y(c) for c in RESP], axis=1).any(axis=1) & gap.between(0, 14)).astype(int)
d['consensus'] = (d['gari30'].astype(bool) & Y(SUS)).astype(int)
d['consensus_strict'] = d['consensus']
TITLE = {'consensus_strict': ('CONSENSUS ARI, AGREEMENT ONLY',
                              'the 61 children where General ARI (0-14 d) and the\n'
                              '                       clinician\'s suspicion disagree are excluded'),
         'gari30': ('GENERAL ARI (0-14 d)',
                    'cough / sore throat / shortness of breath / runny nose,\n'
                    '                       with symptom onset 0-14 days before sampling'),
         'consensus': ('CONSENSUS ARI',
                       'yes only where General ARI (0-14 d) and the clinician\'s\n'
                       '                       suspicion of ARI agree')}[DEF]
SLUG = {'gari30': 'general_ari_30d', 'consensus': 'consensus_ari',
        'consensus_strict': 'consensus_ari_agreement_only'}[DEF]
DISPLAY = {'gari30': 'General ARI (0-14 d)', 'consensus': 'Consensus ARI',
           'consensus_strict': 'Consensus ARI (agreement only)'}[DEF]
if DEF == 'consensus_strict':      # keep only the children both definitions agree on
    d = d[d['gari30'] == Y(SUS).astype(int)].copy()

COV = [('agegrp', '<1 y'), 'male', ('season', 'Dry')]
POS, NEG = f'Meets {DISPLAY}', 'Does not meet'
npos, nneg = int(d[DEF].sum()), int((1 - d[DEF]).sum())
carr = d[d.spn == 1].dropna(subset=['log10dens'])
rows = []

print('=' * 70)
print(f'{TITLE[0]}:  {TITLE[1]}')
print('=' * 70)
print(f'meets {npos}/{len(d)} ({npos/len(d)*100:.1f}%);  does not meet {nneg}')
print(f'symptom-onset gap: median {gap.median():.0f} d, IQR {gap.quantile(.25):.0f}-{gap.quantile(.75):.0f} d')

# ---- 1. carriage PREVALENCE ------------------------------------------------
print('\n--- 1. pneumococcal carriage prevalence ---')
k1, n1 = int(d.loc[d[DEF] == 1, 'spn'].sum()), npos
k0, n0 = int(d.loc[d[DEF] == 0, 'spn'].sum()), nneg
p1, l1, h1 = wilson(k1, n1); p0, l0, h0 = wilson(k0, n0)
tab = pd.crosstab(d[DEF], d.spn).values
fp = stats.fisher_exact(tab)[1]
cOR = (tab[1,1]*tab[0,0])/(tab[1,0]*tab[0,1])
print(f'  meets        {k1}/{n1}  {p1:.1f}% (95% CI {l1:.1f}-{h1:.1f})')
print(f'  does not     {k0}/{n0}  {p0:.1f}% (95% CI {l0:.1f}-{h0:.1f})')
print(f'  difference   {p1-p0:+.1f} percentage points;  crude OR {cOR:.2f};  Fisher p = {fmt_p(fp)}')
X, nm = design(d, [DEF] + COV); aC = logit(X, d.spn.values, nm)[DEF]
print(f'  adjusted aOR {aC["OR"]:.2f} (95% CI {aC["lo"]:.2f}-{aC["hi"]:.2f}), p = {fmt_p(aC["p"])}')
rows += [dict(outcome='carriage prevalence', comparison='meets', value=f'{p1:.1f}%',
              ci=f'{l1:.1f}-{h1:.1f}', n=f'{k1}/{n1}', test='Wilson', p=''),
         dict(outcome='carriage prevalence', comparison='does not meet', value=f'{p0:.1f}%',
              ci=f'{l0:.1f}-{h0:.1f}', n=f'{k0}/{n0}', test='Wilson', p=''),
         dict(outcome='carriage prevalence', comparison='crude OR', value=f'{cOR:.2f}',
              ci='', n='', test="Fisher's exact", p=fmt_p(fp)),
         dict(outcome='carriage prevalence', comparison='adjusted OR', value=f'{aC["OR"]:.2f}',
              ci=f'{aC["lo"]:.2f}-{aC["hi"]:.2f}', n='', test='logistic, adjusted',
              p=fmt_p(aC['p']))]

# ---- 2. carriage DENSITY ---------------------------------------------------
print('\n--- 2. pneumococcal carriage density (carriers only) ---')
a = carr.loc[carr[DEF] == 1, 'log10dens']; b = carr.loc[carr[DEF] == 0, 'log10dens']
mw = stats.mannwhitneyu(a, b)
print(f'  meets        n={len(a)}  median {a.median():.2f} log10 (IQR {a.quantile(.25):.2f}-{a.quantile(.75):.2f})')
print(f'  does not     n={len(b)}  median {b.median():.2f} log10 (IQR {b.quantile(.25):.2f}-{b.quantile(.75):.2f})')
print(f'  difference   {a.median()-b.median():+.2f} log10;  Mann-Whitney U p = {fmt_p(mw.pvalue)}')
X, nm = design(carr, [DEF] + COV); aD = ols(X, carr.log10dens.values, nm)[DEF]
print(f'  adjusted beta {aD["beta"]:+.2f} log10 (95% CI {aD["lo"]:+.2f} to {aD["hi"]:+.2f}), '
      f'p = {fmt_p(aD["p"])};  fold difference x{10**aD["beta"]:.2f} '
      f'({10**aD["lo"]:.2f}-{10**aD["hi"]:.2f})')
rows += [dict(outcome='carriage density', comparison='meets', value=f'{a.median():.2f} log10',
              ci=f'{a.quantile(.25):.2f}-{a.quantile(.75):.2f} (IQR)', n=str(len(a)),
              test='median', p=''),
         dict(outcome='carriage density', comparison='does not meet', value=f'{b.median():.2f} log10',
              ci=f'{b.quantile(.25):.2f}-{b.quantile(.75):.2f} (IQR)', n=str(len(b)),
              test='median', p=''),
         dict(outcome='carriage density', comparison='crude difference',
              value=f'{a.median()-b.median():+.2f} log10', ci='', n='',
              test='Mann-Whitney U', p=fmt_p(mw.pvalue)),
         dict(outcome='carriage density', comparison='adjusted beta',
              value=f'{aD["beta"]:+.2f} log10', ci=f'{aD["lo"]:+.2f} to {aD["hi"]:+.2f}', n='',
              test='linear, adjusted', p=fmt_p(aD['p']))]

# ---- 3. does the definition change the virus estimates? --------------------
print('\n--- 3. effect of adjusting the primary models for this definition ---')
for lab, extra in [('no symptom variable', []), (f'+ {DISPLAY}', [DEF])]:
    X, nm = design(d, ['anyvir'] + extra + COV); r = logit(X, d.spn.values, nm)['anyvir']
    X, nm = design(carr, ['anyvir'] + extra + COV); s = ols(X, carr.log10dens.values, nm)['anyvir']
    print(f'  virus -> carriage aOR {r["OR"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f}) | '
          f'density beta {s["beta"]:+.2f} ({s["lo"]:+.2f} to {s["hi"]:+.2f})   [{lab}]')
    rows += [dict(outcome='virus -> carriage', comparison=lab, value=f'{r["OR"]:.2f}',
                  ci=f'{r["lo"]:.2f}-{r["hi"]:.2f}', n='', test='logistic, adjusted',
                  p=fmt_p(r['p'])),
             dict(outcome='virus -> density', comparison=lab, value=f'{s["beta"]:+.2f} log10',
                  ci=f'{s["lo"]:+.2f} to {s["hi"]:+.2f}', n='', test='linear, adjusted',
                  p=fmt_p(s['p']))]

pd.DataFrame(rows).to_csv(f'{OUT}/{SLUG}_statistics.csv', index=False)
print(f'\nstatistics -> {OUT}/{SLUG}_statistics.csv')


# ---- 4. the comparisons drawn on panels B and C ----------------------------
d['grp4'] = np.select(
    [(d.spn == 0) & (d.anyvir == 0), (d.spn == 1) & (d.anyvir == 0),
     (d.spn == 0) & (d.anyvir == 1), (d.spn == 1) & (d.anyvir == 1)],
    ['Neither', 'Pneumococcus only', 'Virus only', 'Pneumococcus + virus'], default='')
MED = d.loc[d.spn == 1, 'log10dens'].median()
d['densgrp'] = np.where(d.spn == 0, 'Not carrier',
                np.where(d.log10dens <= MED, 'Low density', 'High density'))

print('\n--- 4. pairwise comparisons shown on the figure ---')
prs = []
cnt = {g: (int(d.loc[d.grp4 == g, DEF].sum()), int((d.grp4 == g).sum()))
       for g in ['Neither','Pneumococcus only','Virus only','Pneumococcus + virus']}
print(f'  panel B, overall chi-square p = {fmt_p(overall_p(cnt))}')
for c in pairwise(cnt, [('Neither','Pneumococcus only'), ('Neither','Virus only'),
                        ('Pneumococcus only','Virus only'),
                        ('Neither','Pneumococcus + virus')]):
    print(f"    {c['group_a']:21s} {c['pct_a']:5.1f}%  vs  {c['group_b']:21s} {c['pct_b']:5.1f}%"
          f"   p = {fmt_p(c['p'])}  (Holm {fmt_p(c['p_holm'])})")
    prs.append(dict(panel='B pathogen combination', **c))
cntd = {g: (int(d.loc[d.densgrp == g, DEF].sum()), int((d.densgrp == g).sum()))
        for g in ['Not carrier','Low density','High density']}
print(f'  panel C, overall chi-square p = {fmt_p(overall_p(cntd))}')
for c in pairwise(cntd, [('Not carrier','Low density'), ('Not carrier','High density'),
                         ('Low density','High density')]):
    print(f"    {c['group_a']:21s} {c['pct_a']:5.1f}%  vs  {c['group_b']:21s} {c['pct_b']:5.1f}%"
          f"   p = {fmt_p(c['p'])}  (Holm {fmt_p(c['p_holm'])})")
    prs.append(dict(panel='C carriage density', **c))
pd.DataFrame(prs).to_csv(f'{OUT}/{SLUG}_pairwise.csv', index=False)
print(f'\npairwise -> {OUT}/{SLUG}_pairwise.csv')
print(f'figure   -> {'figures/definitions/figure_def_' + SLUG + '.png'} (drawn by fig_defs.py)')
