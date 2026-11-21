# -*- coding: utf-8 -*-
"""Supplementary Table 2 - virus-specific estimates with Holm-Bonferroni
adjusted p-values.

Three families of seven tests, each virus analysed separately against children
in whom no virus was detected:

  1. pneumococcal carriage prevalence (adjusted OR)   - 335 symptomatic children
  2. pneumococcal carriage density (adjusted beta)    - 222 carriers
  3. consensus ARI (adjusted OR)                      - 274 concordant children

Holm-Bonferroni is applied WITHIN each family, not across all 21 tests: the
three outcomes answer different questions and are not a single search.

Usage:  python3 supplementary_table2.py [Merged_Data_2.xlsx] [out_dir]
Writes: <out_dir>/supplementary_table2_virus_holm.csv  and prints the table.
"""
import sys, os, csv
import numpy as np, pandas as pd
from scipy import stats

SRC = sys.argv[1] if len(sys.argv) > 1 else '../data/Merged_Data_2.xlsx'
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else '../source_data'
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------- models ----
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
    return {n: dict(est=float(np.exp(b[i])), lo=float(np.exp(b[i] - 1.96 * se[i])),
                    hi=float(np.exp(b[i] + 1.96 * se[i])),
                    p=float(2 * (1 - stats.norm.cdf(abs(b[i] / se[i])))))
            for i, n in enumerate(names) if n != '_c'}


def ols(X, y, names):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (r @ r / dof)); t = stats.t.ppf(.975, dof)
    return {n: dict(est=float(b[i]), lo=float(b[i] - t * se[i]), hi=float(b[i] + t * se[i]),
                    p=float(2 * (1 - stats.t.cdf(abs(b[i] / se[i]), dof))))
            for i, n in enumerate(names) if n != '_c'}


def holm(p):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    out = np.empty(n); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (n - i) * p[idx])          # enforce monotonicity
        out[idx] = min(run, 1.0)
    return out


# ----------------------------------------------------------------- cohort ---
d0 = pd.read_excel(SRC)
F = 'A history of self-reported feverishness or measured fever of ≥ 38°C'
SUS = 'Clinical suspicion of Acute Respiratory Infection'
VIR = ['SARS-CoV-2', 'FLUA', 'FLUB', 'HRV', 'RSV', 'HPIV', 'HPeV', 'HMPV', 'EV', 'HAdV', 'HBoV', 'FLUC']
lo, hi = d0.columns.get_loc(F), d0.columns.get_loc('Other symptom')
SYMS = [c for c in d0.columns[lo:hi + 1] if c != SUS]
d0['anysym'] = pd.concat([(d0[c] == 'Yes') for c in SYMS], axis=1).any(axis=1)
d = d0[d0.anysym].copy().reset_index(drop=True)                      # 335
Y = lambda c: (d[c] == 'Yes')

d['spn'] = ((d.lytA_Ct <= 35) & (d.piaB_Ct <= 35) & (d.SP2020_Ct <= 35)).fillna(False).astype(int)
d['log10dens'] = np.log10(pd.to_numeric(d.lytA_quantity, errors='coerce'))
d['agegrp'] = pd.cut(d.age, [-1, 0, 1, 2, 9], labels=['<1 y', '1 y', '2 y', '3–5 y']).astype(str)
d['male'] = (d.Gender.astype(str).str.lower().str[0] == 'm').astype(int)
d['season'] = np.where(pd.to_datetime(d['sample date']).dt.month.isin([11, 12, 1, 2, 3, 4]),
                       'Rainy (Nov–Apr)', 'Dry (May–Oct)')
d['novir'] = ((d[VIR] == 'Positive').sum(axis=1) == 0)

onset = pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str) + ' ' +
                       d['Month symptoms started'].astype(str) + ' ' +
                       d['Year symptoms started'].astype(str), format='%d %b %Y', errors='coerce')
gap = (pd.to_datetime(d['sample date']) - onset).dt.days
RESP = ['Cough', 'Sore Throat', 'Shortness of breath OR Tachypnoea', 'Runny nose (rhinorrhea)']
gari = (pd.concat([Y(c) for c in RESP], axis=1).any(axis=1) & gap.between(0, 14)).astype(int)
d['ari'] = (gari.astype(bool) & Y(SUS)).astype(int)
d['agree'] = (gari == Y(SUS).astype(int))

COV = [('agegrp', '<1 y'), 'male', ('season', 'Dry (May–Oct)')]
carr = d[d.spn == 1].dropna(subset=['log10dens']).copy()             # 222
S = d[d.agree].copy()                                                # 274

NM = {'HRV': 'Rhinovirus', 'FLUA': 'Influenza A', 'HBoV': 'Bocavirus', 'RSV': 'RSV',
      'HMPV': 'Metapneumovirus', 'HAdV': 'Adenovirus', 'SARS-CoV-2': 'SARS-CoV-2'}
ORDER = sorted(NM, key=lambda v: -(d[v] == 'Positive').sum())        # by n positive


# ------------------------------------------------------- the three families -
def family(frame, outcome, kind, restrict=True, firth=None):
    """restrict=True  : each virus compared with the no-virus children only
                        (the Table 2 specification).
       restrict=False : binary virus indicator fitted on the whole frame
                        (the Table 3 specification for consensus ARI).
       firth          : None applies Firth only when fewer than 20 are positive
                        (Table 2); True applies it to every virus (Table 3)."""
    rows = []
    for v in ORDER:
        sub = frame[(frame[v] == 'Positive') | frame.novir].copy() if restrict else frame.copy()
        sub['_v'] = (sub[v] == 'Positive').astype(float)
        npos = int((frame[v] == 'Positive').sum())
        X, nm = design(sub, ['_v'] + COV)
        e = (ols(X, sub[outcome].values, nm) if kind == 'ols'
             else logit(X, sub[outcome].values, nm,
                        firth=(npos < 20) if firth is None else firth))['_v']
        rows.append(dict(virus=NM[v], n=npos, **e))
    for r, q in zip(rows, holm([r['p'] for r in rows])):
        r['holm'] = float(q)
    return rows


FAM = [('Pneumococcal carriage prevalence, adjusted OR', 'or',
        family(d, 'spn', 'logit'), f'{len(d)} symptomatic children'),
       ('Pneumococcal carriage density, adjusted β', 'beta',
        family(carr, 'log10dens', 'ols'), f'{len(carr)} carriers'),
       ('Consensus ARI, adjusted OR', 'or',
        family(S, 'ari', 'logit', restrict=False, firth=True), f'{len(S)} concordant children')]


# ------------------------------------------------------------------ output --
def fmt(r, kind):
    if kind == 'beta':
        return f"{r['est']:+.2f} ({r['lo']:+.2f} to {r['hi']:+.2f})".replace('-', '−')
    return f"{r['est']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})"


pv = lambda p: '<0.001' if p < 0.0005 else f'{p:.3f}'
W = 62
print('\nSupplementary Table 2. Virus-specific estimates with Holm–Bonferroni '
      'adjusted p-values.\n')
print(f"{'Virus':<18}{'n':>4}  {'Estimate (95% CI)':<24}{'p':>8}{'Holm p':>9}")
out = []
for title, kind, rows, denom in FAM:
    print('-' * W); print(f'{title}  ({denom})')
    for r in rows:
        star = '*' if r['p'] < 0.05 else ' '
        hstar = '*' if r['holm'] < 0.05 else ' '
        print(f"  {r['virus']:<16}{r['n']:>4}  {fmt(r, kind):<24}"
              f"{pv(r['p']):>7}{star}{pv(r['holm']):>8}{hstar}")
        out.append(dict(family=title, denominator=denom, virus=r['virus'], n=r['n'],
                        estimate=round(r['est'], 3), ci_lo=round(r['lo'], 3),
                        ci_hi=round(r['hi'], 3), p=round(r['p'], 5),
                        p_holm=round(r['holm'], 5)))
print('-' * W)
print('* p < 0.05.  Holm applied within each family of seven tests.\n')

path = os.path.join(OUTDIR, 'supplementary_table2_virus_holm.csv')
with open(path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print('wrote', path)
