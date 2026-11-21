#!/usr/bin/env python3
"""Compare Symptom_status with Clinical suspicion of ARI.

Usage: compare_symptom_columns.py <Merged_Data_2.xlsx> <out_dir>
Writes <out_dir>/Symptom_status_vs_ARI_suspicion.xlsx (3 sheets) and prints the
agreement statistics, derivation tests and stratified tables to stdout.
"""
import sys, itertools
import numpy as np, pandas as pd
from scipy import stats
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

SRC, OUT = sys.argv[1], sys.argv[2]
S_COL = 'Symptom_status'
A_COL = 'Clinical suspicion of Acute Respiratory Infection'
C_COL = 'Is the participant suspected of COVID-19?'
FEVER = 'A history of self-reported feverishness or measured fever of ≥ 38°C'
VIR = ['SARS-CoV-2','FLUA','FLUB','HRV','RSV','HPIV','HPeV','HMPV','EV','HAdV','HBoV','FLUC']

d = pd.read_excel(SRC)
S = (d[S_COL] == 'Symptomatic').astype(int)
A = (d[A_COL] == 'Yes').astype(int)
n = len(d)

print(f'n = {n};  missing: {S_COL} {d[S_COL].isna().sum()}, ARI {d[A_COL].isna().sum()}')
t = pd.crosstab(S, A).values
po = (t[0,0]+t[1,1])/n
pe = ((t[0].sum()*t[:,0].sum()) + (t[1].sum()*t[:,1].sum()))/n**2
k  = (po-pe)/(1-pe); se = np.sqrt(po*(1-po)/(n*(1-pe)**2))
b, c = int(t[0,1]), int(t[1,0])
print('\n--- 1. agreement ---')
print(pd.crosstab(d[S_COL], d[A_COL]).to_string())
print(f'agreement {po*100:.1f}%  kappa {k:.3f} (95% CI {k-1.96*se:.3f}–{k+1.96*se:.3f})')
print(f'discordant {b+c} (asym/ARI-yes {b}, sym/ARI-no {c}); exact McNemar p = '
      f'{stats.binomtest(b, b+c, 0.5).pvalue:.3f}')

print('\n--- 2. are the columns a function of the recorded symptoms? ---')
lo = d.columns.get_loc(FEVER); hi = d.columns.get_loc('Other symptom')
SYMS = [x for x in d.columns[lo:hi+1] if x != A_COL]
X = pd.DataFrame({x: (d[x].astype(str).str.strip() == 'Yes').astype(int) for x in SYMS})
key = X.astype(str).agg(''.join, axis=1)
print(f'distinct symptom profiles: {key.nunique()}')
for tgt, nm in [(S, S_COL), (A, 'ARI suspicion')]:
    g = pd.DataFrame({'k': key, 'y': tgt}).groupby('k').y.agg(['nunique','count'])
    bad = g[g['nunique'] > 1]
    print(f'  {nm:16s}: {len(bad)} profiles carry both labels, covering '
          f'{int(bad["count"].sum())} children ({bad["count"].sum()/n*100:.0f}%)')

print('\n--- 3. relationship to the COVID-suspect enrolment stream ---')
C = d[C_COL]
print(f'Symptomatic -> COVID-suspect=Yes: {int(((S==1)&(C=="Yes")).sum())}/{int((S==1).sum())}')
print(f'ARI=Yes      -> COVID-suspect=Yes: {int(((A==1)&(C=="Yes")).sum())}/{int((A==1).sum())}')
av = (d[VIR] == 'Positive').any(axis=1)
for arm in ['Yes','No']:
    m = C == arm
    for lab, g in [(S_COL, d[S_COL]), ('ARI suspicion', d[A_COL])]:
        tb = pd.crosstab(g[m], av[m])
        p = stats.fisher_exact(tb.values)[1] if tb.shape == (2,2) else float('nan')
        print(f'  COVID-suspect={arm}, {lab} x virus: {tb.values.tolist()}  Fisher p={p:.3g}')

print('\n--- 4. workbook ---')
d2 = d.copy()
d2.insert(d2.columns.get_loc(A_COL)+1, 'Match', np.where(S == A, 'Agree', 'Disagree'))
SUB = ['Sample_ID','Participant_ID',S_COL,A_COL,'Match','Cough',FEVER,
       'Runny nose (rhinorrhea)','URTI','Pneumonia','Bronchiolitis',C_COL]
disc = d2.loc[d2.Match == 'Disagree', SUB].sort_values([S_COL,'Sample_ID'])
agre = d2.loc[d2.Match == 'Agree',    SUB].sort_values([S_COL,'Sample_ID'])
path = f'{OUT}/Symptom_status_vs_ARI_suspicion.xlsx'
with pd.ExcelWriter(path, engine='openpyxl') as w:
    disc.to_excel(w, sheet_name='Discordant', index=False)
    agre.to_excel(w, sheet_name='Agreement',  index=False)
    d2.to_excel(w,   sheet_name='Full data',  index=False)
    hdr, body = Font(name='Arial', size=10, bold=True, color='FFFFFF'), Font(name='Arial', size=10)
    fill = PatternFill('solid', fgColor='44546A')
    for ws in w.book.worksheets:
        ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = hdr; cell.fill = fill
            cell.alignment = Alignment(vertical='center', wrap_text=True, horizontal='center')
        ws.row_dimensions[1].height = 42
        for row in ws.iter_rows(min_row=2):
            for cell in row: cell.font = body
        for j in range(1, ws.max_column+1):
            vals = [ws.cell(r, j).value for r in range(2, min(ws.max_row, 400)+1)]
            L = max([len(str(v)) for v in vals if v is not None] + [0])
            ws.column_dimensions[get_column_letter(j)].width = min(max(10, L+2), 34)
print(f'wrote {path}  (Discordant {len(disc)}, Agreement {len(agre)}, Full {len(d2)})')
