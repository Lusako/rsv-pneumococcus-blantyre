#!/usr/bin/env python3
"""Check the respiratory panel actually ran on the virus-negative children.

Matches every study child to the laboratory run register and compares the RNase P
internal control and reported panel result between the asymptomatic COVID-suspect
group and everyone else.

Usage: verify_lab_controls.py <Merged_Data_2.xlsx> <lab_run_register.tsv> <out_dir>
"""
import sys
import numpy as np, pandas as pd
from scipy import stats

SRC, LAB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
VIR = ['SARS-CoV-2','FLUA','FLUB','HRV','RSV','HPIV','HPeV','HMPV','EV','HAdV','HBoV','FLUC']

d = pd.read_excel(SRC); d['sid'] = d.Sample_ID.astype(str).str.strip()
lab = pd.read_csv(LAB, sep='\t', skiprows=1, dtype=str)
lab['sid'] = lab['Sample ID'].astype(str).str.strip()
L = lab.drop_duplicates('sid').set_index('sid')

print(f'study children {len(d)};  matched to lab register '
      f'{int(d.sid.isin(L.index).sum())}')
tgt = (d['Symptom_status'] == 'Asymptomatic') & (d['Is the participant suspected of COVID-19?'] == 'Yes')
rows = []
for lb, mask in [('asymptomatic COVID-suspect', tgt), ('all other study children', ~tgt)]:
    sub = L.reindex([s for s in d.loc[mask, 'sid'] if s in L.index])
    rp = pd.to_numeric(sub['RP INTERNAL CONTROL'], errors='coerce')
    res = sub['RESPIRATORY PANEL RESULT'].fillna('MISSING').str.strip().str.lower()
    rows.append(dict(group=lb, n=len(sub), rp_median=round(rp.median(), 1),
                     rp_min=round(rp.min(), 1), rp_max=round(rp.max(), 1),
                     rp_missing=int(rp.isna().sum()),
                     panel_negative=int((res == 'negative').sum()),
                     panel_missing=int((res == 'MISSING'.lower()).sum())))
out = pd.DataFrame(rows)
print('\n' + out.to_string(index=False))

a = pd.to_numeric(L.reindex([s for s in d.loc[tgt,'sid'] if s in L.index])['RP INTERNAL CONTROL'], errors='coerce').dropna()
b = pd.to_numeric(L.reindex([s for s in d.loc[~tgt,'sid'] if s in L.index])['RP INTERNAL CONTROL'], errors='coerce').dropna()
print(f'\ninternal control, group vs rest: Mann–Whitney p = {stats.mannwhitneyu(a, b).pvalue:.3f}')

av = (d[VIR] == 'Positive').any(axis=1)
bg = av[~tgt].mean()
print(f'viral positivity elsewhere {bg*100:.1f}%; observed in the group '
      f'{int(av[tgt].sum())}/{int(tgt.sum())}')
print(f'P(0 of {int(tgt.sum())} | p={bg:.3f}) = {stats.binom.pmf(0, int(tgt.sum()), bg):.2e}')
out.to_csv(f'{OUT}/lab_internal_control_check.csv', index=False)
print(f'\nwrote {OUT}/lab_internal_control_check.csv')
