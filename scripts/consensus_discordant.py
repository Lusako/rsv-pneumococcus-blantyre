#!/usr/bin/env python3
"""Symptoms of the children where General ARI (0-14 d) and clinical suspicion disagree.

Usage: consensus_discordant.py <Merged_Data_2.xlsx> <out_dir>
Writes consensus_discordant_symptoms.csv (symptom frequencies by agreement cell),
consensus_discordant_the5.csv (the suspicion-only children, one row each) and
consensus_onset_date_problems.csv (onset dates outside the 0-14 day window).
"""
import sys
import pandas as pd

SRC, OUT = sys.argv[1], sys.argv[2]
d = pd.read_excel(SRC)
Y = lambda c: (d[c] == 'Yes')
F   = 'A history of self-reported feverishness or measured fever of ≥ 38°C'
SUS = 'Clinical suspicion of Acute Respiratory Infection'
onset = pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str) + ' ' +
                       d['Month symptoms started'].astype(str) + ' ' +
                       d['Year symptoms started'].astype(str), format='%d %b %Y', errors='coerce')
gap = (pd.to_datetime(d['sample date']) - onset).dt.days
RESP = ['Cough', 'Sore Throat', 'Shortness of breath OR Tachypnoea', 'Runny nose (rhinorrhea)']
g = pd.concat([Y(c) for c in RESP], axis=1).any(axis=1) & gap.between(0, 14)
A = Y(SUS)

lo, hi = d.columns.get_loc(F), d.columns.get_loc('Other symptom')
SYMS = [c for c in d.columns[lo:hi+1] if c != SUS and (d[c] == 'Yes').sum() > 0]
CELLS = [('General ARI yes, suspicion no', g & ~A), ('General ARI no, suspicion yes', ~g & A),
         ('Both yes', g & A), ('Both no', ~g & ~A)]
print('agreement cells: ' + ', '.join(f'{nm} n={int(m.sum())}' for nm, m in CELLS))

rows = []
for c in SYMS:
    r = {'Symptom': c}
    for nm, m in CELLS:
        k, n = int((d.loc[m, c] == 'Yes').sum()), int(m.sum())
        r[f'{nm} (n={n})'] = f'{k} ({k/n*100:.0f}%)'
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/consensus_discordant_symptoms.csv', index=False)
print(T.to_string(index=False))

five = d.loc[~g & A, ['Sample_ID', 'Participant_ID'] + SYMS +
             ['URTI', 'Pneumonia', 'Bronchiolitis']].copy()
five.insert(2, 'onset_to_sample_days', gap[~g & A].values)
five.to_csv(f'{OUT}/consensus_discordant_the5.csv', index=False)
print('\nsuspicion-only children, symptoms recorded:')
for _, r in five.iterrows():
    on = [c for c in SYMS if r[c] == 'Yes']
    print(f"  {r['Sample_ID']}  gap {r['onset_to_sample_days']:>4.0f} d  "
          f"{', '.join(on) if on else 'NO symptom recorded'}")

bad = (gap > 30) | (gap < 0)
cols = ['Sample_ID', 'Date Day symptoms started', 'Month symptoms started',
        'Year symptoms started', 'sample date']
prob = d.loc[bad, cols].assign(onset_to_sample_days=gap[bad])
prob.to_csv(f'{OUT}/consensus_onset_date_problems.csv', index=False)
print(f'\nonset dates outside the 0-14 day window ({int(bad.sum())} children):')
print(prob.to_string(index=False))
print(f'\nwrote 3 CSVs to {OUT}/')
