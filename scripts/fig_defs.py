# -*- coding: utf-8 -*-
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panelstats import pairwise, overall_p, draw_brackets, XPAD

SRC = sys.argv[1]; OUT = sys.argv[2]
SD = sys.argv[3] if len(sys.argv) > 3 else None   # optional source-data dir
os.makedirs(OUT, exist_ok=True)
if SD: os.makedirs(SD, exist_ok=True)
SRCROWS = []
STATROWS = []
BLUE, ORANGE, AQUA, GREY, DARK = '#2a78d6', '#eb6834', '#1baf7a', '#9aa3ab', '#14181c'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'savefig.dpi': 400, 'figure.dpi': 140,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

d = pd.read_excel(SRC)
VIR = ['SARS-CoV-2','FLUA','FLUB','HRV','RSV','HPIV','HPeV','HMPV','EV','HAdV','HBoV','FLUC']
d['anyvir'] = (d[VIR] == 'Positive').any(axis=1).astype(int)
d['spn'] = ((d.lytA_Ct <= 35) & (d.piaB_Ct <= 35) & (d.SP2020_Ct <= 35)).fillna(False).astype(int)
d['log10dens'] = np.log10(pd.to_numeric(d.lytA_quantity, errors='coerce'))

# ---- the symptom definitions -------------------------------------------------
Y = lambda c: (d[c] == 'Yes')
F     = 'A history of self-reported feverishness or measured fever of ≥ 38°C'
SUS   = 'Clinical suspicion of Acute Respiratory Infection'
onset = pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str) + ' ' +
                       d['Month symptoms started'].astype(str) + ' ' +
                       d['Year symptoms started'].astype(str), format='%d %b %Y', errors='coerce')
gap   = (pd.to_datetime(d['sample date']) - onset).dt.days
major = Y(F) | Y('Cough') | Y('Shortness of breath OR Tachypnoea') | Y('Loss of smell') | Y('Loss of taste')
addl  = (Y('Sore Throat') | Y('Runny nose (rhinorrhea)') | Y('Headache') | Y('Muscle aches (myalgia)') |
         Y('Fatigue / Malaise') | Y('Chest pain'))
d['clinsus'] = Y(SUS).astype(int)
d['ili']     = (Y(F) & Y('Cough') & gap.between(0, 10)).astype(int)
d['synari']  = (Y(SUS) & gap.between(0, 30) & major & addl).astype(int)
# General ARI: at least one of cough, sore throat, shortness of breath, runny nose
d['genari']  = (Y('Cough') | Y('Sore Throat') |
                Y('Shortness of breath OR Tachypnoea') | Y('Runny nose (rhinorrhea)')).astype(int)
# Any symptom at all (whole symptom block, 27 fields) with onset 0-14 days before sampling
_lo = d.columns.get_loc(F); _hi = d.columns.get_loc('Other symptom')
SYMPTOM_BLOCK = [c for c in d.columns[_lo:_hi+1] if c != SUS]
anysym = pd.concat([Y(c) for c in SYMPTOM_BLOCK], axis=1).any(axis=1)
d['anysym30'] = (anysym & gap.between(0, 14)).astype(int)
# General ARI with the acute window applied
d['genari30'] = (d['genari'].astype(bool) & gap.between(0, 14)).astype(int)
# Consensus ARI: yes only where General ARI (0-14 d) and clinical suspicion agree
d['consensus'] = (d['genari30'].astype(bool) & Y(SUS)).astype(int)
d['consensus_strict'] = d['consensus']   # analysed on the agreement subset only

DEFS = [('clinsus', 'Clinical suspicion of ARI', 'Clinical suspicion of ARI',
         'Suspicion recorded', 'No suspicion', 'figure_def_clinical_suspicion',
         'Pathogens detected, by clinical suspicion of ARI'),
        ('ili', 'WHO influenza-like illness', 'WHO ILI',
         'Meets WHO ILI', 'Does not meet', 'figure_def_who_ili',
         'Pathogens detected, by WHO ILI status'),
        ('synari', 'Syndromic ARI criteria (this analysis)', 'Syndromic ARI',
         'Meets syndromic ARI', 'Does not meet', 'figure_def_syndromic_ari',
         'Pathogens detected, by syndromic ARI status'),
        ('genari', 'General ARI (any of cough, sore throat, shortness of breath, runny nose)',
         'General ARI', 'Meets general ARI', 'Does not meet', 'figure_def_general_ari',
         'Pathogens detected, by general ARI status'),
        ('genari30', 'General ARI with symptom onset 0\u201330 days before sampling',
         'General ARI (0\u201330 d)', 'Meets general ARI, onset \u2264 30 d', 'Does not meet',
         'figure_def_general_ari_30d',
         'Pathogens detected, by general ARI (onset \u2264 30 d) status'),
        ('consensus', 'Consensus ARI: General ARI (0\u201330 d) and clinical suspicion agree',
         'Consensus ARI', 'Both agree: ARI', 'Not both', 'figure_def_consensus_ari',
         'Pathogens detected, by consensus ARI status'),
        ('consensus_strict', 'Consensus ARI',
         'Consensus ARI', 'Consensus ARI', 'Consensus non-ARI',
         'figure_def_consensus_ari_agreement_only',
         'Pathogens detected, by consensus ARI status'),
        ('anysym30', 'Any reported symptom with onset 0\u201330 days before sampling',
         'Any symptom (0\u201330 d)', 'Any symptom, onset \u2264 30 d', 'No symptom in window',
         'figure_def_any_symptom_30d',
         'Pathogens detected, by any-symptom (onset \u2264 30 d) status')]

# ---- pathogen group & density group ---------------------------------------
d['grp4'] = np.select(
    [(d.spn == 0) & (d.anyvir == 0), (d.spn == 1) & (d.anyvir == 0),
     (d.spn == 0) & (d.anyvir == 1), (d.spn == 1) & (d.anyvir == 1)],
    ['Neither', 'Pneumococcus only', 'Virus only', 'Pneumococcus + virus'], default='')
MED = d.loc[d.spn == 1, 'log10dens'].median()
d['densgrp'] = np.where(d.spn == 0, 'Not carrier',
                np.where(d.log10dens <= MED, 'Low density', 'High density'))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan,)*3
    p = k/n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return p*100, max(0, c-h)*100, min(1, c+h)*100

def panel(ax, letter, dx=None, dy=None, pad_x=0.058, pad_y=0.040):
    """Place the panel letter in FIGURE coordinates, so letters in different
    columns line up with each other regardless of axis width."""
    fig = ax.get_figure(); box = ax.get_position()
    fig.text(box.x0 - pad_x, box.y1 + pad_y, f'{letter}.',
             fontsize=12, fontweight='bold', va='bottom', ha='left')

def dotci(ax, labels, est, lo, hi, colors, ns, xlab, xmax=100):
    y = np.arange(len(labels))[::-1]
    for i, yy in enumerate(y):
        ax.plot([lo[i], hi[i]], [yy, yy], color=colors[i], lw=1.8, solid_capstyle='round', zorder=2)
        ax.plot(est[i], yy, 'o', ms=5.5, color=colors[i], mec='white', mew=1.0, zorder=3)
        ax.text(xmax*1.02, yy, str(ns[i]), va='center', ha='left', fontsize=7.0, color='#555')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, XPAD); ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xlab); ax.set_ylim(-0.7, len(labels)-0.3)
    ax.grid(axis='x', color='#e6e9ec', lw=0.7, zorder=0); ax.set_axisbelow(True)

G4 = [('Neither', GREY), ('Pneumococcus only', BLUE), ('Pneumococcus + virus', AQUA), ('Virus only', ORANGE)]
ORDER = ['Neither', 'Pneumococcus only', 'Virus only', 'Pneumococcus + virus']
DORD  = ['Not carrier', 'Low density', 'High density']
cm = dict(G4)

RESTRICT = {'consensus_strict': lambda f: (f['genari30'] == f['clinsus'])}

for col, title, short, poslab, neglab, fname, atitle in DEFS:
    dall = d
    d = d[RESTRICT[col](d)].copy() if col in RESTRICT else d
    npos, nneg = int(d[col].sum()), int((1-d[col]).sum())
    fig = plt.figure(figsize=(7.3, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1], hspace=0.70, wspace=0.60)

    ax = fig.add_subplot(gs[0, :])
    for i, (nm, mask, tot) in enumerate([(f'{poslab}\n(n = {npos})', d[col] == 1, npos),
                                         (f'{neglab}\n(n = {nneg})', d[col] == 0, nneg)]):
        acc = 0
        for k, c in G4:
            w = int(((d.grp4 == k) & mask).sum())/tot*100
            ax.barh(i, w, left=acc, color=c, height=0.5, edgecolor='white', lw=1.2, zorder=2)
            if w > 7:
                ax.text(acc+w/2, i, f'{w:.0f}%', ha='center', va='center', fontsize=8,
                        color='white' if c != GREY else DARK, fontweight='bold')
            acc += w
    ax.set_yticks(range(2))
    ax.set_yticklabels([f'{poslab}\n(n = {npos})', f'{neglab}\n(n = {nneg})'])
    ax.set_xlim(0, 100); ax.set_xlabel('Children (%)'); ax.invert_yaxis()
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in G4], ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), handlelength=1.1, columnspacing=1.4)
    ax.set_title(atitle, fontsize=9)
    panel(ax, 'A', dx=-0.145, dy=1.14)

    ax = fig.add_subplot(gs[1, 0])
    st = [wilson(int(d.loc[d.grp4 == g, col].sum()), int((d.grp4 == g).sum())) for g in ORDER]
    dotci(ax, ORDER, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st],
          [cm[g] for g in ORDER], [int((d.grp4 == g).sum()) for g in ORDER],
          xlab=f'{short} (%)')
    for g, sv in zip(ORDER, st):
        SRCROWS.append(dict(definition=short, panel='B pathogen combination', stratum=g,
                            k=int(d.loc[d.grp4 == g, col].sum()), n=int((d.grp4 == g).sum()),
                            percent=round(sv[0], 2), ci_low=round(sv[1], 2), ci_high=round(sv[2], 2)))
    cnt = {g: (int(d.loc[d.grp4 == g, col].sum()), int((d.grp4 == g).sum())) for g in ORDER}
    comps = pairwise(cnt, [('Neither','Pneumococcus only'), ('Neither','Virus only'),
                           ('Pneumococcus only','Virus only'),
                           ('Neither','Pneumococcus + virus')])
    draw_brackets(ax, ORDER, comps)
    for c in comps:
        STATROWS.append(dict(definition=short, panel='B pathogen combination', **c))
    ax.set_title(f'By pathogen combination (overall p = {overall_p(cnt):.3f})', fontsize=8.6)
    panel(ax, 'B', dx=-0.52)

    ax = fig.add_subplot(gs[1, 1])
    st = [wilson(int(d.loc[d.densgrp == g, col].sum()), int((d.densgrp == g).sum())) for g in DORD]
    dotci(ax, DORD, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st],
          [GREY, BLUE, BLUE], [int((d.densgrp == g).sum()) for g in DORD], xlab=f'{short} (%)')
    for g, sv in zip(DORD, st):
        SRCROWS.append(dict(definition=short, panel='C carriage density', stratum=g,
                            k=int(d.loc[d.densgrp == g, col].sum()), n=int((d.densgrp == g).sum()),
                            percent=round(sv[0], 2), ci_low=round(sv[1], 2), ci_high=round(sv[2], 2)))
    for g, _c in G4:
        SRCROWS.append(dict(definition=short, panel='A pathogen mix (meets definition)', stratum=g,
                            k=int(((d.grp4 == g) & (d[col] == 1)).sum()), n=npos,
                            percent=round(int(((d.grp4 == g) & (d[col] == 1)).sum())/npos*100, 2),
                            ci_low='', ci_high=''))
    cntd = {g: (int(d.loc[d.densgrp == g, col].sum()), int((d.densgrp == g).sum())) for g in DORD}
    compsd = pairwise(cntd, [('Not carrier','Low density'), ('Not carrier','High density'),
                             ('Low density','High density')])
    draw_brackets(ax, DORD, compsd)
    for c in compsd:
        STATROWS.append(dict(definition=short, panel='C carriage density', **c))
    ax.set_title(f'By pneumococcal density (overall p = {overall_p(cntd):.3f})', fontsize=8.6)
    panel(ax, 'C', dx=-0.42)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.005)
    fig.savefig(f'{OUT}/{fname}.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    d = dall
    print(f'{fname}: n meeting = {npos} / {npos+nneg}')

if SD:
    pd.DataFrame(SRCROWS).to_csv(f'{SD}/definition_figures_source_data.csv', index=False)
    print(f'source data -> {SD}/definition_figures_source_data.csv ({len(SRCROWS)} rows)')
if SD:
    pd.DataFrame(STATROWS).to_csv(f'{SD}/definition_figures_pairwise_tests.csv', index=False)
    print(f'pairwise tests -> {SD}/definition_figures_pairwise_tests.csv ({len(STATROWS)} rows)')
