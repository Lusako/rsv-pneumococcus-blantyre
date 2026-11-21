# -*- coding: utf-8 -*-
"""Manuscript figures, symptomatic cohort (n = 335). 600 dpi, fixed aspect, pairwise p-values."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R=json.load(open('nums335.json')); OUT='figs'; os.makedirs(OUT,exist_ok=True)
BLUE,ORANGE,AQUA,GREY,DARK='#2a78d6','#eb6834','#1baf7a','#9aa3ab','#14181c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
 'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.linewidth':0.8,'xtick.major.width':0.8,'ytick.major.width':0.8,
 'savefig.dpi':600,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,
 'legend.frameon':False})

def panel(ax,L,pad_x=0.060,pad_y=0.040):
    f=ax.get_figure(); b=ax.get_position()
    f.text(b.x0-pad_x,b.y1+pad_y,f'{L}.',fontsize=12,fontweight='bold',va='bottom',ha='left')
def fmtp(p): return 'p<0.001' if p<0.001 else (f'p={p:.3f}' if p<0.01 else f'p={p:.2f}')
def brackets(ax,labels,comps,x0=120,step=13,fs=6.5):
    """Draw a comparison bracket only where the pair differs at p < 0.05."""
    comps=[c for c in comps if c['p']<0.05]
    yp={l:y for l,y in zip(labels,np.arange(len(labels))[::-1])}
    for j,c in enumerate(comps):
        x=x0+j*step; ya,yb=yp[c['a']],yp[c['b']]
        ax.plot([x,x],[ya,yb],color=DARK,lw=0.9,clip_on=False,zorder=5)
        for y in (ya,yb): ax.plot([x-3.4,x],[y,y],color=DARK,lw=0.9,clip_on=False,zorder=5)
        ax.text(x+2.2,(ya+yb)/2,fmtp(c['p']),rotation=90,va='center',ha='left',fontsize=fs,
                color=DARK,clip_on=False,fontweight='bold')
def dotci(ax,labels,est,lo,hi,cols,ns,xlab,ref=None,xpad=166):
    y=np.arange(len(labels))[::-1]
    if ref is not None: ax.axvline(ref,color=DARK,lw=0.9,ls='--',alpha=.6,zorder=1)
    for i,yy in enumerate(y):
        ax.plot([lo[i],hi[i]],[yy,yy],color=cols[i],lw=1.9,solid_capstyle='round',zorder=2)
        ax.plot(est[i],yy,'o',ms=5.5,color=cols[i],mec='white',mew=1.0,zorder=3)
        ax.text(114,yy,str(ns[i]),va='center',ha='right',fontsize=7.0,color='#555')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0,xpad); ax.set_xticks([0,20,40,60,80,100])
    ax.set_xlabel(xlab); ax.set_ylim(-0.7,len(labels)-0.3)
    ax.grid(axis='x',color='#e6e9ec',lw=0.7,zorder=0); ax.set_axisbelow(True)

d=pd.read_excel('../data/Merged_Data_2.xlsx')
VIR=['SARS-CoV-2','FLUA','FLUB','HRV','RSV','HPIV','HPeV','HMPV','EV','HAdV','HBoV','FLUC']
NM={'HRV':'Rhinovirus','FLUA':'Influenza A','HBoV':'Bocavirus','RSV':'RSV','HMPV':'Metapneumovirus',
    'HAdV':'Adenovirus','SARS-CoV-2':'SARS-CoV-2','FLUB':'Influenza B','HPIV':'Parainfluenza',
    'HPeV':'Parechovirus','EV':'Enterovirus','FLUC':'Influenza C'}
F='A history of self-reported feverishness or measured fever of ≥ 38°C'
SUS='Clinical suspicion of Acute Respiratory Infection'
lo_=d.columns.get_loc(F); hi_=d.columns.get_loc('Other symptom')
SY=[c for c in d.columns[lo_:hi_+1] if c!=SUS]
d=d[pd.concat([(d[c]=='Yes') for c in SY],axis=1).any(axis=1)].reset_index(drop=True)
d['spn']=((d.lytA_Ct<=35)&(d.piaB_Ct<=35)&(d.SP2020_Ct<=35)).fillna(False).astype(int)
d['log10dens']=np.log10(pd.to_numeric(d.lytA_quantity,errors='coerce'))
d['nvir']=(d[VIR]=='Positive').sum(axis=1)
d['virgrp']=np.select([d.nvir==0,d.nvir==1,d.nvir>1],['No virus','Single virus','Multiple viruses'],default='')
carr=d[d.spn==1].dropna(subset=['log10dens'])
N=len(d)

# ================= FIGURE 1 =================
fig,axes=plt.subplots(1,2,figsize=(8.4,4.0),gridspec_kw=dict(width_ratios=[1,1.12],wspace=0.95))
ax=axes[0]
vs=sorted([(NM[v],int((d[v]=='Positive').sum())) for v in VIR],key=lambda t:t[1])
vs=[(a,b) for a,b in vs if b>0]
yy=np.arange(len(vs))
ax.barh(yy,[b for _,b in vs],color=BLUE,height=0.62,zorder=2)
for i,(a,b) in enumerate(vs):
    ax.text(b+1.2,i,f'{b} ({b/N*100:.1f}%)',va='center',fontsize=7.2,color='#444')
ax.set_yticks(yy); ax.set_yticklabels([a for a,_ in vs])
ax.set_xlim(0,max(b for _,b in vs)*1.34); ax.set_xlabel('Children positive, n')
ax.grid(axis='x',color='#e6e9ec',lw=0.7); ax.set_axisbelow(True)
ax.set_title(f'Respiratory viruses detected (n = {N})',fontsize=9); panel(ax,'A')
ax=axes[1]
rows,est,lo,hi,cols,ns=[],[],[],[],[],[]
for blk in R['table1']:
    if blk['name'] in ('Sex','Season'): continue
    for r in blk['rows']:
        if r['level']=='Not classified': continue
        rows.append(r['level']); est.append(r['p']); lo.append(r['lo']); hi.append(r['hi'])
        ns.append(r['total']); cols.append(AQUA if blk['name']=='Consensus ARI status' else BLUE)
dotci(ax,rows,est,lo,hi,cols,ns,'Pneumococcal carriage (%)',ref=R['carriage_overall']['p'])
brackets(ax,rows,[dict(a='Consensus non-ARI',b='Consensus ARI',p=R['table1'][-1]['p'])],x0=124)
ax.set_title('Carriage by child characteristics',fontsize=9); panel(ax,'B')
fig.savefig(f'{OUT}/figure1.png',bbox_inches='tight',facecolor='white'); plt.close(fig)

# ================= FIGURE 2 =================
fig,axes=plt.subplots(1,2,figsize=(8.4,3.5),gridspec_kw=dict(width_ratios=[1,1.12],wspace=0.95))
ax=axes[0]
g=[r for r in R['table1'] if r['name']=='Respiratory virus'][0]['rows']
labs=[r['level'] for r in g]
dotci(ax,labs,[r['p'] for r in g],[r['lo'] for r in g],[r['hi'] for r in g],
      [GREY,BLUE,AQUA],[r['total'] for r in g],'Pneumococcal carriage (%)',
      ref=R['carriage_overall']['p'],xpad=128)
cnt={r['level']:(r['k'],r['total']) for r in g}
cmp=[]
for a,b in [('No virus','Single virus'),('No virus','Multiple viruses'),('Single virus','Multiple viruses')]:
    ka,na=cnt[a]; kb,nb=cnt[b]
    cmp.append(dict(a=a,b=b,p=float(stats.fisher_exact([[ka,na-ka],[kb,nb-kb]])[1])))
brackets(ax,labs,cmp,x0=124)   # no pair reaches p < 0.05, so none is drawn
ax.set_title('Carriage by virus group',fontsize=9); panel(ax,'A')
ax=axes[1]
T2=[t for t in R['table2'] if 'aOR' in t and t['virus'] not in ('Single virus','Multiple viruses')]
labs=[t['virus'] for t in T2]; y=np.arange(len(labs))[::-1]
ax.axvline(1,color=DARK,lw=0.9)
for i,t in enumerate(T2):
    c=BLUE if t['aOR_p']<0.05 else GREY
    ax.plot([t['aOR_lo'],t['aOR_hi']],[y[i]]*2,color=c,lw=1.9,solid_capstyle='round')
    ax.plot(t['aOR'],y[i],'o',ms=5.5,color=c,mec='white',mew=1.0,zorder=3)
    ax.text(10.5,y[i],f"{t['npos']}",va='center',ha='right',fontsize=7.0,color='#555')
    ax.text(13.0,y[i],fmtp(t['aOR_p']),va='center',ha='left',fontsize=6.6,
            color=DARK if t['aOR_p']<0.05 else '#666',
            fontweight='bold' if t['aOR_p']<0.05 else 'normal')
ax.set_xscale('log'); ax.set_xticks([0.25,0.5,1,2,4,8]); ax.set_xticklabels(['0.25','0.5','1','2','4','8'])
ax.minorticks_off(); ax.set_xlim(0.2,42)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlabel('Adjusted OR for carriage (95% CI)')
ax.grid(axis='x',color='#e6e9ec',lw=0.7); ax.set_axisbelow(True)
ax.set_title('Adjusted carriage odds by virus',fontsize=9); panel(ax,'B')
fig.savefig(f'{OUT}/figure2.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
print('figures 1-2 done')

# ================= FIGURE 3 : density =================
fig=plt.figure(figsize=(9.8,6.6))
gs=fig.add_gridspec(2,2,hspace=0.58,wspace=0.55,height_ratios=[1,1.18])
ax=fig.add_subplot(gs[0,0])
rng=np.random.default_rng(7)
order=['No virus','Single virus','Multiple viruses']
vals={g:carr[carr.virgrp==g].log10dens.values for g in order}
for i,g in enumerate(order):
    v=vals[g]; ax.plot(i+rng.normal(0,0.07,len(v)),v,'o',ms=3.0,color=[GREY,BLUE,AQUA][i],
                       alpha=.45,mec='none',zorder=2)
    q1,me,q3=np.percentile(v,[25,50,75])
    ax.plot([i-.24,i+.24],[me,me],color=DARK,lw=2.2,zorder=4)
    ax.plot([i,i],[q1,q3],color=DARK,lw=1.1,alpha=.7,zorder=3)
    ax.text(i,0.25,f'n={len(v)}',ha='center',fontsize=7,color='#666')
sig3=[(a,b) for a,b in [(0,1),(0,2),(1,2)]
      if float(stats.mannwhitneyu(vals[order[a]],vals[order[b]]).pvalue)<0.05]
for j,(a,b) in enumerate(sig3):
    yh=6.8+j*0.5
    p=float(stats.mannwhitneyu(vals[order[a]],vals[order[b]]).pvalue)
    ax.plot([a,a,b,b],[yh-.12,yh,yh,yh-.12],color=DARK,lw=0.9)
    ax.text((a+b)/2,yh+.05,fmtp(p),ha='center',fontsize=6.8,fontweight='bold',color=DARK)
ax.set_xticks(range(3)); ax.set_xticklabels(['No\nvirus','Single\nvirus','Multiple\nviruses'])
ax.set_xlim(-.6,2.6); ax.set_ylim(0,7.9); ax.set_ylabel('log$_{10}$ lytA copies/mL')
ax.grid(axis='y',color='#e6e9ec',lw=0.7); ax.set_axisbelow(True)
ax.set_title('Density by virus group',fontsize=9); panel(ax,'A')

ax=fig.add_subplot(gs[0,1])
S=pd.read_json('nums335.json',typ='series')
dd=R['ari_dens_desc']
sub=carr.copy()
onset=pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str)+' '+
                     d['Month symptoms started'].astype(str)+' '+d['Year symptoms started'].astype(str),
                     format='%d %b %Y',errors='coerce')
gap=(pd.to_datetime(d['sample date'])-onset).dt.days
RESP=['Cough','Sore Throat','Shortness of breath OR Tachypnoea','Runny nose (rhinorrhea)']
d['gari30']=(pd.concat([(d[c]=='Yes') for c in RESP],axis=1).any(axis=1)&gap.between(0,14)).astype(int)
d['ari']=(d['gari30'].astype(bool)&(d[SUS]=='Yes')).astype(int)
d['agree']=(d['gari30']==(d[SUS]=='Yes').astype(int))
cS=d[(d.agree)&(d.spn==1)].dropna(subset=['log10dens'])
gv=[cS[cS.ari==0].log10dens.values,cS[cS.ari==1].log10dens.values]
for i,(v,c) in enumerate(zip(gv,[GREY,AQUA])):
    ax.plot(i+rng.normal(0,0.07,len(v)),v,'o',ms=3.0,color=c,alpha=.45,mec='none',zorder=2)
    q1,me,q3=np.percentile(v,[25,50,75])
    ax.plot([i-.24,i+.24],[me,me],color=DARK,lw=2.2,zorder=4)
    ax.plot([i,i],[q1,q3],color=DARK,lw=1.1,alpha=.7,zorder=3)
    ax.text(i,0.25,f'n={len(v)}',ha='center',fontsize=7,color='#666')
p=dd['mw_p']
if p<0.05:
    ax.plot([0,0,1,1],[6.9,7.0,7.0,6.9],color=DARK,lw=0.9)
    ax.text(.5,7.05,fmtp(p),ha='center',fontsize=6.8,fontweight='bold',color=DARK)
ax.set_xticks([0,1]); ax.set_xticklabels(['Consensus\nnon-ARI','Consensus\nARI'])
ax.set_xlim(-.6,1.6); ax.set_ylim(0,7.9); ax.set_ylabel('log$_{10}$ lytA copies/mL')
ax.grid(axis='y',color='#e6e9ec',lw=0.7); ax.set_axisbelow(True)
ax.set_title('Density by consensus ARI status',fontsize=9); panel(ax,'B')

# ---- panel C : the raw carrier densities behind panel D ----
ax=fig.add_subplot(gs[1,0])
COL={'Rhinovirus':'HRV','Influenza A':'FLUA','Bocavirus':'HBoV','RSV':'RSV',
     'Metapneumovirus':'HMPV','Adenovirus':'HAdV','SARS-CoV-2':'SARS-CoV-2'}
V7=[t['virus'] for t in R['table2']
    if 'beta' in t and t['virus'] not in ('Any virus','Single virus','Multiple viruses')]
novir=carr[carr.virgrp=='No virus'].log10dens.values
yv=V7[::-1]
for i,v in enumerate(yv):
    vals_v=carr[carr[COL[v]]=='Positive'].log10dens.dropna().values
    ax.plot(vals_v,i+rng.normal(0,0.075,len(vals_v)),'o',ms=3.2,color=ORANGE,
            alpha=.55,mec='none',zorder=2)
    if len(vals_v):
        me=np.median(vals_v)
        ax.plot([me,me],[i-0.26,i+0.26],color=DARK,lw=2.2,zorder=4)
    ax.text(7.15,i,f'{len(vals_v)}',va='center',ha='right',fontsize=7.0,color='#666')
ax.axvline(np.median(novir),color=DARK,lw=0.9,ls='--',alpha=.55)
ax.set_yticks(range(len(yv))); ax.set_yticklabels(yv)
ax.set_xlabel('log$_{10}$ lytA copies/mL'); ax.set_xlim(1.6,7.2)
ax.set_xticks([2,3,4,5,6])
ax.set_ylim(-0.7,len(yv)-0.3)
ax.grid(axis='x',color='#e6e9ec',lw=0.7); ax.set_axisbelow(True)
ax.set_title('Density in carriers, by virus',fontsize=9)
panel(ax,'C')

ax=fig.add_subplot(gs[1,1])
T2=[t for t in R['table2'] if 'beta' in t and t['virus'] not in ('Single virus','Multiple viruses')]
labs=[t['virus'] for t in T2]; y=np.arange(len(labs))[::-1]
ax.axvline(0,color=DARK,lw=0.9)
for i,t in enumerate(T2):
    c=BLUE if t['beta_p']<0.05 else GREY
    ax.plot([t['beta_lo'],t['beta_hi']],[y[i]]*2,color=c,lw=1.9,solid_capstyle='round')
    ax.plot(t['beta'],y[i],'o',ms=5.5,color=c,mec='white',mew=1.0,zorder=3)
    ax.text(1.88,y[i],f"{t['n_carr']}",va='center',ha='right',fontsize=7,color='#555')
    ax.text(2.02,y[i],fmtp(t['beta_p']),va='center',ha='left',fontsize=6.8,
            color=DARK if t['beta_p']<0.05 else '#666',
            fontweight='bold' if t['beta_p']<0.05 else 'normal')
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(-1.3,3.0)
ax.set_xticks([-1,0,1,1.5])
ax.set_xlabel('Adjusted difference in log$_{10}$ density (95% CI)')
ax.grid(axis='x',color='#e6e9ec',lw=0.7); ax.set_axisbelow(True)
ax.set_title('Adjusted density difference by virus',fontsize=9); panel(ax,'D')
fig.savefig(f'{OUT}/figure3.png',bbox_inches='tight',facecolor='white'); plt.close(fig)

# ================= FIGURE 4 : consensus ARI =================
G4=[('Neither',GREY),('Pneumococcus only',BLUE),('Pneumococcus + virus',AQUA),('Virus only',ORANGE)]
fig=plt.figure(figsize=(7.3,6.0))
gs=fig.add_gridspec(2,2,height_ratios=[0.78,1],hspace=0.72,wspace=0.60)
ax=fig.add_subplot(gs[0,:])
for i,(lab,key,tot) in enumerate([(f"Consensus ARI\n(n = {R['n_ari']})",'ari',R['n_ari']),
                                  (f"Consensus non-ARI\n(n = {R['n_non']})",'non',R['n_non'])]):
    acc=0
    for k,c in G4:
        w=R['ari_mix'][key][k]/tot*100
        ax.barh(i,w,left=acc,color=c,height=0.5,edgecolor='white',lw=1.2,zorder=2)
        if w>7: ax.text(acc+w/2,i,f'{w:.0f}%',ha='center',va='center',fontsize=8,
                        color='white' if c!=GREY else DARK,fontweight='bold')
        acc+=w
ax.set_yticks(range(2))
ax.set_yticklabels([f"Consensus ARI\n(n = {R['n_ari']})",f"Consensus non-ARI\n(n = {R['n_non']})"])
ax.set_xlim(0,100); ax.set_xlabel('Children (%)'); ax.invert_yaxis()
ax.legend(handles=[Patch(facecolor=c,label=k) for k,c in G4],ncol=4,loc='upper center',
          bbox_to_anchor=(0.5,-0.30),handlelength=1.1,columnspacing=1.4)
ax.set_title('Pathogens detected, by consensus ARI status',fontsize=9); panel(ax,'A')
ORD=['Neither','Pneumococcus only','Virus only','Pneumococcus + virus']
ax=fig.add_subplot(gs[1,0])
bg=R['ari_by_group']; cm=dict(G4)
dotci(ax,ORD,[bg[g]['p'] for g in ORD],[bg[g]['lo'] for g in ORD],[bg[g]['hi'] for g in ORD],
      [cm[g] for g in ORD],[bg[g]['n'] for g in ORD],'Consensus ARI (%)')
brackets(ax,ORD,R['ari_pairs_group'])
ax.set_title('By pathogen combination',fontsize=9); panel(ax,'B')
DOR=['Not carrier','Low density','High density']
ax=fig.add_subplot(gs[1,1])
bd=R['ari_by_dens']
dotci(ax,DOR,[bd[g]['p'] for g in DOR],[bd[g]['lo'] for g in DOR],[bd[g]['hi'] for g in DOR],
      [GREY,BLUE,BLUE],[bd[g]['n'] for g in DOR],'Consensus ARI (%)')
brackets(ax,DOR,R['ari_pairs_dens'])
ax.set_title('By pneumococcal density',fontsize=9); panel(ax,'C')
fig.savefig(f'{OUT}/figure4.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
print('figures 3-4 done')
