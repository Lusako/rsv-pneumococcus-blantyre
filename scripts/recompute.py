# -*- coding: utf-8 -*-
"""All manuscript statistics on the symptomatic cohort (n = 335)."""
import json
import numpy as np, pandas as pd
from scipy import stats

def wilson(k,n,z=1.96):
    if n==0: return (np.nan,)*3
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p*100,max(0,c-h)*100,min(1,c+h)*100
def logit(X,y,names,firth=False,tol=1e-10,it=400):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p)
        I=np.linalg.pinv(X.T@(X*W[:,None])); g=X.T@(y-p)
        if firth:
            H=(X*W[:,None])@I@X.T; g=g+X.T@(np.diag(H)*(0.5-p))
        s=I@g; b+=s
        if np.max(np.abs(s))<tol: break
    p=1/(1+np.exp(-X@b)); W=p*(1-p)
    se=np.sqrt(np.diag(np.linalg.pinv(X.T@(X*W[:,None]))))
    return {n_:dict(OR=float(np.exp(b[i])),lo=float(np.exp(b[i]-1.96*se[i])),
                    hi=float(np.exp(b[i]+1.96*se[i])),p=float(2*(1-stats.norm.cdf(abs(b[i]/se[i])))))
            for i,n_ in enumerate(names) if n_!='_c'}
def ols(X,y,names):
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; dof=len(y)-X.shape[1]
    se=np.sqrt(np.diag(np.linalg.pinv(X.T@X))*(r@r/dof)); t=stats.t.ppf(.975,dof)
    return {n_:dict(beta=float(b[i]),lo=float(b[i]-t*se[i]),hi=float(b[i]+t*se[i]),
                    p=float(2*(1-stats.t.cdf(abs(b[i]/se[i]),dof)))) for i,n_ in enumerate(names) if n_!='_c'}
def design(df,terms):
    cols,names=[np.ones(len(df))],['_c']
    for t in terms:
        if isinstance(t,tuple):
            v,ref=t
            for lv in sorted(x for x in df[v].unique() if x!=ref):
                cols.append((df[v]==lv).astype(float).values); names.append(f'{v}={lv}')
        else: cols.append(df[t].astype(float).values); names.append(t)
    return np.column_stack(cols),names

d0=pd.read_excel('../data/Merged_Data_2.xlsx')
Y=lambda f,c:(f[c]=='Yes')
F='A history of self-reported feverishness or measured fever of ≥ 38°C'
SUS='Clinical suspicion of Acute Respiratory Infection'
VIR=['SARS-CoV-2','FLUA','FLUB','HRV','RSV','HPIV','HPeV','HMPV','EV','HAdV','HBoV','FLUC']
NM={'HRV':'Rhinovirus','FLUA':'Influenza A','HBoV':'Bocavirus','RSV':'RSV','HMPV':'Metapneumovirus',
    'HAdV':'Adenovirus','SARS-CoV-2':'SARS-CoV-2'}
lo=d0.columns.get_loc(F); hi=d0.columns.get_loc('Other symptom')
SYMS=[c for c in d0.columns[lo:hi+1] if c!=SUS]
d0['anysym']=pd.concat([Y(d0,c) for c in SYMS],axis=1).any(axis=1)
d=d0[d0.anysym].copy().reset_index(drop=True)          # <-- symptomatic cohort
d['anyvir']=(d[VIR]=='Positive').any(axis=1).astype(int)
d['nvir']=(d[VIR]=='Positive').sum(axis=1)
d['spn']=((d.lytA_Ct<=35)&(d.piaB_Ct<=35)&(d.SP2020_Ct<=35)).fillna(False).astype(int)
d['log10dens']=np.log10(pd.to_numeric(d.lytA_quantity,errors='coerce'))
d['agegrp']=pd.cut(d.age,[-1,0,1,2,9],labels=['<1 y','1 y','2 y','3–5 y']).astype(str)
d['male']=(d.Gender.astype(str).str.lower().str[0]=='m').astype(int)
d['season']=np.where(pd.to_datetime(d['sample date']).dt.month.isin([11,12,1,2,3,4]),'Rainy (Nov–Apr)','Dry (May–Oct)')
onset=pd.to_datetime(d['Date Day symptoms started'].astype('Int64').astype(str)+' '+
                     d['Month symptoms started'].astype(str)+' '+d['Year symptoms started'].astype(str),
                     format='%d %b %Y',errors='coerce')
gap=(pd.to_datetime(d['sample date'])-onset).dt.days
RESP=['Cough','Sore Throat','Shortness of breath OR Tachypnoea','Runny nose (rhinorrhea)']
d['gari30']=(pd.concat([Y(d,c) for c in RESP],axis=1).any(axis=1)&gap.between(0,14)).astype(int)
d['ari']=(d['gari30'].astype(bool)&Y(d,SUS)).astype(int)
d['agree']=(d['gari30']==Y(d,SUS).astype(int))
MED=d.loc[d.spn==1,'log10dens'].median()
d['virgrp']=np.select([d.nvir==0,d.nvir==1,d.nvir>1],['No virus','Single virus','Multiple viruses'],default='')
d['grp4']=np.select([(d.spn==0)&(d.anyvir==0),(d.spn==1)&(d.anyvir==0),(d.spn==0)&(d.anyvir==1),
                     (d.spn==1)&(d.anyvir==1)],
                    ['Neither','Pneumococcus only','Virus only','Pneumococcus + virus'],default='')
d['densgrp']=np.where(d.spn==0,'Not carrier',np.where(d.log10dens<=MED,'Low density','High density'))
COV=[('agegrp','<1 y'),'male',('season','Dry (May–Oct)')]
carr=d[d.spn==1].dropna(subset=['log10dens'])
S=d[d.agree].copy(); cS=S[S.spn==1].dropna(subset=['log10dens'])
R={'n':len(d),'n_excluded_nosym':int((~d0.anysym).sum()),'n_carr':int(d.spn.sum()),
   'median_dens':float(MED),'n_sub':len(S),'n_ari':int(S.ari.sum()),'n_non':int((1-S.ari).sum()),
   'n_discordant':int((~d.agree).sum())}
print(f"cohort {R['n']} (excluded {R['n_excluded_nosym']} with no symptom); carriers {R['n_carr']}")
print(f"agreement subset {R['n_sub']}: ARI {R['n_ari']}, non-ARI {R['n_non']}; discordant {R['n_discordant']}")
p,l,h=wilson(int(d.spn.sum()),len(d)); R['carriage_overall']=dict(k=int(d.spn.sum()),n=len(d),p=p,lo=l,hi=h)
print(f"overall carriage {p:.1f} ({l:.1f}-{h:.1f});  any virus {d.anyvir.sum()}/{len(d)} = {d.anyvir.mean()*100:.1f}%")
R['anyvir_pct']=float(d.anyvir.mean()*100); R['anyvir_k']=int(d.anyvir.sum())

# ---------- Table 1 ----------
T1=[]
for name,col in [('Age','agegrp'),('Sex','sexlab'),('Season','season'),
                 ('Respiratory virus','virgrp')]:
    if col=='sexlab': d['sexlab']=np.where(d.male==1,'Male','Female')
    lv=[x for x in (['<1 y','1 y','2 y','3–5 y'] if col=='agegrp' else
                    ['Female','Male'] if col=='sexlab' else
                    ['Rainy (Nov–Apr)','Dry (May–Oct)'] if col=='season' else
                    ['No virus','Single virus','Multiple viruses'])]
    rows=[]
    for v in lv:
        m=d[col]==v; k=int(d.loc[m,'spn'].sum()); n=int(m.sum()); pp,ll,hh=wilson(k,n)
        rows.append(dict(level=v,total=n,pct=round(n/len(d)*100,1),k=k,nc=n-k,p=round(pp,1),lo=round(ll,1),hi=round(hh,1)))
    pv=float(stats.chi2_contingency(pd.crosstab(d[col],d.spn).values)[1])
    T1.append(dict(name=name,p=pv,rows=rows)); print(f"  T1 {name:18s} p={pv:.3f}")
ari_rows=[]
for lab,v in [('Consensus non-ARI',0),('Consensus ARI',1)]:
    m=S.ari==v; k=int(S.loc[m,'spn'].sum()); n=int(m.sum()); pp,ll,hh=wilson(k,n)
    ari_rows.append(dict(level=lab,total=n,pct=round(n/len(d)*100,1),k=k,nc=n-k,p=round(pp,1),lo=round(ll,1),hi=round(hh,1)))
m=~d.agree; k=int(d.loc[m,'spn'].sum()); n=int(m.sum()); pp,ll,hh=wilson(k,n)
ari_rows.append(dict(level='Not classified',total=n,pct=round(n/len(d)*100,1),k=k,nc=n-k,p=round(pp,1),lo=round(ll,1),hi=round(hh,1)))
ari_p=float(stats.fisher_exact(pd.crosstab(S.ari,S.spn).values)[1])
T1.append(dict(name='Consensus ARI status',p=ari_p,rows=ari_rows)); print(f"  T1 Consensus ARI     p={ari_p:.4f}")
R['table1']=T1

# ---------- combined Table 2/3: carriage + density by virus ----------
T2=[]
def blk(label,mask,nm_):
    k=int(d.loc[mask,'spn'].sum()); n=int(mask.sum()); pp,ll,hh=wilson(k,n)
    cm=carr[mask.reindex(carr.index,fill_value=False)]
    return dict(virus=label,npos=n,carr_k=k,carr_p=round(pp,1),carr_lo=round(ll,1),carr_hi=round(hh,1),
                n_carr=len(cm),med=round(cm.log10dens.median(),2) if len(cm) else None)
ref=blk('No virus (ref.)',d.anyvir==0,None); T2.append(ref)
for lab,mask,col in [('Any virus',d.anyvir==1,'anyvir'),
                     ('Single virus',d.virgrp=='Single virus','_sv'),
                     ('Multiple viruses',d.virgrp=='Multiple viruses','_mv')]+\
                    [(NM[v],(d[v]=='Positive'),v) for v in ['HRV','FLUA','HBoV','RSV','HMPV','HAdV','SARS-CoV-2']]:
    d['_x']=mask.astype(int); carr['_x']=mask.reindex(carr.index,fill_value=False).astype(int)
    sel=d[(d.anyvir==0)|mask]; selc=carr[(carr.anyvir==0)|mask.reindex(carr.index,fill_value=False)]
    nfew=int(mask.sum())<20
    X,nmv=design(sel,['_x']+COV); o=logit(X,sel.spn.values,nmv,firth=nfew)['_x']
    X,nmv=design(selc,['_x']+COV); b=ols(X,selc.log10dens.values,nmv)['_x']
    e=blk(lab,mask,col); e.update(aOR=o['OR'],aOR_lo=o['lo'],aOR_hi=o['hi'],aOR_p=o['p'],
                                  beta=b['beta'],beta_lo=b['lo'],beta_hi=b['hi'],beta_p=b['p'],
                                  fold=10**b['beta'],fold_lo=10**b['lo'],fold_hi=10**b['hi'])
    T2.append(e)
    print(f"  {lab:18s} n={e['npos']:3d} carr {e['carr_p']:4.1f}% aOR {o['OR']:.2f} ({o['lo']:.2f}-{o['hi']:.2f}) p={o['p']:.3f} | "
          f"med {e['med']} beta {b['beta']:+.2f} ({b['lo']:+.2f},{b['hi']:+.2f}) p={b['p']:.3f}")
R['table2']=T2
R['dens_kw_p']=float(stats.kruskal(*[g.log10dens.values for _,g in carr.groupby('virgrp')]).pvalue)
R['dens_by_virgrp']={g:dict(n=len(x),med=round(x.log10dens.median(),2),
                            q1=round(x.log10dens.quantile(.25),2),q3=round(x.log10dens.quantile(.75),2))
                     for g,x in carr.groupby('virgrp')}
print('  density by virus group KW p=%.4f'%R['dens_kw_p'],R['dens_by_virgrp'])

# ---------- Consensus ARI analyses ----------
X,nmv=design(S,['ari']+COV); R['ari_carr']=logit(X,S.spn.values,nmv)['ari']
X,nmv=design(cS,['ari']+COV); R['ari_dens']=ols(X,cS.log10dens.values,nmv)['ari']
a,b_=cS[cS.ari==1].log10dens,cS[cS.ari==0].log10dens
R['ari_dens_desc']=dict(med_ari=round(a.median(),2),n_ari=len(a),med_non=round(b_.median(),2),n_non=len(b_),
    q_ari=[round(a.quantile(.25),2),round(a.quantile(.75),2)],q_non=[round(b_.quantile(.25),2),round(b_.quantile(.75),2)],
    mw_p=float(stats.mannwhitneyu(a,b_).pvalue))
ct=pd.crosstab(S.ari,S.anyvir)
R['ari_vir']=dict(k1=int(ct.loc[1,1]),n1=int(ct.loc[1].sum()),k0=int(ct.loc[0,1]),n0=int(ct.loc[0].sum()),
                  p=float(stats.fisher_exact(ct.values)[1]))
R['ari_by_group']={g:dict(k=int(S.loc[S.grp4==g,'ari'].sum()),n=int((S.grp4==g).sum())) for g in
                   ['Neither','Pneumococcus only','Virus only','Pneumococcus + virus']}
R['ari_by_dens']={g:dict(k=int(S.loc[S.densgrp==g,'ari'].sum()),n=int((S.densgrp==g).sum())) for g in
                  ['Not carrier','Low density','High density']}
for k_,v in list(R['ari_by_group'].items())+list(R['ari_by_dens'].items()):
    pp,ll,hh=wilson(v['k'],v['n']); v.update(p=round(pp,1),lo=round(ll,1),hi=round(hh,1))
R['ari_mix']={lab:{g:int(((S.grp4==g)&(S.ari==v)).sum()) for g in
                   ['Neither','Pneumococcus only','Virus only','Pneumococcus + virus']}
              for lab,v in [('ari',1),('non',0)]}
X,nmv=design(S,['spn','anyvir']+COV); m1=logit(X,S.ari.values,nmv)
R['ari_m_spn'],R['ari_m_vir']=m1['spn'],m1['anyvir']
R['ari_m_cov']={k_:v for k_,v in m1.items() if k_ not in ('spn','anyvir')}
S['inter']=S.spn*S.anyvir
X,nmv=design(S,['spn','anyvir','inter']+COV); R['ari_m_int']=logit(X,S.ari.values,nmv)['inter']
X,nmv=design(S,[('densgrp','Not carrier'),'anyvir']+COV); mm=logit(X,S.ari.values,nmv)
R['ari_m_low'],R['ari_m_high']=mm['densgrp=Low density'],mm['densgrp=High density']
R['ari_m_virus']={}
for v in ['HRV','FLUA','HBoV','RSV','HMPV','HAdV','SARS-CoV-2']:
    S['_v']=(S[v]=='Positive').astype(int)
    X,nmv=design(S,['_v']+COV); R['ari_m_virus'][NM[v]]=logit(X,S.ari.values,nmv,firth=True)['_v']
print(f"\n  ARI carriage aOR {R['ari_carr']['OR']:.2f} ({R['ari_carr']['lo']:.2f}-{R['ari_carr']['hi']:.2f}) p={R['ari_carr']['p']:.4f}")
print(f"  ARI density beta {R['ari_dens']['beta']:+.2f} ({R['ari_dens']['lo']:+.2f},{R['ari_dens']['hi']:+.2f}) p={R['ari_dens']['p']:.3f}")
print(f"  ARI vs virus {R['ari_vir']['k1']}/{R['ari_vir']['n1']} vs {R['ari_vir']['k0']}/{R['ari_vir']['n0']} p={R['ari_vir']['p']:.3f}")
for k_ in ['ari_m_spn','ari_m_vir','ari_m_int','ari_m_low','ari_m_high']:
    o=R[k_]; print(f"  {k_:12s} {o['OR']:.2f} ({o['lo']:.2f}-{o['hi']:.2f}) p={o['p']:.4f}")
for v,o in R['ari_m_virus'].items(): print(f"    {v:16s} {o['OR']:.2f} ({o['lo']:.2f}-{o['hi']:.2f}) p={o['p']:.4f}")
print('  by group',{k_:(v['k'],v['n'],v['p']) for k_,v in R['ari_by_group'].items()})
print('  by dens ',{k_:(v['k'],v['n'],v['p']) for k_,v in R['ari_by_dens'].items()})

# pairwise tests for the figure
def pair(dic,pairs):
    out=[]
    for a_,b2 in pairs:
        ka,na=dic[a_]['k'],dic[a_]['n']; kb,nb=dic[b2]['k'],dic[b2]['n']
        out.append(dict(a=a_,b=b2,p=float(stats.fisher_exact([[ka,na-ka],[kb,nb-kb]])[1])))
    return out
R['ari_pairs_group']=pair(R['ari_by_group'],[('Neither','Pneumococcus only'),('Neither','Virus only'),
                          ('Pneumococcus only','Virus only'),('Neither','Pneumococcus + virus')])
R['ari_pairs_dens']=pair(R['ari_by_dens'],[('Not carrier','Low density'),('Not carrier','High density'),
                         ('Low density','High density')])
print('  pairs group',[(x['a'][:12],x['b'][:12],round(x['p'],3)) for x in R['ari_pairs_group']])
print('  pairs dens ',[(x['a'][:12],x['b'][:12],round(x['p'],3)) for x in R['ari_pairs_dens']])
json.dump(R,open('nums335.json','w'),indent=1,default=float)
print('\nwrote nums335.json')
