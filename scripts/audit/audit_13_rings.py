import pathlib
import pandas as pd, json, re, numpy as np
from collections import Counter
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[]).drop_duplicates()
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str).drop_duplicates()
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); m=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(m.group(1)):0{w}d}" if m else None
kyc['u']=kyc.user_id.map(lambda v:norm(v,'USR',5)); mer['m']=mer.merchant_id.map(lambda v:norm(v,'MCH',4))
tx['u']=tx.user_id.map(lambda v:norm(v,'USR',5)); tx['m']=tx.merchant_id.map(lambda v:norm(v,'MCH',4))

print("CANDIDATE RING SIGNAL 1: SHARED SETTLEMENT ACCOUNT ACROSS MERCHANTS"); print("="*84)
sa=mer.settlement_account.astype(str).str.strip().str.upper()
sa=sa.replace({'':None,'NA':None,'NAN':None,'NONE':None,'N/A':None})
mer['sa']=sa
print(f" non-null settlement accounts: {mer.sa.notna().sum():,} / {len(mer):,} ({100*mer.sa.notna().mean():.1f}%)")
def sash(v):
    s=str(v); s=re.sub(r'[A-Za-z]','A',s); s=re.sub(r'[0-9]','9',s); s=re.sub(r'A+','A+',s); s=re.sub(r'9+','9+',s); return s
print(" formats:")
for p,c in mer.sa.dropna().map(sash).value_counts().head(8).items():
    print(f"   {p:<14s} {c:>6,}  e.g. {mer.sa.dropna()[mer.sa.dropna().map(sash)==p].iloc[0]}")
g=mer.dropna(subset=['sa']).groupby('sa').m.nunique()
print(f"\n distinct settlement accounts: {len(g):,}")
print(f" accounts used by >1 DISTINCT merchant: {(g>1).sum():,}  (max {g.max()} merchants)")
if (g>1).sum():
    top=g[g>1].sort_values(ascending=False).head(5)
    print(f"   top shared: \n{top.to_string()}")
    ex=top.index[0]
    print(mer[mer.sa==ex][['merchant_id','merchant_name','merchant_category','city','state','merchant_status','sa']].to_string(index=False))

print("\n\nCANDIDATE RING SIGNAL 2: SHARED PAN / AADHAAR ACROSS USER IDS"); print("="*84)
pan=kyc.pan.astype(str).str.upper().str.replace(r'[^A-Z0-9]','',regex=True).replace('',None)
aad=kyc.aadhaar.astype(str).str.replace(r'[^0-9]','',regex=True).replace('',None)
kyc['panc']=pan; kyc['aadc']=aad
for col,lab in [('panc','PAN'),('aadc','AADHAAR')]:
    s=kyc.dropna(subset=[col])
    g2=s.groupby(col).u.nunique()
    g3=s.groupby(col).size()
    print(f" {lab}: non-null={len(s):,} distinct={len(g2):,} | used by >1 user_id: {(g2>1).sum():,} (max {g2.max()}) | rows repeating a {lab}: {(g3>1).sum():,}")
    if (g2>1).sum():
        t=g2[g2>1].sort_values(ascending=False).head(3)
        print(f"   examples:\n{kyc[kyc[col]==t.index[0]][['user_id','full_name','pan','aadhaar','city','kyc_status']].to_string(index=False)}")
# same person name+dob under different ids
kyc['nk']=kyc.full_name.str.lower().str.replace(r'[^a-z]','',regex=True)
gn=kyc.groupby('nk').u.nunique()
print(f"\n identical normalized NAME under >1 user_id: {(gn>1).sum():,} (max {gn.max()})")

print("\n\nCANDIDATE RING SIGNAL 3: DISPUTE CONCENTRATION (merchant / user level)"); print("="*84)
def n2(v):
    s=re.sub(r'[^0-9]','',str(v)); return int(s) if s else None
cb['t']=cb.txn_id.map(n2); tx['t']=tx.txn_id.map(n2)
txd=tx.drop_duplicates('t').set_index('t')
cb['am']=cb.t.map(txd.m); cb['au']=cb.t.map(txd.u)
L=cb.dropna(subset=['am'])
print(f" chargebacks attributed via txn_id: {len(L):,}")
mc=L.am.value_counts(); uc=L.au.value_counts()
print(f" merchants with >=1 dispute: {len(mc):,}  >=2: {(mc>=2).sum():,}  >=3: {(mc>=3).sum():,}  max={mc.max()}")
print(f" users     with >=1 dispute: {len(uc):,}  >=2: {(uc>=2).sum():,}  >=3: {(uc>=3).sum():,}  max={uc.max()}")
tpm=tx.m.value_counts()
rate=(mc/tpm.reindex(mc.index)).dropna()
print(f"\n merchant chargeback RATIO (disputes/txns): n={len(rate):,} median={rate.median():.3f} max={rate.max():.3f}")
print(f"   merchants with ratio=1.0 (every txn disputed): {(rate>=1).sum():,}")
elig=tpm[tpm>=3]
r2=(mc.reindex(elig.index).fillna(0)/elig)
print(f"   among merchants with >=3 txns (n={len(elig):,}): ratio median={r2.median():.3f} p95={r2.quantile(.95):.3f} max={r2.max():.3f}, ratio==1.0: {(r2>=1).sum():,}")
print("\n TOP merchants by dispute count (attributed):")
print(mc.head(8).to_string())
print(f"\n baseline overall dispute rate = {len(L):,}/{len(tx):,} = {100*len(L)/len(tx):.2f}%")

print("\n\nCANDIDATE RING SIGNAL 4: TEMPORAL BURSTS / VELOCITY"); print("="*84)
MON=r'(?i)^\d{1,2}-[A-Za-z]{3}-\d{4}'
def parse(v):
    s=str(v).strip()
    if s=='': return pd.NaT
    if re.fullmatch(r'-?\d{8,10}',s):
        try: return pd.to_datetime(int(s),unit='s')
        except: return pd.NaT
    if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',s): return pd.to_datetime(s.replace('/','-'),errors='coerce')
    if re.match(MON,s): return pd.to_datetime(s,format='%d-%b-%Y',errors='coerce')
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}',s): return pd.to_datetime(s,dayfirst=True,errors='coerce')
    if re.match(r'^\d{1,2}-\d{1,2}-\d{4}',s): return pd.to_datetime(s,dayfirst=False,errors='coerce')
    return pd.to_datetime(s,errors='coerce')
tx['ts']=tx.timestamp.map(parse)
d=tx.groupby(tx.ts.dt.date).size()
print(f" daily txn counts over {len(d)} days: mean={d.mean():.1f} sd={d.std():.1f} min={d.min()} max={d.max()}  cv={d.std()/d.mean():.3f}")
h=tx.groupby(tx.ts.dt.hour).size()
print(f" hourly profile: min={h.min()} max={h.max()} cv={h.std()/h.mean():.3f}  -> {'FLAT (uniform random)' if h.std()/h.mean()<0.15 else 'has structure'}")
print(f" hour counts: {h.to_dict()}")
# per-merchant daily burst
mb=tx.groupby(['m',tx.ts.dt.date]).size()
print(f" max txns by one merchant in one day: {mb.max()}  (merchants with >=4 in a day: {(mb>=4).sum():,})")
# per-user velocity
ub=tx.sort_values('ts').groupby('u').ts.apply(lambda s: s.diff().dt.total_seconds().min()/60 if len(s)>1 else np.nan).dropna()
print(f" users with >1 txn: {len(ub):,}  min gap between consecutive txns: median={ub.median():.0f} min  under 10min={(ub<10).sum():,}")
