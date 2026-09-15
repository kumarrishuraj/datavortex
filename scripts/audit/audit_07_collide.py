import pathlib
import pandas as pd, json, re
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); m=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(m.group(1)):0{w}d}" if m else None
def namekey(s): return re.sub(r'[^a-z]','',str(s).lower())

kyc['u']=kyc.user_id.map(lambda v:norm(v,'USR',5)); kyc['nk']=kyc.full_name.map(namekey)
g=kyc.groupby('u')
sz=g.size()
multi=sz[sz>1].index
sub=kyc[kyc.u.isin(multi)]
gg=sub.groupby('u')
same_name = gg.nk.nunique().eq(1)
print("KYC REPEATED-ID CLASSIFICATION")
print("-"*70)
print(f" users with >1 row                  : {len(multi):,}  ({len(sub):,} rows)")
print(f"   SAME person (identical name key) : {same_name.sum():,}")
print(f"   DIFFERENT people (ID COLLISION)  : {(~same_name).sum():,}")
# among same-name groups, are they byte-identical?
sn=sub[sub.u.isin(same_name[same_name].index)]
exact = sn.drop(columns=['u','nk']).duplicated(keep=False).groupby(sn.u).all()
print(f"     of same-person groups, byte-identical rows: {exact.sum():,}")
print(f"     of same-person groups, conflicting attrs  : {(~exact).sum():,}")
print(f" rows recoverable by exact-dedup     : {kyc.duplicated().sum():,}")

mer['m']=mer.merchant_id.map(lambda v:norm(v,'MCH',4)); mer['nk']=mer.merchant_name.map(namekey)
g2=mer.groupby('m'); sz2=g2.size(); multi2=sz2[sz2>1].index
sub2=mer[mer.m.isin(multi2)]; gg2=sub2.groupby('m'); same2=gg2.nk.nunique().eq(1)
print("\nMERCHANT REPEATED-ID CLASSIFICATION")
print("-"*70)
print(f" merchants with >1 row              : {len(multi2):,}  ({len(sub2):,} rows)")
print(f"   SAME merchant (identical name)   : {same2.sum():,}")
print(f"   DIFFERENT merchants (COLLISION)  : {(~same2).sum():,}")
print(f" rows recoverable by exact-dedup    : {mer.duplicated().sum():,}")

print("\n"+"="*80); print("AMOUNT FORENSICS"); print("="*80)
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[])
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str)
def shape(v):
    s=str(v); s=re.sub(r'[0-9]','9',s); s=re.sub(r'9+','9',s); return s
for df,col,lab in [(tx,'amount','TX.amount'),(cb,'disputed_amount','CB.disputed_amount'),
                   (mer,'declared_avg_ticket_size','MER.avg_ticket'),(kyc,'monthly_income','KYC.income')]:
    s=df[col].astype(str); vc=s.map(shape).value_counts()
    print(f"\n--- {lab} ({s.nunique():,} distinct) ---")
    for p,c in vc.head(16).items():
        ex=s[s.map(shape)==p].iloc[0]
        print(f"   {p:<18s} {c:>6,}  e.g. {repr(ex)}")

def parse(v):
    s=str(v).strip()
    if s=='' or s.lower() in ('nan','none','na','n/a','null','-'): return ('BLANK',None)
    k=s.lower().replace('k','k')
    neg = s.strip().startswith('-') or (s.strip().startswith('(') and s.strip().endswith(')'))
    t=re.sub(r'[₹,]|rs\.?|inr|\s','',s,flags=re.I)
    mk=re.match(r'^-?([\d.]+)k$',t,flags=re.I)
    if mk: 
        try: return ('K_SUFFIX', float(mk.group(1))*1000*(-1 if neg else 1))
        except: return ('INVALID',None)
    try: return ('OK', float(t))
    except: return ('INVALID',None)

for df,col,lab in [(tx,'amount','TX.amount'),(cb,'disputed_amount','CB.disputed_amount'),
                   (mer,'declared_avg_ticket_size','MER.avg_ticket'),(kyc,'monthly_income','KYC.income')]:
    r=df[col].astype(str).map(parse)
    kind=r.map(lambda x:x[0]); val=r.map(lambda x:x[1])
    print(f"\n{lab}: "+"  ".join(f"{k}={v:,}" for k,v in kind.value_counts().items()))
    ok=val.dropna()
    print(f"   parsed n={len(ok):,} min={ok.min():,.2f} max={ok.max():,.2f} mean={ok.mean():,.2f} "
          f"negatives={(ok<0).sum():,} zeros={(ok==0).sum():,}")
    bad=df[col].astype(str)[kind=='INVALID'].drop_duplicates().head(8).tolist()
    if bad: print(f"   INVALID examples: {bad}")
    if (ok<0).sum(): print(f"   negative examples: {df[col].astype(str)[val.fillna(0)<0].drop_duplicates().head(6).tolist()}")
