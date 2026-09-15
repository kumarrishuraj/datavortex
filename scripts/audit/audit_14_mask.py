import pathlib
import pandas as pd, json, re, numpy as np
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); m=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(m.group(1)):0{w}d}" if m else None
kyc['u']=kyc.user_id.map(lambda v:norm(v,'USR',5)); mer['m']=mer.merchant_id.map(lambda v:norm(v,'MCH',4))

print("MASKING ARTIFACT CHECK — AADHAAR"); print("="*78)
a=kyc.aadhaar.astype(str).str.strip()
masked=a.str.upper().str.contains('X'); full=a.str.replace(r'[^0-9]','',regex=True).str.fullmatch(r'\d{12}')
print(f" masked (XXXX-XXXX-nnnn): {masked.sum():,} | full 12-digit: {full.sum():,} | blank/other: {len(a)-masked.sum()-full.sum():,}")
fk=kyc[full].assign(k=kyc[full].aadhaar.str.replace(r'[^0-9]','',regex=True))
g=fk.groupby('k').u.nunique()
print(f" FULL 12-digit Aadhaar: distinct={len(g):,} | shared across >1 user_id: {(g>1).sum():,} (max {g.max()})")
mk=kyc[masked].assign(k=kyc[masked].aadhaar.str.replace(r'[^0-9]','',regex=True))
gm=mk.groupby('k').u.nunique()
n=len(mk); slots=10000
print(f" MASKED Aadhaar: n={n:,} over only {slots:,} possible last-4 values")
print(f"   shared across >1 user_id: {(gm>1).sum():,} | expected by chance ~{n-slots*(1-(1-1/slots)**n):.0f} colliding keys")
print("   => masked-Aadhaar 'sharing' is a BIRTHDAY COLLISION artifact, NOT identity fraud")

print("\nMASKING ARTIFACT CHECK — SETTLEMENT ACCOUNT"); print("="*78)
sa=mer.settlement_account.astype(str).str.strip().str.upper().replace({'':None,'NA':None,'NAN':None,'NONE':None})
mer['sa']=sa
msk=mer.sa.fillna('').str.startswith('XXXX')
fl =mer.sa.notna() & ~mer.sa.fillna('').str.startswith('XXXX')
print(f" masked XXXXnnnn: {msk.sum():,} | full account nos: {fl.sum():,}")
for sel,lab,slots in [(fl,'FULL account',None),(msk,'MASKED (last4)',10000)]:
    sub=mer[sel]; g=sub.groupby('sa').m.nunique()
    print(f" {lab}: distinct={len(g):,} shared by >1 merchant: {(g>1).sum():,} (max {g.max()})")
    if slots:
        n=len(sub); print(f"    expected colliding keys by chance over {slots:,} slots: ~{n-slots*(1-(1-1/slots)**n):.0f}")
print("\n FULL-account sharing detail (the only defensible signal):")
sub=mer[fl]; g=sub.groupby('sa').m.nunique(); sh=g[g>1]
if len(sh):
    for k in sh.index[:4]:
        print(mer[mer.sa==k][['merchant_id','merchant_name','merchant_category','city','state','merchant_status','sa']].to_string(index=False)); print()
else: print("   none")

print("\nPAN SHARING (full, unmasked)"); print("="*78)
p=kyc.pan.astype(str).str.upper().str.replace(r'[^A-Z0-9]','',regex=True).replace('',None)
pk=kyc.assign(k=p).dropna(subset=['k'])
g=pk.groupby('k').u.nunique()
print(f" distinct PAN={len(g):,} shared across >1 user_id: {(g>1).sum():,}")
print(f" same user_id carrying >1 DIFFERENT PAN: {(pk.groupby('u').k.nunique()>1).sum():,}  <-- the real identity problem")

print("\n\nFINAL CLEAN-ROW ACCOUNTING (raw -> after exact dedup)"); print("="*78)
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[])
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str)
for nm,df in [('transactions',tx),('kyc',kyc[kyc.columns.difference(['u'])]),('merchants',mer[mer.columns.difference(['m','sa'])]),('chargebacks',cb)]:
    print(f" {nm:<13} raw={len(df):>7,}  exact-dupe rows removed={df.duplicated().sum():>6,}  -> {len(df)-df.duplicated().sum():>7,}")
