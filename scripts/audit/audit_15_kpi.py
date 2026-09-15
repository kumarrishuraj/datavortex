import pathlib
import pandas as pd, json, re, numpy as np
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[]).drop_duplicates()
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str).drop_duplicates()
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); m=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(m.group(1)):0{w}d}" if m else None
def amt(v):
    t=re.sub(r'[₹,]|rs\.?|inr|\s','',str(v),flags=re.I)
    try: return float(t)
    except: return np.nan
def st(s):
    s=str(s).strip().upper()
    return 'SUCCESS' if s in {'S','SUCCESS','TXN_SUCCESS','COMPLETED'} else ('FAILED' if s in {'F','FAILED','TXN_FAILED','FAIL','DECLINED'} else 'PENDING')
def ks(s):
    s=str(s).strip().upper()
    if s in {'V','VERIFIED','APPROVED','KYC_DONE','DONE'}: return 'VERIFIED'
    if s in {'P','PENDING','IN_PROGRESS','UNDER REVIEW'}: return 'PENDING'
    return 'REJECTED'
tx['a']=tx.amount.map(amt); tx['aabs']=tx.a.abs(); tx['st']=tx.status.map(st)
tx['u']=tx.user_id.map(lambda v:norm(v,'USR',5)); tx['m']=tx.merchant_id.map(lambda v:norm(v,'MCH',4))
def n2(v):
    s=re.sub(r'[^0-9]','',str(v)); return int(s) if s else None
tx['t']=tx.txn_id.map(n2); cb['t']=cb.txn_id.map(n2)
txd=tx.drop_duplicates('t').set_index('t'); cb['am']=cb.t.map(txd.m); cb['au']=cb.t.map(txd.u)
cb['da']=cb.disputed_amount.map(amt).abs()
kyc['u']=kyc.user_id.map(lambda v:norm(v,'USR',5)); kyc['ks']=kyc.kyc_status.map(ks)
mer['m']=mer.merchant_id.map(lambda v:norm(v,'MCH',4))

n=len(tx)
print("BASELINE KPI PANEL (post exact-dedup, |amount| used, sign flagged)"); print("="*74)
print(f" Total transactions          : {n:,}")
print(f" Total transaction value     : Rs {tx.aabs.sum():,.0f}")
print(f" Average transaction value   : Rs {tx.aabs.mean():,.2f}   median Rs {tx.aabs.median():,.2f}")
for s in ['SUCCESS','FAILED','PENDING']:
    c=(tx.st==s).sum(); print(f"   {s:<8}: {c:>7,}  ({100*c/n:>5.2f}%)   value Rs {tx.loc[tx.st==s,'aabs'].sum():>14,.0f}")
print(f" Date span                   : {90} days (2026-01-01 .. 2026-03-31, all Q1-2026)")
L=cb.dropna(subset=['am'])
print(f"\n Chargebacks (deduped)       : {len(cb):,}")
print(f"   linked to a transaction   : {len(L):,} ({100*len(L)/len(cb):.2f}%)")
print(f"   unlinked (orphan txn_id)  : {len(cb)-len(L):,}")
print(f" Disputed amount (sum,|x|)   : Rs {cb.da.sum():,.0f}  (mean Rs {cb.da.mean():,.0f}, {cb.da.isna().sum():,} missing)")
print(f" Chargeback-to-txn ratio     : {len(L):,}/{n:,} = {100*len(L)/n:.2f}%")
print(f" Distinct merchants disputed : {L.am.nunique():,} / {tx.m.nunique():,} ({100*L.am.nunique()/tx.m.nunique():.1f}%)")
print(f" Distinct users disputed     : {L.au.nunique():,} / {tx.u.nunique():,}")

print(f"\n KYC rows {len(kyc):,} -> distinct users {kyc.u.nunique():,}")
d=kyc.ks.value_counts()
print("   status mix (row level): "+", ".join(f"{k}={v:,} ({100*v/len(kyc):.1f}%)" for k,v in d.items()))
print(f"   KYC completion rate = VERIFIED/total = {100*d.get('VERIFIED',0)/len(kyc):.2f}%")
print(f"   KYC rejection rate  = REJECTED/total = {100*d.get('REJECTED',0)/len(kyc):.2f}%")

print("\n COVERAGE OF ANALYTICAL DIMENSIONS (the binding constraint)"); print("-"*74)
uset=set(kyc.u.dropna()); mset=set(mer.m.dropna())
print(f"   txns with a KYC record      : {tx.u.isin(uset).sum():,} ({100*tx.u.isin(uset).mean():.2f}%)")
print(f"   txns with a merchant record : {tx.m.isin(mset).sum():,} ({100*tx.m.isin(mset).mean():.2f}%)")
print(f"   txn VALUE covered by merchant master: Rs {tx.loc[tx.m.isin(mset),'aabs'].sum():,.0f} ({100*tx.loc[tx.m.isin(mset),'aabs'].sum()/tx.aabs.sum():.2f}%)")
print(f"   chargebacks -> merchant category resolvable: {L.am.isin(mset).sum():,} ({100*L.am.isin(mset).mean():.2f}% of linked)")
print(f"   chargebacks -> KYC resolvable (via txn user): {L.au.isin(uset).sum():,} ({100*L.au.isin(uset).mean():.2f}% of linked)")

print("\n QUALITY FLAG COUNTS (dashboard 'data quality proof' panel)"); print("-"*74)
flags={
 'txn rows raw':20400,'txn rows after exact dedup':n,
 'amount negative (sign corrupted)':int((tx.a<0).sum()),
 'amount unparseable':int(tx.a.isna().sum()),
 'utr missing':int((tx.utr.str.strip()=='').sum()),
 'utr format-repaired (spaces)':int(tx.utr.str.contains(' ').sum()),
 'mcc missing/unusable in txn':int(tx.mcc.str.strip().eq('').sum()),
 'txn unmatched to KYC':int((~tx.u.isin(uset)).sum()),
 'txn unmatched to merchant master':int((~tx.m.isin(mset)).sum()),
 'chargebacks unlinked to txn':int(len(cb)-len(L)),
 'chargeback disputed_amount missing':int(cb.da.isna().sum()),
 'KYC rows':len(kyc),'KYC user_id collisions (diff people, same id)':5341,
 'KYC PAN invalid (pre-repair)':int((~kyc.pan.str.upper().str.replace(r'[^A-Z0-9]','',regex=True).str.fullmatch(r'[A-Z]{5}[0-9]{4}[A-Z]')).sum()),
 'merchant rows':len(mer),'merchant_id collisions (diff merchants)':1310,
}
for k,v in flags.items(): print(f"   {k:<48s} {v:>8,}")
