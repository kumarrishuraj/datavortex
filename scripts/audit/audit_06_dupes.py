import pathlib
import pandas as pd, json, re
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx  = pd.read_csv(f"{D}/track1_upi_transactions.csv", dtype=str, keep_default_na=False, na_values=[])
kyc = pd.read_csv(f"{D}/track1_kyc_records.csv", dtype=str, keep_default_na=False, na_values=[])
mer = pd.read_csv(f"{D}/track1_merchants_master.csv", dtype=str, keep_default_na=False, na_values=[])
cb  = pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json", encoding='utf-8'))).astype(str)
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); m=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(m.group(1)):0{w}d}" if m else None

print("="*92); print("DUPLICATE FORENSICS — TRANSACTIONS"); print("="*92)
print(f" total rows              : {len(tx):,}")
print(f" exact full-row dupes    : {tx.duplicated(keep=False).sum():,}  (extra rows: {tx.duplicated().sum():,})")
d = tx[tx.duplicated('txn_id', keep=False)].sort_values('txn_id')
print(f" rows sharing a txn_id   : {len(d):,}  across {d.txn_id.nunique():,} txn_ids")
g = d.groupby('txn_id')
def agree(col): return g[col].nunique().eq(1).sum()
print(f"   of those {d.txn_id.nunique():,} duplicated txn_ids, how many have IDENTICAL values per field:")
for c in ['user_id','merchant_id','amount','utr','mcc','status','timestamp']:
    print(f"     {c:<12s}: {agree(c):>5,} identical / {d.txn_id.nunique():,}")
core = ['txn_id','user_id','merchant_id']
same_core = g[core].nunique().eq(1).all(axis=1).sum()
print(f"   same txn_id AND same user AND same merchant: {same_core:,}")
print("\n  SAMPLE duplicate txn_id pairs:")
for t in d.txn_id.drop_duplicates().head(4):
    print(d[d.txn_id==t].to_string(index=False)); print()

print("="*92); print("DUPLICATE FORENSICS — CHARGEBACKS"); print("="*92)
print(f" total rows            : {len(cb):,}")
print(f" exact full-row dupes  : {cb.duplicated(keep=False).sum():,} (extra: {cb.duplicated().sum():,})")
dc = cb[cb.duplicated('complaint_id', keep=False)].sort_values('complaint_id')
print(f" rows sharing compl_id : {len(dc):,} across {dc.complaint_id.nunique():,} ids")
gc=dc.groupby('complaint_id')
for c in ['txn_id','user_id','merchant_id','disputed_amount','reason_code','resolution_status','severity']:
    print(f"     {c:<18s}: {gc[c].nunique().eq(1).sum():>4,} identical / {dc.complaint_id.nunique():,}")
print("\n  SAMPLE:"); 
for t in dc.complaint_id.drop_duplicates().head(2):
    print(dc[dc.complaint_id==t].drop(columns=['complaint_text']).to_string(index=False)); print()
# txn_id reuse across different complaints
cb['t']=cb.txn_id.map(lambda v:norm(v,'TXN',8))
vt=cb.dropna(subset=['t']).t.value_counts(); 
print(f" distinct txns disputed more than once: {(vt>1).sum():,} (max {vt.max()} complaints on one txn)")

print("="*92); print("MASTER-DATA CONFLICTS — KYC"); print("="*92)
kyc['u']=kyc.user_id.map(lambda v:norm(v,'USR',5))
print(f" rows={len(kyc):,}  distinct raw={kyc.user_id.nunique():,}  distinct normalized={kyc.u.nunique():,}")
print(f" exact full-row dupes  : {kyc.duplicated().sum():,}")
dk=kyc[kyc.duplicated('u',keep=False)]
print(f" rows with repeated user (normalized): {len(dk):,} across {dk.u.nunique():,} users (max {kyc.u.value_counts().max()}x)")
gk=dk.groupby('u')
for c in ['full_name','pan','aadhaar','kyc_status','risk_segment','monthly_income','city','state','occupation','date_of_birth','signup_timestamp']:
    conf = gk[c].nunique().gt(1).sum()
    print(f"     {c:<18s}: {conf:>5,} users have CONFLICTING values ({100*conf/dk.u.nunique():.1f}%)")
print("\n  SAMPLE conflicting user:")
cand=gk['kyc_status'].nunique(); cand=cand[cand>1]
if len(cand): print(kyc[kyc.u==cand.index[0]].to_string(index=False))

print("\n"+"="*92); print("MASTER-DATA CONFLICTS — MERCHANTS"); print("="*92)
mer['m']=mer.merchant_id.map(lambda v:norm(v,'MCH',4))
print(f" rows={len(mer):,}  distinct raw={mer.merchant_id.nunique():,}  distinct normalized={mer.m.nunique():,}")
print(f" exact full-row dupes  : {mer.duplicated().sum():,}")
dm=mer[mer.duplicated('m',keep=False)]
print(f" rows with repeated merchant: {len(dm):,} across {dm.m.nunique():,} merchants (max {mer.m.value_counts().max()}x)")
gm=dm.groupby('m')
for c in ['merchant_name','mcc','merchant_category','business_type','city','state','merchant_status','onboarding_date','settlement_account','declared_avg_ticket_size']:
    conf=gm[c].nunique().gt(1).sum()
    print(f"     {c:<24s}: {conf:>5,} merchants CONFLICT ({100*conf/dm.m.nunique():.1f}%)")
print("\n  SAMPLE conflicting merchant:")
cand=gm['merchant_status'].nunique(); cand=cand[cand>1]
if len(cand): print(mer[mer.m==cand.index[0]].to_string(index=False))
