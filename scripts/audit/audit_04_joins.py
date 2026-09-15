import pathlib
import pandas as pd, json, re
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx  = pd.read_csv(f"{D}/track1_upi_transactions.csv", dtype=str, keep_default_na=False, na_values=[])
kyc = pd.read_csv(f"{D}/track1_kyc_records.csv", dtype=str, keep_default_na=False, na_values=[])
mer = pd.read_csv(f"{D}/track1_merchants_master.csv", dtype=str, keep_default_na=False, na_values=[])
cb  = pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json", encoding='utf-8'))).astype(str)

# case check on transaction ids
for c in ["txn_id","user_id","merchant_id","utr"]:
    s=tx[c].astype(str); low=(s!=s.str.upper()).sum()
    print(f"TX.{c}: rows not already uppercase = {low:,}")
print()

def norm(v, prefix, width):
    s = str(v).strip().upper()
    s = re.sub(r'[\s\-_\.]', '', s)
    m = re.match(rf'^(?:{prefix})?0*(\d+)$', s)
    return f"{prefix}{int(m.group(1)):0{width}d}" if m else None

tx["u"]=tx.user_id.map(lambda v:norm(v,"USR",5)); tx["m"]=tx.merchant_id.map(lambda v:norm(v,"MCH",4)); tx["t"]=tx.txn_id.map(lambda v:norm(v,"TXN",8))
kyc["u"]=kyc.user_id.map(lambda v:norm(v,"USR",5))
mer["m"]=mer.merchant_id.map(lambda v:norm(v,"MCH",4))
cb["u"]=cb.user_id.map(lambda v:norm(v,"USR",5)); cb["m"]=cb.merchant_id.map(lambda v:norm(v,"MCH",4)); cb["t"]=cb.txn_id.map(lambda v:norm(v,"TXN",8))

def rel(left, lkey, lraw, right, rkey, rraw, name):
    print("="*92); print(f"RELATIONSHIP: {name}"); print("="*92)
    L=left[lkey]; R=set(right[rkey].dropna())
    Lraw=left[lraw].astype(str); Rraw=set(right[rraw].astype(str))
    raw_hit = Lraw.isin(Rraw).sum()
    nrm_hit = L.isin(R).sum()
    n=len(left)
    print(f"  left rows                : {n:,}")
    print(f"  RAW match (exact string) : {raw_hit:,}  ({100*raw_hit/n:.2f}%)")
    print(f"  NORMALIZED match         : {nrm_hit:,}  ({100*nrm_hit/n:.2f}%)")
    print(f"  gain from normalization  : +{nrm_hit-raw_hit:,} rows ({100*(nrm_hit-raw_hit)/n:.2f} pp)")
    unm = left[~L.isin(R)]
    print(f"  UNMATCHED rows           : {len(unm):,}  ({100*len(unm)/n:.2f}%)  distinct keys={unm[lkey].nunique():,}")
    if len(unm): print(f"     examples: {unm[lraw].astype(str).drop_duplicates().head(8).tolist()}")
    # right-side key duplication -> fan-out risk
    dupr = right[rkey].value_counts()
    dupr = dupr[dupr>1]
    print(f"  RIGHT key duplicates     : {len(dupr):,} keys appear >1x (max {dupr.max() if len(dupr) else 0}x), "
          f"{int(dupr.sum()) if len(dupr) else 0:,} rows involved")
    if len(dupr):
        fan = L.isin(set(dupr.index)).sum()
        print(f"  LEFT rows hitting dup key: {fan:,} -> naive merge would inflate to ~{n - fan + int(L.map(right[rkey].value_counts()).fillna(1).sum() - (n-fan)):,} rows")
    print()

rel(tx,"u","user_id",kyc,"u","user_id","transactions -> KYC (user_id)")
rel(tx,"m","merchant_id",mer,"m","merchant_id","transactions -> MERCHANTS (merchant_id)")
rel(cb,"t","txn_id",tx,"t","txn_id","chargebacks -> transactions (txn_id)")
rel(cb,"u","user_id",kyc,"u","user_id","chargebacks -> KYC (user_id)")
rel(cb,"m","merchant_id",mer,"m","merchant_id","chargebacks -> MERCHANTS (merchant_id)")

print("="*92); print("ID SPACE OVERLAP DIAGNOSIS"); print("="*92)
print(f"tx distinct users     : {tx.u.nunique():,}")
print(f"kyc distinct users    : {kyc.u.nunique():,}")
print(f"overlap               : {len(set(tx.u)&set(kyc.u)):,}")
print(f"tx-only users         : {len(set(tx.u)-set(kyc.u)):,}")
print(f"kyc-only users        : {len(set(kyc.u)-set(tx.u)):,}")
print()
print(f"tx distinct merchants : {tx.m.nunique():,}")
print(f"mer distinct merchants: {mer.m.nunique():,}")
print(f"overlap               : {len(set(tx.m)&set(mer.m)):,}")
print(f"tx-only merchants     : {len(set(tx.m)-set(mer.m)):,}")
print(f"mer-only merchants    : {len(set(mer.m)-set(tx.m)):,}")
import numpy as np
tn=tx.m.dropna().map(lambda s:int(s[3:])); mn=mer.m.dropna().map(lambda s:int(s[3:]))
print(f"\ntx merchant numeric range : {tn.min()}..{tn.max()}")
print(f"mer merchant numeric range: {mn.min()}..{mn.max()}")
un=tx.u.dropna().map(lambda s:int(s[3:])); kn=kyc.u.dropna().map(lambda s:int(s[3:]))
print(f"tx user numeric range     : {un.min()}..{un.max()}")
print(f"kyc user numeric range    : {kn.min()}..{kn.max()}")
