import pathlib
import pandas as pd, json, re, numpy as np
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[])
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str)
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); m=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(m.group(1)):0{w}d}" if m else None
def mcc(v):
    s=str(v).strip().upper()
    if s in ('','NA','UNKNOWN','MISC','N/A','NONE','NAN'): return None
    s=re.sub(r'^MCC[-_ ]?','',s); s=re.sub(r'\.0$','',s); s=s.lstrip('0')
    return s.zfill(4) if s.isdigit() else None
tx['mccn']=tx.mcc.map(mcc); mer['mccn']=mer.mcc.map(mcc)
tx['m']=tx.merchant_id.map(lambda v:norm(v,'MCH',4))
mer['m']=mer.merchant_id.map(lambda v:norm(v,'MCH',4))
print("MCC — CORRECTED CANONICALIZATION"); print("="*80)
print(f" TX  canonical set : {sorted(tx.mccn.dropna().unique())}")
print(f" MER canonical set : {sorted(mer.mccn.dropna().unique())}")
print(f" TX  null: {tx.mccn.isna().sum():,} ({100*tx.mccn.isna().mean():.2f}%)   MER null: {mer.mccn.isna().sum():,} ({100*mer.mccn.isna().mean():.2f}%)")
print(f"\n TX  mcc dist:  "+", ".join(f"{k}={v:,}" for k,v in tx.mccn.value_counts().items()))
print(f" MER mcc dist:  "+", ".join(f"{k}={v:,}" for k,v in mer.mccn.value_counts().items()))
mm=mer.dropna(subset=['mccn']).drop_duplicates('m').set_index('m').mccn
tx['mer_mcc']=tx.m.map(mm); c=tx.dropna(subset=['mccn','mer_mcc'])
agree=(c.mccn==c.mer_mcc)
ptx=tx.mccn.value_counts(normalize=True); pme=mer.mccn.value_counts(normalize=True)
exp=sum(ptx.get(v,0)*pme.get(v,0) for v in set(ptx.index)|set(pme.index))
print(f"\n both known: {len(c):,} | AGREE {agree.sum():,} ({100*agree.mean():.2f}%) | expected-if-independent {100*exp:.2f}%")
print(" -> tx.mcc and master.mcc are INDEPENDENT random draws" if abs(agree.mean()-exp)<0.02 else " -> some real signal")

print("\n\nCHARGEBACK ATTRIBUTION HYPOTHESIS"); print("="*80)
def n2(v):
    s=re.sub(r'[^0-9]','',str(v)); return int(s) if s else None
cb['t']=cb.txn_id.map(n2); tx['t']=tx.txn_id.map(n2)
cb['cbm']=cb.merchant_id.map(lambda v:norm(v,'MCH',4)); cb['cbu']=cb.user_id.map(lambda v:norm(v,'USR',5))
tx['u']=tx.user_id.map(lambda v:norm(v,'USR',5))
txd=tx.drop_duplicates('t').set_index('t')
cb['tx_m']=cb.t.map(txd.m); cb['tx_u']=cb.t.map(txd.u)
linked=cb.dropna(subset=['tx_m'])
print(f" chargebacks linked to a transaction via txn_id : {len(linked):,} / {len(cb):,} ({100*len(linked)/len(cb):.2f}%)")
print(f"   cb.merchant_id == linked txn's merchant : {(linked.cbm==linked.tx_m).sum():,} ({100*(linked.cbm==linked.tx_m).mean():.3f}%)")
print(f"   cb.user_id     == linked txn's user     : {(linked.cbu==linked.tx_u).sum():,} ({100*(linked.cbu==linked.tx_u).mean():.3f}%)")
print(f"   (chance if independent: merchant {100/9000:.3f}%, user {100/90000:.4f}%)")
mset=set(mer.m.dropna())
print(f"\n cb.merchant_id present in master  : {cb.cbm.isin(mset).sum():,} ({100*cb.cbm.isin(mset).mean():.2f}%)  [master covers {100*len(mset)/9000:.1f}% of id space]")
print(f" ATTRIBUTED merchant (via txn) in master: {linked.tx_m.isin(mset).sum():,} ({100*linked.tx_m.isin(mset).mean():.2f}%)")
print("\n disputed_amount vs linked txn amount:")
def amt(v):
    t=re.sub(r'[₹,]|rs\.?|inr|\s','',str(v),flags=re.I)
    try: return float(t)
    except: return np.nan
tx['a']=tx.amount.map(amt); cb['da']=cb.disputed_amount.map(amt); cb['tx_a']=cb.t.map(txd.amount.map(amt))
z=cb.dropna(subset=['da','tx_a'])
print(f"   n={len(z):,}  exact equal: {(z.da.round(2)==z.tx_a.round(2)).sum():,}   |disputed|<=|txn|: {(z.da.abs()<=z.tx_a.abs()).sum():,} ({100*(z.da.abs()<=z.tx_a.abs()).mean():.1f}%)")
print(f"   median disputed={z.da.abs().median():,.0f}  median txn={z.tx_a.abs().median():,.0f}")

print("\n\nPAN REPAIR TEST (O/0, I/1 confusion)"); print("="*80)
raw=kyc.pan.astype(str).str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
nb=raw[raw!='']
valid=nb.str.fullmatch(r'[A-Z]{5}[0-9]{4}[A-Z]')
print(f" non-blank PAN: {len(nb):,}  valid as-is: {valid.sum():,} ({100*valid.mean():.2f}%)  invalid: {(~valid).sum():,}")
bad=nb[~valid]
print(f" invalid length distribution: "+", ".join(f"len{k}={v:,}" for k,v in bad.str.len().value_counts().items()))
def repair(s):
    if len(s)!=10: return None
    h=s[:5].translate(str.maketrans('01258','OIZSB')); d=s[5:9].translate(str.maketrans('OIZSB','01258')); l=s[9].translate(str.maketrans('01258','OIZSB'))
    r=h+d+l
    return r if re.fullmatch(r'[A-Z]{5}[0-9]{4}[A-Z]',r) else None
rep=bad[bad.str.len()==10].map(repair)
print(f" of len-10 invalid ({(bad.str.len()==10).sum():,}), repairable by O/0-I/1-Z/2-S/5-B/8 substitution: {rep.notna().sum():,}")
print(f"   examples: {[(a,b) for a,b in zip(bad[bad.str.len()==10].head(5), rep.head(5))]}")
print(f" UNREPAIRABLE (len!=10): {(bad.str.len()!=10).sum():,}  e.g. {bad[bad.str.len()!=10].head(5).tolist()}")
