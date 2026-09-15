import pathlib
import pandas as pd, json, re, numpy as np
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[])
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str)
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])
MON=r'(?i)^\d{1,2}-[A-Za-z]{3}-\d{4}'
def parse(v):
    s=str(v).strip()
    if s=='' or s.lower() in ('nan','none','na','n/a','null'): return pd.NaT
    if re.fullmatch(r'-?\d{8,10}', s):
        try: return pd.to_datetime(int(s),unit='s')
        except: return pd.NaT
    if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}', s): return pd.to_datetime(s.replace('/','-'),errors='coerce')
    if re.match(MON,s): return pd.to_datetime(s,format='%d-%b-%Y',errors='coerce')
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', s): return pd.to_datetime(s,dayfirst=True,errors='coerce')
    if re.match(r'^\d{1,2}-\d{1,2}-\d{4}', s): return pd.to_datetime(s,dayfirst=False,errors='coerce')
    return pd.to_datetime(s,errors='coerce')
def n2(v):
    s=re.sub(r'[^0-9]','',str(v)); return int(s) if s else None

tx['ts']=tx.timestamp.map(parse); tx['t']=tx.txn_id.map(n2)
cb['t']=cb.txn_id.map(n2); cb['cb_txn_ts']=cb.transaction_timestamp.map(parse)
cb['rep']=cb.reported_timestamp.map(parse); cb['bank']=cb.bank_response_timestamp.map(parse)
txu=tx.drop_duplicates('t').set_index('t')
cb['tx_ts']=cb.t.map(txu.ts)

print("TEMPORAL INTEGRITY — CHARGEBACKS"); print("="*84)
m=cb.dropna(subset=['tx_ts','cb_txn_ts'])
d=(m.cb_txn_ts - m.tx_ts).dt.total_seconds().abs()
print(f" cb.transaction_timestamp vs joined txn.timestamp  (n={len(m):,})")
print(f"   same calendar DAY : {(d<86400).sum():,} ({100*(d<86400).mean():.2f}%)   exact: {(d==0).sum():,}")
print(f"   median abs diff   : {d.median()/86400:,.1f} days")
print("   -> cb.transaction_timestamp is a DATE-TRUNCATED copy of the txn timestamp" if (d<86400).mean()>.9 else "   -> INCONSISTENT")

for src,lab in [('tx_ts','vs TRUE txn.timestamp (via txn_id)'), ('cb_txn_ts','vs cb.transaction_timestamp')]:
    x=cb.dropna(subset=[src,'rep'])
    delay=(x.rep - x[src]).dt.total_seconds()/86400
    print(f"\n reporting_delay {lab}: n={len(x):,}")
    print(f"   median={delay.median():.2f}d  mean={delay.mean():.2f}d  min={delay.min():.2f}d  max={delay.max():.2f}d")
    print(f"   NEGATIVE (reported before txn): {(delay<0).sum():,} ({100*(delay<0).mean():.2f}%)")
    print(f"   >7 days: {(delay>7).sum():,}   >30 days: {(delay>30).sum():,}")
x=cb.dropna(subset=['rep','bank'])
bd=(x.bank-x.rep).dt.total_seconds()/86400
print(f"\n bank_response - reported: n={len(x):,} median={bd.median():.2f}d min={bd.min():.2f}d neg={(bd<0).sum():,}")

print("\n\nTEMPORAL INTEGRITY — KYC / MERCHANT vs TRANSACTIONS"); print("="*84)
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); mm=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(mm.group(1)):0{w}d}" if mm else None
kyc['u']=kyc.user_id.map(lambda v:norm(v,'USR',5)); kyc['signup']=kyc.signup_timestamp.map(parse); kyc['dob']=kyc.date_of_birth.map(parse)
mer['m']=mer.merchant_id.map(lambda v:norm(v,'MCH',4)); mer['onb']=mer.onboarding_date.map(parse)
tx['u']=tx.user_id.map(lambda v:norm(v,'USR',5)); tx['m']=tx.merchant_id.map(lambda v:norm(v,'MCH',4))
ks=kyc.dropna(subset=['signup']).groupby('u').signup.min()
ms=mer.dropna(subset=['onb']).groupby('m').onb.min()
tx['signup']=tx.u.map(ks); tx['onb']=tx.m.map(ms)
a=tx.dropna(subset=['signup'])
print(f" txns with a KYC signup date  : {len(a):,}")
print(f"   txn BEFORE user signup     : {(a.ts<a.signup).sum():,} ({100*(a.ts<a.signup).mean():.2f}%)  <-- impossible")
b=tx.dropna(subset=['onb'])
print(f" txns with merchant onboarding: {len(b):,}")
print(f"   txn BEFORE onboarding      : {(b.ts<b.onb).sum():,} ({100*(b.ts<b.onb).mean():.2f}%)  <-- impossible")
age=((pd.Timestamp('2026-01-01')-kyc.dob).dt.days/365.25)
print(f"\n KYC age at 2026-01-01: n={age.notna().sum():,} min={age.min():.1f} max={age.max():.1f}  under18={(age<18).sum():,}  over100={(age>100).sum():,}")
print(f" DOB after signup date: {(kyc.dob>kyc.signup).sum():,}")

print("\n\nPAN / AADHAAR FORENSICS"); print("="*84)
def sh(v):
    s=str(v); s=re.sub(r'[A-Za-z]','A',s); s=re.sub(r'[0-9]','9',s); s=re.sub(r'A+','A+',s); s=re.sub(r'9+','9+',s); return s
for col in ['pan','aadhaar']:
    s=kyc[col].astype(str); vc=s.map(sh).value_counts()
    print(f"\n--- KYC.{col} ({s.nunique():,} distinct) ---")
    for p,c in vc.head(12).items():
        print(f"   {p:<22s} {c:>7,}  e.g. {repr(s[s.map(sh)==p].iloc[0])}")
pan=kyc.pan.astype(str).str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
print(f"\n PAN valid ^[A-Z]{{5}}[0-9]{{4}}[A-Z]$ after cleanup: {pan.str.fullmatch(r'[A-Z]{5}[0-9]{4}[A-Z]').sum():,} / {len(kyc):,}")
aad=kyc.aadhaar.astype(str).str.replace(r'[^0-9X]','',regex=True).str.upper()
print(f" Aadhaar 12 digits after cleanup : {aad.str.fullmatch(r'\d{12}').sum():,}")
print(f" Aadhaar masked (contains X)     : {aad.str.contains('X').sum():,}")
print(f" Aadhaar other/blank             : {len(kyc)-aad.str.fullmatch(r'\d{12}').sum()-aad.str.contains('X').sum():,}")

print("\n\nMCC CONSISTENCY: transactions vs merchant master"); print("="*84)
def mcc(v):
    s=str(v).strip().upper()
    if s in ('','NA','UNKNOWN','MISC','N/A','NONE','NAN'): return None
    s=re.sub(r'^MCC[-_ ]?','',s); s=re.sub(r'\.0$','',s)
    return s.zfill(4) if s.isdigit() else None
tx['mccn']=tx.mcc.map(mcc); mer['mccn']=mer.mcc.map(mcc)
print(f" TX mcc canonical values : {sorted(tx.mccn.dropna().unique())}")
print(f" MER mcc canonical values: {sorted(mer.mccn.dropna().unique())}")
print(f" TX mcc null after canon : {tx.mccn.isna().sum():,} ({100*tx.mccn.isna().mean():.2f}%)")
print(f" MER mcc null after canon: {mer.mccn.isna().sum():,} ({100*mer.mccn.isna().mean():.2f}%)")
mm=mer.dropna(subset=['mccn']).drop_duplicates('m').set_index('m').mccn
tx['mer_mcc']=tx.m.map(mm)
cmp=tx.dropna(subset=['mccn','mer_mcc'])
print(f"\n txns where BOTH tx.mcc and master mcc known: {len(cmp):,}")
print(f"   AGREE: {(cmp.mccn==cmp.mer_mcc).sum():,} ({100*(cmp.mccn==cmp.mer_mcc).mean():.2f}%)")
print(f"   DISAGREE: {(cmp.mccn!=cmp.mer_mcc).sum():,}")
