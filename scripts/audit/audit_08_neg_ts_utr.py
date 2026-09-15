import pathlib
import pandas as pd, json, re, numpy as np
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[])
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str)
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])
def amt(v):
    t=re.sub(r'[₹,]|rs\.?|inr|\s','',str(v),flags=re.I)
    try: return float(t)
    except: return np.nan
tx['a']=tx.amount.map(amt); tx['neg']=tx.a<0
def canon(s):
    s=str(s).strip().upper()
    if s in {'S','SUCCESS','TXN_SUCCESS','COMPLETED'}: return 'SUCCESS'
    if s in {'F','FAILED','TXN_FAILED','FAIL','DECLINED'}: return 'FAILED'
    if s in {'PENDING','PROCESSING','INITIATED'}: return 'PENDING'
    return 'OTHER:'+s
tx['st']=tx.status.map(canon)
print("NEGATIVE TRANSACTION AMOUNTS — REFUND OR CORRUPTION?")
print("-"*78)
print(f" negative rows: {tx.neg.sum():,} / {len(tx):,} ({100*tx.neg.mean():.2f}%)")
ct=pd.crosstab(tx.st, tx.neg, normalize='index').mul(100).round(2)
print("\n status vs negative-amount share (%):"); print(pd.concat([pd.crosstab(tx.st,tx.neg), ct.add_suffix('_%')],axis=1).to_string())
def n2(v): 
    s=re.sub(r'[^0-9]','',str(v)); return int(s) if s else None
tx['t']=tx.txn_id.map(n2); cb['t']=cb.txn_id.map(n2)
disputed=set(cb.t.dropna())
tx['disp']=tx.t.isin(disputed)
print(f"\n disputed share among negative-amount txns : {100*tx[tx.neg].disp.mean():.2f}%")
print(f" disputed share among positive-amount txns : {100*tx[~tx.neg].disp.mean():.2f}%")
print(f" |amount| distribution  neg: median={tx[tx.neg].a.abs().median():,.0f}  pos: median={tx[~tx.neg].a.median():,.0f}")
print(f" missing UTR among neg: {100*tx[tx.neg].utr.eq('').mean():.2f}%   among pos: {100*tx[~tx.neg].utr.eq('').mean():.2f}%")
# does a matching positive twin exist (reversal signature)?
pos=tx[~tx.neg]
key=set(zip(pos.user_id,pos.merchant_id,pos.a.round(2)))
twin=sum(1 for r in tx[tx.neg].itertuples() if (r.user_id,r.merchant_id,round(-r.a,2)) in key)
print(f" negative txns having an exact positive twin (same user+merchant+|amt|): {twin:,} / {tx.neg.sum():,}")

print("\n"+"="*78); print("TIMESTAMP FORENSICS"); print("="*78)
def tshape(v):
    s=str(v).strip()
    if s=='': return 'BLANK'
    if re.fullmatch(r'\d{9,10}', s): return 'UNIX_EPOCH'
    s2=re.sub(r'\d','9',s); s2=re.sub(r'9+','9',s2)
    s2=re.sub(r'[A-Za-z]+','MON',s2)
    return s2
for df,col,lab in [(tx,'timestamp','TX.timestamp'),(kyc,'signup_timestamp','KYC.signup'),(kyc,'date_of_birth','KYC.dob'),
                   (mer,'onboarding_date','MER.onboarding'),(cb,'transaction_timestamp','CB.txn_ts'),
                   (cb,'reported_timestamp','CB.reported_ts'),(cb,'bank_response_timestamp','CB.bank_ts')]:
    s=df[col].astype(str); vc=s.map(tshape).value_counts()
    print(f"\n--- {lab} ({s.nunique():,} distinct) ---")
    for p,c in vc.items():
        ex=s[s.map(tshape)==p].iloc[0]
        print(f"   {p:<24s} {c:>7,}  e.g. {repr(ex)}")

print("\n"+"="*78); print("AMBIGUOUS DAY/MONTH TEST (dd/mm vs mm/dd)"); print("="*78)
def slashparts(v):
    m=re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})', str(v).strip())
    return (int(m.group(1)),int(m.group(2))) if m else None
for df,col,lab in [(kyc,'signup_timestamp','KYC.signup'),(mer,'onboarding_date','MER.onboarding'),
                   (cb,'transaction_timestamp','CB.txn_ts'),(cb,'reported_timestamp','CB.reported_ts'),
                   (cb,'bank_response_timestamp','CB.bank_ts'),(kyc,'date_of_birth','KYC.dob')]:
    p=df[col].astype(str).map(slashparts).dropna()
    if not len(p): continue
    a=p.map(lambda x:x[0]); b=p.map(lambda x:x[1])
    a_gt12=(a>12).sum(); b_gt12=(b>12).sum(); both=((a<=12)&(b<=12)).sum()
    print(f" {lab:<18s} n={len(p):>6,} | first>12 (=> DD first): {a_gt12:>5,} | second>12 (=> MM first): {b_gt12:>5,} | ambiguous: {both:>5,}")

print("\n"+"="*78); print("UTR FORENSICS"); print("="*78)
u=tx.utr.astype(str)
print(f" blank        : {(u.str.strip()=='').sum():,} ({100*(u.str.strip()=='').mean():.2f}%)")
clean=u.str.replace(r'[\s\-_]','',regex=True).str.upper()
print(f" has space    : {u.str.contains(' ').sum():,}")
print(f" matches UTR+10digits after cleanup : {clean.str.fullmatch(r'UTR\d{10}').sum():,}")
print(f" other shapes : {(~clean.str.fullmatch(r'UTR\d{10}') & (u.str.strip()!='')).sum():,}")
print(f" distinct raw={u.nunique():,}  distinct cleaned(non-blank)={clean[clean!=''].nunique():,}")
vc=clean[clean!=''].value_counts()
print(f" UTRs used by >1 transaction: {(vc>1).sum():,} (max {vc.max()}x)")
dupu=set(vc[vc>1].index)
sub=tx[clean.isin(dupu)]
print(f"   rows involved: {len(sub):,};  of these, how many are exact-duplicate txn rows? {sub.duplicated(subset=tx.columns.difference(['a','neg','st','t','disp']).tolist(),keep=False).sum():,}")
