import pathlib
import pandas as pd, json, re, numpy as np
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[])
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str)
kyc=pd.read_csv(f"{D}/track1_kyc_records.csv",dtype=str,keep_default_na=False,na_values=[])
mer=pd.read_csv(f"{D}/track1_merchants_master.csv",dtype=str,keep_default_na=False,na_values=[])

FIELDS=[(tx,'timestamp','TX.timestamp'),(kyc,'signup_timestamp','KYC.signup'),(kyc,'date_of_birth','KYC.dob'),
        (mer,'onboarding_date','MER.onboarding'),(cb,'transaction_timestamp','CB.txn_ts'),
        (cb,'reported_timestamp','CB.reported_ts'),(cb,'bank_response_timestamp','CB.bank_ts')]

print("SEPARATOR-SPLIT DAY/MONTH DISAMBIGUATION")
print("="*94)
print(f"{'field':<18}{'sep':<7}{'n':>7}{'p1>12 (DD1st)':>16}{'p2>12 (MM1st)':>16}{'ambig':>8}  verdict")
print("-"*94)
for df,col,lab in FIELDS:
    s=df[col].astype(str).str.strip()
    for sep,pat in [('slash',r'^(\d{1,2})/(\d{1,2})/(\d{4})'), ('hyphen',r'^(\d{1,2})-(\d{1,2})-(\d{4})')]:
        m=s.str.extract(pat).dropna()
        if not len(m): continue
        p1=m[0].astype(int); p2=m[1].astype(int)
        a=(p1>12).sum(); b=(p2>12).sum(); amb=((p1<=12)&(p2<=12)).sum()
        verdict = "DD/MM/YYYY" if a>0 and b==0 else ("MM/DD/YYYY" if b>0 and a==0 else "!! MIXED !!")
        print(f"{lab:<18}{sep:<7}{len(m):>7,}{a:>16,}{b:>16,}{amb:>8,}  {verdict}")

print("\nISO-LIKE (YYYY-..-.. / YYYY/../..) — year first, unambiguous")
print("-"*94)
for df,col,lab in FIELDS:
    s=df[col].astype(str).str.strip()
    n=s.str.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}').sum()
    m=s.str.extract(r'^\d{4}[-/](\d{1,2})[-/](\d{1,2})').dropna()
    if len(m):
        p1=m[0].astype(int);p2=m[1].astype(int)
        print(f"{lab:<18} n={n:>7,}  pos2>12: {(p1>12).sum():>5,}  pos3>12: {(p2>12).sum():>5,}  -> "
              f"{'YYYY-MM-DD' if (p1>12).sum()==0 else 'YYYY-DD-MM?'}")

print("\n\nCONTROLLED PARSER — COVERAGE & RANGE")
print("="*94)
MON=r'(?i)^\d{1,2}-[A-Za-z]{3}-\d{4}$'
def parse(v):
    s=str(v).strip()
    if s=='' or s.lower() in ('nan','none','na','n/a','null'): return (None,'BLANK')
    if re.fullmatch(r'-?\d{8,10}', s):
        try: return (pd.to_datetime(int(s),unit='s',errors='raise'),'UNIX')
        except: return (None,'UNIX_BAD')
    if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}', s):
        return (pd.to_datetime(s.replace('/','-'),errors='coerce'),'ISO')
    if re.match(MON,s):
        return (pd.to_datetime(s,format='%d-%b-%Y',errors='coerce'),'DMON_Y')
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', s):
        return (pd.to_datetime(s,dayfirst=True,errors='coerce'),'SLASH_DMY')
    if re.match(r'^\d{1,2}-\d{1,2}-\d{4}', s):
        return (pd.to_datetime(s,dayfirst=False,errors='coerce'),'HYPH_MDY')
    return (pd.to_datetime(s,errors='coerce'),'FALLBACK')

for df,col,lab in FIELDS:
    r=df[col].astype(str).map(parse)
    dt=pd.Series([x[0] for x in r]); meth=pd.Series([x[1] for x in r])
    ok=dt.notna()
    print(f"\n{lab}  n={len(df):,}")
    print(f"   parsed={ok.sum():,} ({100*ok.mean():.2f}%)   methods: "+", ".join(f"{k}={v:,}" for k,v in meth.value_counts().items()))
    if ok.sum():
        d=pd.to_datetime(dt[ok])
        print(f"   range: {d.min()}  ->  {d.max()}")
        fut=(d>pd.Timestamp('2026-09-14')).sum(); old=(d<pd.Timestamp('1900-01-01')).sum()
        print(f"   after today(2026-09-14): {fut:,}   before 1900: {old:,}")
