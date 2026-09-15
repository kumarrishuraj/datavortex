import pathlib
import pandas as pd, json, re
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx  = pd.read_csv(f"{D}/track1_upi_transactions.csv", dtype=str, keep_default_na=False, na_values=[])
kyc = pd.read_csv(f"{D}/track1_kyc_records.csv", dtype=str, keep_default_na=False, na_values=[])
mer = pd.read_csv(f"{D}/track1_merchants_master.csv", dtype=str, keep_default_na=False, na_values=[])
cb  = pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json", encoding='utf-8'))).astype(str)

def shape(v):
    s = str(v)
    s = re.sub(r'[A-Za-z]', 'A', s); s = re.sub(r'[0-9]', '9', s)
    s = re.sub(r'A+', 'A+', s);      s = re.sub(r'9+', '9+', s)
    return s

def idshapes(series, label, top=25):
    s = series.astype(str)
    sh = s.map(shape).value_counts()
    print(f"\n--- {label}  (n={len(s):,}, distinct raw={s.nunique():,}) ---")
    for pat, c in sh.head(top).items():
        ex = s[s.map(shape)==pat].iloc[0]
        print(f"   {pat:<24s} {c:>7,}   e.g. {repr(ex)}")

print("#"*92); print("ID FORMAT SHAPES (A+=letters run, 9+=digits run)"); print("#"*92)
idshapes(tx["txn_id"],      "TX.txn_id")
idshapes(tx["user_id"],     "TX.user_id")
idshapes(tx["merchant_id"], "TX.merchant_id")
idshapes(tx["utr"],         "TX.utr")
idshapes(kyc["user_id"],    "KYC.user_id")
idshapes(mer["merchant_id"],"MER.merchant_id")
idshapes(cb["complaint_id"],"CB.complaint_id")
idshapes(cb["txn_id"],      "CB.txn_id")
idshapes(cb["user_id"],     "CB.user_id")
idshapes(cb["merchant_id"], "CB.merchant_id")

print("\n"+"#"*92); print("NORMALIZATION IMPACT TEST"); print("#"*92)
def norm_id(v, prefix, width):
    s = str(v).strip().upper()
    s = re.sub(r'[\s\-_\.]', '', s)
    m = re.match(rf'^(?:{prefix})?0*(\d+)$', s)
    if not m: return None
    return f"{prefix}{int(m.group(1)):0{width}d}"

for df, col, pre, w, lab in [(tx,"user_id","USR",5,"TX.user_id"), (kyc,"user_id","USR",5,"KYC.user_id"),
                             (cb,"user_id","USR",5,"CB.user_id"),
                             (tx,"merchant_id","MCH",4,"TX.merchant_id"), (mer,"merchant_id","MCH",4,"MER.merchant_id"),
                             (cb,"merchant_id","MCH",4,"CB.merchant_id"),
                             (tx,"txn_id","TXN",8,"TX.txn_id"), (cb,"txn_id","TXN",8,"CB.txn_id"),
                             (cb,"complaint_id","CBK",7,"CB.complaint_id")]:
    raw = df[col].astype(str)
    nrm = raw.map(lambda v: norm_id(v, pre, w))
    fails = nrm.isna().sum()
    print(f"{lab:<20s} raw_distinct={raw.nunique():>7,}  norm_distinct={nrm.nunique():>7,}  unparseable={fails:>5,}"
          f"  collapse={raw.nunique()-nrm.nunique():>6,}")
    if fails:
        print(f"      unparseable examples: {raw[nrm.isna()].drop_duplicates().head(6).tolist()}")
