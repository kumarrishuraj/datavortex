import pathlib
import pandas as pd, numpy as np, json, os, re, sys
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")

def load():
    tx  = pd.read_csv(f"{D}/track1_upi_transactions.csv", dtype=str, keep_default_na=False, na_values=[])
    kyc = pd.read_csv(f"{D}/track1_kyc_records.csv", dtype=str, keep_default_na=False, na_values=[])
    mer = pd.read_csv(f"{D}/track1_merchants_master.csv", dtype=str, keep_default_na=False, na_values=[])
    cb  = pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json", encoding='utf-8')))
    cb  = cb.astype(str)
    return tx, kyc, mer, cb

tx, kyc, mer, cb = load()

print("="*100)
print("A. FILE INVENTORY")
print("="*100)
for name, df, f in [("track1_upi_transactions.csv", tx, "track1_upi_transactions.csv"),
                    ("track1_kyc_records.csv", kyc, "track1_kyc_records.csv"),
                    ("track1_merchants_master.csv", mer, "track1_merchants_master.csv"),
                    ("track1_chargebacks.json", cb, "track1_chargebacks.json")]:
    sz = os.path.getsize(f"{D}/{f}")/1024/1024
    print(f"{name:36s} rows={len(df):>7,}  cols={df.shape[1]:>3}  size={sz:>6.2f} MB")
    print(f"   columns: {list(df.columns)}")

def profile(df, name, examples=4):
    print("\n" + "="*100)
    print(f"B. SCHEMA PROFILE — {name}  ({len(df):,} rows)")
    print("="*100)
    n = len(df)
    rows=[]
    for c in df.columns:
        s = df[c].astype(str)
        blank = s.str.strip().isin(["", "nan", "None", "NaN", "null", "NULL", "NA", "N/A", "na", "-"]).sum()
        uniq = s.nunique()
        ex = [repr(v)[:34] for v in s[~s.str.strip().isin(["","nan","None"])].drop_duplicates().head(examples).tolist()]
        rows.append((c, n-blank, blank, 100*blank/n, uniq, " | ".join(ex)))
    out = pd.DataFrame(rows, columns=["column","non_blank","blank_like","blank_%","n_unique","examples"])
    out["blank_%"] = out["blank_%"].round(2)
    print(out.to_string(index=False))

for df, nm in [(tx,"TRANSACTIONS"), (kyc,"KYC"), (mer,"MERCHANTS"), (cb,"CHARGEBACKS")]:
    profile(df, nm)
