import pathlib
import pandas as pd, json
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx  = pd.read_csv(f"{D}/track1_upi_transactions.csv", dtype=str, keep_default_na=False, na_values=[])
kyc = pd.read_csv(f"{D}/track1_kyc_records.csv", dtype=str, keep_default_na=False, na_values=[])
mer = pd.read_csv(f"{D}/track1_merchants_master.csv", dtype=str, keep_default_na=False, na_values=[])
cb  = pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json", encoding='utf-8'))).astype(str)

def dom(df, col, label, top=100):
    vc = df[col].value_counts(dropna=False)
    print(f"\n--- {label}.{col}  ({vc.shape[0]} distinct) ---")
    for v, c in vc.head(top).items():
        print(f"   {repr(v):<42s} {c:>7,}")

print("#"*90); print("TRANSACTION CATEGORICAL DOMAINS"); print("#"*90)
dom(tx,"status","TX"); dom(tx,"mcc","TX")

print("\n"+"#"*90); print("KYC CATEGORICAL DOMAINS"); print("#"*90)
dom(kyc,"kyc_status","KYC"); dom(kyc,"risk_segment","KYC"); dom(kyc,"occupation","KYC"); dom(kyc,"state","KYC"); dom(kyc,"city","KYC",50)

print("\n"+"#"*90); print("MERCHANT CATEGORICAL DOMAINS"); print("#"*90)
dom(mer,"merchant_status","MER"); dom(mer,"business_type","MER"); dom(mer,"mcc","MER"); dom(mer,"merchant_category","MER",90); dom(mer,"state","MER"); dom(mer,"city","MER",50)

print("\n"+"#"*90); print("CHARGEBACK CATEGORICAL DOMAINS"); print("#"*90)
dom(cb,"reason_code","CB"); dom(cb,"resolution_status","CB"); dom(cb,"severity","CB"); dom(cb,"channel","CB")
print(f"\n--- CB.complaint_text ({cb['complaint_text'].nunique()} distinct) ---")
for v,c in cb["complaint_text"].value_counts().items():
    print(f"   {c:>5,}  {v[:110]}")
