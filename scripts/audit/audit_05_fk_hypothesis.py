import pathlib
import pandas as pd, json, re, numpy as np
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx  = pd.read_csv(f"{D}/track1_upi_transactions.csv", dtype=str, keep_default_na=False, na_values=[])
kyc = pd.read_csv(f"{D}/track1_kyc_records.csv", dtype=str, keep_default_na=False, na_values=[])
mer = pd.read_csv(f"{D}/track1_merchants_master.csv", dtype=str, keep_default_na=False, na_values=[])
cb  = pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json", encoding='utf-8'))).astype(str)
def num(v):
    s=re.sub(r'[^0-9]','',str(v)); return int(s) if s else None
txu=set(tx.user_id.map(num).dropna()); kyu=set(kyc.user_id.map(num).dropna())
txm=set(tx.merchant_id.map(num).dropna()); mem=set(mer.merchant_id.map(num).dropna())
cbt=set(cb.txn_id.map(num).dropna()); txt=set(tx.txn_id.map(num).dropna())

print("H1: IS THE OVERLAP CONSISTENT WITH INDEPENDENT RANDOM DRAWS?")
print("-"*80)
def test(A,B,lo,hi,lab):
    space=hi-lo+1
    exp=len(A)*len(B)/space
    obs=len(A&B)
    sd=np.sqrt(exp*(1-len(B)/space))
    print(f" {lab}")
    print(f"   id space {lo}..{hi} = {space:,} slots | |A|={len(A):,} |B|={len(B):,}")
    print(f"   expected overlap if independent = {exp:,.0f} (sd {sd:.0f}) | OBSERVED = {obs:,} | z = {(obs-exp)/sd:+.2f}")
test(txu,kyu,10000,99999,"tx.user vs kyc.user")
test(txm,mem,1000,9999,"tx.merchant vs mer.merchant")
test(cbt,txt,1,99999999,"cb.txn vs tx.txn  (space = observed TXN range)")
print(f"   cb.txn within tx.txn range? tx range={min(txt)}..{max(txt)}  cb range={min(cbt)}..{max(cbt)}")

print("\nH2: DO CHARGEBACK FKs POINT AT THE SAME UNIVERSE AS THEIR OWN TXN?")
print("-"*80)
cbn=cb.copy(); cbn["t"]=cbn.txn_id.map(num); cbn["u"]=cbn.user_id.map(num); cbn["m"]=cbn.merchant_id.map(num)
txn=tx.copy(); txn["t"]=txn.txn_id.map(num); txn["u"]=txn.user_id.map(num); txn["m"]=txn.merchant_id.map(num)
txd=txn.drop_duplicates("t").set_index("t")[["u","m"]]
j=cbn.join(txd, on="t", rsuffix="_tx")
ok=j.dropna(subset=["u_tx"])
print(f" chargebacks joined to a txn: {len(ok):,}")
print(f"   user_id agrees with txn's user_id    : {(ok.u==ok.u_tx).sum():,} / {len(ok):,}  ({100*(ok.u==ok.u_tx).mean():.2f}%)")
print(f"   merchant_id agrees with txn's mch_id : {(ok.m==ok.m_tx).sum():,} / {len(ok):,}  ({100*(ok.m==ok.m_tx).mean():.2f}%)")
dis=ok[ok.u!=ok.u_tx].head(5)[["complaint_id","txn_id","user_id","u","u_tx","merchant_id","m","m_tx"]]
print("   disagreement examples:"); print(dis.to_string(index=False))

print("\nH3: TXN_ID WIDTH ANOMALY IN CHARGEBACKS")
print("-"*80)
cb["tl"]=cb.txn_id.map(lambda s: len(re.sub(r'[^0-9]','',str(s))))
print(cb.tl.value_counts().to_string())
print(" tx txn_id digit lengths:"); print(tx.txn_id.map(lambda s: len(re.sub(r'[^0-9]','',str(s)))).value_counts().to_string())
short=cb[cb.tl==5]
print(f"\n 5-digit cb txn_ids: {len(short):,}  examples {short.txn_id.head(6).tolist()}")
print(f"   do their numeric values fall in tx range 1..{max(txt)}? "
      f"{sum(1 for v in short.txn_id.map(num) if v in txt):,} of {len(short):,} hit an existing txn")

print("\nH4: MERCHANT MASTER COVERAGE OF TRANSACTION VOLUME")
print("-"*80)
covered=tx.merchant_id.map(num).isin(mem)
print(f" transactions on a merchant present in master : {covered.sum():,} ({100*covered.mean():.2f}%)")
print(f" distinct merchants in tx     : {len(txm):,} / 9000 possible ({100*len(txm)/9000:.1f}% of id space)")
print(f" distinct merchants in master : {len(mem):,} / 9000 possible ({100*len(mem)/9000:.1f}% of id space)")
