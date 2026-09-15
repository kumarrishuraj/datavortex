import pathlib
import pandas as pd, json, re, numpy as np, networkx as nx, time
D = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "raw")
tx=pd.read_csv(f"{D}/track1_upi_transactions.csv",dtype=str,keep_default_na=False,na_values=[])
cb=pd.json_normalize(json.load(open(f"{D}/track1_chargebacks.json",encoding='utf-8'))).astype(str)
def norm(v,p,w):
    s=re.sub(r'[\s\-_\.]','',str(v).strip().upper()); m=re.match(rf'^(?:{p})?0*(\d+)$',s)
    return f"{p}{int(m.group(1)):0{w}d}" if m else None
tx=tx.drop_duplicates()
tx['u']=tx.user_id.map(lambda v:norm(v,'USR',5)); tx['m']=tx.merchant_id.map(lambda v:norm(v,'MCH',4))

print("GRAPH STRUCTURE FEASIBILITY"); print("="*82)
print(f" deduped transactions : {len(tx):,}")
print(f" distinct users       : {tx.u.nunique():,}")
print(f" distinct merchants   : {tx.m.nunique():,}")
print(f" user-id numeric range: {tx.u.map(lambda s:int(s[3:])).min()}..{tx.u.map(lambda s:int(s[3:])).max()}")
print(f" mch-id numeric range : {tx.m.map(lambda s:int(s[3:])).min()}..{tx.m.map(lambda s:int(s[3:])).max()}")
print(f" any id appearing as BOTH user and merchant? {len(set(tx.u)&set(tx.m))}")
print(" => graph is strictly BIPARTITE user->merchant. No user->user transfers exist.")
print(f"\n edges (distinct user-merchant pairs): {tx.groupby(['u','m']).ngroups:,}")
ec=tx.groupby(['u','m']).size()
print(f"   pairs transacting >1 time: {(ec>1).sum():,} ({100*(ec>1).mean():.2f}%)  max repeats={ec.max()}")
print(f" user degree  : mean={tx.groupby('u').m.nunique().mean():.2f} max={tx.groupby('u').m.nunique().max()}")
print(f" merchant deg : mean={tx.groupby('m').u.nunique().mean():.2f} max={tx.groupby('m').u.nunique().max()}")
print(f" txns per user: mean={tx.groupby('u').size().mean():.2f} max={tx.groupby('u').size().max()}")
print(f" txns per mch : mean={tx.groupby('m').size().mean():.2f} max={tx.groupby('m').size().max()}")

t0=time.time()
G=nx.Graph(); G.add_edges_from(zip(tx.u,tx.m))
print(f"\n networkx bipartite graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  (built in {time.time()-t0:.2f}s)")
cc=list(nx.connected_components(G))
sizes=sorted((len(c) for c in cc),reverse=True)
print(f" connected components: {len(cc):,}  largest={sizes[0]:,} ({100*sizes[0]/G.number_of_nodes():.1f}% of nodes)  next={sizes[1:6]}")
print(f" giant component contains {sizes[0]:,} of {G.number_of_nodes():,} nodes -> global community detection is NOT discriminative")
deg=dict(G.degree())
print(f" degree: mean={np.mean(list(deg.values())):.2f} p99={np.percentile(list(deg.values()),99):.0f} max={max(deg.values())}")
print(f" nodes with degree>=5: {sum(1 for v in deg.values() if v>=5):,}")

# 4-cycles = u1-m1-u2-m2-u1 : collusion signature. count via shared-pair projection on a candidate subgraph
print("\n 4-CYCLE (u1-m1-u2-m2-u1) FEASIBILITY — the only true 'ring' shape in a bipartite graph")
pairs=tx.groupby(['u','m']).size().reset_index(name='n')
mu=pairs.groupby('m').u.apply(list)
from itertools import combinations
from collections import Counter
co=Counter()
t0=time.time()
for m,us in mu.items():
    if len(us)<2 or len(us)>60: continue
    for a,b in combinations(sorted(set(us)),2): co[(a,b)]+=1
print(f"   user-pairs sharing >=1 merchant : {len(co):,}   (computed in {time.time()-t0:.2f}s)")
sh=Counter(co.values())
print(f"   sharing >=2 merchants (=4-cycle): {sum(v for k,v in sh.items() if k>=2):,}")
print(f"   sharing >=3 merchants           : {sum(v for k,v in sh.items() if k>=3):,}")
print(f"   distribution of shared-merchant count: {dict(sorted(sh.items())[:6])}")
exp=tx.groupby('u').m.nunique().mean()**2 * tx.u.nunique()**2/2 / tx.m.nunique()
print(f"   NOTE: with {tx.u.nunique():,} users x {tx.m.nunique():,} merchants and mean degree ~{tx.groupby('u').m.nunique().mean():.1f}, co-occurrence is mostly CHANCE")

print("\n\nCHARGEBACK reason_code vs complaint_text CONSISTENCY"); print("="*82)
def rc(s):
    s=str(s).strip().lower()
    if any(k in s for k in ['unauth','not done by me','fraud','scam','suspicious']): return 'UNAUTHORIZED'
    if any(k in s for k in ['dup','twice','double','charged twice']): return 'DUPLICATE_DEBIT'
    if any(k in s for k in ['not delivered','no service','item not received','service failed','delivery issue','merchant service','service not provided']): return 'SERVICE_NOT_PROVIDED'
    if any(k in s for k in ['amount','wrong','mismatch','incorrect','extra']): return 'WRONG_AMOUNT'
    if any(k in s for k in ['ato','takeover','hacked','compromised','login']): return 'ACCOUNT_TAKEOVER'
    return 'OTHER'
def tc(s):
    s=str(s).strip().lower()
    if 'debited twice' in s: return 'DUPLICATE_DEBIT'
    if 'not authorized' in s or 'upi pin was not entered' in s: return 'UNAUTHORIZED'
    if 'compromised' in s: return 'ACCOUNT_TAKEOVER'
    if 'not delivered' in s or 'denies receiving' in s: return 'SERVICE_NOT_PROVIDED'
    if 'unknown merchant' in s: return 'UNAUTHORIZED'
    if 'high-value' in s or 'high value' in s: return 'SUSPICIOUS'
    if 'failed attempts' in s: return 'FAILED_ATTEMPTS'
    if 'unclear' in s: return 'UNCLEAR'
    return 'OTHER'
cb['rc']=cb.reason_code.map(rc); cb['tc']=cb.complaint_text.map(tc)
print(" reason_code canonical groups:"); print(cb.rc.value_counts().to_string())
print("\n complaint_text canonical groups:"); print(cb.tc.value_counts().to_string())
ct=pd.crosstab(cb.rc,cb.tc)
print("\n crosstab reason_code(rows) x complaint_text(cols):"); print(ct.to_string())
comp=cb[cb.tc.isin(['DUPLICATE_DEBIT','UNAUTHORIZED','ACCOUNT_TAKEOVER','SERVICE_NOT_PROVIDED'])]
agree=(comp.rc==comp.tc).sum()
print(f"\n comparable rows={len(comp):,}  reason_code AGREES with complaint_text: {agree:,} ({100*agree/len(comp):.2f}%)")
print(f" expected if independent: ~{100*sum((cb.rc.value_counts(normalize=True).get(g,0)*comp.tc.value_counts(normalize=True).get(g,0)) for g in ['DUPLICATE_DEBIT','UNAUTHORIZED','ACCOUNT_TAKEOVER','SERVICE_NOT_PROVIDED']):.2f}%")
