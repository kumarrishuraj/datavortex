# Track 1 — Forensic Dataset Audit
**TransOrg AgentIQ Datathon 2026 · UPI Fraud Ring & Merchant Analytics**

Stage 1 deliverable. Every number below is computed from the actual bundle, not assumed.

> **Corrections after Stage 1 (final pass, 2026-09-15).** This document is kept as the original
> audit record. The pipeline now recomputes the figures below on cleaned data and stores them in
> `data/processed/audit_facts.parquet`; the dashboard quotes those values, not this document.
>
> | Stage 1 statement | Pipeline value | Why they differ |
> |---|---|---|
> | Masked Aadhaar last-4 shared: 319 vs ~388 expected, "below chance" | 319 vs **326.5** expected — at chance | The Stage 1 expectation used an incorrect birthday-collision formula. Corrected in `audit_facts.expected_shared_keys` and unit-tested |
> | Masked settlement last-4 shared: 71 vs ~79 expected, "below chance" | 71 vs **69.9** expected — at chance | Same formula correction |
> | 4,864 user_ids carry more than one distinct PAN | **4,422** carry more than one distinct *valid* PAN | Stage 1 counted raw PAN strings, so formatting variants of one PAN counted as different PANs (reproduced: raw strings give 4,864, cleaned PANs give 4,422) |
> | Delay against the linked transaction: 45.29% negative | **45.11%** | Recomputed on cleaned, de-duplicated complaints |
> | Delay against the file's own timestamp: 3.82% negative | **3.85%** of complaints with a computable delay | Recomputed on cleaned data |
> | Transaction MCC agrees with master 9.72% vs 9.98% expected (7,778 rows) | **10.12%** vs **9.76%** expected | Pipeline measure on cleaned transactions; the conclusion (independent draws) is unchanged |
> | Reason code vs complaint text agree 22.32% vs 21.41% expected | **16.83%** vs **16.51%** expected, z = +0.38, n = 1,973 (hypothesis register) | The register uses a stricter comparability rule; the conclusion (no information) is unchanged |
>
> Full PANs quoted in the Stage 1 draft have been masked in this document.
Audit scripts: `scripts/audit/` · Environment: Python 3.13.14, pandas 3.0.1 (duckdb NOT installed)

---

## A. File inventory

| File | Type | Rows | Cols | Size | Purpose |
|---|---|---|---|---|---|
| track1_upi_transactions.csv | CSV | 20,400 | 8 | 1.65 MB | Core UPI transaction log (the fact table) |
| track1_kyc_records.csv | CSV | 36,400 | 12 | 4.15 MB | Customer KYC master |
| track1_merchants_master.csv | CSV | 6,210 | 11 | 0.64 MB | Merchant master |
| track1_chargebacks.json | JSON array | 2,884 | 13 | 1.35 MB | Disputes / chargebacks |
| track1_dataset_notes.txt | TXT | 111 lines | — | 5 KB | Spec: joins, metrics, example questions |

The competition PDF was **not** present in the bundle — only the five files above.

---

## B. Schema profile

### track1_upi_transactions.csv (20,400 rows)

| Column | Blank | Blank % | Unique | Examples | Business meaning |
|---|---|---|---|---|---|
| txn_id | 0 | 0.00 | 20,000 | TXN00011869 | PK — 400 exact-duplicate rows |
| timestamp | 0 | 0.00 | 19,083 | 2026-01-15 00:11:30, 1770063471 | 5 formats incl. Unix epoch |
| user_id | 0 | 0.00 | 17,878 | USR45826 | FK to KYC (uniformly clean format) |
| merchant_id | 0 | 0.00 | 8,051 | MCH7045 | FK to merchants (uniformly clean) |
| amount | 0 | 0.00 | 19,900 | 15722.34, Rs. 6362.9, ₹16,466.93, -23820.57 | Txn value INR |
| utr | 1,024 | 5.02 | 19,001 | UTR6498104698, "UTR 2787678319" | Bank reference number |
| mcc | 2,926 | 14.34 | 7 | 5411, 05411 | **Unreliable — see D5** |
| status | 0 | 0.00 | 14 | COMPLETED, TXN_FAILED, S | Transaction outcome |

### track1_kyc_records.csv (36,400 rows)

| Column | Blank | Blank % | Unique | Notes |
|---|---|---|---|---|
| user_id | 0 | 0.00 | 32,165 | 5 format variants → 28,920 after normalization |
| full_name | 0 | 0.00 | 33,921 | Mixed case |
| pan | 1,896 | 5.21 | 33,165 | 92.33% valid; homoglyph + truncation damage |
| aadhaar | 2,664 | 7.32 | 32,094 | 27,553 full / 2,922 masked (XXXX-XXXX-nnnn) |
| date_of_birth | 2,944 | 8.09 | 29,285 | 7 formats incl. **negative Unix epochs** (pre-1970) |
| city | 0 | 0.00 | 41 | 12 real cities × alias/case variants |
| state | 0 | 0.00 | 9 | Clean |
| monthly_income | 2,933 | 8.06 | 26,035 | ₹/Rs./INR/commas/`27.3k`/`Not Available`/negatives |
| occupation | 0 | 0.00 | 9 | Clean |
| signup_timestamp | 2,910 | 7.99 | 14,608 | 6 formats |
| kyc_status | 0 | 0.00 | 16 | → 3 canonical |
| risk_segment | 0 | 0.00 | 12 | → 4 canonical (LOW/MEDIUM/HIGH/UNKNOWN) |

### track1_merchants_master.csv (6,210 rows)

| Column | Blank | Blank % | Unique | Notes |
|---|---|---|---|---|
| merchant_id | 0 | 0.00 | 5,083 | 4 variants → 4,343 normalized |
| merchant_name | 0 | 0.00 | 5,732 | Untrimmed whitespace, `Gh0sh` digit-in-name |
| mcc | 514 | 8.28 | 44 | zero-pad / `.0` float / `MCC-` prefix / NA / UNKNOWN / misc |
| merchant_category | 0 | 0.00 | 82 | → 11 canonical |
| business_type | 0 | 0.00 | 14 | → 4 canonical |
| city / state | 0 | 0.00 | 41 / 9 | Same alias problem as KYC |
| onboarding_date | 499 | 8.04 | 4,400 | 6 formats |
| settlement_account | 2,451 | 39.47 | 3,550 | 2,476 full / 1,283 masked |
| merchant_status | 0 | 0.00 | 15 | → 4 canonical |
| declared_avg_ticket_size | 371 | 5.97 | 5,521 | Currency noise + 501 negatives |

### track1_chargebacks.json (2,884 rows)

| Column | Blank | Blank % | Unique | Notes |
|---|---|---|---|---|
| complaint_id | 0 | 0.00 | 2,800 | PK — 84 exact dupes |
| txn_id | 81 | 2.81 | 2,582 | **The only trustworthy FK** |
| user_id | 0 | 0.00 | 2,454 | **Noise — see D4** |
| merchant_id | 0 | 0.00 | 2,051 | **Noise — see D4** |
| transaction_timestamp | 235 | 8.15 | 1,238 | Does not match linked txn |
| reported_timestamp | 209 | 7.25 | 1,329 | Coherent with the line above |
| disputed_amount | 183 | 6.35 | 2,609 | Independent of txn amount |
| reason_code | 0 | 0.00 | 34 | → 6 canonical |
| complaint_text | 0 | 0.00 | 84 | 10 templates × case × noise suffix |
| resolution_status | 0 | 0.00 | 13 | → 6 canonical |
| bank_response_timestamp | 718 | 24.90 | 1,149 | Highest missingness in bundle |
| severity | 0 | 0.00 | 16 | L/M/H/CRIT + P1–P4 → 4 canonical |
| channel | 0 | 0.00 | 8 | → 5 canonical |

---

## C. Data quality assessment

### Duplicates (exact, byte-identical rows)

| File | Raw | Exact dupes removed | After |
|---|---|---|---|
| transactions | 20,400 | 400 | 20,000 |
| kyc | 36,400 | 278 | 36,122 |
| merchants | 6,210 | 12 | 6,198 |
| chargebacks | 2,884 | 84 | 2,800 |

All 400 duplicated `txn_id` values are **byte-identical across all 8 columns** (verified field by field) — double-ingestion, not a business event. Same for all 84 duplicated `complaint_id` values. These are safe to collapse; nothing else is.

### Canonical domains (derived from observed values only)

- `status` 14 → 3: SUCCESS 17,395 (85.27%) | FAILED 1,990 (9.76%) | PENDING 1,015 (4.98%)
  - SUCCESS ← S, Success, TXN_SUCCESS, COMPLETED, SUCCESS
  - FAILED ← FAILED, TXN_FAILED, Fail, Declined, F
  - PENDING ← PENDING, Pending, PROCESSING, Initiated
- `kyc_status` 16 → 3: VERIFIED (Verified/VERIFIED/APPROVED/KYC_DONE/V/Done) | PENDING (PENDING/Pending/P/IN_PROGRESS/Under Review) | REJECTED (Rejected/R/REJECTED/Reject/FAILED)
- `risk_segment` 12 → 4 (pure case variants): LOW / MEDIUM / HIGH / UNKNOWN
- `merchant_status` 15 → 4: ACTIVE (Active/ACTIVE/A/Enabled/Live) | INACTIVE (Inactive/INACTIVE/I/Disabled/Closed) | SUSPENDED (Suspended/SUSPENDED/S/Hold) | BLOCKED
- `business_type` 14 → 4: INDIVIDUAL / PARTNERSHIP / PRIVATE_LIMITED / SOLE_PROPRIETOR
- `merchant_category` 82 → 11: Grocery, Restaurant, Hotel, Telecom, Transport, Apparel, Pharmacy, Books/Stationery, Department Store, Retail Other, Misc
- `severity` 16 → 4: LOW / MEDIUM / HIGH / CRITICAL (L/M/H/CRIT plus P4/P3/P2/P1)
- `resolution_status` 13 → 6: OPEN / IN_PROGRESS (incl. WIP) / PENDING_BANK / RESOLVED / CLOSED / REJECTED
- `channel` 8 → 5: IVR / CHATBOT / APP / EMAIL / BRANCH / CALL_CENTER
- `city` 41 → 12 real cities. Aliases proven: Bombay/Mumbay/MUMBAI→Mumbai, Poona→Pune, Calcutta→Kolkata, Madras→Chennai, Dilli/New Delhi→Delhi, Jalandar→Jalandhar, LDH→Ludhiana, ASR→Amritsar, JPR→Jaipur, LKO→Lucknow, BLR/Bangalore→Bengaluru, Hyd→Hyderabad. Every alias is confirmed by a consistent `state` value.

### Amounts

All four monetary fields parse to 100% of non-blank values with one regex (strip `₹ | Rs. | INR | , | whitespace`):

| Field | Parsed | Blank | Invalid | Negatives | Range |
|---|---|---|---|---|---|
| tx.amount | 20,400 | 0 | 0 | 429 (2.10%) | −24,847.86 … 24,998.12 |
| cb.disputed_amount | 2,701 | 183 | 0 | 231 (8.55%) | −19,036.69 … 45,384.61 |
| mer.declared_avg_ticket_size | 5,839 | 371 | 0 | 501 (8.58%) | −12,917.19 … 23,536.65 |
| kyc.monthly_income | 31,677 | 2,933 | 1,790 (`Not Available`) | 1,456 | −300,000 … 235,221 |

`kyc.monthly_income` additionally has 2,834 values in `27.3k` shorthand (multiply by 1000).

**Negative transaction amounts are sign corruption, not refunds.** Evidence:

1. No refund/reversal/credit value exists anywhere in the `status` domain.
2. **0 of 429** negative rows have a matching positive twin (same user + merchant + |amount|).
3. Negative rate is flat across statuses — SUCCESS 2.07%, FAILED 2.46%, PENDING 1.97% — the signature of uniform random corruption, not a business process.
4. |amount| distribution is identical to positives (median 12,136 vs 12,505).
5. Dispute rate among negatives is *lower*, not higher (10.02% vs 12.31%).

Repair with `abs()`, retain an `amount_sign_invalid` flag. Negative income and negative declared-ticket-size have no valid interpretation at all — flag invalid, do not repair.

### Timestamps

7 date fields, 5–7 formats each. A separator-based rule disambiguates day/month with **zero counterexamples across all 7 fields**:

| Rule | Evidence (part > 12 counts) |
|---|---|
| `d/m/Y` → **DD/MM/YYYY** | 18,385 rows prove first > 12; **0** rows prove second > 12 |
| `d-m-Y` → **MM-DD-YYYY** | 9,133 rows prove second > 12; **0** rows prove first > 12 |
| `Y-m-d` / `Y/m/d` → ISO | 0 rows with month > 12 |
| `d-Mon-Y` → textual | unambiguous |
| 8–10 bare digits → Unix epoch | incl. **negative epochs** in `date_of_birth` (pre-1970) |

Parser coverage: tx.timestamp **100.00%**; all other fields 91.8–92.8% (remainder is genuinely blank); cb.bank_response 75.10% (24.9% blank).

Ranges: transactions **2026-01-01 → 2026-03-31 (exactly Q1 2026, 90 days)**; KYC signup 2024-01-01 → 2026-03-31; DOB 1960 → 2007 (age 18–66, **no impossible ages**); merchant onboarding 2023-01-01 → 2026-02-28; chargeback reported 2025-12-30 → 2026-05-16.

**Impossible timestamps found:** 301 transactions (4.86% of those joinable) occur *before* the user's KYC signup; 83 (0.90%) occur *before* the merchant's onboarding date.

### UTR

1,024 blank (5.02%); 1,881 contain an internal space. After removing whitespace, **100% of non-blank UTRs match `UTR\d{10}`** — zero malformed. 376 UTRs appear twice, and all 752 rows involved are the exact-duplicate transaction rows. **There is no UTR collision signal** once duplicates are removed.

### PAN

92.33% of non-blank values valid. Of the 2,646 invalid: **819 are length-10 and 100% repairable** by homoglyph substitution (0↔O, 1↔I, 2↔Z, 5↔S, 8↔B applied positionally); **1,827 are length-9** — a truncated trailing check letter, unrepairable. Post-repair validity 94.71%.

### Aadhaar

27,553 full 12-digit, 2,922 masked, 2,664 blank, remainder malformed.

---

## D. Relationship forensics

| Relationship | Raw match | Normalized match | Gain | Unmatched | Verdict |
|---|---|---|---|---|---|
| transactions → KYC (user_id) | 22.40% | **32.44%** | +10.03 pp | 13,783 | Random — no designed integrity |
| transactions → merchants (merchant_id) | 36.10% | **48.08%** | +11.98 pp | 10,591 | Random — no designed integrity |
| chargebacks → transactions (txn_id) | 89.36% | **93.03%** | +3.68 pp | 201 | **Real and designed** |
| chargebacks → KYC (user_id) | 15.33% | 31.76% | +16.44 pp | 1,968 | Column is noise — do not use |
| chargebacks → merchants (merchant_id) | 25.07% | 46.32% | +21.26 pp | 1,548 | Column is noise — do not use |

### D1. The residual mismatch is irreducible, not a cleaning failure

Overlap size was tested against the null hypothesis of independent random draws over the observed ID space:

| Pair | ID space | Expected overlap if independent | Observed | z |
|---|---|---|---|---|
| tx.user × kyc.user | 90,000 | 5,745 (sd 62) | 5,799 | **+0.87** |
| tx.merchant × mer.merchant | 9,000 | 3,885 (sd 45) | 3,893 | **+0.18** |
| cb.txn × tx.txn | — | ~1 | 2,451 | **+3420** |

The first two are statistically indistinguishable from chance. **No normalization can raise them** — the generator drew user and merchant IDs independently for each file. The third is overwhelming, confirming `txn_id` is the one deliberately constructed foreign key.

### D2. ID normalization rule (evidence-backed)

```
upper → strip [space, hyphen, underscore, dot] → strip leading zeros
     → re-pad to canonical width

USR + 5 digits | MCH + 4 digits | TXN + 8 digits | CBK + 7 digits
```

Observed variants: `USR12345`, `usr12345`, `USR-12345`, `USR 12345`, `usr_12345`, `12345` (bare numeric). Identical family for MCH. Unparseable: **0**, except 81 blank `cb.txn_id`.

Transaction-file IDs are already uniformly clean (100% uppercase, no separators) — **all messiness lives in KYC, merchants and chargebacks.**

Collapse achieved: KYC 32,165 → 28,920 keys; merchants 5,083 → 4,343; chargebacks user 2,454 → 2,294, merchant 2,051 → 1,855.

**120 chargeback `txn_id` values are 5-digit (`TXN65742`) against the 8-digit standard; none resolve to any transaction. Structurally invalid, unrecoverable — flag, retain, exclude from linked metrics.**

### D3. Repeated IDs are entity collisions, not duplicate records — the most consequential finding

| | KYC | Merchants |
|---|---|---|
| IDs appearing more than once | 6,288 (13,768 rows) | 1,412 (3,279 rows) |
| **Same entity** (identical name) | 947 | 102 |
| **Different entity — ID COLLISION** | **5,341 (85.0%)** | **1,310 (92.8%)** |

`USR10043` is simultaneously *rehaan buch* (PAN TW*******F, Ludhiana, Verified, Medium) and *inaya lanka* (PAN HG*******L, Mumbai, Pending, LOW) — different name, PAN, Aadhaar, DOB, city, state and income.

`MCH1007` is both *Sunder, Wason and Pau* (eating place, Jaipur, Enabled) and *Jain-Wadhwa* (hotel lodging, Chennai, Disabled).

Corroborating: **4,864 user_ids carry more than one distinct PAN**, while **0 PANs are shared across user_ids**.

A naive `drop_duplicates(subset='user_id')` silently destroys 5,341 real identities and fabricates a false golden record. These IDs are **ambiguous, not duplicated** — they must be flagged, and any attribute read through them marked low-confidence.

### D4. Chargeback denormalized columns are noise — attribute via txn_id only

Of the 2,683 chargebacks that link to a transaction:

- `cb.user_id` equals the linked transaction's user: **0 (0.000%)** — chance would give 0.0011%
- `cb.merchant_id` equals the linked transaction's merchant: **0 (0.000%)** — chance would give 0.011%
- `cb.transaction_timestamp` falls on the same day as the linked transaction: **55 (2.23%)**, median gap 26 days
- `cb.disputed_amount` equals the linked transaction amount: **0**; medians ₹1,823 vs ₹12,231

The dataset notes propose joining chargebacks to KYC and merchants on `user_id` / `merchant_id`. **Those columns do not describe the transaction they point at.** All merchant and user attribution must flow `chargeback → txn_id → transaction → user_id / merchant_id`.

**But the chargeback file is internally coherent in time.** Reporting delay measured against the *linked transaction* gives 45.29% negative values (impossible); measured against the file's own `transaction_timestamp` it gives **3.82% negative, median 3.00 days, max 46.69 days** — a plausible dispute-ageing curve. So: use `txn_id` for *entity* attribution, and the file's own timestamp pair for *delay* measurement. `bank_response − reported` = median 15.00 days, 16 negatives.

### D5. Transaction `mcc` is not the merchant's MCC

`tx.mcc` uses only 5 canonical codes {4131, 5411, 5812, 5912, 7011}; the master uses 10. Where both are known (7,778 rows) they agree **9.72%** of the time against **9.98% expected under independence**. Independent draws. Merchant master MCC is therefore the sole authority for category; `tx.mcc` is retained as a flagged attribute only.

### D6. Fan-out risk

Right-side key duplication would inflate a naive merge: tx→KYC 20,400 → ~22,219 rows; tx→merchants 20,400 → ~24,678 rows. Every dimension must be collapsed to one row per key **before** joining, or transaction counts and sums will silently overstate.

---

## E. Fraud-pattern feasibility — the findings that constrain the solution

### E1. The transaction graph is a sparse forest. It contains no cycles.

After deduplication: 20,000 transactions across 17,878 users and 8,051 merchants.

- **20,000 transactions = 20,000 distinct (user, merchant) pairs. Not one pair repeats.**
- Mean user degree 1.12 (max 5); mean merchant degree 2.48 (max 10)
- 5,929 connected components; **largest holds 65 of 25,929 nodes (0.3%)**
- No ID appears as both user and merchant — strictly **bipartite**, no user-to-user transfers exist
- **User pairs sharing ≥2 merchants: 0 → zero 4-cycles → the bipartite graph is acyclic**

A directed `A→B→C→A` money-movement cycle **cannot be formed from these transactions**, so the available data does not provide statistically or structurally supported evidence for the tested fraud-ring hypothesis. Money moving outside this dataset is not observed. Global community detection is not discriminative when the largest component is 0.3% of the graph.

### E2. No temporal anomaly signal

- Daily volume across 90 days: mean 222.2, sd 14.2, **CV 0.064** — uniform, no spikes
- Hourly profile flat (~790/hr) apart from an hour-0 count of 1,802 caused by date-only timestamps parsing to midnight — a **parsing artifact, not user behaviour**
- Max transactions by one merchant in one day: **3**. Merchants with ≥4 in a day: **0**
- Of 1,952 users with more than one transaction, minimum inter-transaction gap: median 24 days, **0 under 10 minutes**

No velocity, burst or spike signal exists.

### E3. No shared-identity evidence (both candidate signals are masking artifacts)

| Signal | Observed | Expected by chance | Verdict |
|---|---|---|---|
| Full 12-digit Aadhaar shared across user_ids | **0** | — | none |
| Masked Aadhaar (last-4) shared | 319 | ~388 | below chance |
| Full settlement account shared across merchants | **0** | — | none |
| Masked settlement account (last-4) shared | 71 | ~79 | below chance |
| Full PAN shared across user_ids | **0** | — | none |

Collisions appear only where masking reduces the key to 4 digits (10,000 slots), and even then occur *below* the birthday-collision rate. The data therefore provides no supported evidence of identity farming across IDs.

### E4. `reason_code` and `complaint_text` are independent

Agreement on comparable rows: **22.32%** against **21.41% expected under independence**. The brief suggests treating reason/text mismatch as an anomaly signal — the data shows it is noise everywhere, so it carries no per-record information and must not be scored.

### E5. What *is* real

- Chargeback-to-transaction linkage: **2,607 of 2,800 (93.11%)**
- Baseline dispute rate **13.04%**; 381 merchants with ≥2 disputes, 58 with ≥3, max 5; 184 users with ≥2, 10 with ≥3, max 4
- Among merchants with ≥3 transactions (n=3,451): dispute ratio median 0.000, p95 0.500, max 1.250
- Reporting-delay distribution: median 3.00 d, 595 disputes beyond 7 days, 127 beyond 30 days
- All data-quality flags (missing UTR, sign-corrupted amounts, invalid PAN, KYC status, risk segment, unmatched FKs)

---

## F. Baseline KPI panel (post-dedup, |amount|, Q1 2026)

| KPI | Value |
|---|---|
| Total transactions | 20,000 |
| Total transaction value | ₹249,772,509 |
| Average transaction value | ₹12,488.63 (median ₹12,478.21) |
| Success / Failed / Pending | 85.27% / 9.78% / 4.96% |
| Chargebacks (deduped) | 2,800 — 2,607 linked, 193 orphan |
| Disputed amount | ₹8,001,121 (179 missing) |
| Chargeback-to-transaction ratio | 13.04% |
| KYC completion rate | 76.27% |
| KYC rejection rate | 8.71% |
| Merchants with at least one dispute | 2,164 of 8,051 (26.9%) |

**Dimension coverage — the binding constraint on every segmented KPI:**

| Analysis | Coverage |
|---|---|
| Transactions with a KYC record | 6,478 (32.39%) |
| Transactions with a merchant record | 9,631 (48.16%) |
| Transaction *value* covered by merchant master | ₹120,241,723 (48.14%) |
| Chargebacks resolvable to merchant category | 1,270 (48.71% of linked) |
| Chargebacks resolvable to KYC | 895 (34.33% of linked) |

Any "by merchant category" or "by KYC status" metric is computed on roughly a third to a half of the data. Coverage must be displayed next to the number, every time.

---

## G. Data-quality proof panel (generated, never hardcoded)

| Metric | Value |
|---|---|
| Transaction rows raw | 20,400 |
| Transaction rows after exact dedup | 20,000 |
| Amount negative (sign corrupted) | 420 |
| Amount unparseable | 0 |
| UTR missing | 1,000 |
| UTR format-repaired (whitespace) | 1,842 |
| MCC missing/unusable in transactions | 2,872 |
| Transactions unmatched to KYC | 13,522 |
| Transactions unmatched to merchant master | 10,369 |
| Chargebacks unlinked to a transaction | 193 |
| Chargeback disputed_amount missing | 179 |
| KYC rows | 36,400 |
| KYC user_id collisions (different people, same id) | 5,341 |
| KYC PAN invalid (pre-repair) | 4,542 |
| Merchant rows | 6,210 |
| Merchant_id collisions (different merchants) | 1,310 |

Counts above are post-dedup where the label says so, which is why a few differ slightly from the raw-file figures in section C.
