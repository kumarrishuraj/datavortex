# DataVortex — UPI Fraud & Identity Intelligence

**Turning messy UPI transaction data into explainable risk, dispute and identity intelligence.**

TransOrg AgentIQ Datathon 2026 · Track 1 — *UPI Fraud Ring & Merchant Analytics*

> **Live dashboard:** *link to be added after deployment.*

---

## 1. Project Overview

DataVortex takes the four deliberately broken Track 1 files (UPI transactions, KYC records,
a merchant master and a chargeback log) and turns them into a validated star schema, a
tested analytics layer, a 7-page Streamlit dashboard and a rule-based AI Investigator.

The short version of what we learned: the most important work was not the dashboard. It
was finding out which parts of the data could be trusted before building anything on top of
it. Several of the "obvious" steps — deduplicating on `user_id`, joining chargebacks on their
own `merchant_id` — would have quietly produced wrong answers.

| | |
|---|---|
| Raw → clean transactions | 20,400 → **20,000** |
| Transaction value | **₹249,772,508.82** |
| Chargeback complaints | **2,800** (2,884 raw) |
| Chargeback rate | **12.26%** (distinct disputed transactions / transactions) |
| Merchant-master coverage | **48.16%** of transactions |
| KYC coverage | **32.39%** of transactions |
| Customer IDs shared by different people | **5,341** |
| Merchant IDs shared by different businesses | **1,310** |
| Dashboard pages | **7** |
| AI Investigator evaluation | **66 / 66** questions pass |
| Automated tests | **387** passing |

The dataset has **no confirmed fraud label**. So this project does not label anyone as
fraudulent. It tests fraud hypotheses, reports which ones the data supports and which it
does not, and gives reviewers an explainable way to prioritise their work.

---

## 2. Problem We Chose

From the competition brief: a National Payments Authority needs to analyse UPI
micro-transaction data to identify compromised merchant accounts, synthetic identity fraud
and circular money-laundering rings. The data arrives with missing UTR numbers, mismatched
PAN/Aadhaar formats, OCR errors and currency symbols embedded in numeric columns.

## 3. Why This Problem

- UPI is high-volume and low-value. A control that fires on noise creates review work
  without catching anything, so telling signal from noise matters more than flagging a lot.
- Dispute attribution decides who carries the loss. Pinning a chargeback on the wrong
  merchant is both a financial and a regulatory mistake.
- KYC identifiers sit under almost every financial-crime control in India. If an identifier
  is ambiguous, every control built on it inherits the ambiguity.
- It is the track where messy data does the most damage: a wrong join produces a confident,
  wrong answer that looks like a finding.

We chose this track because we wanted to work on a financial problem where data quality
directly affects the reliability of fraud and dispute analysis. The combination of messy
transaction data, identity resolution and explainable risk analysis gave us an opportunity to
build something practical rather than only train a model.

---

## 4. What We Found in the Raw Data

We did not build a single chart in the first stage. We wrote 15 audit scripts
(`scripts/audit/`) and profiled every file first. The full record is in
[`docs/01_forensic_audit.md`](docs/01_forensic_audit.md).

| File | Rows | Columns |
|---|---|---|
| `track1_upi_transactions.csv` | 20,400 | 8 |
| `track1_kyc_records.csv` | 36,400 | 12 |
| `track1_merchants_master.csv` | 6,210 | 11 |
| `track1_chargebacks.json` | 2,884 | 13 |
| `track1_dataset_notes.txt` | 111 lines | — |

What was wrong with it:

- **Identifiers** — five spellings of the same user ID (`USR12345`, `usr12345`, `USR-12345`,
  `USR 12345`, `usr_12345`, plus bare `12345`) and four merchant ID spellings. The
  transaction file's IDs were clean; the damage was in the KYC, merchant and chargeback files.
- **Amounts** — `₹`, `Rs.`, `INR`, thousands separators, `27.3k` shorthand, `Not Available`,
  blanks and negative values across four money columns.
- **Timestamps** — seven date fields with five to seven formats each, including Unix epochs
  and negative epochs (pre-1970 dates of birth).
- **Categoricals** — 14 transaction-status spellings, 16 KYC statuses, 15 merchant statuses,
  82 merchant-category spellings, 34 chargeback reason codes and 16 severity values on two
  different scales.
- **Text** — 41 city spellings for 12 real cities (Bombay, Mumbay, Poona, Calcutta, Madras,
  LDH, BLR, Dilli…).
- **Damaged identifiers** — PANs with OCR look-alike characters or a missing last letter,
  partly masked Aadhaar numbers, and UTRs with whitespace inside.
- **Duplicates** — 400 transaction rows, 84 complaint rows, 278 KYC rows and 12 merchant rows,
  all byte-identical.

---

## 5. Our Approach

A few rules we held ourselves to, because each one came out of something that went wrong or
nearly went wrong:

1. **Audit before building.** Every cleaning rule has a measurement behind it.
2. **Only delete what is provably a duplicate.** Rows are removed only when identical on
   every column. Everything else is flagged, not dropped.
3. **Repair when unambiguous, never fabricate.** A PAN missing its last letter stays missing.
   A missing UTR is never invented.
4. **Keep what doesn't join.** Unresolved foreign keys go to an explicit `UNKNOWN` member, so
   every metric still reconciles to 20,000 transactions.
5. **Test relationships instead of trusting column names.** A column called `merchant_id`
   is not evidence that it identifies the merchant.
6. **Test hypotheses before reporting them.** A ranking is shown as a finding only if it
   survives a significance test.
7. **Define each metric once.** A semantic registry holds every metric definition; the
   dashboard, tests and AI Investigator all read from it.
8. **No typed numbers in the app.** Page text reads the pipeline's own measurements. A test
   fails the build if a previously typed figure reappears.

---

## 6. Architecture

```
data/raw/            organisers' files (not in this repository; never written to)
   │
   ├── scripts/run_pipeline.py        Stage 2 — data rescue
   │     src/ingestion/               read every column as text
   │     src/cleaning/  (9 modules)   ids, amounts, timestamps, statuses, kyc,
   │                                  merchants, chargebacks, duplicates, validation
   │     src/transformation/          star schema + audit facts
   │     → data/processed/*.parquet   → docs/data_quality_report.md
   │
   ├── scripts/build_analytics.py     Stage 3 — analytics
   │     src/analytics/semantic.py        metric and dimension registry
   │     src/analytics/query_engine.py    runs registered metrics
   │     src/analytics/kpis.py            value + denominator + coverage
   │     src/analytics/shrinkage.py       denominator floor + empirical Bayes
   │     src/analytics/significance.py    chi-square, Wilson intervals, BH correction
   │     src/analytics/hypotheses.py      hypothesis register
   │     src/analytics/network.py         transaction graph structure
   │     src/analytics/risk_indicators.py Risk Indicator Score
   │     src/analytics/forecast.py        backtested baseline forecast
   │     → data/processed/analytics/*.parquet   → docs/kpi_validation_report.md
   │
   └── app/dashboard.py               Stage 4 — Streamlit dashboard (7 pages)
         app/components/  app/views/  src/agent/ (planner, chart selector, executor)
```

Data flows one way. The dashboard reads only Parquet and defines no metric of its own.

There is **no SQL engine** in the project. The materialised Parquet aggregates play the role
that SQL views would in a warehouse, and the AI Investigator therefore has no query surface
to inject into.

---

## 7. Data Cleaning & Data Rescue

| Decision | Why we believe it |
|---|---|
| Remove only byte-identical duplicates | All 400 duplicate transaction rows are identical on every column; checked before collapsing |
| Negative amounts are sign corruption, not refunds | No refund status exists in the data; **0 of 429** negative amounts have a matching positive twin; the negative rate is flat across SUCCESS / FAILED / PENDING (2.07% / 2.46% / 1.97%). We store the magnitude and keep a flag |
| Slash dates are `DD/MM/YYYY`, hyphen dates are `MM-DD-YYYY` | 18,385 slash-date rows have a first part above 12 and **0** have a second part above 12; 9,133 hyphen-date rows show the reverse and **0** contradict it. We use explicit formats, never a global `dayfirst` |
| Never invent a UTR | 1,000 are missing and flagged |
| Repair PANs with OCR look-alikes | 0↔O, 1↔I, 2↔Z, 5↔S, 8↔B by position, kept only if the result is a valid PAN: **687** repaired |
| Never complete a truncated PAN | **1,454** PANs are 9 characters long — inventing the last letter would invent an identity |
| Use the merchant master for categories | The transaction file's own MCC matches the master **10.12%** of the time, against **9.76%** expected by chance alone |
| Send unresolved keys to `UNKNOWN` | 13,522 transactions have no KYC match and 10,369 no merchant-master match. They stay in every total |

**The totals reconcile.** Transaction value changes only for the two documented steps:

| Step | Total ₹ | Change |
|---|---|---|
| Raw, signed | 244,248,045.17 | — |
| After sign repair | 254,958,886.61 | +10,710,841.44 |
| After duplicate removal | 249,772,508.82 | −5,186,377.79 |
| `FACT_TRANSACTIONS` | **249,772,508.82** | **+0.00** |

**Row accounting** — zero rows unaccounted for:

| Source | Raw | Exact duplicates | Kept as the record for their ID | Kept in the collision bridge | Unaccounted |
|---|---|---|---|---|---|
| KYC | 36,400 | 278 | 28,920 | 7,202 | **0** |
| Merchants | 6,210 | 12 | 4,343 | 1,855 | **0** |

Every step is logged into [`docs/data_quality_report.md`](docs/data_quality_report.md), which
the pipeline regenerates on each run.

---

## 8. Identity Collision Intelligence

One of the first things we checked was whether `user_id` could safely be treated as a unique
identity. It couldn't. After normalising the five spellings, the same ID often pointed at
genuinely different people — different name, PAN, Aadhaar, date of birth, city and income.
`USR10043`, for example, is two different people. A normal `drop_duplicates(subset="user_id")`
would have deleted one of them and merged the other into a record that belongs to nobody.

| Resolution | Customers | Merchants |
|---|---|---|
| Unique ID | 22,806 | 2,936 |
| Same entity, conflicting rows | 773 | 97 |
| **Different entities sharing one ID** | **5,341** | **1,310** |

Corroboration: **4,422** customer IDs carry more than one distinct valid PAN, while **0** PANs
are shared across customer IDs. The identifier is ambiguous; the person is not.

What we built instead: each ID gets one deterministic survivor for joins (most complete
record, then latest signup, then earliest source row), flagged `identity_ambiguous=True` with
low resolution confidence. **All 16,578 candidate rows** are kept in
`BRIDGE_IDENTITY_COLLISION`, and the dashboard's collision explorer shows them side by side.
`drop_duplicates(subset="user_id")` does not appear anywhere in the code.

*A number we had to correct:* our first audit reported 4,864 customer IDs with more than one
PAN. It had counted raw PAN strings, so two formatting variants of the same PAN looked like
two PANs. Counting cleaned PANs gives 4,422. We reproduced both numbers to be sure.

---

## 9. Chargeback Attribution

The dataset notes suggest linking chargebacks to users and merchants through the chargeback
file's own `user_id` and `merchant_id`. We tested that before using it.

Of the **2,607** complaints whose `txn_id` links to a transaction:

| Check | Result |
|---|---|
| Complaint `user_id` equals the linked transaction's customer | **0** complaints |
| Complaint `merchant_id` equals the linked transaction's merchant | **0** complaints |
| Complaint's own transaction timestamp within a day of the linked transaction | **2.21%** |

Meanwhile `txn_id` links 2,607 of 2,800 complaints (**93.11%**). The 2,451 distinct linked
transaction IDs compare with a chance expectation of about 1, a z-score of about **+3,420**.

So every dispute is attributed `chargeback → txn_id → transaction → customer / merchant`. The
chargeback file's own ID columns are cleaned, kept and flagged, but never joined on. Following
the notes literally would have blamed an unrelated set of merchants.

There was one exception. The chargeback file *is* consistent with itself in time. Measuring
reporting delay from the linked transaction would make **45.11%** of delays negative, which is
impossible. Measured from the complaint's own transaction timestamp, **3.85%** are negative
(flagged, not clipped) and the median is 3.00 days. So we use `txn_id` for *who* and the
complaint file's own timestamps for *when*.

193 complaints (6.89%) cannot be attributed to any transaction, 116 of them because their
`txn_id` is a malformed 5-digit value. They stay in the complaint counts.

---

## 10. Analytics & KPIs

**Star schema (Parquet):** `FACT_TRANSACTIONS` (20,000; one row per `txn_id`),
`FACT_CHARGEBACKS` (2,800; one row per complaint), `DIM_USERS` (28,920 + UNKNOWN),
`DIM_MERCHANTS` (4,343 + UNKNOWN), `DIM_DATE`, `BRIDGE_IDENTITY_COLLISION` (16,578), plus
reconciliation tables and `audit_facts` — 29 forensic measurements that the dashboard quotes
instead of hardcoding.

**Semantic registry.** `src/analytics/semantic.py` defines **22 metrics** and **15 dimensions**
once — definition, formula, grain, coverage basis, source table, aggregation and synonyms.
`src/analytics/query_engine.py` runs any registered metric by dimension, time grain and time
window, and checks that additive breakdowns sum back to the total. A test runs all 17 headline
KPIs through the engine and requires an exact match with the materialised values. A second,
independent recomputation checks 16 KPIs a different way ([`docs/kpi_validation_report.md`](docs/kpi_validation_report.md)).

**Every KPI carries its coverage.** Values come back with their denominator and coverage
percentage, and the dashboard shows a coverage badge next to any number that describes only
part of the book.

**Are the documented joins real?** We compared observed ID overlap with what two independent
random draws from the same ID space would produce:

| Join | Expected if unrelated | Observed | z |
|---|---|---|---|
| transactions × KYC (user) | 5,745 (sd 62.4) | 5,799 | **+0.87** |
| transactions × merchant master | 3,885 (sd 44.8) | 3,893 | **+0.18** |
| chargebacks × transactions (`txn_id`) | 1 | 2,451 | **+3,420** |

The first two cannot be told apart from chance. ID normalisation raised the KYC match from
22.37% to 32.39%, but no amount of cleaning can raise it further, and we stopped trying.

| Headline KPI | Value | Coverage |
|---|---|---|
| Transactions | 20,000 | 100% |
| Transaction value | ₹249,772,509 | 100% |
| Average transaction value | ₹12,489 | 100% |
| Success / Failed / Pending | 85.27% / 9.78% / 4.96% | 100% |
| Chargeback complaints | 2,800 | 100% |
| Disputed value | ₹8,001,121 | 93.61% |
| Chargeback rate | 12.26% | 100% |
| Complaints per transaction | 0.130 | 100% |
| Average reporting delay | 6.47 days | 85.29% |
| Disputes reported after 7 days | 579 | 85.29% |
| KYC completion / rejection | 77.43% / 8.10% | 100% |
| Missing UTR rate | 5.00% | 100% |
| Merchant-master coverage | 48.16% | 100% |
| KYC coverage | 32.39% | 100% |

The chargeback rate counts **distinct disputed transactions** (2,451), not complaints (2,607).
151 transactions carry more than one complaint, and counting complaints broke one of our tests
(see §12).

---

## 11. Risk Indicator Score

The brief asks for a risk score. The data has disputes, but **no confirmed fraud label**, so
we cannot train a supervised model or validate any score as a fraud predictor. What we built
instead is a transparent points total over observed conditions that justify a manual review.

**Risk Indicator ≠ confirmed fraud.** The score is **not a probability of fraud**. It is
review priority, and every score is shown with the reasons that produced it.

The score runs from 0 to 100 and has four components, each worth up to 25 points:

| Customer component | Points |
|---|---|
| Dispute history | 25 × min(disputed transactions, 2) / 2 |
| Disputed value | 25 × percentile rank of disputed amount among customers with any disputed value |
| KYC status | REJECTED 25 · PENDING 12.5 · VERIFIED 0 · no KYC record 0 (marked "not assessable") |
| Identity ambiguity | 25 if the customer ID is shared by different people |

| Merchant component | Points |
|---|---|
| Dispute volume | 25 × min(disputed transactions, 3) / 3 |
| Disputed value | 25 × percentile rank of disputed amount among merchants with any disputed value |
| Merchant status | BLOCKED or SUSPENDED 25 · INACTIVE 12.5 · ACTIVE 0 · no master record 0 |
| Identity ambiguity | 25 if the merchant ID is shared by different businesses |

Bands: Low 0–24 · Moderate 25–49 · Elevated 50–74 · High 75–100.

| Band | Customers | Merchants |
|---|---|---|
| High | 6 (0.03%) | 10 (0.12%) |
| Elevated | 205 (1.15%) | 211 (2.62%) |
| Moderate | 2,370 (13.26%) | 1,793 (22.27%) |
| Low | 15,297 (85.56%) | 6,037 (74.98%) |
| Total | 17,878 transacting customers | 8,051 transacting merchants |

Why it is built this way:

- **Equal weights.** With no label, there is nothing to learn weights from. Unequal weights
  would imply one condition predicts fraud better than another — a claim this data cannot back.
- **Dispute volume, not dispute rate.** We first ranked merchants by dispute rate and got a list
  of merchants with one transaction: 304 merchants show a 100% rate. A minimum of 3
  transactions leaves 3 of them, and empirical-Bayes shrinkage cuts the spread across eligible
  merchants from 17.16 to 0.69 percentage points. Then the overdispersion test showed there is
  no merchant-level dispute propensity to measure at all (§12). A rate component would be
  scoring noise.
- **A missing record scores zero.** No KYC record or no merchant master record is a coverage
  gap, not evidence.
- **Missing UTR and reporting delay are left out.** The hypothesis tests found neither is
  associated with disputes.

Tests check the 0–100 bounds, that the components add up to the score, that bands follow the
thresholds, and that no output column is named as fraud.

---

## 12. Statistical Hypothesis Testing

The dataset notes include a list headed *"Example insights students may discover."* It would
have been easy to build a chart for each one and present it as a finding. We tested each one
instead, and added three checks of our own.

| Verdict | Hypothesis | Evidence |
|---|---|---|
| NOT SUPPORTED | Certain merchant categories have disproportionately high chargebacks | χ² = 5.241, df = 11, p = 0.919 |
| NOT SUPPORTED | Some users appear repeatedly in disputes | 39 observed vs 33.9 expected, z = +0.87, p = 0.384 |
| NOT SUPPORTED | Merchants show sudden transaction spikes followed by disputes | variance/mean = 0.901, max merchant-day = 3, p = 0.736 |
| NOT SUPPORTED | Missing UTRs correlate with failed or disputed transactions | dispute χ² = 0.117, failure χ² = 0.007, p = 0.730 |
| BORDERLINE | Unverified or rejected KYC users show higher dispute risk | χ² = 7.894, df = 3, p = 0.047, but the smallest corrected pairwise p = 0.053 |
| **CONTRADICTED** | Delayed reporting indicates account takeover | takeover disputes are reported **earlier**: 5.45 vs 6.62 days, t = −2.45, p = 0.014 |
| QUANTIFIED | Duplicate transactions inflate revenue and dispute metrics | 400 duplicate rows worth ₹5,186,378 |
| NOT SUPPORTED | Individual merchants differ in how often they are disputed *(ours)* | χ²/df = 0.9688 across 3,451 merchants, p = 0.904 |
| NOT SUPPORTED | Transaction volume follows an intraday pattern *(ours)* | χ² = 17.5, df = 23, CV = 0.031, p = 0.785 |
| NOT SUPPORTED | A mismatch between reason code and complaint text is an anomaly signal *(ours)* | agreement 16.83% vs 16.51% expected, z = +0.38, n = 1,973, p = 0.705 |

Of the seven insights in the dataset notes: **4 not supported, 1 borderline, 1 contradicted,
1 quantified.** "Not supported" means this data did not show the effect; it does not mean the
effect cannot exist in real UPI traffic. The borderline KYC result should not be acted on,
because no pairwise comparison survives correction for multiple testing.

Methods: chi-square homogeneity tests, two-proportion z-tests with Benjamini-Hochberg
correction, Wilson confidence intervals, a beta-binomial overdispersion test, Welch's t-test
and a Poisson index of dispersion — implemented directly in `src/analytics/` without SciPy and
unit-tested against synthetic data where the right answer is known.

**Two bugs we caught in our own statistics:**

- The account-takeover hypothesis first came out as SUPPORTED, because the p-value was
  significant — while the effect ran in the opposite direction to the claim. We added a
  direction check and a test for it. Without it we would have published a backwards finding.
- The merchant overdispersion test first reported strong signal (χ²/df = 1.127,
  p = 1.8 × 10⁻⁷). It was counting complaints rather than disputed transactions; with 151
  transactions carrying more than one complaint, some merchants exceeded a ratio of 1, which
  breaks the test's assumptions. Counting distinct disputed transactions removed the "signal".

We also found a midnight activity spike that turned out to be our own parsing: 1,000 date-only
timestamps parse to 00:00, creating a peak of 1,802 transactions against a baseline near 790
per hour. Excluding them, the hourly profile is flat (CV 0.031).

---

## 13. Network Intelligence

The brief mentions circular money-laundering rings, so we modelled transactions as a graph of
customers and merchants and measured its structure.

| Measurement | Value |
|---|---|
| Nodes / edges | 25,929 / 20,000 |
| Distinct customers / merchants | 17,878 / 8,051 |
| Connected components | 5,929 |
| Largest component | 65 nodes (0.25% of the graph) |
| Bipartite (no ID on both sides) | Yes |
| Repeated customer–merchant pairs | 0 |
| Customer pairs sharing two or more merchants (4-cycles) | 0 |
| Acyclic | Yes — edges = nodes − components exactly |

Customers only ever pay merchants, and the graph contains no cycle of any length. A circular
A → B → C → A money path cannot be formed from these transactions. **The available data did
not provide statistically or structurally supported evidence for the tested fraud-ring
hypothesis**, so the dashboard reports that result instead of drawing a ring.

We also looked for shared identifiers, which is how synthetic identities often show up:

- 0 full Aadhaar numbers shared across customer IDs
- 0 full settlement accounts shared across merchants
- 0 PANs shared across customer IDs

Sharing only appears where masking cuts a value down to its last four digits, and there it
matches chance: 319 shared masked Aadhaar values against 326.5 expected, and 71 shared masked
settlement accounts against 69.9 expected.

*Another correction:* our first audit said masked-value sharing was *below* chance (~388 and
~79 expected). The expected counts came from a wrong birthday-collision formula. The corrected
formula is in `src/transformation/audit_facts.py` with a hand-calculated unit test.

**Clustering:** we considered it and decided against it. With no repeated pairs, no 4-cycles
and a largest component of 65 nodes, there is no shared structure for a clustering or
community-detection method to recover — it would partition noise.

---

## 14. Forecasting

`src/analytics/forecast.py` forecasts three daily series: transactions, disputed transactions
and transaction value. We deliberately used simple, checkable methods:

1. Fit four baselines — flat mean, last value, same weekday last week, linear trend — on the
   earlier part of the 90-day history.
2. Score each on the last **28 days**, which none of them saw.
3. Pick the lowest mean absolute error and project **14 days** ahead with a 95% band.
4. Record how often that band contained the real values in the holdout.

| Series | Selected method | Holdout MAE | Flat-mean MAE | Band held |
|---|---|---|---|---|
| Transactions per day | flat mean | 13.449 | 13.449 | 96.4% |
| Disputed transactions per day | same weekday last week | 4.571 | 4.675 | 92.9% |
| Transaction value per day | last value carried forward | ₹174,654 | ₹179,858 | 96.4% |

The honest reading: these series are flat. No method beats the flat mean by much, the trend
tests are not significant (p = 0.977, 0.476, 0.672) and neither are the day-of-week tests
(p = 0.701 for transactions, 0.241 for disputed transactions). The forecast is useful for
planning dispute-team capacity, not for predicting any single transaction.

---

## 15. Dashboard

Streamlit + Plotly, 7 pages, reading only Parquet.

| Page | What it shows |
|---|---|
| **1. Executive Overview** | 12 KPI cards with coverage badges, daily trends, outcome mix, top exposure, and key findings generated from the current build |
| **2. Transaction Analytics** | Daily volume and value, outcome mix, hour of day (date-only rows excluded), category charts, category chargeback rate with 95% confidence intervals and an automatic significance caveat, and the validated forecast |
| **3. Merchant & User Risk** | Merchant exposure rankings, raw-vs-shrunk dispute rates, and merchant and customer Risk Indicator tabs with the formula, bands, components and a reason for every score |
| **4. Data Quality & Hypotheses** | Defect scorecard, before/after reconciliation, row and value accounting, timestamp parsing, and the full hypothesis register with the CONTRADICTED result highlighted |
| **5. Identity & Network** | Identity collision explorer, network structure, cluster explorer and the fraud-ring evidence table |
| **6. Chargeback Intelligence** | Attribution integrity, reason / severity / resolution / channel breakdowns, reporting-delay analysis and top exposure |
| **7. AI Investigator** | Ask questions, see the evaluation matrix, how answers are built, and the metric registry |

Sidebar filters (date, status, category, state) recompute KPIs on pages 1, 2, 3 and 6 through
the same KPI functions the analytics layer uses. `UNKNOWN` stays visible in breakdowns rather
than being filtered out.

*A deployment bug we hit:* the page modules originally lived in `app/pages/`. Streamlit
automatically turns a `pages/` folder next to the entry script into extra navigation, so a
judge would have seen seven duplicate links that opened blank pages. We renamed the folder to
`app/views/`.

*A consistency bug we hit:* some page text still contained numbers we had typed in during
development. One callout said reason code and complaint text agree 22.32% of the time; the
pipeline now computes 16.83%. We replaced every such number with values read from the pipeline
and added a test that scans the app source for them.

---

## 16. AI Investigator

**What it is:** a question box on page 7 that answers questions about this dataset with a
number, a chart, a short explanation and the caveats the data requires.

**What it is not:** a large language model. The current planner is **rule-based**. It needs
no API key and makes no network calls, so a live demo cannot fail on one.

```
natural-language question
   → typed intent        (measure, dimension, time grain, window, series, ranking)
   → semantic registry validation   (22 metrics, 15 dimensions, allowed charts and limits)
   → query engine        (computes the answer in pandas over Parquet)
   → chart selection     (rules on the shape of the answer)
   → narrative + caveats (coverage, significance, risk disclaimer) + provenance
```

The planner resolves each part of a question against the registry's own synonyms, so it is
not keyed to specific sentences. No number in an answer is written by the planner: every
figure comes from the query engine, and the "How this answer was generated" panel shows the
metric, formula, source table, grain, window, chart reason and rows analysed.

**Chart selection** follows fixed rules: anything grouped by day, week or month is a **line**;
rankings of individual merchants or customers, or more than 8 categories, are horizontal bars;
small comparisons are bars; record lists are tables; distributions are histograms; single
figures are KPI cards.

**Evaluation.** `scripts/run_agent_eval.py` plans and executes every question below and writes
[`docs/agent_test_matrix.md`](docs/agent_test_matrix.md). **66 of 66 pass.**

| Source | Questions | Passed |
|---|---|---|
| Dataset notes — example agent queries | 12 | 12 |
| Dataset notes — example dashboard questions | 15 | 15 |
| Competition PDF examples | 4 | 4 |
| Team demo questions | 10 | 10 |
| Other phrasings ("over time", "daily", "timeline", "success vs failure trend", …) | 12 | 12 |
| Safety probes that must be refused | 13 | 13 |

A question passes only if the intent, metric, dimension, time grain, chart type and required
caveat are all correct. Charts produced across the matrix: 15 line, 19 horizontal bar, 7 bar,
6 table, 5 KPI card, and one correct "no result" — *"Compare Q3 sales by region"*, because the
data only covers Q1 2026. A separate test runs 15 trend phrasings and requires a line chart with
real data every time.

**Safety.** The Investigator refuses statement-shaped SQL, input containing any of 21 blocked
tokens (`drop`, `delete`, `exec`, `__import__`, …), prompt-injection phrasing, requests to reveal
PAN / Aadhaar / account numbers, and requests to label anyone fraudulent. Coverage caveats (for
example, only 32.39% of transactions resolve to KYC) and significance caveats stay attached to
the answers that need them.

*How it got here:* an earlier version answered only 5 of the 12 dataset-notes agent queries
cleanly and could not draw a line chart at all — trend questions were refused. We rebuilt the
planner around the registry, added the query engine and the chart rules, and turned the example
questions into a test matrix. The final safety review also caught `SELECT * FROM dim_users`
being quietly reinterpreted as a ranking instead of refused; SQL-shaped input is now rejected.

**Limitation:** because the current planner is deterministic, unsupported phrasing outside the
registry's synonym set may not be understood. It also has no scatter-chart intent.

---

## 17. Key Findings

1. **Two of the documented joins behave like random overlap** — transactions to KYC (z = 0.87)
   and transactions to the merchant master (z = 0.18). 13,522 and 10,369 transactions are kept
   under an UNKNOWN member rather than dropped.
2. **5,341 customer IDs and 1,310 merchant IDs are shared by different entities.** Deduplicating
   on the ID would have destroyed real records.
3. **The chargeback file's own customer and merchant IDs never match the transaction they
   reference** (0 of 2,607). Attribution has to go through `txn_id`.
4. **The transaction file's MCC does not provide reliable agreement with the merchant master
   MCC** (10.12% agreement vs 9.76% expected).
5. **Merchant dispute rates show no merchant-level propensity** (χ²/df = 0.969, p = 0.904).
6. **Category chargeback rankings are within sampling noise** (χ² = 5.24, p = 0.919). Transport
   ranks first at 13.36% (95% CI 11.46–15.52%), and every category's interval overlaps.
7. **The tested fraud-ring hypothesis is not supported** by the available data: the graph is
   bipartite and acyclic, with no 4-cycles and no repeated pairs.
8. **One suggested insight runs backwards:** account-takeover disputes are reported 1.17 days
   *earlier* than other disputes, not later (p = 0.014).
9. **The midnight activity spike is a parsing artifact** from 1,000 date-only timestamps.
10. **Duplicates would have overstated transaction value by ₹5,186,378.**
11. **The daily series have no significant trend or weekly pattern**, so a flat forecast is the
    honest one.

---

## 18. Data Coverage & Limitations

- **The data is synthetic.** Findings describe this dataset, not real UPI behaviour.
- **Coverage limits segmented analysis.** Category metrics describe 48.16% of transactions and
  KYC-segmented metrics 32.39%. Both limits come from the source data and cannot be cleaned away.
- **No fraud ground truth.** There is no supervised model, no precision or recall, and no claim
  about detection performance. The Risk Indicator Score is not validated as a fraud predictor
  and cannot be with this data.
- **Some values cannot be recovered:** 1,454 truncated PANs and 1,000 missing UTRs.
- **193 complaints (6.89%) cannot be attributed** to a transaction; 412 have no computable
  reporting delay; 92 have a negative delay and are flagged.
- **The BORDERLINE KYC result should not be acted on** (§12).
- **The forecast rests on 90 days of history** and cannot capture seasonality longer than that.
- **The AI Investigator is deterministic and registry-bound** (§16).
- **Row counts are pinned** in `src/config.py`. A different input file makes the pipeline fail
  loudly by design.
- **Running the pipeline without the raw files** stops with a clear message pointing to
  `data/raw/README.md`, but it is printed as a Python traceback.

---

## 19. Privacy

**The raw competition files are not included in this repository.** The competition asks for
code and data dictionaries to be published, says data must not be shared outside the team, and
the KYC file contains full (synthetic) PAN and Aadhaar values.

What is committed under `data/processed/` is privacy-safe according to our final audit:

- PAN is stored only in masked form (first two characters and the last one).
- Aadhaar is stored only as its last four digits.
- Settlement accounts are stored as `XXXX` plus the last four digits.
- No identifier hashes and no dates of birth are persisted.

The final audit compared every file intended for the repository — all text files and 461,664
distinct values in the Parquet files — against every raw PAN, Aadhaar and settlement-account
value, and found none. The automated tests scan every Parquet file for PAN- and Aadhaar-shaped
values, and scan the text files against the raw values whenever the raw files are present
locally. No `.env` file or API key is needed or committed.

**To reproduce the pipeline from raw data:**

1. Obtain the competition-provided dataset through the official competition or team channel.
2. Copy the five required files into `data/raw/`.
3. Run the pipeline (see §21).
4. [`data/raw/README.md`](data/raw/README.md) lists the exact file names expected.

The dashboard, the notebook and the tests do not need the raw files.

---

## 20. Project Structure

```
.
├── app/
│   ├── dashboard.py              Streamlit entry point
│   ├── components/               charts, data loading, theme, UI helpers
│   └── views/                    the 7 pages (p1_overview … p7_investigator)
├── src/
│   ├── config.py                 paths, ID specs, expected row counts
│   ├── ingestion/                raw file loading (everything as text)
│   ├── cleaning/                 9 cleaning modules
│   ├── transformation/           star schema, audit facts
│   ├── analytics/                semantic registry, query engine, KPIs, shrinkage,
│   │                             significance, hypotheses, network, risk indicators,
│   │                             forecast, aggregate views
│   └── agent/                    planner, chart selector, executor, evaluation cases
├── scripts/
│   ├── run_pipeline.py           raw → star schema + data quality report
│   ├── build_analytics.py        aggregates, risk scores, forecast, KPI validation report
│   ├── build_report.py           data quality report writer (called by the pipeline)
│   ├── build_data_dictionary.py  docs/data_dictionary.md
│   ├── run_agent_eval.py         AI Investigator evaluation matrix
│   ├── build_notebook.py         builds and executes the walkthrough notebook
│   └── audit/                    15 Stage 1 forensic audit scripts
├── notebooks/
│   └── datavortex_walkthrough.ipynb
├── tests/                        5 test files, 387 tests
├── docs/                         audit, data dictionary, reports, test matrix, checklists
├── data/
│   ├── raw/README.md             raw files go here (not committed)
│   └── processed/                privacy-safe Parquet used by the dashboard (37 files)
├── .streamlit/config.toml
├── requirements.txt              runtime dependencies
├── requirements-dev.txt          + pytest and notebook tooling
├── .env.example
└── .gitignore
```

Key documents:

| Document | Contents |
|---|---|
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Every column of the star schema: type, nulls, distinct values, example, meaning |
| [`docs/data_quality_report.md`](docs/data_quality_report.md) | Every cleaning step with before/after counts and reconciliation |
| [`docs/kpi_validation_report.md`](docs/kpi_validation_report.md) | KPI definitions and independent recomputation |
| [`docs/01_forensic_audit.md`](docs/01_forensic_audit.md) | The original Stage 1 audit, with a corrections table |
| [`docs/agent_test_matrix.md`](docs/agent_test_matrix.md) | All 66 AI Investigator evaluation questions and results |
| [`notebooks/datavortex_walkthrough.ipynb`](notebooks/datavortex_walkthrough.ipynb) | Commented walkthrough of the audit, cleaning evidence, two statistical analyses and the forecast backtest |

---

## 21. How to Run

Python 3.13 is recommended. The project was developed and verified on Python 3.13, including a
clean install from `requirements.txt` in a fresh virtual environment.

**1. Get the code and create an environment**

```bash
git clone <repository-url>
cd <repository-folder>
python --version        # should report Python 3.13.x
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt        # enough to run the dashboard and the pipeline
pip install -r requirements-dev.txt    # adds pytest and the notebook tooling
```

**3. Run the dashboard** — works straight away on the committed Parquet:

```bash
streamlit run app/dashboard.py
```

**4. Rebuild everything from raw data (optional)**

Place the five competition files in `data/raw/` (see §19), then:

```bash
python scripts/run_pipeline.py          # cleaning, star schema, audit facts, data quality report
python scripts/build_analytics.py       # aggregates, risk scores, forecast, KPI validation report
python scripts/build_data_dictionary.py # data dictionary
```

**5. Run the tests**

```bash
pytest tests/ -q
```

**6. Regenerate the AI Investigator matrix and the notebook (optional)**

These need only the committed `data/processed/`:

```bash
python scripts/run_agent_eval.py
python scripts/build_notebook.py
```

---

## 22. Reproducibility

- Every path is relative, derived from `src/config.py`.
- The pipeline asserts the expected raw row counts and output grains, and fails loudly if they
  change.
- The cleaning rules are deterministic, including identity-survivor selection.

We checked this end to end in fresh virtual environments, separate from our development setup,
on a copy containing exactly the files that go into the repository:

| Check | Result |
|---|---|
| Install from `requirements.txt` only | clean; `pip check` passes; pytest is not needed to run the app |
| Rebuild with the raw files copied in | all pipeline assertions and 16/16 KPI checks pass; all 37 Parquet files identical in content to the committed ones; the 4 generated reports identical |
| Dashboard | all 7 pages render with no errors; `streamlit run` serves the app |
| AI Investigator matrix | 66 / 66 |
| `pytest tests/ -q` with raw files | 387 passed |
| `pytest tests/ -q` in a fresh clone without raw files | 380 passed, 7 skipped with a stated reason |
| Notebook | re-executed with 0 errors and the same outputs |

That fresh-clone run also caught a real problem: four pipeline tests crashed with
`FileNotFoundError` instead of skipping when the raw files were absent. They now skip with a
reason in that case and still run in full when the files are present.

The full rebuild takes under a minute on our machines.

---

## 23. Testing

```bash
pytest tests/ -q
```

| File | Tests | What it covers |
|---|---|---|
| `tests/test_cleaning_units.py` | 94 | ID, amount, timestamp, status, PAN, MCC and city cleaning rules (test identifiers are made-up values, not dataset records) |
| `tests/test_pipeline_integration.py` | 34 | Grain, reconciliation to raw, no join fan-out, UNKNOWN routing, attribution, privacy of the processed tables |
| `tests/test_analytics.py` | 46 | Shrinkage, significance tests, hypothesis grading, registry, aggregates |
| `tests/test_dashboard.py` | 60 | All 7 pages rendered with Streamlit's AppTest, KPI parity, coverage badges, privacy, AI Investigator planning and execution |
| `tests/test_submission_safety.py` | 153 | All 66 evaluation questions, trend phrasings must draw lines, chart rules, query engine vs every headline KPI, Risk Indicator formula and bounds, forecast selection, audit facts, PAN / Aadhaar / account scans, credential scan, a ban on `eval` / `exec` / `subprocess` / SQL engines, and a scan for typed figures in the app |

**387 tests pass.** Without the raw files, 380 pass and 7 skip (4 raw-reconciliation tests, 2
dataset-notes coverage tests and 1 raw-identifier scan).

Statistical routines are tested against synthetic data where the answer is known: a population
with one shared rate must return *no signal*, and a genuinely varied one must be detected.

---

## 24. Competition Rubric Mapping

| Criterion | Implementation | Evidence |
|---|---|---|
| Data dictionary | Generated from the Parquet schema for every star-schema column | [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| Proof of data cleaning | Raw vs clean counts asserted in the pipeline; row, value and duplicate reconciliation | [`docs/data_quality_report.md`](docs/data_quality_report.md), dashboard page 4, notebook §1 |
| Missing values & duplicates | Only byte-identical duplicates removed; collisions preserved; unresolved keys routed to UNKNOWN; missing UTRs and truncated PANs flagged, never fabricated | §7, §8 |
| Standardisation | Date format rule with zero counterexamples; currency parsing; 41 city spellings → 12; categorical canonicalisation. The Track 1 data has no genuinely multilingual text — regional city names are the closest case | §7, `src/cleaning/` |
| Reproducibility | Documented commands; verified in fresh environments; notebook runs on committed data. Rebuilding needs the raw files copied into `data/raw/` | §21, §22 |
| Notebook with explained decisions *(bonus)* | Executed walkthrough that imports the production code | [`notebooks/datavortex_walkthrough.ipynb`](notebooks/datavortex_walkthrough.ipynb) |
| Interactivity & UX | Sidebar filters, tabs, explorers, selectors, drill-down tables | Dashboard |
| Core KPIs | Revenue and 16 other KPIs, independently validated; Risk Indicator Score. Churn does not apply — the data has no customer lifecycle field | §10, §11, [`docs/kpi_validation_report.md`](docs/kpi_validation_report.md) |
| Storytelling | Findings generated from the build; rescue → identity → attribution → hypotheses → risk → investigator | Dashboard page 1, §17 |
| Innovative dashboard approaches *(bonus)* | Coverage badges, confidence-interval rankings, generated significance caveats, hypothesis register, structural fraud-ring evidence, forecast bands, in-app test matrix | Pages 2–7 |
| Code elegance & architecture | Modular cleaning, semantic registry, query engine, one-way data flow, 387 tests. No SQL views: Parquet aggregates are the view layer | §6, §20, §23 |
| Advanced insights | Identity collisions, attribution tests, random-join detection, hypothesis testing, backtested forecasting. Clustering considered and rejected with evidence | §8–§14 |
| AI: natural-language understanding *(bonus)* | **Partly met.** Rule-based planner over the registry, 66/66 evaluation questions; not an LLM, so phrasing outside the synonym set may fail | §16, [`docs/agent_test_matrix.md`](docs/agent_test_matrix.md) |
| AI: chart selection *(bonus)* | Rule-based: line for time series, bars for comparisons, tables for records; no scatter intent | §16 |
| AI: text summary *(bonus)* | Headline, explanation, caveats and provenance with every answer | §16 |

Track 1 bonus query — *"Which merchant category has the highest chargeback-to-transaction ratio
this quarter?"* — is answered with Transport at 13.36%, a confidence-interval chart, and the
caveat that category differences are not statistically significant (χ² = 5.24, p = 0.919).

---

## 25. Team

| Name | Registration No. | Role / Contributions |
|---|---|---|
| Rishuraj Kumar | 12400878 | Project development, data cleaning & forensic audit, analytics, dashboard and AI Investigator |
| Amit Kumar | 12400871 | Project development, analytics, dashboard and testing |
| Satyam Saurabh | 12310274 | Project development, data analysis, testing and presentation |

---

## 26. Future Improvements

- **An optional LLM intent parser** that produces the same validated intent object, keeping the
  rule-based planner as the fallback and keeping all computation in the query engine.
- **A relationship intent** so questions about how two measures relate can be drawn as scatter
  plots.
- **More dimensions for complaints**, such as day of week of reporting.
- **A friendlier pipeline message** when raw files are missing, instead of a traceback.
- **Range checks instead of pinned row counts**, so the pipeline can accept a new data drop.
- **A column-level dictionary for the analytics tables**, not only the star schema.
- **A columnar engine (for example DuckDB) and benchmarks** if the data grew to bank scale.
- **Validation of the Risk Indicator Score** if confirmed fraud outcomes ever became available.
