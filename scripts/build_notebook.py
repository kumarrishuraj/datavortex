"""Build and execute notebooks/datavortex_walkthrough.ipynb.

    python scripts/build_notebook.py

The notebook is a reading guide to the evidence, not a second implementation.
Every cell either reads a Parquet table the pipeline wrote or calls a production
function, so it cannot drift from the dashboard. It needs only data/processed/,
not the raw files, and it never displays a customer-level row. Outputs are
embedded so the notebook can be read on GitHub without running it.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "datavortex_walkthrough.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


CELLS = [
    md("""
# DataVortex — analysis walkthrough

**Track 1 · UPI Fraud Ring & Merchant Analytics**

This notebook walks through the evidence behind the dashboard: what arrived, what the
cleaning pipeline changed, two statistical tests that decide how rankings may be read,
the hypothesis register, and the forecast backtest.

It reads the Parquet tables written by `scripts/run_pipeline.py` and
`scripts/build_analytics.py` and calls the same modules the dashboard uses. Nothing
here re-implements a metric. Rebuild it with `python scripts/build_notebook.py`.

1. Data audit
2. Cleaning evidence
3. Statistical analysis A: do merchant categories differ in dispute rate?
4. Statistical analysis B: do individual merchants differ?
5. Hypothesis register
6. Forecast validation
7. Key findings and limitations
"""),
    code("""
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():          # opened from inside notebooks/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from IPython.display import display

from src.analytics import query_engine as QE
from src.analytics.shrinkage import MIN_DENOMINATOR, estimate_prior_strength
from src.analytics.significance import homogeneity_test
from src.config import PROCESSED_DIR

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 160)

# Read the validated Parquet layer the pipeline wrote, so this walkthrough and the
# dashboard resolve every figure from the same canonical facts and dimensions.
star = {p.stem: pd.read_parquet(p) for p in PROCESSED_DIR.glob("*.parquet")}
agg = {p.stem: pd.read_parquet(p) for p in (PROCESSED_DIR / "analytics").glob("*.parquet")}
print(f"{len(star)} star-schema tables and {len(agg)} analytics tables loaded")
"""),
    md("""
## 1. Data audit

### Row counts, raw to clean

Only byte-identical duplicate rows are removed. KYC and merchant records reduce to one row
per key only after every candidate row is kept in the identity bridge table.
"""),
    code("""
# Raw-to-clean row counts. The reduction is deduplication and identity resolution only:
# no row is dropped for failing to match another table.
star["recon_rows"]
"""),
    md("""
### Are the documented joins real?

Two ID columns drawn independently from the same ID space still overlap a lot by chance.
The test compares observed overlap with the overlap expected under independence. A z-score
near zero means the join cannot be told apart from random.
"""),
    code("""
# This test decides how a dispute may be attributed: a z-score near zero means the
# documented join is indistinguishable from random overlap and cannot carry attribution.
star["recon_independence"][["pair", "expected_overlap", "observed_overlap", "z_score", "verdict"]]
"""),
    md("""
### Forensic facts measured by the pipeline

These are the values the dashboard quotes in its text. No page types them in by hand.
"""),
    code("""
# Each claim the dashboard makes in prose is measured here first, so the narrative and
# the data cannot drift apart when the pipeline is rerun.
star["audit_facts"][["fact", "value", "unit", "description"]]
"""),
    md("""
## 2. Cleaning evidence

Damaged values are repaired where the repair is unambiguous and flagged either way. The
flags stay on the rows, so every exclusion in an analysis can be counted.
"""),
    code("""
# Defects are recorded as boolean flag columns instead of deleted rows, so an analysis can
# exclude a class of records and still report exactly how many it excluded.
def flag_summary(name, frame):
    flags = frame.select_dtypes(include=["bool", "boolean"])
    return pd.DataFrame({
        "table": name,
        "flag": flags.columns,
        "rows_flagged": flags.sum().astype(int).values,
        "pct_of_rows": (100 * flags.mean()).astype(float).round(2).values,
    })

tx, cb = star["fact_transactions"], star["fact_chargebacks"]
pd.concat([flag_summary("fact_transactions", tx), flag_summary("fact_chargebacks", cb)],
          ignore_index=True)
"""),
    code("""
# The treatment column is the decision taken for each defect class: repaired and flagged,
# flagged only, or routed to UNKNOWN and retained. Nothing here was deleted.
agg["agg_data_quality"]
"""),
    code("""
# Repeated IDs are reported as candidates rather than merged here, because a shared
# identifier can represent different real-world entities: collapsing on the key would
# destroy one of them and fabricate a record that was never in the source.
users, merchants = star["dim_users"], star["dim_merchants"]
bridge = star["bridge_identity_collision"]
print(f"customer IDs shared by different people:      {int(users['identity_ambiguous'].sum()):,}")
print(f"merchant IDs shared by different businesses:  {int(merchants['identity_ambiguous'].sum()):,}")
print(f"candidate rows preserved in the bridge table: {len(bridge):,}")
"""),
    md("""
## Cleaning decisions and why

One rule governs the pipeline: **repair when the repair is unambiguous; otherwise flag and
retain**. Cleaning is built to preserve auditability, not to raise join coverage by deleting
rows that do not match. The tables above are the evidence; below is the reasoning.

### Duplicate handling
**Decision:** remove only byte-identical duplicate rows — 400 transactions, 84 complaints,
278 KYC and 12 merchant rows. Repeated KYC and merchant IDs are never collapsed on the ID.

**Why:** an identical row carries no information the original lacks, but two rows sharing an
ID may be two different entities; collapsing on the key would destroy one of them.

**Consequence:** every candidate row survives in `BRIDGE_IDENTITY_COLLISION` and stays
inspectable.

### ID normalization
**Decision:** normalise identifiers to their canonical form; refuse and flag a value that
cannot be repaired safely rather than truncating it to fit (`src/cleaning/ids.py`).

**Why:** a value with more significant digits than the canonical width belongs to a
different ID space, so truncating it would invent a join to the wrong entity.

**Consequence:** malformed identifiers remain visible as flags instead of becoming
confident false matches.

### Timestamp parsing
**Decision:** apply the separator rule established in `docs/01_forensic_audit.md` and flag
whatever will not parse (`src/cleaning/timestamps.py`).

**Why:** the rule is measured, not assumed — 18,385 slash-dated rows have a first part above
12 and **0** have a second part above 12, while hyphen-dated rows show exactly the reverse.
Zero counterexamples justify the interpretation; anything outside it is not guessed.

**Consequence:** 1,000 date-only timestamps are excluded from hourly analysis and 412
complaint delays are marked unknown rather than invented.

### Amount sign repair
**Decision:** treat a negative amount as a corrupted sign, repair it to magnitude and flag
it (`src/cleaning/amounts.py`). It is not read as a refund.

**Why:** the 429 raw negative amounts have **0** matching positive twins, and the negative
rate is near-identical across success, failed and pending. A refund would leave a paired
original; these do not, so the sign is damage rather than meaning.

**Consequence:** 420 rows in the cleaned fact table carry the sign-repair flag, so any
analysis can exclude them and say how many it excluded.

### KYC / PAN repair
**Decision:** repair only unambiguous OCR substitutions; leave truncated values unrepaired
(`src/cleaning/kyc.py`).

**Why:** a deterministic substitution (0↔O, 1↔I, 2↔Z, 5↔S, 8↔B) that yields a valid PAN can
be justified, but a truncated PAN is missing characters outright and inventing them would
create an identity that is not in the source.

**Consequence:** 687 PANs repaired and 1,454 left unrepaired and flagged, both readable in
`pan_status` on `DIM_USERS`.
"""),
    md("""
### Identity collision handling
**Decision:** never merge records only because `user_id` or `merchant_id` repeats.

**Why:** 5,341 customer IDs and 1,310 merchant IDs are shared by genuinely different
entities, corroborated by 4,422 IDs carrying more than one distinct valid PAN while **0**
PANs are shared across IDs. The identifier is ambiguous; the person is not.

**Consequence:** one survivor per key in the dimension, every other candidate preserved in
the bridge table, and the ID flagged low-confidence wherever it is used.

### UNKNOWN routing
**Decision:** route unmatched fact rows to an explicit UNKNOWN dimension member instead of
dropping them (`src/transformation/star_schema.py`).

**Why:** dropping them would shrink every total and make coverage look better than it is.
Two of the three documented joins are indistinguishable from random overlap, so the
unmatched share is large and has to stay visible.

**Consequence:** 13,522 transactions without a KYC match, 10,369 without a merchant-master
match and 193 unlinked complaints stay in the totals and are reported as coverage.

### Chargeback attribution
**Decision:** attribute a complaint to its transaction through `txn_id`, taking the customer
and merchant from that transaction — never from the complaint's own `user_id` or
`merchant_id` (`src/cleaning/chargebacks.py`).

**Why:** only the complaint-to-transaction join survives the independence test above. The
complaint's own `user_id` agrees with its linked transaction in **0 of 2,607** cases and its
`merchant_id` in **0 of 2,607**. Joining on those columns would attribute disputes to the
wrong customers and merchants with complete confidence.

**Consequence:** 2,607 complaints are attributed through the validated path and 193 stay
unlinked rather than being forced onto a transaction.

### Why records are retained rather than dropped
Every decision above resolves the same way: what cannot be repaired is flagged and kept. The
flags are ordinary columns, so any analysis can exclude a class of rows and state how many it
excluded — see `docs/data_quality_report.md` and `docs/kpi_validation_report.md`.
"""),
    md("""
## 3. Statistical analysis A — do merchant categories differ in dispute rate?

A dashboard can always sort categories by rate. The question is whether the ordering is
real. A chi-square test of homogeneity compares every category's disputed-transaction
count with what one shared rate would produce. The UNKNOWN group (transactions with no
merchant master record) stays in the test, as it does on the dashboard.
"""),
    code("""
# The grain is distinct disputed transactions, not complaints: 151 transactions carry more
# than one complaint, and counting complaints would treat them as independent events and
# break the binomial assumption the test rests on. UNKNOWN stays in so unmatched rows
# remain visible in the denominator instead of being silently removed.
cat = agg["agg_category"]
category_test = homogeneity_test(cat["disputed_transactions"], cat["transactions"])
display(cat[["category", "transactions", "disputed_transactions", "rate_pct",
             "ci_low_pct", "ci_high_pct", "p_adjusted", "significant"]])
category_test
"""),
    md("""
Cross-check: the semantic query engine the AI Investigator uses computes the same
numerators and denominators as the materialized aggregate.
"""),
    code("""
# Asserted rather than eyeballed: if the notebook and the AI Investigator's query engine
# ever disagreed on a numerator, this cell fails instead of quietly diverging.
frames = QE.prepare_frames(star)
engine = QE.run(QE.QuerySpec(metric="chargeback_to_transaction_ratio",
                             dimension="merchant_category"), frames).frame
check = (engine.set_index("dimension")[["numerator", "denominator"]]
               .join(cat.set_index("category")[["disputed_transactions", "transactions"]]))
assert (check["numerator"] == check["disputed_transactions"]).all()
assert (check["denominator"] == check["transactions"]).all()
print(f"query engine and aggregate agree for all {len(check)} categories")
"""),
    md("""
## 4. Statistical analysis B — do individual merchants differ?

If merchants had different underlying dispute propensities, their observed rates would
spread out more than binomial sampling alone allows. The overdispersion statistic
(chi-square divided by degrees of freedom) sits near 1 when there is no extra spread.
Only merchants above the denominator floor are tested.
"""),
    code("""
# Only merchants above the denominator floor are tested: a 100% rate on a single
# transaction is sampling noise, and leaving those rows in would dominate the statistic
# and manufacture a merchant ranking out of thin denominators.
mer = agg["agg_merchant"]
eligible = mer[~mer["below_floor"]]
observed = estimate_prior_strength(eligible["disputed_transactions"], eligible["transactions"])
observed_ratio = observed["chi_square"] / observed["df"]
print(f"merchants with at least {MIN_DENOMINATOR} transactions: {len(eligible):,}")
print(f"chi-square / df = {observed_ratio:.4f}, p = {observed['p_value']:.3f}, "
      f"overdispersed = {observed['overdispersed']}")
"""),
    md("""
Calibration by simulation: give every merchant the same dispute rate, keep each merchant's
real transaction count, and recompute the statistic 200 times. If the observed value sits
inside that range, merchant-level differences cannot be told apart from noise.
"""),
    code("""
# Calibration: simulate one shared dispute rate at the real transaction counts, so the
# observed statistic is judged against noise this dataset could actually produce. The seed
# is fixed so the published interval is reproducible.
rng = np.random.default_rng(2026)
n = eligible["transactions"].to_numpy()
shared_rate = observed["mean_rate"]
simulated = []
for _ in range(200):
    draw = estimate_prior_strength(pd.Series(rng.binomial(n, shared_rate)), pd.Series(n))
    simulated.append(draw["chi_square"] / draw["df"])
simulated = np.array(simulated)
low, high = np.quantile(simulated, [0.025, 0.975])
print(f"observed chi-square / df:               {observed_ratio:.4f}")
print(f"one shared rate, middle 95% of 200 runs: {low:.4f} to {high:.4f}")
print(f"simulations at or above observed:        {(simulated >= observed_ratio).mean():.0%}")
print(f"raw rate spread {eligible['dispute_rate_raw_pct'].std():.2f} pp -> "
      f"after shrinkage {eligible['dispute_rate_shrunk_pct'].std():.2f} pp")
"""),
    md("""
## 5. Hypothesis register

Each insight suggested by the dataset notes, plus the team's own checks, with the test
used and the verdict. CONTRADICTED means the data points the other way.
"""),
    code("""
# NOT SUPPORTED and CONTRADICTED verdicts are kept in the register and shown: a hypothesis
# the data refused is a result, not a gap to quietly drop from the report.
agg["agg_hypothesis_register"][["hypothesis", "source", "test", "statistic", "p_value", "verdict"]]
"""),
    md("""
## 6. Forecast validation

Four simple methods are fitted on the earlier weeks and scored on a holdout they never saw.
The lowest holdout error is selected. The last column shows how often the 95% band
contained the actual holdout value.
"""),
    code("""agg["agg_forecast_backtest"]"""),
    code("""
# Methods are scored on a holdout they never saw rather than on in-sample fit, so a method
# is selected only if it beat the alternatives on unseen days.
agg["agg_forecast_diagnostics"][["label", "history_days", "holdout_days", "selected_method_label",
                                 "selected_mae", "mean_baseline_mae", "slope_per_day", "slope_p",
                                 "weekday_p", "holdout_band_coverage_pct"]]
"""),
    md("""
## 7. Key findings

Generated from the tables above, so the sentences change if the data changes.
"""),
    code("""
# The findings are generated from the tables above rather than typed by hand, so the
# wording cannot drift from the data if the pipeline is rerun.
register = agg["agg_hypothesis_register"]
notes = register[register["source"] == "track1_dataset_notes.txt"]
z = star["recon_independence"].set_index("pair")["z_score"]
verdicts = ", ".join(f"{count} {verdict}" for verdict, count in notes["verdict"].value_counts().items())
findings = [
    f"The transaction-to-KYC and transaction-to-merchant joins overlap no more than chance "
    f"(z = {z['tx.user x kyc.user']:.2f} and {z['tx.merchant x mer.merchant']:.2f}); only the "
    f"complaint-to-transaction join is real (z = {z['cb.txn x tx.txn']:.0f}).",
    f"{int(users['identity_ambiguous'].sum()):,} customer IDs and "
    f"{int(merchants['identity_ambiguous'].sum()):,} merchant IDs each belong to more than one entity.",
    f"Category dispute rates do not differ beyond chance (chi-square {category_test['chi_square']:.2f}, "
    f"df {category_test['df']}, p = {category_test['p_value']:.3f}).",
    f"Merchants show no dispute propensity beyond binomial noise "
    f"(chi-square/df {observed_ratio:.3f}, p = {observed['p_value']:.3f}).",
    f"Of the {len(notes)} insights suggested by the dataset notes: {verdicts}.",
]
for number, text in enumerate(findings, 1):
    print(f"{number}. {text}")
"""),
    md("""
### Limitations

- The data has no fraud label, so no supervised fraud model can be trained or validated.
  The Risk Indicator Score is a transparent review-priority total, not a fraud probability.
- Two of the documented joins behave like random overlap, so any breakdown by customer or
  merchant attributes covers only the rows that resolve. Each chart states its coverage.
- The forecast rests on one quarter of daily history. It can show that a series has no
  trend or weekly pattern. It cannot capture seasonality longer than the data.
- Clustering was tried and dropped: the transaction graph is a forest with no repeated
  customer-merchant pairs, so there is no structure for a clustering method to recover.
"""),
]


def main() -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote and executed {OUT.relative_to(ROOT)} ({len(nb.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
