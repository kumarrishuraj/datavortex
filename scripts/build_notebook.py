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
    code("""star["recon_rows"]"""),
    md("""
### Are the documented joins real?

Two ID columns drawn independently from the same ID space still overlap a lot by chance.
The test compares observed overlap with the overlap expected under independence. A z-score
near zero means the join cannot be told apart from random.
"""),
    code("""star["recon_independence"][["pair", "expected_overlap", "observed_overlap", "z_score", "verdict"]]"""),
    md("""
### Forensic facts measured by the pipeline

These are the values the dashboard quotes in its text. No page types them in by hand.
"""),
    code("""star["audit_facts"][["fact", "value", "unit", "description"]]"""),
    md("""
## 2. Cleaning evidence

Damaged values are repaired where the repair is unambiguous and flagged either way. The
flags stay on the rows, so every exclusion in an analysis can be counted.
"""),
    code("""
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
    code("""agg["agg_data_quality"]"""),
    code("""
users, merchants = star["dim_users"], star["dim_merchants"]
bridge = star["bridge_identity_collision"]
print(f"customer IDs shared by different people:      {int(users['identity_ambiguous'].sum()):,}")
print(f"merchant IDs shared by different businesses:  {int(merchants['identity_ambiguous'].sum()):,}")
print(f"candidate rows preserved in the bridge table: {len(bridge):,}")
"""),
    md("""
## 3. Statistical analysis A — do merchant categories differ in dispute rate?

A dashboard can always sort categories by rate. The question is whether the ordering is
real. A chi-square test of homogeneity compares every category's disputed-transaction
count with what one shared rate would produce. The UNKNOWN group (transactions with no
merchant master record) stays in the test, as it does on the dashboard.
"""),
    code("""
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
    code("""agg["agg_hypothesis_register"][["hypothesis", "source", "test", "statistic", "p_value", "verdict"]]"""),
    md("""
## 6. Forecast validation

Four simple methods are fitted on the earlier weeks and scored on a holdout they never saw.
The lowest holdout error is selected. The last column shows how often the 95% band
contained the actual holdout value.
"""),
    code("""agg["agg_forecast_backtest"]"""),
    code("""
agg["agg_forecast_diagnostics"][["label", "history_days", "holdout_days", "selected_method_label",
                                 "selected_mae", "mean_baseline_mae", "slope_per_day", "slope_p",
                                 "weekday_p", "holdout_band_coverage_pct"]]
"""),
    md("""
## 7. Key findings

Generated from the tables above, so the sentences change if the data changes.
"""),
    code("""
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
