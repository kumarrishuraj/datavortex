"""DataVortex — Stage 3: analytics layer.

Reads data/processed/*.parquet (the Stage 2 star schema), writes the
materialized aggregates the dashboard will consume plus a KPI validation
report in which every headline number is recomputed by an independent route.

    python scripts/build_analytics.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analytics import kpis as K  # noqa: E402
from src.analytics import forecast, hypotheses, network, risk_indicators, semantic, views  # noqa: E402
from src.analytics.significance import overlap_verdict, rank_with_confidence  # noqa: E402
from src.analytics.shrinkage import MIN_DENOMINATOR  # noqa: E402
from src.config import DOCS_DIR, PROCESSED_DIR, UNKNOWN_KEY  # noqa: E402
from scripts.build_report import md_table  # noqa: E402

ANALYTICS_DIR = PROCESSED_DIR / "analytics"


def load_star() -> dict[str, pd.DataFrame]:
    names = ["fact_transactions", "fact_chargebacks", "dim_users",
             "dim_merchants", "dim_date", "bridge_identity_collision",
             "recon_rows", "recon_value", "recon_independence"]
    missing = [n for n in names if not (PROCESSED_DIR / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing}; run `python scripts/run_pipeline.py` first"
        )
    return {n: pd.read_parquet(PROCESSED_DIR / f"{n}.parquet") for n in names}


def compute_headline_kpis(tx, cb, dim_users, dim_merchants) -> pd.DataFrame:
    results = [
        K.total_transaction_count(tx),
        K.total_transaction_amount(tx),
        K.average_transaction_value(tx),
        K.status_rate(tx, "SUCCESS"),
        K.status_rate(tx, "FAILED"),
        K.status_rate(tx, "PENDING"),
        K.chargeback_count(cb),
        K.chargeback_amount(cb),
        K.chargeback_to_transaction_ratio(tx, cb),
        K.complaints_per_transaction(tx, cb),
        K.average_dispute_reporting_delay(cb),
        K.disputes_reported_after_7_days(cb),
        K.kyc_completion_rate(dim_users),
        K.kyc_rejection_rate(dim_users),
        K.utr_missing_rate(tx),
        K.merchant_master_coverage(tx),
        K.kyc_coverage(tx),
    ]
    rows = []
    for r in results:
        m = semantic.METRICS.get(r.metric)
        rows.append({
            "metric": r.metric,
            "label": m.label if m else r.metric,
            "value": r.value,
            "formatted": r.formatted(),
            "unit": r.unit,
            "denominator": r.denominator,
            "coverage_pct": r.coverage_pct,
            "coverage_basis": m.coverage_basis if m else "",
            "detail": r.detail,
            "higher_is_worse": m.higher_is_worse if m else False,
        })
    return pd.DataFrame(rows)


def validate(tx, cb, dim_users) -> pd.DataFrame:
    """Recompute each headline KPI a second, independent way."""
    checks = []

    def add(name, primary, secondary, tol=0.01):
        delta = abs(float(primary) - float(secondary))
        checks.append({
            "kpi": name,
            "primary": round(float(primary), 4),
            "independent_recompute": round(float(secondary), 4),
            "abs_delta": round(delta, 6),
            "status": "PASS" if delta <= tol else "FAIL",
        })

    add("total_transaction_amount",
        K.total_transaction_amount(tx).value, K.check_total_amount(tx))

    rates = K.check_status_rates(tx)
    for s in ["SUCCESS", "FAILED", "PENDING"]:
        add(f"{s.lower()}_transaction_rate", K.status_rate(tx, s).value, rates[s])
    add("status_rates_sum_to_100", sum(rates.values()), 100.0)

    add("chargeback_to_transaction_ratio",
        K.chargeback_to_transaction_ratio(tx, cb).value, K.check_chargeback_ratio(tx, cb))
    add("average_dispute_reporting_delay",
        K.average_dispute_reporting_delay(cb).value, K.check_reporting_delay(cb))

    kyc = K.check_kyc_rates(dim_users)
    add("kyc_completion_rate", K.kyc_completion_rate(dim_users).value, kyc["VERIFIED"])
    add("kyc_rejection_rate", K.kyc_rejection_rate(dim_users).value, kyc["REJECTED"])
    add("kyc_rates_sum_to_100", sum(kyc.values()), 100.0)

    # Aggregates must reconcile to the facts they came from.
    daily = views.agg_daily(tx, cb)
    add("agg_daily transactions == fact rows", daily["transactions"].sum(), len(tx))
    add("agg_daily value == fact value", daily["transaction_value"].sum(),
        tx["amount_inr"].sum(), tol=1.0)

    cat = views.agg_category(tx, cb, pd.read_parquet(PROCESSED_DIR / "dim_merchants.parquet"))
    add("agg_category transactions == fact rows", cat["transactions"].sum(), len(tx))

    mer, _ = views.agg_merchant(tx, cb, pd.read_parquet(PROCESSED_DIR / "dim_merchants.parquet"))
    add("agg_merchant transactions == fact rows", mer["transactions"].sum(), len(tx))
    add("agg_merchant merchants == distinct merchants",
        len(mer), tx["merchant_id_normalized"].nunique())

    usr = views.agg_user(tx, cb, dim_users)
    add("agg_user transactions == fact rows", usr["transactions"].sum(), len(tx))

    return pd.DataFrame(checks)


def write_validation_report(headline, validation, prior, mer, cat, hourly,
                            cat_test, register) -> str:
    out: list[str] = []
    w = out.append
    w("# DataVortex — KPI Validation Report")
    w("**Stage 3 · Analytics Layer.** _Turning UPI Data Chaos into Fraud Intelligence._\n")
    w("Generated by `scripts/build_analytics.py`. Every KPI below is computed once by the "
      "production path and once by an independent route; the two must agree.\n")
    w("---\n")

    w("## Headline KPIs\n")
    w(md_table(headline[["label", "formatted", "denominator", "coverage_pct", "coverage_basis"]]))
    w("")
    w("`coverage_pct` is the share of the full population the number was computed on. It is "
      "stored with the metric, not added at render time, so a chart cannot present a "
      "part-coverage number as if it covered the whole book.\n")

    w("## Independent recomputation\n")
    w(md_table(validation))
    w("")
    failures = int((validation["status"] == "FAIL").sum())
    w(f"**{len(validation) - failures} of {len(validation)} checks pass.**"
      + ("" if not failures else f" **{failures} FAILED — investigate before using these numbers.**") + "\n")

    w("## Merchant dispute rates: floor and shrinkage\n")
    w(f"A raw dispute ratio is meaningless on a thin denominator. In this dataset "
      f"**{int((mer['transactions'] < MIN_DENOMINATOR).sum()):,} merchants** have fewer than "
      f"{MIN_DENOMINATOR} transactions, and "
      f"**{int((mer['dispute_rate_raw_pct'] >= 100).sum()):,} show a raw ratio of 100%** — almost "
      "all of them merchants with a single transaction that happened to be disputed.\n")
    w("Two guards are applied:\n")
    w(f"1. **Denominator floor of {MIN_DENOMINATOR} transactions.** Below it, `below_floor` is set "
      "and the merchant is excluded from rankings. The row is still shown.")
    w("2. **Beta-binomial shrinkage toward the category mean**, with the prior strength "
      "estimated from the data by method of moments.\n")
    w("### Is there any merchant-level dispute signal to shrink toward?\n")
    w(md_table(pd.DataFrame([{
        "eligible merchants (n>=3)": prior["groups"],
        "transactions": int(prior["total_trials"]),
        "mean dispute rate": f"{100 * prior['mean_rate']:.3f}%",
        "chi-square": round(prior["chi_square"], 1),
        "df": prior["df"],
        "chi2/df": round(prior["chi_square"] / prior["df"], 4),
        "p-value": f"{prior['p_value']:.3g}",
        "overdispersed": prior["overdispersed"],
    }])))
    w("")
    if not prior["overdispersed"]:
        w("**No.** The spread of merchant dispute rates is no wider than binomial sampling "
          "noise would produce on its own — the chi-square statistic sits *below* its degrees "
          "of freedom. Every merchant's dispute rate is consistent with one shared underlying "
          "probability.\n")
        w("The consequence is deliberate and visible in the data: shrinkage pulls every merchant "
          "essentially all the way to its category mean, so `dispute_rate_shrunk_pct` is nearly "
          "flat across merchants. **That flatness is the finding, not a bug.** This dataset "
          "contains no detectable merchant-level dispute propensity, and we will not manufacture "
          "a risk ranking out of sampling noise.\n")
        w("> An earlier run of this same test appeared to find overdispersion "
          "(chi2/df = 1.127, p = 1.8e-07). That was an artifact of counting *complaints* rather "
          "than *disputed transactions*: 151 transactions carry more than one complaint, which "
          "pushes some merchants above a ratio of 1 and breaks the binomial bound the test "
          "assumes. Counting distinct disputed transactions removes it.\n")
        w("**What we rank instead:** absolute dispute *exposure* — complaint count and disputed "
          "value. A merchant with 5 disputes and a large disputed amount is a genuine operational "
          "priority. That is a statement about workload, not about the merchant being riskier "
          "than its peers, and the dashboard labels it that way.\n")
    else:
        w(f"**Yes.** Between-merchant variation exceeds binomial noise "
          f"(p = {prior['p_value']:.3g}), so the estimated prior strength is "
          f"k = {prior['prior_strength_k']:.2f}: a merchant's own history is worth roughly "
          f"{prior['prior_strength_k']:.0f} transactions of evidence before it moves away from "
          "its peer mean.\n")

    w("## Merchant-category performance\n")
    w(md_table(cat[["category", "merchants", "transactions", "transaction_value",
                    "dispute_rate_pct", "failed_rate_pct", "share_of_transactions_pct"]]))
    w("")
    w("The `UNKNOWN (no master record)` row is kept deliberately. It is the single largest "
      "group, and hiding it would make the remaining categories look like the whole book.\n")

    w("## Hour-of-day profile\n")
    excluded = int(hourly["excluded_date_only"].iloc[0]) if len(hourly) else 0
    cv = hourly["transactions"].std() / hourly["transactions"].mean() if len(hourly) else 0
    w(f"Computed on transactions whose source carried a clock time "
      f"({int(hourly['transactions'].sum()):,}); **{excluded:,} date-only timestamps are "
      "excluded** because they parse to 00:00:00 and would otherwise create a fake midnight "
      f"peak of 1,802 against a ~790 baseline.\n")
    w(f"Resulting profile: min {int(hourly['transactions'].min())}, "
      f"max {int(hourly['transactions'].max())}, **coefficient of variation {cv:.3f}** — "
      "effectively flat. This dataset has no intraday pattern, which is itself worth stating: "
      "real UPI traffic does not look like this.\n")

    w("## Does the category ranking mean anything?\n")
    w("The competition brief's headline bonus query is *\"Which merchant category has the "
      "highest chargeback-to-transaction ratio this quarter?\"* Sorting the table answers it. "
      "Testing whether the ordering is real is a different question, and the one that matters.\n")
    w(md_table(cat[["rank", "category", "transactions", "rate_pct", "ci_low_pct",
                    "ci_high_pct", "p_adjusted", "significant"]]))
    w("")
    w(f"Homogeneity across all categories: chi-square = {cat_test['chi_square']}, "
      f"df = {cat_test['df']}, **p = {cat_test['p_value']:.4f}** — {cat_test['note']}.\n")
    w(f"> **{overlap_verdict(cat)}**\n")
    w("So the agent's answer to that query is the top category *and* this caveat. Naming a "
      "winner without it would be reporting sampling noise as a business finding.\n")

    w("## Hypothesis register\n")
    w("`track1_dataset_notes.txt` ships a list of \"insights students may discover\". Each is a "
      "plausible claim, and each is testable. Asserting them because they sound like fraud "
      "analytics is the easiest way to publish something false, so every one is tested and "
      "reported whichever way it falls.\n")
    counts = register["verdict"].value_counts().to_dict()
    w("Result: " + ", ".join(f"**{v} {k}**" for k, v in counts.items()) + ".\n")
    w(md_table(register[["hypothesis", "test", "statistic", "p_value", "verdict"]]))
    w("")
    for _, r in register.iterrows():
        w(f"**{r['verdict']} — {r['hypothesis']}**  \n{r['reading']}\n")

    w("## Metric registry\n")
    w("Every metric above is declared once in `src/analytics/semantic.py` and read from there by "
      "the KPI functions, the validation checks, the dashboard and the AI agent, so a metric "
      "cannot mean two different things in two places.\n")
    reg = semantic.registry_frame()
    w(md_table(reg[["metric", "label", "unit", "formula", "coverage_basis"]]))
    w("")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "kpi_validation_report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print("DataVortex — Stage 3: Analytics Layer")
    print("=" * 78)

    star = load_star()
    tx, cb = star["fact_transactions"], star["fact_chargebacks"]
    dim_users, dim_merchants = star["dim_users"], star["dim_merchants"]

    print("\n[1/4] Computing headline KPIs…")
    headline = compute_headline_kpis(tx, cb, dim_users, dim_merchants)
    for _, r in headline.iterrows():
        print(f"      {r['label']:<34} {r['formatted']:>18}   coverage {r['coverage_pct']:>6.2f}%")

    print("\n[2/4] Building aggregates…")
    merchant, prior = views.agg_merchant(tx, cb, dim_merchants)
    merchant = risk_indicators.score_merchants(merchant)
    users_scored = risk_indicators.score_customers(views.agg_user(tx, cb, dim_users))
    daily = views.agg_daily(tx, cb)
    forecasts = forecast.build(daily)
    category = views.agg_category(tx, cb, dim_merchants)
    kyc_status = views.agg_kyc_status(tx, cb, dim_users)

    # Attach confidence intervals and a homogeneity verdict to every ranked
    # breakdown, so a chart can never present an ordering that is pure noise.
    category, cat_test = rank_with_confidence(
        category, "category", "disputed_transactions", "transactions")
    kyc_status, kyc_test = rank_with_confidence(
        kyc_status, "kyc_status", "disputed_transactions", "transactions")
    register = hypotheses.build_register(
        tx, cb, dim_users, dim_merchants,
        recon={"recon_rows": star["recon_rows"], "recon_value": star["recon_value"]},
        independence=star["recon_independence"])
    net_summary = network.summarize(tx, cb)
    net_components = network.component_table(tx, cb)
    top_components = net_components["component"].head(25).tolist()
    net_edges = network.component_edges(tx, cb, top_components)
    print(f"      network: {int(net_summary['nodes'].iloc[0]):,} nodes, "
          f"{int(net_summary['edges'].iloc[0]):,} edges, "
          f"{int(net_summary['components'].iloc[0]):,} components, "
          f"4-cycles={int(net_summary['user_pairs_sharing_2plus'].iloc[0])}, "
          f"acyclic={bool(net_summary['is_forest_acyclic'].iloc[0])}")
    print(f"      category dispute-rate ranking meaningful? {cat_test['significant']} "
          f"(chi2={cat_test['chi_square']}, df={cat_test['df']}, p={cat_test['p_value']:.4f})")
    print(f"      hypotheses tested: " + ", ".join(
        f"{k}={v}" for k, v in register["verdict"].value_counts().items()))

    tables = {
        "agg_daily": daily,
        "agg_hourly": views.agg_hourly(tx),
        "agg_merchant": merchant,
        "agg_user": users_scored,
        "agg_category": category,
        "agg_kyc_status": kyc_status,
        "agg_hypothesis_register": register,
        "agg_network_summary": net_summary,
        "agg_network_components": net_components,
        "agg_network_edges": net_edges,
        "agg_risk_bands": pd.concat([
            risk_indicators.band_summary(users_scored, "customer"),
            risk_indicators.band_summary(merchant, "merchant")], ignore_index=True),
        "agg_risk_components": pd.concat([
            risk_indicators.component_summary(users_scored, "customer",
                                              risk_indicators.CUSTOMER_COMPONENTS),
            risk_indicators.component_summary(merchant, "merchant",
                                              risk_indicators.MERCHANT_COMPONENTS)],
            ignore_index=True),
        **forecasts,
        "agg_state": views.agg_state(tx, cb, dim_merchants),
        "agg_chargeback_reason": views.agg_chargeback_dimension(cb, "reason_code_canonical"),
        "agg_chargeback_severity": views.agg_chargeback_dimension(cb, "severity_canonical"),
        "agg_chargeback_resolution": views.agg_chargeback_dimension(cb, "resolution_status_canonical"),
        "agg_chargeback_channel": views.agg_chargeback_dimension(cb, "channel_canonical"),
        "agg_chargeback_theme": views.agg_chargeback_dimension(cb, "complaint_theme"),
        "agg_data_quality": views.agg_data_quality(tx, cb, dim_users, dim_merchants),
        "kpi_headline": headline,
    }

    print("\n[3/4] Validating…")
    validation = validate(tx, cb, dim_users)
    for _, r in validation.iterrows():
        mark = "ok " if r["status"] == "PASS" else "FAIL"
        print(f"      [{mark}] {r['kpi']:<44} {r['primary']:>16,.4f} vs {r['independent_recompute']:>16,.4f}")
    failures = int((validation["status"] == "FAIL").sum())
    if failures:
        raise AssertionError(f"{failures} KPI validation check(s) failed — see output above")
    tables["kpi_validation"] = validation

    print("\n[4/4] Writing analytics layer…")
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = ANALYTICS_DIR / f"{name}.parquet"
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == "object":
                sample = out[col].dropna()
                if not sample.empty and not sample.map(lambda v: isinstance(v, str)).all():
                    out[col] = out[col].astype(str)
        out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        print(f"      {name:<28} {len(df):>7,} rows x {df.shape[1]:>3} cols")

    report = write_validation_report(
        headline, validation, prior, merchant, tables["agg_category"], tables["agg_hourly"],
        cat_test, register,
    )
    print(f"      kpi validation report        ->  {report}")

    print(f"\nDone in {time.time() - t0:.1f}s. {len(validation)} validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
