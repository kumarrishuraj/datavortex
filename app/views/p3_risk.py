"""Page 3 — Merchant & User Risk Intelligence.

Terminology is deliberate throughout. Nothing here is called fraudulent. The
merchant ranking is labelled operational exposure, because the hypothesis
register tests merchant-level dispute propensity and the verdict is read from
there rather than written here. The Risk Indicator Score is an explainable
review-priority total, shown with every component that produced it, and never
presented as a fraud probability.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.theme import SERIES
from app.components.ui import (
    callout, empty_state, fmt_inr, fmt_int, kpi_card, section,
)
from src.analytics import risk_indicators as RI
from src.analytics.shrinkage import MIN_DENOMINATOR


def _register_row(agg, needle: str):
    reg = agg["agg_hypothesis_register"]
    hit = reg[reg["hypothesis"].str.contains(needle, case=False, regex=False)]
    return hit.iloc[0] if len(hit) else None


def render(ctx) -> None:
    agg = ctx["agg"]
    st.markdown("# Merchant & User Risk Intelligence")

    row = _register_row(agg, "individual merchants differ")
    if row is not None:
        supported = row["verdict"] == "SUPPORTED"
        callout(
            "<strong>How to read this page.</strong> The hypothesis register tests whether "
            "merchants differ in how often their transactions are disputed: "
            f"<strong>{row['verdict']}</strong> ({row['statistic']}, p = {row['p_value']}). "
            + ("Merchant rankings here therefore represent <strong>operational exposure</strong> — "
               "where the disputes team should spend its time — not a measured difference in "
               "merchant risk. " if not supported else
               "Merchant dispute rates differ by more than noise, so shrunk rates are informative. ")
            + "The two Risk Indicator tabs add an explainable review-priority score, which is "
              "<strong>not</strong> a fraud probability.")

    tabs = st.tabs(["Merchant exposure", "Merchant risk indicators", "Customer risk indicators"])
    with tabs[0]:
        _merchants(ctx)
    with tabs[1]:
        _risk(ctx, "merchant")
    with tabs[2]:
        _risk(ctx, "user")


# --------------------------------------------------------------------------
def _merchants(ctx) -> None:
    mer = ctx["agg"]["agg_merchant"].copy()

    section("Merchant population")
    eligible = mer[~mer["below_floor"]]
    c = st.columns(4, gap="small")
    with c[0]:
        kpi_card("Transacting Merchants", fmt_int(len(mer)),
                 "all merchants with at least one transaction", 100.0,
                 "full transacting population")
    with c[1]:
        in_master = int(mer["in_master"].sum())
        kpi_card("With Master Record", fmt_int(in_master),
                 "have category, status and location",
                 100 * in_master / max(len(mer), 1),
                 "merchants present in the merchant master")
    with c[2]:
        kpi_card("Rateable", fmt_int(len(eligible)),
                 f"at least {MIN_DENOMINATOR} transactions",
                 100 * len(eligible) / max(len(mer), 1),
                 "merchants above the denominator floor")
    with c[3]:
        kpi_card("Below Floor", fmt_int(int(mer["below_floor"].sum())),
                 "kept and shown, never ranked on rate")

    raw_100 = int((mer["dispute_rate_raw_pct"] >= 100).sum())
    elig_100 = int((eligible["dispute_rate_raw_pct"] >= 100).sum())
    callout(
        f"<strong>{raw_100} merchants show a raw dispute ratio of 100%.</strong> Almost all "
        f"have a single transaction that happened to be disputed. Applying the "
        f"{MIN_DENOMINATOR}-transaction floor leaves <strong>{elig_100}</strong>. Ranking on "
        "the raw ratio would produce a list of merchants with n = 1.", kind="warn")

    metric = st.radio(
        "Rank merchants by", ["Disputed value", "Complaint count", "Transaction value"],
        horizontal=True, key="mer_metric",
        help="All three are absolute exposure measures. Rate-based ranking is not "
             "supported by the data — see the note at the top of this page.")
    col = {"Disputed value": "disputed_amount", "Complaint count": "complaints",
           "Transaction value": "transaction_value"}[metric]

    top = mer.nlargest(15, col)
    if top.empty:
        empty_state("No merchants to rank.")
        return
    labels = [f"{r.merchant_id_normalized} · {str(r.merchant_name_clean)[:28]}"
              for r in top.itertuples()]
    fmt = fmt_inr if col != "complaints" else fmt_int
    section(f"Top merchants by {metric.lower()}",
            "Absolute exposure. This is a workload queue, not a risk score.")
    st.plotly_chart(
        charts.hbar(labels, top[col].tolist(), title=f"{metric} by merchant",
                    text=[fmt(v) for v in top[col]], xtitle=metric.lower()),
        width="stretch", config={"displayModeBar": False})

    section(
        "Raw versus shrunk dispute rate",
        "Every eligible merchant, with its raw ratio against its shrunk estimate. When the "
        "overdispersion test finds nothing to separate merchants, the shrunk values collapse "
        "onto their category means — that flatness is the finding.")
    sample = eligible.sample(min(len(eligible), 1500), random_state=7) if len(eligible) else eligible
    if not sample.empty:
        st.plotly_chart(
            charts.scatter(
                sample["dispute_rate_raw_pct"].tolist(),
                sample["dispute_rate_shrunk_pct"].tolist(),
                labels=sample["merchant_id_normalized"].tolist(),
                title="Raw dispute ratio vs shrunk estimate (eligible merchants)",
                xtitle="raw ratio (%)", ytitle="shrunk estimate (%)", diagonal=True),
            width="stretch", config={"displayModeBar": False})
        callout(
            f"Raw ratios range {eligible['dispute_rate_raw_pct'].min():.0f}–"
            f"{eligible['dispute_rate_raw_pct'].max():.0f}% with a standard deviation of "
            f"{eligible['dispute_rate_raw_pct'].std():.2f} points. After shrinkage the spread "
            f"is {eligible['dispute_rate_shrunk_pct'].std():.2f} points. The dotted diagonal "
            "is where the two would agree; the distance from it is how much of each raw "
            "ratio was sampling noise.")

    section("Merchant detail", "Sortable. Below-floor merchants are retained and flagged.")
    show = mer.nlargest(300, col)[[
        "merchant_id_normalized", "merchant_name_clean", "merchant_category_canonical",
        "merchant_status_canonical", "transactions", "transaction_value",
        "complaints", "disputed_transactions", "disputed_amount",
        "dispute_rate_raw_pct", "dispute_rate_shrunk_pct", "peer_mean_pct",
        "below_floor", "in_master", "identity_ambiguous", "failed_rate_pct",
    ]].rename(columns={
        "merchant_id_normalized": "Merchant", "merchant_name_clean": "Name",
        "merchant_category_canonical": "Category", "merchant_status_canonical": "Status",
        "transactions": "Txns", "transaction_value": "Value ₹",
        "complaints": "Complaints", "disputed_transactions": "Disputed txns",
        "disputed_amount": "Disputed ₹", "dispute_rate_raw_pct": "Raw %",
        "dispute_rate_shrunk_pct": "Shrunk %", "peer_mean_pct": "Category benchmark %",
        "below_floor": "Below floor", "in_master": "In master",
        "identity_ambiguous": "ID ambiguous", "failed_rate_pct": "Failed %",
    })
    st.dataframe(
        show, hide_index=True, width="stretch", height=420,
        column_config={
            "Value ₹": st.column_config.NumberColumn(format="%.0f"),
            "Disputed ₹": st.column_config.NumberColumn(format="%.0f"),
            "Raw %": st.column_config.NumberColumn(
                format="%.2f", help="Unadjusted. Unreliable below the floor."),
            "Shrunk %": st.column_config.NumberColumn(
                format="%.3f", help="Empirical-Bayes estimate, shrunk toward the category mean."),
            "Category benchmark %": st.column_config.NumberColumn(format="%.3f"),
            "Below floor": st.column_config.CheckboxColumn(
                help=f"Fewer than {MIN_DENOMINATOR} transactions — shown, never ranked on rate."),
            "ID ambiguous": st.column_config.CheckboxColumn(
                help="This merchant_id is shared by more than one distinct business."),
        })


# --------------------------------------------------------------------------
def _formula(entity: str) -> str:
    points = RI.POINTS
    if entity == "merchant":
        status = ", ".join(f"{k} {v:g}" for k, v in RI.MERCHANT_STATUS_POINTS.items())
        return (
            f"Four components of up to **{points:g} points** each, total 0–100.\n\n"
            f"| Component | Points |\n|---|---|\n"
            f"| Dispute volume | {points:g} × min(disputed transactions, {RI.MERCHANT_DISPUTE_CAP}) "
            f"/ {RI.MERCHANT_DISPUTE_CAP} |\n"
            f"| Disputed value | {points:g} × percentile rank of disputed amount among merchants "
            "with any disputed value |\n"
            f"| Merchant status | {status}; 0 and flagged when there is no master record |\n"
            f"| Identity ambiguity | {points:g} when the merchant ID is shared by different "
            "businesses |\n\n"
            "Dispute **rate** is deliberately excluded: the overdispersion test decides whether a "
            "merchant rate carries information, and a rate component would otherwise score noise.")
    kyc = ", ".join(f"{k} {v:g}" for k, v in RI.KYC_POINTS.items())
    return (
        f"Four components of up to **{points:g} points** each, total 0–100.\n\n"
        f"| Component | Points |\n|---|---|\n"
        f"| Dispute history | {points:g} × min(disputed transactions, {RI.CUSTOMER_DISPUTE_CAP}) "
        f"/ {RI.CUSTOMER_DISPUTE_CAP} |\n"
        f"| Disputed value | {points:g} × percentile rank of disputed amount among customers with "
        "any disputed value |\n"
        f"| KYC status | {kyc}; 0 and flagged 'not assessable' when there is no KYC record |\n"
        f"| Identity ambiguity | {points:g} when the customer ID is shared by different people |\n\n"
        "Weights are equal because the data has no fraud label to learn them from. Missing UTR "
        "and reporting delay are excluded because the hypothesis register finds neither "
        "associated with disputes.")


def _risk(ctx, entity: str) -> None:
    agg = ctx["agg"]
    table = agg["agg_merchant" if entity == "merchant" else "agg_user"]
    if "risk_indicator_score" not in table.columns:
        empty_state("The Risk Indicator Score has not been built.",
                    "Run python scripts/build_analytics.py.")
        return
    noun = "merchants" if entity == "merchant" else "customers"
    band_entity = "merchant" if entity == "merchant" else "customer"
    id_col = "merchant_id_normalized" if entity == "merchant" else "user_id_normalized"
    name_col = "merchant_name_clean" if entity == "merchant" else "full_name_clean"
    status_col = "merchant_status_canonical" if entity == "merchant" else "kyc_status_canonical"
    components = RI.MERCHANT_COMPONENTS if entity == "merchant" else RI.CUSTOMER_COMPONENTS

    callout(f"<strong>Risk Indicator ≠ confirmed fraud.</strong> {RI.DISCLAIMER}", kind="crit")
    with st.expander("How the score is calculated"):
        st.markdown(_formula(entity))

    bands = agg["agg_risk_bands"]
    bands = bands[bands["entity"] == band_entity]
    cols = st.columns(len(bands), gap="small")
    for col, (_, r) in zip(cols, bands.iterrows()):
        with col:
            kpi_card(f"{r['band']} priority", fmt_int(r["entities"]),
                     f"{r['share_pct']:.2f}% of {noun}")

    left, right = st.columns(2, gap="medium")
    with left:
        comp = agg["agg_risk_components"]
        comp = comp[comp["entity"] == band_entity]
        st.plotly_chart(
            charts.hbar(comp["label"].tolist(), comp["entities_with_points"].tolist(),
                        title=f"{noun.capitalize()} scoring on each component",
                        text=[fmt_int(v) for v in comp["entities_with_points"]], xtitle=noun),
            width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(
            charts.histogram(table["risk_indicator_score"].tolist(),
                             title="Risk Indicator Score distribution",
                             xtitle="score (0–100)", nbins=20, colour=SERIES[0]),
            width="stretch", config={"displayModeBar": False})

    if entity == "user":
        repeat = _register_row(agg, "appear repeatedly in disputes")
        if repeat is not None:
            callout(f"Hypothesis register — <strong>{repeat['verdict']}</strong>: "
                    f"{repeat['reading']} ({repeat['statistic']}, p = {repeat['p_value']})",
                    kind="warn")

    section("Highest review priority",
            "Every score is shown with the components that produced it. The list says where to "
            "look, not who is at fault.")
    f1, f2 = st.columns([2, 1], gap="small")
    with f1:
        chosen = st.multiselect("Band", list(RI.BAND_ORDER), default=[], key=f"band_{entity}",
                                help="Empty means all bands.")
    with f2:
        min_disputes = st.number_input("Minimum disputed transactions", min_value=0,
                                       max_value=10, value=0, key=f"min_disp_{entity}")
    subset = table
    if chosen:
        subset = subset[subset["risk_indicator_band"].isin(chosen)]
    if min_disputes:
        subset = subset[subset["disputed_transactions"] >= min_disputes]
    subset = subset.sort_values(["risk_indicator_score", "disputed_amount", id_col],
                                ascending=[False, False, True]).head(300)
    if subset.empty:
        empty_state("No records match these filters.")
        return
    columns = [id_col, name_col, "risk_indicator_score", "risk_indicator_band",
               *components, "risk_indicator_reasons", "disputed_transactions",
               "disputed_amount", status_col]
    rename = {id_col: "Merchant" if entity == "merchant" else "Customer", name_col: "Name",
              "risk_indicator_score": "Score", "risk_indicator_band": "Band",
              "risk_indicator_reasons": "Why", "disputed_transactions": "Disputed txns",
              "disputed_amount": "Disputed ₹", status_col: "Status", **components}
    config = {c: st.column_config.NumberColumn(format="%.1f") for c in components.values()}
    config.update({
        "Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100,
                                                 help="Review priority, not a fraud probability."),
        "Disputed ₹": st.column_config.NumberColumn(format="%.0f"),
        "Why": st.column_config.TextColumn(width="large"),
    })
    st.dataframe(subset[[c for c in columns if c in subset.columns]].rename(columns=rename),
                 hide_index=True, width="stretch", height=440, column_config=config)
    st.caption(
        "Names are shown from the master records. PAN and Aadhaar are never displayed — the "
        "processed layer holds only a masked PAN and the last four Aadhaar digits.")
