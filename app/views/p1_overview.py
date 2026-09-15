"""Page 1 — Executive Overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.theme import SERIES, STATUS
from app.components.ui import (
    callout, fmt_inr, fmt_int, fmt_pct, insight_card, kpi_card, section,
)


def render(ctx) -> None:
    agg, star = ctx["agg"], ctx["star"]
    kpis = ctx["kpis"]
    tx, cb = ctx["tx"], ctx["cb"]

    st.markdown("# Executive Overview")
    st.markdown(
        '<div class="dv-sec-note" style="max-width:56rem">'
        "DataVortex turns a deliberately corrupted UPI dataset into an auditable analytics "
        "layer. It rescues four messy source files into a validated star schema, resolves "
        "identifiers that different entities share, attributes every dispute through the one "
        "foreign key that survives testing, and reports which fraud hypotheses the data "
        "actually supports — including the ones it does not."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- headline KPIs -----------------------------------------------------
    section("Transaction book", "Computed from the full cleaned population of "
                               f"{fmt_int(kpis['count'].value)} transactions.")
    c = st.columns(4, gap="small")
    with c[0]:
        kpi_card("Transactions", fmt_int(kpis["count"].value),
                 "unique txn_id, post-deduplication",
                 kpis["count"].coverage_pct, "all cleaned transactions")
    with c[1]:
        kpi_card("Transaction Value", fmt_inr(kpis["amount"].value, compact=True),
                 f"{fmt_inr(kpis['amount'].value)} total",
                 kpis["amount"].coverage_pct, "transactions with a parseable amount")
    with c[2]:
        kpi_card("Average Value", fmt_inr(kpis["average"].value),
                 "per transaction", kpis["average"].coverage_pct, "all cleaned transactions")
    with c[3]:
        kpi_card("Success Rate", fmt_pct(kpis["success"].value),
                 kpis["success"].detail, kpis["success"].coverage_pct,
                 "all cleaned transactions")

    c = st.columns(4, gap="small")
    with c[0]:
        kpi_card("Chargebacks", fmt_int(kpis["cb_count"].value),
                 "unique complaint_id", kpis["cb_count"].coverage_pct,
                 "all cleaned complaints")
    with c[1]:
        kpi_card("Disputed Value", fmt_inr(kpis["cb_amount"].value, compact=True),
                 f"{fmt_inr(kpis['cb_amount'].value)} total",
                 kpis["cb_amount"].coverage_pct, "complaints with a parseable amount")
    with c[2]:
        kpi_card("Chargeback Rate", fmt_pct(kpis["cb_ratio"].value),
                 kpis["cb_ratio"].detail, kpis["cb_ratio"].coverage_pct,
                 "distinct disputed transactions / all transactions")
    with c[3]:
        kpi_card("Avg Reporting Delay", f"{kpis['delay'].value:.2f} days",
                 kpis["delay"].detail, kpis["delay"].coverage_pct,
                 "complaints with both timestamps parseable")

    # ---- enrichment coverage ----------------------------------------------
    section(
        "Enrichment coverage",
        "Two of the five joins the dataset notes propose are statistically indistinguishable "
        "from random. Any metric segmented by merchant category or KYC status is therefore "
        "computed on part of the book, and every such number on this dashboard says so.",
    )
    c = st.columns(4, gap="small")
    with c[0]:
        kpi_card("Merchant Master Coverage", fmt_pct(kpis["merchant_cov"].value),
                 kpis["merchant_cov"].detail, kpis["merchant_cov"].value,
                 "transactions resolving to a merchant record")
    with c[1]:
        kpi_card("KYC Coverage", fmt_pct(kpis["kyc_cov"].value),
                 kpis["kyc_cov"].detail, kpis["kyc_cov"].value,
                 "transactions resolving to a KYC record")
    with c[2]:
        linked = int((~cb["txn_unlinked"]).sum())
        kpi_card("Chargeback Attribution", fmt_pct(100 * linked / max(len(cb), 1)),
                 f"{fmt_int(linked)} of {fmt_int(len(cb))} linked via txn_id",
                 100 * linked / max(len(cb), 1), "complaints linked to a transaction")
    with c[3]:
        kpi_card("Missing UTR", fmt_pct(kpis["utr_missing"].value),
                 kpis["utr_missing"].detail, kpis["utr_missing"].coverage_pct,
                 "all cleaned transactions")

    # ---- trends ------------------------------------------------------------
    daily = agg["agg_daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"])

    section("Daily activity", "The full observed window, day by day.")
    left, right = st.columns(2, gap="medium")
    with left:
        st.plotly_chart(
            charts.line(daily["date"], daily["transactions"],
                        title="Transactions per day", fill=True),
            width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(
            charts.line(daily["date"], daily["transaction_value"],
                        title="Transaction value per day (₹)",
                        colour=SERIES[1], fill=True, yfmt=",.0f"),
            width="stretch", config={"displayModeBar": False})

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.plotly_chart(
            charts.line(daily["date"], daily["disputed_transactions"],
                        title="Disputed transactions per day",
                        colour=STATUS["critical"]),
            width="stretch", config={"displayModeBar": False})
    with right:
        counts = {s: int((tx["status_canonical"] == s).sum())
                  for s in ["SUCCESS", "FAILED", "PENDING"]}
        st.plotly_chart(
            charts.status_composition(counts, title="Transaction outcome mix"),
            width="stretch", config={"displayModeBar": False})
        st.markdown(
            f'<div class="dv-kpi-sub">Success {fmt_pct(kpis["success"].value)} · '
            f'Failed {fmt_pct(kpis["failed"].value)} · '
            f'Pending {fmt_pct(kpis["pending"].value)}</div>',
            unsafe_allow_html=True)

    # ---- operational exposure ---------------------------------------------
    mer = agg["agg_merchant"]
    top = mer.nlargest(10, "disputed_amount")
    section(
        "Top operational exposure",
        "Ranked by absolute disputed value, not by dispute rate. Stage 3 found no "
        "statistically detectable merchant-level dispute propensity, so this is a workload "
        "queue for the disputes team — not a list of riskier merchants.",
    )
    labels = [f"{r.merchant_id_normalized} · {str(r.merchant_name_clean)[:26]}"
              for r in top.itertuples()]
    st.plotly_chart(
        charts.hbar(labels, top["disputed_amount"].tolist(),
                    title="Disputed value by merchant (₹)",
                    text=[fmt_inr(v) for v in top["disputed_amount"]],
                    xtitle="disputed value (₹)"),
        width="stretch", config={"displayModeBar": False})

    # ---- key findings ------------------------------------------------------
    section("Key findings", "Generated from the current analytics build — no fixed text.")
    _render_findings(ctx)


def _render_findings(ctx) -> None:
    agg, star = ctx["agg"], ctx["star"]
    tx, cb = ctx["tx"], ctx["cb"]
    reg = agg["agg_hypothesis_register"]
    net = agg["agg_network_summary"].iloc[0]
    dq = agg["agg_data_quality"].set_index("metric")["value"]
    users = star["dim_users"]
    merchants = star["dim_merchants"]

    from src.transformation.audit_facts import lookup as fact

    notes = reg[reg["source"] == "track1_dataset_notes.txt"]
    not_supported = int((notes["verdict"] == "NOT SUPPORTED").sum())
    contradicted = reg[reg["verdict"] == "CONTRADICTED"]
    cat = agg["agg_category"]
    top_cat = cat.iloc[0]
    z = star["recon_independence"].set_index("pair")["z_score"]
    facts = star["audit_facts"]
    linked = fact(facts, "chargebacks_linked", 0.0)
    user_agree = fact(facts, "cb_user_id_agrees_with_linked_txn", 0.0)
    merchant_agree = fact(facts, "cb_merchant_id_agrees_with_linked_txn", 0.0)
    category_row = reg[reg["hypothesis"].str.contains("merchant categories", case=False,
                                                      regex=False)]
    category_clause = ""
    if len(category_row) and category_row.iloc[0]["verdict"] != "SUPPORTED":
        r = category_row.iloc[0]
        category_clause = (f" but differ no more than chance would produce "
                           f"({r['statistic']}, p = {float(r['p_value']):.3f})")

    cards = [
        ("Two of the five documented joins are random",
         "Transaction-to-KYC and transaction-to-merchant overlap match what independent "
         f"random draws produce (z = {z.get('tx.user x kyc.user', float('nan')):.2f} and "
         f"z = {z.get('tx.merchant x mer.merchant', float('nan')):.2f}). No amount of ID "
         "normalisation can "
         f"raise them, so {fmt_int(dq.get('Merchant unresolved', 0))} transactions carry an "
         "UNKNOWN merchant and are kept rather than dropped."),
        ("Repeated IDs are different entities, not duplicate rows",
         f"{fmt_int(int(users['identity_ambiguous'].sum()))} customer IDs and "
         f"{fmt_int(int(merchants['identity_ambiguous'].sum()))} merchant IDs are each shared "
         "by more than one real entity. A drop_duplicates on the ID would have silently "
         "destroyed them; every candidate row is preserved in a bridge table instead."),
        ("Dispute attribution has exactly one trustworthy path",
         f"{fmt_int(linked)} complaints link to a transaction via txn_id. The chargeback "
         f"file's own user_id agrees with that transaction's customer {fmt_int(user_agree)} "
         f"times and its merchant_id {fmt_int(merchant_agree)} times, so both are kept as "
         "flagged attributes and never joined on."),
        (f"{not_supported} of the dataset notes' {len(notes)} suggested insights are not supported",
         f"Category dispute rates span {cat['rate_pct'].min():.2f}–{cat['rate_pct'].max():.2f}%"
         f"{category_clause}. {top_cat['category']} ranks first"
         + (" and that ranking carries no information." if category_clause else ".")),
        ("No fraud ring exists in this data — structurally",
         f"The transaction graph is bipartite, has {fmt_int(net['components'])} components, "
         f"and contains {int(net['user_pairs_sharing_2plus'])} four-cycles. It is a forest: "
         "a circular A→B→C→A money path cannot exist here, so none is reported."),
    ]
    if len(contradicted):
        row = contradicted.iloc[0]
        cards.append((
            "One expected signal runs backwards",
            f"{row['hypothesis']} The data shows the opposite — {row['statistic']}, "
            f"p = {row['p_value']}. A control built on the brief's assumption would have "
            "looked in the wrong place."))

    for i in range(0, len(cards), 3):
        cols = st.columns(3, gap="medium")
        for col, (heading, body) in zip(cols, cards[i:i + 3]):
            with col:
                insight_card(heading, body)
