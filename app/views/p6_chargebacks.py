"""Page 6 — Chargeback Intelligence.

Attribution on this page flows chargeback → txn_id → transaction → user/merchant,
and never through the chargeback file's own user_id / merchant_id columns. Those
do not agree with the transaction they point at (measured in audit_facts).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.theme import SERIES, STATUS
from app.components.ui import (
    callout, empty_state, fmt_inr, fmt_int, fmt_pct, kpi_card, section,
)


def render(ctx) -> None:
    agg, star = ctx["agg"], ctx["star"]
    cb, tx = ctx["cb"], ctx["tx"]
    kpis = ctx["kpis"]

    st.markdown("# Chargeback Intelligence")
    st.markdown(
        '<div class="dv-sec-note">Every entity attribution on this page flows through '
        '<code>txn_id</code> to the transaction, and from there to the customer and merchant.'
        '</div>', unsafe_allow_html=True)

    if cb.empty:
        empty_state("No chargebacks match the selected filters.")
        return

    linked = cb[~cb["txn_unlinked"]]

    c = st.columns(5, gap="small")
    with c[0]:
        kpi_card("Chargebacks", fmt_int(kpis["cb_count"].value), "unique complaint_id")
    with c[1]:
        kpi_card("Disputed Value", fmt_inr(kpis["cb_amount"].value, compact=True),
                 kpis["cb_amount"].detail, kpis["cb_amount"].coverage_pct,
                 "complaints with a parseable amount")
    with c[2]:
        kpi_card("Chargeback Rate", fmt_pct(kpis["cb_ratio"].value),
                 kpis["cb_ratio"].detail, kpis["cb_ratio"].coverage_pct,
                 "distinct disputed transactions / transactions")
    with c[3]:
        kpi_card("Avg Reporting Delay", f"{kpis['delay'].value:.2f} d",
                 kpis["delay"].detail, kpis["delay"].coverage_pct,
                 "complaints with both timestamps parseable")
    with c[4]:
        kpi_card("Reported After 7 Days", fmt_int(kpis["delay_7d"].value),
                 "of complaints with a computable delay",
                 kpis["delay_7d"].coverage_pct, "complaints with a computable delay")

    # ---- attribution explainer --------------------------------------------
    conflict_user = int(linked["cb_userid_conflicts_txn"].sum())
    conflict_mer = int(linked["cb_merchantid_conflicts_txn"].sum())
    callout(
        f"<strong>Attribution integrity.</strong> {fmt_int(len(linked))} of "
        f"{fmt_int(len(cb))} complaints link to a transaction via <code>txn_id</code>. For "
        f"<strong>{fmt_int(conflict_user)} of {fmt_int(len(linked))}</strong> of those, the "
        f"complaint's own <code>user_id</code> disagrees with the linked transaction's "
        f"customer; the same holds for <code>merchant_id</code> in "
        f"<strong>{fmt_int(conflict_mer)}</strong> cases. The dataset notes propose joining "
        f"on those columns. Doing so would attribute disputes to a different, wrong set of "
        f"entities, so this dashboard never does.", kind="warn")

    # ---- trend -------------------------------------------------------------
    section("Dispute volume over time", "By the date the customer reported the dispute.")
    reported = cb.dropna(subset=["reported_date"]).copy()
    if not reported.empty:
        daily = (reported.assign(day=pd.to_datetime(reported["reported_date"]))
                         .groupby("day")
                         .agg(complaints=("complaint_key", "size"),
                              disputed=("disputed_amount_inr", "sum"))
                         .reset_index())
        left, right = st.columns(2, gap="medium")
        with left:
            st.plotly_chart(
                charts.line(daily["day"], daily["complaints"],
                            title="Complaints reported per day",
                            colour=STATUS["critical"], fill=True),
                width="stretch", config={"displayModeBar": False})
        with right:
            st.plotly_chart(
                charts.line(daily["day"], daily["disputed"],
                            title="Disputed value reported per day (₹)",
                            colour=SERIES[1], fill=True),
                width="stretch", config={"displayModeBar": False})

    # ---- breakdowns --------------------------------------------------------
    section("Dispute composition",
            "Reason codes, severity levels, resolution states and intake channels, each "
            "canonicalised from the observed spellings with originals retained.")
    left, right = st.columns(2, gap="medium")
    with left:
        _dimension_chart(cb, "reason_code_canonical", "Complaints by reason code", SERIES[0])
        _dimension_chart(cb, "resolution_status_canonical", "Complaints by resolution status",
                         SERIES[2])
    with right:
        _dimension_chart(cb, "severity_canonical", "Complaints by severity", SERIES[1],
                         order=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        _dimension_chart(cb, "channel_canonical", "Complaints by intake channel", SERIES[6])

    with st.expander("Reason code versus complaint text"):
        register = agg["agg_hypothesis_register"]
        row = register[register["hypothesis"].str.contains("reason code and complaint text",
                                                          case=False, regex=False)]
        if len(row):
            r = row.iloc[0]
            st.markdown(f"**{r['verdict']}** — {r['hypothesis']}\n\n{r['reading']}\n\n"
                        f"_{r['test']}: {r['statistic']}, p = {float(r['p_value']):.3f}_")
        cross = pd.crosstab(cb["reason_code_canonical"], cb["complaint_theme"])
        st.dataframe(cross, width="stretch")

    # ---- reporting delay ---------------------------------------------------
    section(
        "Reporting delay",
        "Measured from the complaint's own transaction timestamp to its reported timestamp. "
        "Measuring against the linked transaction's timestamp instead yields "
        f"{_fact(star, 'delay_negative_vs_linked_txn_pct', float('nan')):.2f}% "
        "impossible negative delays, because the complaint file's denormalized timestamp "
        "does not describe the transaction it points at.")
    delays = cb["reporting_delay_days"].dropna()
    if delays.empty:
        empty_state("No complaints in this selection have a computable reporting delay.")
    else:
        left, right = st.columns([3, 2], gap="medium")
        with left:
            st.plotly_chart(
                charts.histogram(delays.tolist(),
                                 title="Reporting delay distribution (days)",
                                 xtitle="days from transaction to report",
                                 vline=7, vline_label="7-day mark"),
                width="stretch", config={"displayModeBar": False})
        with right:
            stats = pd.DataFrame([
                ("Median", f"{delays.median():.2f} days"),
                ("Mean", f"{delays.mean():.2f} days"),
                ("90th percentile", f"{delays.quantile(0.9):.2f} days"),
                ("Maximum", f"{delays.max():.2f} days"),
                ("Reported after 7 days", fmt_int(int((delays > 7).sum()))),
                ("Reported after 30 days", fmt_int(int((delays > 30).sum()))),
                ("Negative (flagged)", fmt_int(int(cb["delay_negative"].sum()))),
                ("Not computable", fmt_int(int(cb["delay_missing"].sum()))),
            ], columns=["Statistic", "Value"])
            st.dataframe(stats, hide_index=True, width="stretch")

        by_reason = (cb.dropna(subset=["reporting_delay_days"])
                       .groupby("reason_code_canonical")["reporting_delay_days"]
                       .mean().sort_values(ascending=False))
        st.plotly_chart(
            charts.hbar(by_reason.index.tolist(), by_reason.round(2).tolist(),
                        title="Mean reporting delay by reason code (days)",
                        text=[f"{v:.2f} d" for v in by_reason], xtitle="days"),
            width="stretch", config={"displayModeBar": False})
        register = agg["agg_hypothesis_register"]
        takeover = register[register["hypothesis"].str.contains("account takeover", case=False,
                                                                regex=False)]
        if len(takeover):
            r = takeover.iloc[0]
            callout(f"<strong>{r['verdict']}</strong> — {r['reading']} "
                    f"({r['statistic']}, p = {float(r['p_value']):.3f})",
                    kind={"CONTRADICTED": "crit", "SUPPORTED": "good"}.get(r["verdict"], "warn"))

    # ---- exposure ----------------------------------------------------------
    section("Top exposure by attributed entity",
            "Attributed through txn_id. Ranked by absolute disputed value — the disputes "
            "team's workload, not a measured risk difference.")
    left, right = st.columns(2, gap="medium")
    with left:
        mer = ctx["agg"]["agg_merchant"].nlargest(10, "disputed_amount")
        st.plotly_chart(
            charts.hbar([f"{r.merchant_id_normalized}" for r in mer.itertuples()],
                        mer["disputed_amount"].tolist(),
                        title="Merchants by disputed value (₹)",
                        text=[fmt_inr(v, compact=True) for v in mer["disputed_amount"]],
                        xtitle="disputed value (₹)"),
            width="stretch", config={"displayModeBar": False})
    with right:
        usr = ctx["agg"]["agg_user"].nlargest(10, "disputed_amount")
        st.plotly_chart(
            charts.hbar([f"{r.user_id_normalized}" for r in usr.itertuples()],
                        usr["disputed_amount"].tolist(),
                        title="Customers by disputed value (₹)", colour=SERIES[1],
                        text=[fmt_inr(v, compact=True) for v in usr["disputed_amount"]],
                        xtitle="disputed value (₹)"),
            width="stretch", config={"displayModeBar": False})

    with st.expander("Complaint detail"):
        cols = ["complaint_key", "txn_key", "attributed_user_id", "attributed_merchant_id",
                "reason_code_canonical", "severity_canonical", "resolution_status_canonical",
                "channel_canonical", "disputed_amount_inr", "reporting_delay_days",
                "txn_unlinked", "txn_id_malformed"]
        st.dataframe(
            cb[[c for c in cols if c in cb.columns]].head(500),
            hide_index=True, width="stretch", height=380,
            column_config={
                "disputed_amount_inr": st.column_config.NumberColumn("Disputed ₹", format="%.2f"),
                "reporting_delay_days": st.column_config.NumberColumn("Delay (d)", format="%.2f"),
                "attributed_user_id": st.column_config.TextColumn(
                    "Customer", help="Resolved via txn_id, not the complaint's own user_id."),
                "attributed_merchant_id": st.column_config.TextColumn(
                    "Merchant", help="Resolved via txn_id, not the complaint's own merchant_id."),
            })


def _dimension_chart(cb, column, title, colour, order=None) -> None:
    counts = cb[column].value_counts()
    if order:
        counts = counts.reindex([o for o in order if o in counts.index])
    if counts.empty:
        empty_state(f"No values for {column}.")
        return
    st.plotly_chart(
        charts.hbar(counts.index.tolist(), counts.tolist(), title=title, colour=colour,
                    text=[fmt_int(v) for v in counts], xtitle="complaints"),
        width="stretch", config={"displayModeBar": False})


def _fact(star, name, default=None):
    from src.transformation.audit_facts import lookup
    return lookup(star.get("audit_facts"), name, default)
