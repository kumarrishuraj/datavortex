"""Page 2 — Transaction & Business Analytics."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.theme import SERIES, STATUS
from app.components.ui import (
    callout, empty_state, fmt_inr, fmt_int, fmt_pct, kpi_card, section,
    significance_caveat,
)
from src.analytics.significance import rank_with_confidence


def render(ctx) -> None:
    tx, cb = ctx["tx"], ctx["cb"]
    agg = ctx["agg"]
    kpis = ctx["kpis"]

    st.markdown("# Transaction & Business Analytics")
    st.markdown(
        '<div class="dv-sec-note">Filters in the sidebar apply to every chart on this page. '
        'KPIs are recomputed through the Stage 3 metric functions on the filtered rows — the '
        'dashboard never re-derives a formula of its own.</div>',
        unsafe_allow_html=True)

    if tx.empty:
        empty_state("No transactions match the selected filters.",
                    "Widen the date range or clear a category selection.")
        return

    if ctx["filters_active"]:
        callout(
            f"Filtered view — <strong>{fmt_int(len(tx))}</strong> of "
            f"<strong>{fmt_int(ctx['total_transactions'])}</strong> transactions "
            f"({100 * len(tx) / ctx['total_transactions']:.1f}%). All figures below reflect "
            "the current selection.")

    c = st.columns(5, gap="small")
    with c[0]:
        kpi_card("Transactions", fmt_int(kpis["count"].value))
    with c[1]:
        kpi_card("Value", fmt_inr(kpis["amount"].value, compact=True))
    with c[2]:
        kpi_card("Average Value", fmt_inr(kpis["average"].value))
    with c[3]:
        kpi_card("Failed Rate", fmt_pct(kpis["failed"].value), kpis["failed"].detail)
    with c[4]:
        kpi_card("Pending Rate", fmt_pct(kpis["pending"].value), kpis["pending"].detail)

    # ---- daily trends ------------------------------------------------------
    daily = (tx.assign(day=pd.to_datetime(tx["timestamp_clean"]).dt.date)
               .groupby("day")
               .agg(transactions=("txn_key", "size"),
                    value=("amount_inr", "sum"),
                    avg_value=("amount_inr", "mean"))
               .reset_index())
    daily["day"] = pd.to_datetime(daily["day"])

    section("Daily trend", "Count and value are plotted separately: putting two different "
                           "scales on one axis would invent a relationship that is not there.")
    left, right = st.columns(2, gap="medium")
    with left:
        st.plotly_chart(charts.line(daily["day"], daily["transactions"],
                                    title="Transactions per day", fill=True),
                        width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(charts.line(daily["day"], daily["value"],
                                    title="Transaction value per day (₹)",
                                    colour=SERIES[1], fill=True),
                        width="stretch", config={"displayModeBar": False})

    left, right = st.columns(2, gap="medium")
    with left:
        status_daily = (tx.assign(day=pd.to_datetime(tx["timestamp_clean"]).dt.date)
                          .groupby(["day", "status_canonical"]).size()
                          .unstack(fill_value=0))
        series = {s: status_daily[s].tolist() for s in ["SUCCESS", "FAILED", "PENDING"]
                  if s in status_daily.columns}
        st.plotly_chart(
            charts.multi_line(pd.to_datetime(status_daily.index), series,
                              title="Outcome mix per day"),
            width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(charts.line(daily["day"], daily["avg_value"],
                                    title="Average transaction value per day (₹)",
                                    colour=SERIES[2]),
                        width="stretch", config={"displayModeBar": False})

    _forecast_section(ctx)

    # ---- hour of day -------------------------------------------------------
    known = tx[tx["timestamp_time_known"]]
    excluded = len(tx) - len(known)
    section(
        "Hour of day",
        f"Restricted to the {fmt_int(len(known))} transactions whose source carried a clock "
        f"time. {fmt_int(excluded)} date-only timestamps parse to 00:00 and are excluded — "
        "including them produces a midnight spike that is a parsing artifact, not behaviour.")
    if known.empty:
        empty_state("No transactions with a known clock time in this selection.")
    else:
        hourly = known.groupby("hour").size().reindex(range(24), fill_value=0)
        cv = hourly.std() / hourly.mean() if hourly.mean() else 0
        st.plotly_chart(
            charts.line(list(hourly.index), hourly.tolist(),
                        title="Transactions by hour of day", colour=SERIES[0], height=280),
            width="stretch", config={"displayModeBar": False})
        callout(
            f"Coefficient of variation <strong>{cv:.3f}</strong> across the 24 hours — the "
            "profile is effectively flat. This dataset has no intraday pattern, which is "
            "worth knowing before anyone builds an off-hours rule.")

    # ---- merchant category -------------------------------------------------
    cat = (tx.groupby("category")
             .agg(transactions=("txn_key", "size"),
                  value=("amount_inr", "sum"),
                  merchants=("merchant_id_normalized", "nunique"))
             .reset_index())
    disputed_keys = set(cb.loc[~cb["txn_unlinked"], "txn_key"].dropna())
    cat_disp = (tx.assign(d=tx["txn_key"].isin(disputed_keys))
                  .groupby("category")["d"].sum().rename("disputed_transactions"))
    cat = cat.merge(cat_disp, on="category", how="left")
    cat["disputed_transactions"] = cat["disputed_transactions"].fillna(0)

    section(
        "Merchant category",
        "Category comes from the merchant master, which covers "
        f"{fmt_pct(kpis['merchant_cov'].value)} of transactions. The UNKNOWN row is the "
        "remainder and is shown rather than hidden — it is the largest single group.")

    left, right = st.columns(2, gap="medium")
    top_value = cat.nlargest(12, "value")
    with left:
        st.plotly_chart(
            charts.hbar(top_value["category"].tolist(), top_value["value"].tolist(),
                        title="Transaction value by category (₹)",
                        text=[fmt_inr(v, compact=True) for v in top_value["value"]],
                        xtitle="value (₹)"),
            width="stretch", config={"displayModeBar": False})
    with right:
        top_count = cat.nlargest(12, "transactions")
        st.plotly_chart(
            charts.hbar(top_count["category"].tolist(), top_count["transactions"].tolist(),
                        title="Transaction count by category", colour=SERIES[1],
                        text=[fmt_int(v) for v in top_count["transactions"]],
                        xtitle="transactions"),
            width="stretch", config={"displayModeBar": False})

    # ---- category dispute rate, with the statistical caveat ---------------
    section(
        "Chargeback rate by category",
        "Bars carry 95% Wilson confidence intervals. When the intervals overlap, the "
        "ordering is not a real ranking — and the caveat below is generated from the test, "
        "not written by hand.")
    ranked, test = rank_with_confidence(
        cat[["category", "disputed_transactions", "transactions"]],
        "category", "disputed_transactions", "transactions")
    eligible = ranked[ranked["transactions"] >= 3]
    if eligible.empty:
        empty_state("No category has enough transactions in this selection to rate.")
    else:
        baseline = 100 * cat["disputed_transactions"].sum() / max(cat["transactions"].sum(), 1)
        st.plotly_chart(
            charts.hbar_with_ci(
                eligible["category"].tolist(), eligible["rate_pct"].tolist(),
                eligible["ci_low_pct"].tolist(), eligible["ci_high_pct"].tolist(),
                baseline=baseline,
                title="Chargeback rate by merchant category (95% CI)",
                xtitle="disputed transactions / transactions (%)"),
            width="stretch", config={"displayModeBar": False})
        significance_caveat(eligible, test, "category", "merchant categories")

    with st.expander("Category detail table"):
        show = ranked.copy()
        show["value"] = show["category"].map(cat.set_index("category")["value"])
        show["merchants"] = show["category"].map(cat.set_index("category")["merchants"])
        st.dataframe(
            show[["rank", "category", "merchants", "transactions", "value",
                  "disputed_transactions", "rate_pct", "ci_low_pct", "ci_high_pct",
                  "p_adjusted", "significant"]],
            hide_index=True, width="stretch",
            column_config={
                "value": st.column_config.NumberColumn("Value (₹)", format="%.0f"),
                "rate_pct": st.column_config.NumberColumn("Rate %", format="%.2f"),
                "ci_low_pct": st.column_config.NumberColumn("CI low %", format="%.2f"),
                "ci_high_pct": st.column_config.NumberColumn("CI high %", format="%.2f"),
                "p_adjusted": st.column_config.NumberColumn("p (adj.)", format="%.3f"),
                "significant": st.column_config.CheckboxColumn("Differs from rest?"),
            })


def _forecast_section(ctx) -> None:
    """Backtested forecast. Always built on the full, unfiltered history."""
    agg = ctx["agg"]
    forecast = agg.get("agg_forecast")
    diagnostics = agg.get("agg_forecast_diagnostics")
    backtest = agg.get("agg_forecast_backtest")
    if forecast is None or diagnostics is None or forecast.empty:
        return
    section("Validated forecast",
            "Four simple methods are compared on the most recent weeks, which none of them was "
            "fitted to. The method with the lowest holdout error projects the next two weeks "
            "with a 95% band. The forecast always uses the full, unfiltered history.")
    labels = dict(zip(diagnostics["series"], diagnostics["label"]))
    series = st.selectbox("Series", list(labels), format_func=labels.get, key="forecast_series")
    d = diagnostics[diagnostics["series"] == series].iloc[0]
    rows = forecast[forecast["series"] == series].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    actual = rows[rows["kind"] == "actual"]
    future = rows[rows["kind"] == "forecast"]
    left, right = st.columns([4, 3], gap="medium")
    with left:
        st.plotly_chart(
            charts.forecast_chart(actual["date"], actual["value"], future["date"],
                                  future["value"], future["lower_95"], future["upper_95"],
                                  title=f"{d['label']}: actual and {len(future)}-day projection"),
            width="stretch", config={"displayModeBar": False})
    with right:
        table = backtest[backtest["series"] == series][
            ["method_label", "mae", "mape_pct", "holdout_band_coverage_pct", "selected"]].rename(
            columns={"method_label": "Method", "mae": "MAE", "mape_pct": "MAPE %",
                     "holdout_band_coverage_pct": "Band %", "selected": "Chosen"})
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={
                         "MAE": st.column_config.NumberColumn(
                             format="%.2f", help="Mean absolute error on the holdout weeks"),
                         "MAPE %": st.column_config.NumberColumn(
                             format="%.2f", help="Mean absolute percentage error on the holdout"),
                         "Band %": st.column_config.NumberColumn(
                             format="%.1f", help="Share of holdout days inside the 95% band"),
                         "Chosen": st.column_config.CheckboxColumn(
                             help="Lowest holdout MAE; used for the projection"),
                     })
    money = fmt_inr if d["unit"] == "INR" else fmt_int
    trend = "no significant trend" if float(d["slope_p"]) >= 0.05 else "a significant trend"
    weekday = (f"; a day-of-week test gives p = {float(d['weekday_p']):.3f}"
               if pd.notna(d["weekday_p"]) else "")
    callout(
        f"Selected method: <strong>{d['selected_method_label']}</strong>, holdout MAE "
        f"{money(d['selected_mae'])} against {money(d['mean_baseline_mae'])} for the flat mean. "
        f"The 95% band held {float(d['holdout_band_coverage_pct']):.0f}% of the "
        f"{int(d['holdout_days'])} holdout days. The series shows {trend} "
        f"(slope {float(d['slope_per_day']):+.3f} per day, p = {float(d['slope_p']):.3f}){weekday}. "
        "A near-flat projection is the honest result for a series without structure. It is "
        "useful for planning dispute-team capacity, not for predicting any single transaction.")
