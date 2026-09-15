"""DataVortex — Streamlit entry point.

    streamlit run app/dashboard.py

Reads only the Stage 2 star schema and the Stage 3 materialized aggregates.
If they are missing, the app says how to build them rather than failing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="DataVortex — UPI Fraud Intelligence",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components import data as D  # noqa: E402
from app.components.theme import CSS  # noqa: E402
from app.components.ui import callout  # noqa: E402
from app.views import (  # noqa: E402
    p1_overview, p2_transactions, p3_risk, p4_quality,
    p5_identity, p6_chargebacks, p7_investigator,
)

PAGES = {
    "Executive Overview": p1_overview.render,
    "Transaction Analytics": p2_transactions.render,
    "Merchant & User Risk": p3_risk.render,
    "Data Quality & Hypotheses": p4_quality.render,
    "Identity & Network": p5_identity.render,
    "Chargeback Intelligence": p6_chargebacks.render,
    "AI Investigator": p7_investigator.render,
}

# Pages whose content is global rather than a slice of the transaction book.
# Applying the transaction filters to them would be misleading.
UNFILTERED_PAGES = {"Data Quality & Hypotheses", "Identity & Network", "AI Investigator"}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    missing = D.missing_tables()
    if missing:
        st.title("DataVortex")
        st.error("The analytics layer has not been built yet.")
        st.markdown(
            "Run the two build steps, then reload:\n\n"
            "```bash\n"
            "python scripts/run_pipeline.py      # Stage 2 — star schema\n"
            "python scripts/build_analytics.py   # Stage 3 — aggregates\n"
            "```")
        with st.expander("Missing tables"):
            st.write(missing)
        return

    try:
        star = D.load_star()
        agg = D.load_agg()
        tx_all = D.enriched_transactions()
        options = D.filter_options()
    except Exception as exc:  # noqa: BLE001
        st.title("DataVortex")
        st.error(f"Could not load the analytics layer: {type(exc).__name__}: {exc}")
        st.markdown("Rebuild with `python scripts/run_pipeline.py && "
                    "python scripts/build_analytics.py`.")
        return

    page = _sidebar(options, agg)

    filters_active = False
    tx = tx_all
    if page not in UNFILTERED_PAGES:
        tx, filters_active = _apply_sidebar_filters(tx_all, options)

    cb_all = star["fact_chargebacks"]
    cb = D.linked_chargebacks(cb_all, tx) if filters_active else cb_all

    ctx = {
        "star": star,
        "agg": agg,
        "tx": tx,
        "tx_all": tx_all,
        "cb": cb,
        "cb_all": cb_all,
        "kpis": D.headline_kpis(tx, cb, star["dim_users"]),
        "options": options,
        "filters_active": filters_active,
        "total_transactions": len(tx_all),
        "page": page,
    }

    try:
        PAGES[page](ctx)
    except Exception as exc:  # noqa: BLE001 — a page error must not blank the app
        st.error(f"This page could not be rendered: {type(exc).__name__}: {exc}")
        with st.expander("Details"):
            import traceback
            st.code(traceback.format_exc(), language="text")


def _sidebar(options, agg) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="dv-brand">'
            '<div class="dv-brand-name">🌀 DataVortex</div>'
            '<div class="dv-brand-tag">Turning UPI Data Chaos into Fraud Intelligence</div>'
            "</div>", unsafe_allow_html=True)

        page = st.radio("Section", list(PAGES), label_visibility="collapsed")

        st.markdown("---")
        head = agg["kpi_headline"].set_index("metric")
        st.markdown(
            f'<div class="dv-kpi-label">Data build</div>'
            f'<div class="dv-kpi-sub" style="line-height:1.7">'
            f'{int(head.loc["total_transaction_count", "value"]):,} transactions<br>'
            f'{int(head.loc["chargeback_count", "value"]):,} chargebacks<br>'
            f'{options["date_min"]:%d %b %Y} – {options["date_max"]:%d %b %Y}</div>',
            unsafe_allow_html=True)
        return page


def _apply_sidebar_filters(tx_all, options):
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="dv-kpi-label">Filters</div>', unsafe_allow_html=True)

        date_range = st.date_input(
            "Date range", value=(options["date_min"], options["date_max"]),
            min_value=options["date_min"], max_value=options["date_max"],
            help="Defaults to the full observed transaction window.")
        statuses = st.multiselect(
            "Transaction status", options["statuses"], default=[],
            help="Empty means no filter.")
        categories = st.multiselect(
            "Merchant category", options["categories"], default=[],
            help=f"Category comes from the merchant master, which covers "
                 f"{options['merchant_coverage_pct']:.2f}% of transactions. "
                 f"UNKNOWN is a real selectable value.")
        states = st.multiselect(
            "State", options["states"], default=[],
            help="Location comes from the merchant master.")

        active = bool(statuses or categories or states)
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            active = active or (date_range[0] != options["date_min"]
                                or date_range[1] != options["date_max"])
        else:
            date_range = None

        if active and st.button("Clear filters", width="stretch"):
            st.rerun()

    tx = D.apply_filters(tx_all, date_range=date_range, categories=categories,
                         statuses=statuses, states=states)
    return tx, active


if __name__ == "__main__":
    main()
