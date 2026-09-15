"""Cached data access for the dashboard.

The dashboard is a presentation layer. It reads the Stage 2 star schema and the
Stage 3 materialized aggregates from Parquet, and computes KPIs by calling the
Stage 3 KPI functions — it never redefines a metric formula of its own. The raw
CSV and JSON files are never touched here.

Everything is cached, so a judge clicking between pages during a live demo pays
the load cost once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import kpis as K  # noqa: E402
from src.config import PROCESSED_DIR, UNKNOWN_KEY  # noqa: E402

ANALYTICS_DIR = PROCESSED_DIR / "analytics"

STAR_TABLES = [
    "fact_transactions", "fact_chargebacks", "dim_users",
    "dim_merchants", "dim_date", "bridge_identity_collision",
    # Reconciliation frames, written by the pipeline so the dashboard can show
    # before/after counts without re-reading raw files or hardcoding them.
    "recon_rows", "recon_accounting", "recon_value",
    # Forensic facts and relationship tests quoted in page text, computed by the pipeline.
    "audit_facts", "recon_fk", "recon_independence",
]

AGG_TABLES = [
    "agg_daily", "agg_hourly", "agg_merchant", "agg_user", "agg_category",
    "agg_kyc_status", "agg_state", "agg_chargeback_reason",
    "agg_chargeback_severity", "agg_chargeback_resolution",
    "agg_chargeback_channel", "agg_chargeback_theme", "agg_data_quality",
    "agg_hypothesis_register", "agg_network_summary", "agg_network_components",
    "agg_network_edges", "kpi_headline", "kpi_validation",
    "agg_risk_bands", "agg_risk_components",
    "agg_forecast", "agg_forecast_backtest", "agg_forecast_diagnostics",
]


class DataNotBuilt(FileNotFoundError):
    """Raised when the pipeline has not been run yet."""


def missing_tables() -> list[str]:
    missing = [t for t in STAR_TABLES if not (PROCESSED_DIR / f"{t}.parquet").exists()]
    missing += [t for t in AGG_TABLES if not (ANALYTICS_DIR / f"{t}.parquet").exists()]
    return missing


@st.cache_data(show_spinner=False)
def load_star() -> dict[str, pd.DataFrame]:
    missing = [t for t in STAR_TABLES if not (PROCESSED_DIR / f"{t}.parquet").exists()]
    if missing:
        raise DataNotBuilt(f"missing star tables: {missing}")
    return {t: pd.read_parquet(PROCESSED_DIR / f"{t}.parquet") for t in STAR_TABLES}


@st.cache_data(show_spinner=False)
def load_agg() -> dict[str, pd.DataFrame]:
    missing = [t for t in AGG_TABLES if not (ANALYTICS_DIR / f"{t}.parquet").exists()]
    if missing:
        raise DataNotBuilt(f"missing aggregates: {missing}")
    return {t: pd.read_parquet(ANALYTICS_DIR / f"{t}.parquet") for t in AGG_TABLES}


@st.cache_data(show_spinner=False)
def filter_options() -> dict[str, list]:
    """Distinct filter values, computed once."""
    star, agg = load_star(), load_agg()
    tx = star["fact_transactions"]
    dim_m = star["dim_merchants"].set_index("merchant_key")
    cats = sorted(agg["agg_category"]["category"].unique().tolist())
    states = sorted(agg["agg_state"]["state"].unique().tolist())
    return {
        "categories": cats,
        "states": states,
        "statuses": ["SUCCESS", "FAILED", "PENDING"],
        "merchant_statuses": sorted(
            dim_m["merchant_status_canonical"].dropna().unique().tolist()
        ),
        "kyc_statuses": sorted(agg["agg_kyc_status"]["kyc_status"].unique().tolist()),
        "date_min": pd.to_datetime(tx["timestamp_clean"]).min().date(),
        "date_max": pd.to_datetime(tx["timestamp_clean"]).max().date(),
        "merchant_coverage_pct": 100 * float((~tx["merchant_unresolved"]).mean()),
        "kyc_coverage_pct": 100 * float((~tx["user_unresolved"]).mean()),
    }


def enrich_transactions(tx: pd.DataFrame, dim_merchants: pd.DataFrame,
                        dim_users: pd.DataFrame) -> pd.DataFrame:
    """Attach category / state / KYC status, keeping UNKNOWN as a real value.

    Grouping keys stay the entity identifiers (`*_id_normalized`) so merchants
    and users absent from the master are still counted; only their descriptive
    attributes fall back to UNKNOWN.
    """
    cat = dim_merchants.set_index("merchant_key")["merchant_category_canonical"]
    state = dim_merchants.set_index("merchant_key")["state_clean"]
    mstatus = dim_merchants.set_index("merchant_key")["merchant_status_canonical"]
    kyc = dim_users.set_index("user_key")["kyc_status_canonical"]
    out = tx.copy()
    out["category"] = out["merchant_id_normalized"].map(cat).fillna("UNKNOWN (no master record)")
    out["state"] = out["merchant_id_normalized"].map(state).fillna(UNKNOWN_KEY)
    out["merchant_status"] = out["merchant_id_normalized"].map(mstatus).fillna(UNKNOWN_KEY)
    out["kyc_status"] = out["user_id_normalized"].map(kyc).fillna(UNKNOWN_KEY)
    return out


@st.cache_data(show_spinner=False)
def enriched_transactions() -> pd.DataFrame:
    star = load_star()
    return enrich_transactions(
        star["fact_transactions"], star["dim_merchants"], star["dim_users"]
    )


def apply_filters(tx: pd.DataFrame, date_range=None, categories=None,
                  statuses=None, states=None) -> pd.DataFrame:
    """Apply the sidebar filters. Empty selections mean 'no filter'."""
    out = tx
    if date_range and len(date_range) == 2:
        lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        out = out[(out["timestamp_clean"] >= lo) & (out["timestamp_clean"] < hi)]
    if categories:
        out = out[out["category"].isin(categories)]
    if statuses:
        out = out[out["status_canonical"].isin(statuses)]
    if states:
        out = out[out["state"].isin(states)]
    return out


def linked_chargebacks(cb: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Complaints attributable to the given transactions, via txn_key ONLY.

    The chargeback file's own user_id / merchant_id columns are never used for
    attribution: they do not agree with the transaction they point at (see
    audit_facts), so joining on them would produce a different, wrong entity universe.
    """
    return cb[cb["txn_key"].isin(set(tx["txn_key"]))]


def headline_kpis(tx: pd.DataFrame, cb: pd.DataFrame, dim_users: pd.DataFrame) -> dict:
    """Compute headline KPIs via the Stage 3 functions, never re-derived here."""
    return {
        "count": K.total_transaction_count(tx),
        "amount": K.total_transaction_amount(tx),
        "average": K.average_transaction_value(tx),
        "success": K.status_rate(tx, "SUCCESS"),
        "failed": K.status_rate(tx, "FAILED"),
        "pending": K.status_rate(tx, "PENDING"),
        "cb_count": K.chargeback_count(cb),
        "cb_amount": K.chargeback_amount(cb),
        "cb_ratio": K.chargeback_to_transaction_ratio(tx, cb),
        "delay": K.average_dispute_reporting_delay(cb),
        "delay_7d": K.disputes_reported_after_7_days(cb),
        "kyc_complete": K.kyc_completion_rate(dim_users),
        "kyc_reject": K.kyc_rejection_rate(dim_users),
        "utr_missing": K.utr_missing_rate(tx),
        "merchant_cov": K.merchant_master_coverage(tx),
        "kyc_cov": K.kyc_coverage(tx),
    }


@st.cache_data(show_spinner=False)
def load_agent_eval() -> pd.DataFrame | None:
    """The agent evaluation matrix written by scripts/run_agent_eval.py, if present.

    Read from Parquet rather than recomputed on page load, so the test-matrix tab
    shows exactly what the evaluation script measured.
    """
    path = ANALYTICS_DIR / "agent_eval.parquet"
    return pd.read_parquet(path) if path.exists() else None
