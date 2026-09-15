"""Materialized aggregates.

The dashboard reads only these tables, never the facts. Three rules hold
everywhere:

1. **Every aggregate carries `coverage_pct`** — the share of the full
   transaction population the row was computed on — so a chart cannot present
   a 48%-coverage number as if it were the whole book.

2. **Entity aggregation groups by `*_id_normalized`; attributes join on
   `*_key`.** Merchant exposure therefore covers all 8,051 transacting
   merchants, while category and status are only available for the 3,893 with
   a master record. Grouping by the FK would collapse the other 4,158 into one
   UNKNOWN bucket and hide half the disputes.

3. **UNKNOWN is a visible member**, never filtered out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.kpis import disputed_transaction_keys
from src.analytics.shrinkage import (
    MIN_DENOMINATOR,
    binomial_pvalue,
    estimate_prior_strength,
    excess_over_peer,
    shrink_rates,
)
from src.config import UNKNOWN_KEY

UNKNOWN_LABEL = "UNKNOWN (no master record)"


def _flag_disputed(tx: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """Attach per-transaction dispute facts to the transaction fact table."""
    out = tx.copy()
    disputed = disputed_transaction_keys(cb)
    out["is_disputed"] = out["txn_key"].isin(disputed)
    linked = cb[~cb["txn_unlinked"]]
    out["complaint_count"] = out["txn_key"].map(linked["txn_key"].value_counts()).fillna(0).astype(int)
    amounts = linked.groupby("txn_key")["disputed_amount_inr"].sum()
    out["disputed_amount_inr"] = out["txn_key"].map(amounts).fillna(0.0)
    return out


def _coverage(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


# --------------------------------------------------------------------------
def agg_daily(tx: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """Daily volume, value and outcome mix."""
    t = _flag_disputed(tx, cb)
    total = len(t)
    g = t.groupby("date", dropna=False)
    out = g.agg(
        transactions=("txn_key", "size"),
        transaction_value=("amount_inr", "sum"),
        avg_transaction_value=("amount_inr", "mean"),
        disputed_transactions=("is_disputed", "sum"),
        disputed_amount=("disputed_amount_inr", "sum"),
        complaints=("complaint_count", "sum"),
    ).reset_index()
    for status in ["SUCCESS", "FAILED", "PENDING"]:
        counts = t[t["status_canonical"] == status].groupby("date").size()
        out[f"{status.lower()}_count"] = out["date"].map(counts).fillna(0).astype(int)
        out[f"{status.lower()}_rate_pct"] = (100 * out[f"{status.lower()}_count"] / out["transactions"]).round(2)
    out["chargeback_rate_pct"] = (100 * out["disputed_transactions"] / out["transactions"]).round(2)
    out["coverage_pct"] = 100.0
    out["coverage_basis"] = "all transactions"
    return out.sort_values("date").reset_index(drop=True)


def agg_hourly(tx: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day profile, EXCLUDING date-only timestamps.

    1,000 transactions carry a date but no clock time and parse to 00:00:00.
    Including them creates a spurious midnight peak of 1,802 against a ~790
    baseline. Restricted to rows with a known time, the profile is flat
    (CV 0.031) — this dataset has no intraday pattern, and the aggregate says so
    rather than letting a parsing artifact read as customer behaviour.
    """
    known = tx[tx["timestamp_time_known"]]
    total = len(tx)
    out = known.groupby("hour").agg(
        transactions=("txn_key", "size"),
        transaction_value=("amount_inr", "sum"),
    ).reset_index()
    out["excluded_date_only"] = total - len(known)
    out["coverage_pct"] = _coverage(len(known), total)
    out["coverage_basis"] = "transactions with a clock time in the source"
    return out


def agg_merchant(tx: pd.DataFrame, cb: pd.DataFrame, dim_merchants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-merchant exposure, over the FULL transacting population.

    Returns (frame, prior) where `prior` holds the overdispersion diagnostics
    so the dashboard and the validation report can show the statistical basis
    for the shrinkage rather than asserting it.
    """
    t = _flag_disputed(tx, cb)
    total = len(t)
    g = t.groupby("merchant_id_normalized", dropna=False)
    out = g.agg(
        transactions=("txn_key", "size"),
        transaction_value=("amount_inr", "sum"),
        avg_ticket=("amount_inr", "mean"),
        disputed_transactions=("is_disputed", "sum"),
        complaints=("complaint_count", "sum"),
        disputed_amount=("disputed_amount_inr", "sum"),
        failed=("status_canonical", lambda s: int((s == "FAILED").sum())),
        utr_missing=("utr_missing", "sum"),
        first_seen=("timestamp_clean", "min"),
        last_seen=("timestamp_clean", "max"),
    ).reset_index()
    out["failed_rate_pct"] = (100 * out["failed"] / out["transactions"]).round(2)

    # Attributes come from the master where one exists; UNKNOWN stays visible.
    attrs = dim_merchants[[
        "merchant_key", "merchant_name_clean", "merchant_category_canonical",
        "merchant_status_canonical", "business_type_canonical", "city_clean",
        "state_clean", "identity_ambiguous", "resolution_confidence",
    ]].rename(columns={"merchant_key": "merchant_id_normalized"})
    out = out.merge(attrs, on="merchant_id_normalized", how="left")
    out["in_master"] = out["merchant_name_clean"].notna()
    out["merchant_category_canonical"] = out["merchant_category_canonical"].fillna(UNKNOWN_LABEL)
    out["merchant_status_canonical"] = out["merchant_status_canonical"].fillna(UNKNOWN_KEY)
    out["state_clean"] = out["state_clean"].fillna(UNKNOWN_KEY)
    out["merchant_name_clean"] = out["merchant_name_clean"].fillna("(not in merchant master)")
    out["identity_ambiguous"] = out["identity_ambiguous"].fillna(False).astype(bool)

    shrunk = shrink_rates(
        out, "disputed_transactions", "transactions",
        group_col="merchant_category_canonical", min_denominator=MIN_DENOMINATOR,
    )
    prior = shrunk.attrs["prior"]
    shrunk["dispute_rate_raw_pct"] = (100 * shrunk["rate_raw"]).round(2)
    shrunk["dispute_rate_shrunk_pct"] = (100 * shrunk["rate_shrunk"]).round(3)
    shrunk["peer_mean_pct"] = (100 * shrunk["peer_mean"]).round(3)
    shrunk["excess_over_peer_pp"] = excess_over_peer(shrunk).round(3)

    baseline = float(shrunk["disputed_transactions"].sum() / max(shrunk["transactions"].sum(), 1))
    eligible = shrunk[~shrunk["below_floor"]]
    shrunk["dispute_pvalue"] = np.nan
    shrunk.loc[eligible.index, "dispute_pvalue"] = binomial_pvalue(
        eligible["disputed_transactions"], eligible["transactions"], baseline
    ).round(4)

    shrunk["coverage_pct"] = 100.0
    shrunk["coverage_basis"] = "all transacting merchants"
    shrunk["attributes_available"] = shrunk["in_master"]
    shrunk = shrunk.drop(columns=["rate_raw", "rate_shrunk", "peer_mean"])
    return shrunk.sort_values("disputed_amount", ascending=False).reset_index(drop=True), prior


def agg_user(tx: pd.DataFrame, cb: pd.DataFrame, dim_users: pd.DataFrame) -> pd.DataFrame:
    """Per-customer exposure over the full transacting population."""
    t = _flag_disputed(tx, cb)
    g = t.groupby("user_id_normalized", dropna=False)
    out = g.agg(
        transactions=("txn_key", "size"),
        transaction_value=("amount_inr", "sum"),
        disputed_transactions=("is_disputed", "sum"),
        complaints=("complaint_count", "sum"),
        disputed_amount=("disputed_amount_inr", "sum"),
        merchants_used=("merchant_id_normalized", "nunique"),
        first_seen=("timestamp_clean", "min"),
        last_seen=("timestamp_clean", "max"),
    ).reset_index()

    attrs = dim_users[[
        "user_key", "full_name_clean", "kyc_status_canonical", "risk_segment_canonical",
        "city_clean", "state_clean", "occupation_clean", "monthly_income_inr",
        "identity_ambiguous", "resolution_confidence",
    ]].rename(columns={"user_key": "user_id_normalized"})
    out = out.merge(attrs, on="user_id_normalized", how="left")
    out["in_kyc"] = out["full_name_clean"].notna()
    out["kyc_status_canonical"] = out["kyc_status_canonical"].fillna(UNKNOWN_KEY)
    out["risk_segment_canonical"] = out["risk_segment_canonical"].fillna(UNKNOWN_KEY)
    out["state_clean"] = out["state_clean"].fillna(UNKNOWN_KEY)
    out["identity_ambiguous"] = out["identity_ambiguous"].fillna(False).astype(bool)
    out["dispute_rate_raw_pct"] = (100 * out["disputed_transactions"] / out["transactions"]).round(2)
    out["below_floor"] = out["transactions"] < MIN_DENOMINATOR
    out["coverage_pct"] = 100.0
    out["coverage_basis"] = "all transacting customers"
    return out.sort_values("disputed_amount", ascending=False).reset_index(drop=True)


def agg_category(tx: pd.DataFrame, cb: pd.DataFrame, dim_merchants: pd.DataFrame) -> pd.DataFrame:
    """Merchant-category performance, with UNKNOWN kept as a visible member."""
    t = _flag_disputed(tx, cb)
    total = len(t)
    cat = dim_merchants.set_index("merchant_key")["merchant_category_canonical"]
    t = t.assign(category=t["merchant_id_normalized"].map(cat).fillna(UNKNOWN_LABEL))
    out = t.groupby("category").agg(
        transactions=("txn_key", "size"),
        transaction_value=("amount_inr", "sum"),
        avg_transaction_value=("amount_inr", "mean"),
        disputed_transactions=("is_disputed", "sum"),
        complaints=("complaint_count", "sum"),
        disputed_amount=("disputed_amount_inr", "sum"),
        merchants=("merchant_id_normalized", "nunique"),
        failed=("status_canonical", lambda s: int((s == "FAILED").sum())),
    ).reset_index()
    out["dispute_rate_pct"] = (100 * out["disputed_transactions"] / out["transactions"]).round(2)
    out["failed_rate_pct"] = (100 * out["failed"] / out["transactions"]).round(2)
    out["share_of_transactions_pct"] = (100 * out["transactions"] / total).round(2)
    out["is_unknown"] = out["category"] == UNKNOWN_LABEL
    known = int(out.loc[~out["is_unknown"], "transactions"].sum())
    out["coverage_pct"] = _coverage(known, total)
    out["coverage_basis"] = "merchant master coverage; UNKNOWN row shown separately"
    return out.sort_values("transactions", ascending=False).reset_index(drop=True)


def agg_chargeback_dimension(cb: pd.DataFrame, column: str) -> pd.DataFrame:
    """Complaint breakdown by any canonical chargeback dimension."""
    total = len(cb)
    out = cb.groupby(column, dropna=False).agg(
        complaints=("complaint_key", "size"),
        disputed_amount=("disputed_amount_inr", "sum"),
        avg_disputed_amount=("disputed_amount_inr", "mean"),
        avg_reporting_delay_days=("reporting_delay_days", "mean"),
        linked_to_transaction=("txn_unlinked", lambda s: int((~s).sum())),
    ).reset_index()
    out["share_pct"] = (100 * out["complaints"] / total).round(2)
    out["avg_reporting_delay_days"] = out["avg_reporting_delay_days"].round(2)
    out["coverage_pct"] = 100.0
    out["coverage_basis"] = "all cleaned complaints"
    return out.sort_values("complaints", ascending=False).reset_index(drop=True)


def agg_kyc_status(tx: pd.DataFrame, cb: pd.DataFrame, dim_users: pd.DataFrame) -> pd.DataFrame:
    """Transaction behaviour by KYC status, with UNKNOWN visible.

    UNKNOWN is the largest group here (67.61% of transactions). Dropping it
    would make the remaining segments look like the whole customer base.
    """
    t = _flag_disputed(tx, cb)
    total = len(t)
    status = dim_users.set_index("user_key")["kyc_status_canonical"]
    t = t.assign(kyc_status=t["user_id_normalized"].map(status).fillna(UNKNOWN_KEY))
    out = t.groupby("kyc_status").agg(
        transactions=("txn_key", "size"),
        transaction_value=("amount_inr", "sum"),
        avg_transaction_value=("amount_inr", "mean"),
        disputed_transactions=("is_disputed", "sum"),
        disputed_amount=("disputed_amount_inr", "sum"),
        customers=("user_id_normalized", "nunique"),
    ).reset_index()
    out["dispute_rate_pct"] = (100 * out["disputed_transactions"] / out["transactions"]).round(2)
    out["share_of_transactions_pct"] = (100 * out["transactions"] / total).round(2)
    out["is_unknown"] = out["kyc_status"] == UNKNOWN_KEY
    known = int(out.loc[~out["is_unknown"], "transactions"].sum())
    out["coverage_pct"] = _coverage(known, total)
    out["coverage_basis"] = "KYC coverage; UNKNOWN row shown separately"
    return out.sort_values("transactions", ascending=False).reset_index(drop=True)


def agg_state(tx: pd.DataFrame, cb: pd.DataFrame, dim_merchants: pd.DataFrame) -> pd.DataFrame:
    """Geography via the merchant master, with UNKNOWN visible."""
    t = _flag_disputed(tx, cb)
    total = len(t)
    state = dim_merchants.set_index("merchant_key")["state_clean"]
    t = t.assign(state=t["merchant_id_normalized"].map(state).fillna(UNKNOWN_KEY))
    out = t.groupby("state").agg(
        transactions=("txn_key", "size"),
        transaction_value=("amount_inr", "sum"),
        disputed_transactions=("is_disputed", "sum"),
        disputed_amount=("disputed_amount_inr", "sum"),
        merchants=("merchant_id_normalized", "nunique"),
    ).reset_index()
    out["dispute_rate_pct"] = (100 * out["disputed_transactions"] / out["transactions"]).round(2)
    out["is_unknown"] = out["state"] == UNKNOWN_KEY
    known = int(out.loc[~out["is_unknown"], "transactions"].sum())
    out["coverage_pct"] = _coverage(known, total)
    out["coverage_basis"] = "merchant master coverage; UNKNOWN row shown separately"
    return out.sort_values("transactions", ascending=False).reset_index(drop=True)


def agg_data_quality(tx, cb, dim_users, dim_merchants) -> pd.DataFrame:
    """The data-quality proof panel, generated rather than hardcoded."""
    rows = [
        ("Transactions (cleaned)", len(tx), "", 100.0),
        ("Amount sign-corrupted", int(tx["amount_sign_invalid"].sum()), "repaired to magnitude + flagged",
         round(100 * tx["amount_sign_invalid"].mean(), 2)),
        ("UTR missing", int(tx["utr_missing"].sum()), "never fabricated",
         round(100 * tx["utr_missing"].mean(), 2)),
        ("Timestamp date-only", int((~tx["timestamp_time_known"]).sum()), "excluded from hourly analysis",
         round(100 * (~tx["timestamp_time_known"]).mean(), 2)),
        ("Merchant unresolved", int(tx["merchant_unresolved"].sum()), "routed to UNKNOWN, retained",
         round(100 * tx["merchant_unresolved"].mean(), 2)),
        ("User unresolved", int(tx["user_unresolved"].sum()), "routed to UNKNOWN, retained",
         round(100 * tx["user_unresolved"].mean(), 2)),
        ("Complaints (cleaned)", len(cb), "", 100.0),
        ("Complaints unlinked", int(cb["txn_unlinked"].sum()), "no attributable transaction",
         round(100 * cb["txn_unlinked"].mean(), 2)),
        ("Complaint delay unknown", int(cb["delay_missing"].sum()), "timestamp unparseable",
         round(100 * cb["delay_missing"].mean(), 2)),
        ("Complaint delay negative", int(cb["delay_negative"].sum()), "flagged, not clipped",
         round(100 * cb["delay_negative"].mean(), 2)),
        ("Customer identities", len(dim_users) - 1, "", 100.0),
        ("Customers with ambiguous ID", int(dim_users["identity_ambiguous"].sum()),
         "different people share the ID", round(100 * dim_users["identity_ambiguous"].mean(), 2)),
        ("Merchant identities", len(dim_merchants) - 1, "", 100.0),
        ("Merchants with ambiguous ID", int(dim_merchants["identity_ambiguous"].sum()),
         "different businesses share the ID", round(100 * dim_merchants["identity_ambiguous"].mean(), 2)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "treatment", "pct"])
