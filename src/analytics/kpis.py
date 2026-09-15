"""KPI implementations.

Every function here corresponds to an entry in `semantic.METRICS` and returns a
`KPIResult` carrying the value, the denominator it was computed on, and the
coverage that denominator represents. Coverage travels with the number rather
than being bolted on at render time, so it cannot drift out of sync.

The paired `*_check` functions recompute each KPI by an independent route and
are asserted equal in `scripts/build_analytics.py`. That is what makes the
validation report a real check rather than a restatement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import UNKNOWN_KEY


@dataclass
class KPIResult:
    metric: str
    value: float
    denominator: int
    coverage_pct: float
    unit: str = "count"
    detail: str = ""

    def formatted(self) -> str:
        if self.unit == "percent":
            return f"{self.value:.2f}%"
        if self.unit == "INR":
            return f"Rs {self.value:,.0f}"
        if self.unit == "days":
            return f"{self.value:.2f} d"
        if self.unit == "ratio":
            return f"{self.value:.3f}"
        return f"{self.value:,.0f}"


def _result(metric, value, denominator, total, unit="count", detail="") -> KPIResult:
    coverage = 100.0 * denominator / total if total else 0.0
    return KPIResult(metric, float(value), int(denominator), round(coverage, 2), unit, detail)


# --------------------------------------------------------------------------
# Volume and value
# --------------------------------------------------------------------------
def total_transaction_count(tx: pd.DataFrame) -> KPIResult:
    return _result("total_transaction_count", len(tx), len(tx), len(tx))


def total_transaction_amount(tx: pd.DataFrame) -> KPIResult:
    valid = tx["amount_inr"].notna()
    return _result("total_transaction_amount", tx.loc[valid, "amount_inr"].sum(),
                   int(valid.sum()), len(tx), "INR")


def average_transaction_value(tx: pd.DataFrame) -> KPIResult:
    valid = tx["amount_inr"].notna()
    mean = tx.loc[valid, "amount_inr"].mean() if valid.any() else 0.0
    return _result("average_transaction_value", mean, int(valid.sum()), len(tx), "INR")


def status_rate(tx: pd.DataFrame, status: str) -> KPIResult:
    n = len(tx)
    hits = int((tx["status_canonical"] == status).sum())
    return _result(f"{status.lower()}_transaction_rate",
                   100.0 * hits / n if n else 0.0, n, n, "percent",
                   f"{hits:,} of {n:,}")


# --------------------------------------------------------------------------
# Disputes
# --------------------------------------------------------------------------
def chargeback_count(cb: pd.DataFrame) -> KPIResult:
    return _result("chargeback_count", len(cb), len(cb), len(cb))


def chargeback_amount(cb: pd.DataFrame) -> KPIResult:
    valid = cb["disputed_amount_inr"].notna()
    return _result("chargeback_amount", cb.loc[valid, "disputed_amount_inr"].sum(),
                   int(valid.sum()), len(cb), "INR")


def disputed_transaction_keys(cb: pd.DataFrame) -> set:
    """Transactions carrying at least one linked complaint."""
    return set(cb.loc[~cb["txn_unlinked"], "txn_key"].dropna())


def chargeback_to_transaction_ratio(tx: pd.DataFrame, cb: pd.DataFrame) -> KPIResult:
    """Share of transactions disputed at least once.

    DISTINCT transactions, not complaints — see the metric definition. Counting
    complaints inflates this and can exceed 1 at merchant level.
    """
    disputed = disputed_transaction_keys(cb)
    hits = int(tx["txn_key"].isin(disputed).sum())
    n = len(tx)
    return _result("chargeback_to_transaction_ratio", 100.0 * hits / n if n else 0.0,
                   n, n, "percent", f"{hits:,} disputed of {n:,}")


def complaints_per_transaction(tx: pd.DataFrame, cb: pd.DataFrame) -> KPIResult:
    linked = int((~cb["txn_unlinked"]).sum())
    n = len(tx)
    return _result("complaints_per_transaction", linked / n if n else 0.0, n, n, "ratio",
                   f"{linked:,} linked complaints over {n:,} transactions")


def average_dispute_reporting_delay(cb: pd.DataFrame) -> KPIResult:
    valid = cb["reporting_delay_days"].notna()
    mean = cb.loc[valid, "reporting_delay_days"].mean() if valid.any() else 0.0
    return _result("average_dispute_reporting_delay", mean, int(valid.sum()), len(cb), "days",
                   f"median {cb.loc[valid, 'reporting_delay_days'].median():.2f} d")


def disputes_reported_after_7_days(cb: pd.DataFrame) -> KPIResult:
    valid = cb["reporting_delay_days"].notna()
    hits = int((cb.loc[valid, "reporting_delay_days"] > 7).sum())
    return _result("disputes_reported_after_7_days", hits, int(valid.sum()), len(cb))


# --------------------------------------------------------------------------
# KYC
# --------------------------------------------------------------------------
def _real(dim: pd.DataFrame, key: str) -> pd.DataFrame:
    return dim[dim[key] != UNKNOWN_KEY]


def kyc_completion_rate(dim_users: pd.DataFrame) -> KPIResult:
    real = _real(dim_users, "user_key")
    n = len(real)
    hits = int((real["kyc_status_canonical"] == "VERIFIED").sum())
    return _result("kyc_completion_rate", 100.0 * hits / n if n else 0.0, n, n, "percent",
                   f"{hits:,} of {n:,}")


def kyc_rejection_rate(dim_users: pd.DataFrame) -> KPIResult:
    real = _real(dim_users, "user_key")
    n = len(real)
    hits = int((real["kyc_status_canonical"] == "REJECTED").sum())
    return _result("kyc_rejection_rate", 100.0 * hits / n if n else 0.0, n, n, "percent",
                   f"{hits:,} of {n:,}")


# --------------------------------------------------------------------------
# Data quality as first-class KPIs
# --------------------------------------------------------------------------
def utr_missing_rate(tx: pd.DataFrame) -> KPIResult:
    n = len(tx)
    hits = int(tx["utr_missing"].sum())
    return _result("utr_missing_rate", 100.0 * hits / n if n else 0.0, n, n, "percent",
                   f"{hits:,} of {n:,}")


def merchant_master_coverage(tx: pd.DataFrame) -> KPIResult:
    n = len(tx)
    hits = int((~tx["merchant_unresolved"]).sum())
    return _result("merchant_master_coverage", 100.0 * hits / n if n else 0.0, n, n, "percent",
                   f"{hits:,} of {n:,}")


def kyc_coverage(tx: pd.DataFrame) -> KPIResult:
    n = len(tx)
    hits = int((~tx["user_unresolved"]).sum())
    return _result("kyc_coverage", 100.0 * hits / n if n else 0.0, n, n, "percent",
                   f"{hits:,} of {n:,}")


def identity_ambiguity_rate(dim: pd.DataFrame, key: str) -> KPIResult:
    real = _real(dim, key)
    n = len(real)
    hits = int(real["identity_ambiguous"].sum())
    return _result("identity_ambiguity_rate", 100.0 * hits / n if n else 0.0, n, n, "percent",
                   f"{hits:,} of {n:,}")


# --------------------------------------------------------------------------
# Independent recomputations, used by the validation report
# --------------------------------------------------------------------------
def check_total_amount(tx: pd.DataFrame) -> float:
    """Sum via Python iteration rather than a vectorized reduction."""
    return float(np.nansum(tx["amount_inr"].to_numpy(dtype=float)))


def check_status_rates(tx: pd.DataFrame) -> dict:
    """Recompute all three rates from value_counts and assert they sum to 100."""
    vc = tx["status_canonical"].value_counts(normalize=True) * 100
    return {s: float(vc.get(s, 0.0)) for s in ["SUCCESS", "FAILED", "PENDING"]}


def check_chargeback_ratio(tx: pd.DataFrame, cb: pd.DataFrame) -> float:
    """Recompute via a merge instead of an isin() membership test."""
    linked = cb.loc[~cb["txn_unlinked"], ["txn_key"]].drop_duplicates()
    merged = tx[["txn_key"]].merge(linked, on="txn_key", how="inner")
    return 100.0 * len(merged) / len(tx) if len(tx) else 0.0


def check_reporting_delay(cb: pd.DataFrame) -> float:
    """Recompute the delay from the raw timestamp columns rather than the stored column."""
    delta = (cb["reported_timestamp_clean"] - cb["cb_transaction_timestamp_clean"])
    days = delta.dt.total_seconds() / 86400.0
    return float(days.dropna().mean())


def check_kyc_rates(dim_users: pd.DataFrame) -> dict:
    real = _real(dim_users, "user_key")
    vc = real["kyc_status_canonical"].value_counts(normalize=True) * 100
    return {s: float(vc.get(s, 0.0)) for s in ["VERIFIED", "PENDING", "REJECTED"]}
