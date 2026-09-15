"""Categorical canonicalization.

Every mapping below is built from the *observed* value domain only — no value
was invented, and no observed value is left unmapped. Counts in the comments
are the raw occurrence counts from the forensic audit, so a reviewer can check
that a mapping accounts for all of them.

The original value is always retained alongside the canonical one, so no
information is destroyed by canonicalizing.
"""
from __future__ import annotations

import pandas as pd

UNKNOWN = "UNKNOWN"


def _build(mapping: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Invert {canonical: (variants,)} into {variant_upper: canonical}."""
    out = {}
    for canonical, variants in mapping.items():
        for v in variants:
            out[v.strip().upper()] = canonical
    return out


# --- transaction status: 14 observed variants -> 3 canonical -----------------
# SUCCESS 17,395 (85.27%) | FAILED 1,990 (9.76%) | PENDING 1,015 (4.98%)
TXN_STATUS = _build(
    {
        "SUCCESS": ("S", "SUCCESS", "Success", "TXN_SUCCESS", "COMPLETED"),
        "FAILED": ("F", "FAILED", "TXN_FAILED", "Fail", "Declined"),
        "PENDING": ("PENDING", "Pending", "PROCESSING", "Initiated"),
    }
)

# --- KYC status: 16 observed variants -> 3 canonical -------------------------
KYC_STATUS = _build(
    {
        "VERIFIED": ("V", "VERIFIED", "Verified", "APPROVED", "KYC_DONE", "Done"),
        "PENDING": ("P", "PENDING", "Pending", "IN_PROGRESS", "Under Review"),
        "REJECTED": ("R", "REJECTED", "Rejected", "Reject", "FAILED"),
    }
)

# --- risk segment: 12 observed variants -> 4 (pure case differences) ---------
RISK_SEGMENT = _build(
    {
        "LOW": ("LOW", "low", "Low"),
        "MEDIUM": ("MEDIUM", "medium", "Medium"),
        "HIGH": ("HIGH", "high", "High"),
        "UNKNOWN": ("UNKNOWN", "unknown", "Unknown"),
    }
)

# --- merchant status: 15 observed variants -> 4 canonical --------------------
MERCHANT_STATUS = _build(
    {
        "ACTIVE": ("A", "ACTIVE", "Active", "Enabled", "Live"),
        "INACTIVE": ("I", "INACTIVE", "Inactive", "Disabled", "Closed"),
        "SUSPENDED": ("S", "SUSPENDED", "Suspended", "Hold"),
        "BLOCKED": ("Blocked",),
    }
)

# --- business type: 14 observed variants -> 4 canonical ---------------------
BUSINESS_TYPE = _build(
    {
        "INDIVIDUAL": ("INDIVIDUAL", "individual", "Individual"),
        "PARTNERSHIP": ("PARTNERSHIP", "partnership", "Partnership"),
        "PRIVATE_LIMITED": (
            "PRIVATE_LIMITED",
            "PRIVATE-LIMITED",
            "private_limited",
            "Private Limited",
        ),
        "SOLE_PROPRIETOR": (
            "SOLE_PROPRIETOR",
            "SOLE-PROPRIETOR",
            "sole_proprietor",
            "Sole Proprietor",
        ),
    }
)

# --- chargeback severity: 16 observed variants -> 4 canonical ---------------
# Both a word scale (L/M/H/CRIT) and a priority scale (P4..P1) are present.
SEVERITY = _build(
    {
        "LOW": ("L", "LOW", "Low", "P4"),
        "MEDIUM": ("M", "MEDIUM", "Medium", "P3"),
        "HIGH": ("H", "HIGH", "High", "P2"),
        "CRITICAL": ("CRIT", "CRITICAL", "Critical", "P1"),
    }
)

# --- chargeback resolution status: 13 observed variants -> 6 canonical -------
RESOLUTION_STATUS = _build(
    {
        "OPEN": ("OPEN", "Open"),
        "IN_PROGRESS": ("IN_PROGRESS", "In Progress", "WIP"),
        "PENDING_BANK": ("PENDING_BANK", "Pending Bank"),
        "RESOLVED": ("RESOLVED", "Resolved"),
        "CLOSED": ("CLOSED", "Closed"),
        "REJECTED": ("REJECTED", "Rejected"),
    }
)

# --- chargeback channel: 8 observed variants -> 6 canonical -----------------
CHANNEL = _build(
    {
        "IVR": ("IVR", "ivr"),
        "CHATBOT": ("CHATBOT", "chatbot"),
        "APP": ("App",),
        "EMAIL": ("Email",),
        "BRANCH": ("Branch",),
        "CALL_CENTER": ("Call Center",),
    }
)

# --- chargeback reason code: 34 observed variants -> 6 canonical ------------
# Grouped by stated meaning. 'OTHER' holds the genuinely non-specific codes
# ("complaint", "customer issue", "dispute raised", "Customer Dispute") rather
# than being a dumping ground for anything unmapped.
REASON_CODE = _build(
    {
        "UNAUTHORIZED": (
            "Unauthorized Transaction",
            "unauthorized_transaction",
            "unauth txn",
            "UNAUTHORISED",
            "not done by me",
        ),
        "DUPLICATE_DEBIT": (
            "Duplicate Debit",
            "DUP_DEBIT",
            "double debit",
            "charged twice",
        ),
        "SERVICE_NOT_PROVIDED": (
            "Service Not Provided",
            "Merchant Not Delivered",
            "not delivered",
            "no service",
            "item not received",
            "delivery issue",
            "service failed",
            "merchant service issue",
        ),
        "WRONG_AMOUNT": (
            "Wrong Amount",
            "amount mismatch",
            "incorrect amount",
            "extra amount deducted",
        ),
        "ACCOUNT_TAKEOVER": (
            "Account Takeover",
            "ATO",
            "account hacked",
            "login compromised",
        ),
        "FRAUD_SUSPECTED": (
            "FRAUD",
            "fraud",
            "Fraud Suspected",
            "scam",
            "suspicious transaction",
        ),
        "OTHER": (
            "complaint",
            "customer issue",
            "dispute raised",
            "Customer Dispute",
        ),
    }
)


def canonicalize(series: pd.Series, mapping: dict[str, str], default: str = UNKNOWN) -> pd.Series:
    """Map a raw categorical column onto its canonical domain.

    Values outside the mapping become `default` rather than raising, but the
    pipeline separately asserts the unmapped count is zero for every column
    where the audit enumerated the full domain.
    """
    return series.fillna("").astype(str).str.strip().str.upper().map(mapping).fillna(default)


def unmapped_values(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Which raw values a mapping fails to cover — used as a pipeline guard."""
    norm = series.fillna("").astype(str).str.strip().str.upper()
    missing = norm[~norm.isin(mapping) & (norm != "")]
    return missing.value_counts()
