"""Chargeback parsing and attribution.

The decisive finding from the forensic audit (section D4): the chargeback
file's own `user_id`, `merchant_id` and `transaction_timestamp` columns do NOT
describe the transaction that `txn_id` points at. Of 2,683 linked chargebacks:

    cb.user_id     == linked transaction's user      ->  0  (0.000%)
    cb.merchant_id == linked transaction's merchant  ->  0  (0.000%)
    cb.transaction_timestamp on the same day         -> 55  (2.23%)
    cb.disputed_amount == transaction amount         ->  0

Chance alone would produce ~0.011% merchant agreement, so this is not damage —
those columns were generated independently. Meanwhile `txn_id` matches the
transaction table at 93.03% against a chance expectation of ~1 row (z = +3420),
confirming it is the one deliberately constructed foreign key.

Therefore:
  * entity attribution flows chargeback -> txn_id -> transaction -> user/merchant
  * the denormalized columns are retained, normalized and flagged, never joined

But the chargeback file IS internally coherent in time. Reporting delay measured
against the linked transaction gives a large share of negative (impossible) values;
measured against the file's own transaction_timestamp only a few percent are
negative and the distribution is a plausible dispute-ageing curve (audit_facts). So delay uses
the file's own timestamp pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_chargebacks(path: Path) -> pd.DataFrame:
    """Parse the JSON array into a flat frame, all columns as strings."""
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    df = pd.json_normalize(records)
    for col in df.columns:
        df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
    return df


def compute_reporting_delay(
    txn_ts: pd.Series, reported_ts: pd.Series
) -> pd.DataFrame:
    """Delay in days between the dispute's own transaction and report times.

    Deliberately does NOT use the transaction table's timestamp — see module
    docstring. Negative delays are flagged, not clipped or dropped.
    """
    delay = (reported_ts - txn_ts).dt.total_seconds() / 86400.0
    return pd.DataFrame(
        {
            "reporting_delay_days": delay.round(3),
            "delay_negative": delay.notna() & (delay < 0),
            "delay_missing": delay.isna(),
        },
        index=txn_ts.index,
    )


def compute_bank_response_delay(
    reported_ts: pd.Series, bank_ts: pd.Series
) -> pd.DataFrame:
    """Days from customer report to bank response."""
    delay = (bank_ts - reported_ts).dt.total_seconds() / 86400.0
    return pd.DataFrame(
        {
            "bank_response_days": delay.round(3),
            "bank_response_missing": delay.isna(),
        },
        index=reported_ts.index,
    )


# Ten complaint-text templates observed, plus case variants and trailing call
# centre noise ('urgent', 'NA', 'call dropped', 'details missing',
# 'pls check asap') and abbreviations ('txn', 'cust').
_TEXT_THEMES = [
    ("debited twice", "DUPLICATE_DEBIT"),
    ("not authorized", "UNAUTHORIZED"),
    ("upi pin was not entered", "UNAUTHORIZED"),
    ("unknown merchant", "UNAUTHORIZED"),
    ("compromised", "ACCOUNT_TAKEOVER"),
    ("not delivered", "SERVICE_NOT_PROVIDED"),
    ("denies receiving", "SERVICE_NOT_PROVIDED"),
    ("high-value", "SUSPICIOUS_HIGH_VALUE"),
    ("failed attempts", "REPEATED_FAILED_ATTEMPTS"),
    ("unclear", "UNCLEAR_NOTES"),
]

_NOISE_SUFFIXES = ["urgent", "na", "call dropped", "details missing", "pls check asap"]


def clean_complaint_text(value) -> str | None:
    """Trim call-centre noise suffixes and collapse whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = " ".join(str(value).split()).strip()
    if not s:
        return None
    lowered = s.lower()
    for suffix in _NOISE_SUFFIXES:
        if lowered.endswith(" " + suffix):
            s = s[: -(len(suffix) + 1)].strip()
            lowered = s.lower()
    return s or None


def classify_complaint_text(value) -> str:
    """Theme of the free-text complaint, independent of the reason code.

    Reported as its own dimension, NOT scored as an anomaly when it disagrees
    with reason_code: the audit measured 22.32% agreement against 21.41%
    expected under independence, so the two fields carry no mutual information
    and a mismatch means nothing.
    """
    if value is None:
        return "UNKNOWN"
    s = str(value).lower()
    for needle, theme in _TEXT_THEMES:
        if needle in s:
            return theme
    return "OTHER"
