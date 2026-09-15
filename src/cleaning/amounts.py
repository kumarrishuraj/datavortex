"""Monetary value parsing.

Observed forms across tx.amount, cb.disputed_amount, mer.declared_avg_ticket_size
and kyc.monthly_income (forensic audit section C):

    15722.34        Rs. 6362.9      INR 13,312      Rs. 548
    Rs. 1,250       INR 1250        1,250           1250.00
    27.3k           Not Available   ''              -23820.57

One regex handles all of them; every non-blank value in all four fields parses.

Negative transaction amounts are SIGN CORRUPTION, not refunds. Five independent
checks support this (audit section C):
  1. no refund/reversal/credit value exists anywhere in the status domain
  2. 0 of 429 negatives have a matching positive twin (user+merchant+|amount|)
  3. negative rate is flat across statuses (SUCCESS 2.07 / FAILED 2.46 / PENDING 1.97 %)
     -- the signature of uniform random corruption, not a business process
  4. |amount| distribution is identical to positives (median 12,136 vs 12,505)
  5. dispute rate among negatives is lower, not higher (10.02% vs 12.31%)

So transaction amounts are repaired with abs() and flagged. Negative income and
negative declared ticket size have no valid interpretation at all and are
flagged invalid rather than repaired.
"""
from __future__ import annotations

import re

import pandas as pd

# Strip currency symbols, thousands separators and whitespace. Kept as one
# compiled pattern so the same rule provably applies to every money column.
_CURRENCY = re.compile(r"(?:₹|rs\.?|inr|,|\s)", flags=re.IGNORECASE)
_K_SUFFIX = re.compile(r"^(-?[\d.]+)\s*k$", flags=re.IGNORECASE)
_NULLISH = {"", "nan", "none", "null", "na", "n/a", "-", "not available"}

# Parse outcome codes, surfaced as amount_parse_method in the fact tables.
OK = "OK"
K_SUFFIX = "K_SUFFIX"
BLANK = "BLANK"
INVALID = "INVALID"


def parse_amount(value) -> tuple[float | None, str]:
    """Parse one monetary value. Returns (value, parse_method).

    >>> parse_amount("₹1,250.50")
    (1250.5, 'OK')
    >>> parse_amount("Rs. 6362.9")
    (6362.9, 'OK')
    >>> parse_amount("27.3k")
    (27300.0, 'K_SUFFIX')
    >>> parse_amount("Not Available")
    (None, 'INVALID')
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, BLANK
    raw = str(value).strip()
    if raw.lower() in _NULLISH:
        return (None, INVALID) if raw.lower() == "not available" else (None, BLANK)

    cleaned = _CURRENCY.sub("", raw)

    k = _K_SUFFIX.match(cleaned)
    if k:
        try:
            return float(k.group(1)) * 1000, K_SUFFIX
        except ValueError:
            return None, INVALID
    try:
        return float(cleaned), OK
    except ValueError:
        return None, INVALID


def parse_amount_series(series: pd.Series) -> pd.DataFrame:
    """Parse a whole column into value + method frames."""
    parsed = series.map(parse_amount)
    return pd.DataFrame(
        {
            "value": [p[0] for p in parsed],
            "method": [p[1] for p in parsed],
        },
        index=series.index,
    )


def repair_sign(values: pd.Series) -> pd.DataFrame:
    """Apply the sign-corruption repair to a parsed transaction amount column.

    Returns the repaired magnitude plus the flag, never mutating in place.
    The original string column is always retained alongside by the caller.
    """
    negative = values.notna() & (values < 0)
    return pd.DataFrame(
        {
            "amount_inr": values.abs(),
            "amount_sign_invalid": negative.fillna(False),
        },
        index=values.index,
    )


def sign_repair_stats(values: pd.Series) -> dict:
    """Before / repaired / after counts for the data-quality report."""
    negative_before = int((values.notna() & (values < 0)).sum())
    repaired = values.abs()
    return {
        "negative_before": negative_before,
        "repaired": negative_before,
        "negative_after": int((repaired.notna() & (repaired < 0)).sum()),
        "sum_before": float(values.dropna().sum()),
        "sum_after": float(repaired.dropna().sum()),
    }
