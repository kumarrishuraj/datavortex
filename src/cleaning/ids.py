"""Identifier normalization.

Rule established in docs/01_forensic_audit.md section D2, from the observed
variants across all four files:

    USR12345 | usr12345 | USR-12345 | USR 12345 | usr_12345 | 12345
    MCH1234  | mch1234  | MCH-1234  | MCH 1234  | 1234
    TXN00012345 | txn-00012345
    CBK0001234

    uppercase -> strip [space hyphen underscore dot] -> strip leading zeros
              -> re-pad to the canonical width

The transaction file's own IDs were already uniformly clean; every variant
above comes from the KYC, merchant and chargeback files. Normalization is
deliberately conservative: it only rewrites *formatting*, never identity.
A value that does not reduce to "optional prefix + digits" returns None rather
than being coerced, so we never invent a key.
"""
from __future__ import annotations

import re

import pandas as pd

from src.config import ID_SPECS

_SEPARATORS = re.compile(r"[\s\-_.]")
_NULLISH = {"", "NAN", "NONE", "NULL", "NA", "N/A", "-"}


def normalize_id(value, prefix: str, width: int) -> str | None:
    """Return the canonical form of one identifier, or None if unparseable.

    >>> normalize_id("usr_00012345", "USR", 5)
    'USR12345'
    >>> normalize_id("12345", "USR", 5)
    'USR12345'
    >>> normalize_id("not-an-id", "USR", 5) is None
    True
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = _SEPARATORS.sub("", str(value).strip().upper())
    if s in _NULLISH:
        return None
    match = re.fullmatch(rf"(?:{prefix})?0*(\d+)", s)
    if not match:
        return None
    digits = match.group(1)
    # A value with more significant digits than the canonical width is a
    # different ID space, not a padding problem -> refuse rather than truncate.
    if len(digits) > width:
        return None
    return f"{prefix}{int(digits):0{width}d}"


def normalize_series(series: pd.Series, kind: str) -> pd.Series:
    """Vectorised wrapper. `kind` is a key of config.ID_SPECS."""
    if kind not in ID_SPECS:
        raise KeyError(f"unknown id kind {kind!r}; expected one of {list(ID_SPECS)}")
    prefix, width = ID_SPECS[kind]
    return series.map(lambda v: normalize_id(v, prefix, width))


def id_format_flag(original, normalized) -> str:
    """Classify what normalization had to do — kept for auditability."""
    if normalized is None:
        return "UNPARSEABLE" if str(original).strip() else "MISSING"
    if str(original).strip() == normalized:
        return "ALREADY_CANONICAL"
    return "REPAIRED"


def normalize_utr(value) -> tuple[str | None, bool]:
    """Normalize a UTR and report whether it matches the valid shape.

    The audit found every non-blank UTR valid once internal whitespace is
    removed, and proved that reused UTRs are duplicate rows rather than a
    fraud signal. So this reports format only and invents nothing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, False
    s = _SEPARATORS.sub("", str(value).strip().upper())
    if s in _NULLISH:
        return None, False
    return s, bool(re.fullmatch(r"UTR\d{10}", s))
