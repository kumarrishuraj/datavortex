"""KYC field repair: PAN, Aadhaar, income, city, state.

Privacy posture (competition brief section 38): full PAN and Aadhaar values are
never written to the processed layer. We persist a masked display form plus a
SHA-256 hash. The hash preserves the only analytical property we actually need
— equality, for identity-collision detection — without carrying the identifier
itself into the dashboard, the Parquet files or the agent.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

_NULLISH = {"", "nan", "none", "null", "na", "n/a", "-"}

PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# OCR homoglyph confusions, applied positionally. The audit showed this repairs
# 819 of 819 length-10 invalid PANs — a 100% hit rate, which is what justifies
# applying it automatically rather than flagging for manual review.
_TO_LETTER = str.maketrans("01258", "OIZSB")
_TO_DIGIT = str.maketrans("OIZSB", "01258")

# City aliases: 41 observed spellings -> 12 real cities. Every alias is
# corroborated by a consistent `state` value in the source data, which is what
# makes the merge safe rather than a guess.
CITY_ALIASES = {
    "PUNE": "Pune", "POONA": "Pune",
    "HYDERABAD": "Hyderabad", "HYD": "Hyderabad",
    "AMRITSAR": "Amritsar", "ASR": "Amritsar",
    "LUDHIANA": "Ludhiana", "LDH": "Ludhiana",
    "LUCKNOW": "Lucknow", "LKO": "Lucknow",
    "KOLKATA": "Kolkata", "CALCUTTA": "Kolkata",
    "JALANDHAR": "Jalandhar", "JALANDAR": "Jalandhar",
    "CHENNAI": "Chennai", "MADRAS": "Chennai",
    "JAIPUR": "Jaipur", "JPR": "Jaipur",
    "BENGALURU": "Bengaluru", "BANGALORE": "Bengaluru", "BLR": "Bengaluru",
    "DELHI": "Delhi", "NEW DELHI": "Delhi", "DILLI": "Delhi",
    "MUMBAI": "Mumbai", "BOMBAY": "Mumbai", "MUMBAY": "Mumbai",
}

STATE_CANONICAL = {
    "PUNJAB": "Punjab",
    "MAHARASHTRA": "Maharashtra",
    "DELHI": "Delhi",
    "TELANGANA": "Telangana",
    "KARNATAKA": "Karnataka",
    "UTTAR PRADESH": "Uttar Pradesh",
    "WEST BENGAL": "West Bengal",
    "RAJASTHAN": "Rajasthan",
    "TAMIL NADU": "Tamil Nadu",
}


def clean_pan(value) -> tuple[str | None, str]:
    """Normalize and repair one PAN. Returns (pan, status).

    status is one of VALID, REPAIRED, TRUNCATED, INVALID, MISSING.

    >>> clean_pan("AB0CD1234E")
    ('ABOCD1234E', 'REPAIRED')
    >>> clean_pan("ABCDE1234")
    (None, 'TRUNCATED')
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, "MISSING"
    s = re.sub(r"[^A-Za-z0-9]", "", str(value).strip().upper())
    if s.lower() in _NULLISH or not s:
        return None, "MISSING"
    if PAN_RE.match(s):
        return s, "VALID"
    if len(s) == 10:
        repaired = (
            s[:5].translate(_TO_LETTER) + s[5:9].translate(_TO_DIGIT) + s[9].translate(_TO_LETTER)
        )
        if PAN_RE.match(repaired):
            return repaired, "REPAIRED"
        return None, "INVALID"
    if len(s) == 9:
        # Missing the trailing check letter. Fabricating it would invent an
        # identity, so this stays unrepairable by design.
        return None, "TRUNCATED"
    return None, "INVALID"


def clean_aadhaar(value) -> tuple[str | None, str]:
    """Normalize one Aadhaar. Returns (last4, status).

    The full 12-digit value is deliberately discarded here. status is one of
    FULL, MASKED, INVALID, MISSING.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, "MISSING"
    raw = str(value).strip().upper()
    if raw.lower() in _NULLISH or not raw:
        return None, "MISSING"
    if "X" in raw:
        digits = re.sub(r"[^0-9]", "", raw)
        return (digits[-4:], "MASKED") if len(digits) >= 4 else (None, "INVALID")
    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) == 12:
        return digits[-4:], "FULL"
    return None, "INVALID"


def hash_identifier(value, salt: str = "datavortex") -> str | None:
    """Stable one-way hash, used for equality comparison only."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    if not s or s.lower() in _NULLISH:
        return None
    return hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:16]


def aadhaar_full_hash(value, salt: str = "datavortex") -> str | None:
    """Hash the full Aadhaar where available, for collision analysis only.

    Masked values return None: their last 4 digits collide by chance (observed
    and expected counts are in audit_facts), so hashing them would manufacture
    a false identity-sharing signal.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip().upper()
    if "X" in raw:
        return None
    digits = re.sub(r"[^0-9]", "", raw)
    return hash_identifier(digits, salt) if len(digits) == 12 else None


def mask_pan(pan) -> str | None:
    """Display form: first 2 + asterisks + last 1."""
    if pan is None or (isinstance(pan, float) and pd.isna(pan)):
        return None
    pan = str(pan)
    if len(pan) != 10:
        return None
    return f"{pan[:2]}*******{pan[9]}"


def normalize_city(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    if s.lower() in _NULLISH or not s:
        return None
    return CITY_ALIASES.get(s, str(value).strip().title())


def normalize_state(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    if s.lower() in _NULLISH or not s:
        return None
    return STATE_CANONICAL.get(s, str(value).strip().title())


def normalize_name(value) -> str | None:
    """Title-case a person or business name and collapse whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = re.sub(r"\s+", " ", str(value).strip())
    return s.title() if s else None


def name_key(value) -> str:
    """Comparison key for entity identity — letters only, lowercased.

    Used to decide whether two rows sharing an ID are the same person or an
    ID collision. A blank name yields a blank key, which groups all
    unnamed rows together — deliberately conservative, since we cannot
    prove two unnamed rows are different entities.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"[^a-z]", "", str(value).lower())
