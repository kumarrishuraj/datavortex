"""Merchant master normalization: MCC and merchant category.

The merchant master's MCC/category is the AUTHORITATIVE category source.
Transaction-level `mcc` is deliberately not used for category, because the
audit proved the two are independent draws: their agreement is no higher than
independent assignment would give (the pipeline measures both rates in
audit_facts), and the transaction file uses only 5 codes against the master's 10.
"""
from __future__ import annotations

import re

import pandas as pd

_NULLISH = {"", "nan", "none", "null", "na", "n/a", "-", "unknown", "misc"}

# MCC -> canonical category, from the 10 codes present in the merchant master.
MCC_TO_CATEGORY = {
    "4131": "Transport",
    "4814": "Telecom",
    "5311": "Department Store",
    "5411": "Grocery",
    "5699": "Apparel",
    "5812": "Restaurant",
    "5912": "Pharmacy",
    "5942": "Books & Stationery",
    "5999": "Retail",
    "7011": "Hotel",
}

# 82 observed category spellings -> 11 canonical. Keys are uppercased, so the
# lower/title-case variants in the source collapse onto these automatically.
CATEGORY_ALIASES = {
    "GROCERY": "Grocery", "GROCERIES": "Grocery", "GROCERY STORES": "Grocery",
    "GROCERY_STORE": "Grocery", "KIRANA": "Grocery",
    "RESTAURANT": "Restaurant", "RESTAURANTS": "Restaurant", "FOOD": "Restaurant",
    "EATING PLACE": "Restaurant", "FOOD_SERVICES": "Restaurant",
    "HOTEL": "Hotel", "HOTELS": "Hotel", "HOTEL_LODGING": "Hotel",
    "HOSPITALITY": "Hotel",
    "TELECOM": "Telecom", "PHONE SERVICE": "Telecom", "MOBILE RECHARGE": "Telecom",
    "TRANSPORT": "Transport", "TRANSPRT": "Transport", "TRANSPORTATION": "Transport",
    "BUS/TAXI": "Transport", "TRAVEL": "Transport",
    "APPAREL": "Apparel", "CLOTHS": "Apparel", "CLOTHING": "Apparel",
    "GARMENTS": "Apparel", "FASHION": "Apparel",
    "PHARMACY": "Pharmacy", "PHARMACIES": "Pharmacy", "CHEMIST": "Pharmacy",
    "MEDICAL": "Pharmacy", "MEDICAL_STORE": "Pharmacy",
    "BOOKS": "Books & Stationery", "BOOK STORE": "Books & Stationery",
    "STATIONERY": "Books & Stationery", "BOOKS_STATIONERY": "Books & Stationery",
    "DEPARTMENT STORE": "Department Store", "DEPARTMENT STORES": "Department Store",
    "DEPT_STORE": "Department Store",
    "RETAIL": "Retail", "RETAIL OTHER": "Retail",
    "MISCELLANEOUS": "Misc", "MISC RETAIL": "Misc", "OTHER": "Misc",
}


def normalize_mcc(value) -> str | None:
    """Canonicalize an MCC to 4 digits, or None.

    Handles the observed damage: 'MCC-5999', '05912', '5311.0', 'misc',
    'UNKNOWN', 'NA', blank.

    >>> normalize_mcc("MCC-5999")
    '5999'
    >>> normalize_mcc("05912")
    '5912'
    >>> normalize_mcc("5311.0")
    '5311'
    >>> normalize_mcc("misc") is None
    True
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    if s.lower() in _NULLISH:
        return None
    s = re.sub(r"^MCC[-_ ]?", "", s)
    s = re.sub(r"\.0+$", "", s)
    s = s.lstrip("0")
    if not s.isdigit():
        return None
    return s.zfill(4) if len(s) <= 4 else None


def normalize_category(value) -> str | None:
    """Map a raw merchant_category string onto the canonical 11."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = re.sub(r"\s+", " ", str(value).strip()).upper()
    if s.lower() in _NULLISH:
        return None
    return CATEGORY_ALIASES.get(s)


def resolve_category(mcc: str | None, raw_category) -> tuple[str | None, str]:
    """Derive the authoritative category for a merchant master row.

    MCC wins where present because it is a controlled code; the free-text
    category is the fallback. Returns (category, source) so the provenance of
    every category value stays visible.
    """
    from_mcc = MCC_TO_CATEGORY.get(mcc) if mcc else None
    from_text = normalize_category(raw_category)
    if from_mcc:
        return from_mcc, "MCC"
    if from_text:
        return from_text, "CATEGORY_TEXT"
    return None, "UNRESOLVED"


def category_conflict(mcc: str | None, raw_category) -> bool:
    """True when MCC and the text category disagree — reported, not silently fixed."""
    from_mcc = MCC_TO_CATEGORY.get(mcc) if mcc else None
    from_text = normalize_category(raw_category)
    return bool(from_mcc and from_text and from_mcc != from_text)


def normalize_settlement_account(value) -> tuple[str | None, str]:
    """Return (display_value, status) where status is FULL, MASKED or MISSING.

    Masked accounts keep only 4 digits, so their observed sharing sits at the
    rate chance alone produces (observed and expected counts are in
    audit_facts). It is a birthday collision and must not be scored as a
    shell-merchant signal. Full account numbers are reduced to XXXX + last 4.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, "MISSING"
    s = str(value).strip().upper()
    if s.lower() in _NULLISH:
        return None, "MISSING"
    if s.startswith("XXXX"):
        return s, "MASKED"
    # Full account numbers never leave this function: the display form keeps only
    # the last four characters, matching the source's own masking convention.
    return ("XXXX" + s[-4:]) if len(s) > 4 else "XXXX", "FULL"
