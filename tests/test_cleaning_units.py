"""Unit tests for the cleaning primitives.

Each test encodes a rule that the forensic audit established from the data, so
a regression here means the pipeline has stopped honouring an evidenced
decision — not merely that a number moved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import amounts as am
from src.cleaning import duplicates as dup
from src.cleaning import kyc as kycmod
from src.cleaning import merchants as mermod
from src.cleaning import statuses as st
from src.cleaning import timestamps as ts
from src.cleaning.chargebacks import classify_complaint_text, clean_complaint_text, compute_reporting_delay
from src.cleaning.ids import normalize_id, normalize_series, normalize_utr


# --------------------------------------------------------------------------
# 1-3. ID normalization
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    ["USR12345", "usr12345", "USR-12345", "USR 12345", "usr_12345", "12345", " usr.12345 "],
)
def test_user_id_variants_collapse_to_one_key(raw):
    """All six observed user_id spellings must reduce to the same key."""
    assert normalize_id(raw, "USR", 5) == "USR12345"


@pytest.mark.parametrize(
    "raw", ["MCH1234", "mch1234", "MCH-1234", "MCH 1234", "1234", "mch_1234"]
)
def test_merchant_id_variants_collapse_to_one_key(raw):
    assert normalize_id(raw, "MCH", 4) == "MCH1234"


@pytest.mark.parametrize(
    "raw,expected",
    [("TXN00012345", "TXN00012345"), ("txn-00012345", "TXN00012345"), ("12345", "TXN00012345")],
)
def test_transaction_id_normalization(raw, expected):
    assert normalize_id(raw, "TXN", 8) == expected


def test_complaint_id_normalization():
    assert normalize_id("cbk-0001234", "CBK", 7) == "CBK0001234"


def test_id_normalization_refuses_unparseable_rather_than_guessing():
    """A value that is not 'optional prefix + digits' must return None."""
    for bad in ["", "   ", "NOT-AN-ID", "USR", "ABC123XYZ", None]:
        assert normalize_id(bad, "USR", 5) is None


def test_id_normalization_refuses_oversized_digit_runs():
    """More significant digits than the canonical width is a different ID space.

    Truncating would silently merge two unrelated entities.
    """
    assert normalize_id("MCH123456", "MCH", 4) is None


def test_normalize_series_matches_scalar_behaviour():
    s = pd.Series(["usr_1", "USR-00002", "bad"])
    out = normalize_series(s, "user")
    assert out.iloc[0] == "USR00001"
    assert out.iloc[1] == "USR00002"
    # pandas represents the unparseable result as a null, not a literal None
    assert pd.isna(out.iloc[2])


# --------------------------------------------------------------------------
# 4-5. Amounts
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("15722.34", 15722.34),
        ("Rs. 6362.9", 6362.9),
        ("₹16,466.93", 16466.93),
        ("INR 13,312", 13312.0),
        ("17,833.99", 17833.99),
        ("₹ 1,250.50", 1250.50),
        ("1250.00", 1250.0),
        ("-23820.57", -23820.57),
    ],
)
def test_amount_parser_handles_every_observed_form(raw, expected):
    value, method = am.parse_amount(raw)
    assert method == am.OK
    assert value == pytest.approx(expected)


def test_amount_parser_expands_k_suffix():
    value, method = am.parse_amount("27.3k")
    assert method == am.K_SUFFIX
    assert value == pytest.approx(27300.0)


def test_amount_parser_distinguishes_blank_from_invalid():
    assert am.parse_amount("")[1] == am.BLANK
    assert am.parse_amount("Not Available")[1] == am.INVALID


def test_negative_amounts_are_repaired_and_flagged_not_dropped():
    """Sign corruption: magnitude is kept, the flag records what happened."""
    values = pd.Series([100.0, -250.0, 75.0])
    out = am.repair_sign(values)
    assert list(out["amount_inr"]) == [100.0, 250.0, 75.0]
    assert list(out["amount_sign_invalid"]) == [False, True, False]


def test_sign_repair_stats_report_before_and_after():
    values = pd.Series([100.0, -250.0, -75.0])
    stats = am.sign_repair_stats(values)
    assert stats["negative_before"] == 2
    assert stats["negative_after"] == 0
    assert stats["sum_after"] == pytest.approx(425.0)


# --------------------------------------------------------------------------
# 6. Timestamps
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,method,iso",
    [
        ("2026-01-15 00:11:30", ts.ISO, "2026-01-15 00:11:30"),
        ("2026/02/02", ts.ISO, "2026-02-02 00:00:00"),
        # slash = DD/MM/YYYY: 25 must be the DAY
        ("25/02/2026 00:53:02", ts.SLASH_DMY, "2026-02-25 00:53:02"),
        # hyphen = MM-DD-YYYY: 03 must be the MONTH
        ("03-10-2026 09:27:31 PM", ts.HYPH_MDY, "2026-03-10 21:27:31"),
        ("01-31-2024", ts.HYPH_MDY, "2024-01-31 00:00:00"),
        ("03-Feb-2026", ts.DMON_Y, "2026-02-03 00:00:00"),
        ("10/11/2024 04:26 PM", ts.SLASH_DMY, "2024-11-10 16:26:00"),
    ],
)
def test_separator_rule_picks_the_right_day_month_order(raw, method, iso):
    parsed, got, _time_known = ts.parse_timestamp(raw)
    assert got == method
    assert parsed == pd.Timestamp(iso)


def test_unix_epoch_is_recognised():
    parsed, method, _ = ts.parse_timestamp("1770063471")
    assert method == ts.UNIX
    assert parsed.year == 2026


def test_negative_unix_epoch_parses_as_pre_1970_date_of_birth():
    parsed, method, _ = ts.parse_timestamp("-160078671")
    assert method == ts.UNIX
    assert parsed.year == 1964


def test_blank_timestamp_is_blank_not_unparseable():
    assert ts.parse_timestamp("")[1] == ts.BLANK
    assert ts.parse_timestamp(None)[1] == ts.BLANK


def test_date_only_sources_report_that_the_time_is_unknown():
    """Without this flag, date-only rows pile up at 00:00 and fake a midnight peak."""
    assert ts.parse_timestamp("2026/02/02")[2] is False
    assert ts.parse_timestamp("01-31-2024")[2] is False
    assert ts.parse_timestamp("2026-01-15 00:11:30")[2] is True
    assert ts.parse_timestamp("1770063471")[2] is True


def test_calendar_parts_are_derived_consistently():
    s = pd.Series([pd.Timestamp("2026-03-31 14:02:37")])
    cal = ts.derive_calendar(s)
    assert cal["quarter"].iloc[0] == "2026-Q1"
    assert cal["month_name"].iloc[0] == "March"
    assert cal["hour"].iloc[0] == 14


# --------------------------------------------------------------------------
# 7. Status normalization
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("S", "SUCCESS"), ("Success", "SUCCESS"), ("TXN_SUCCESS", "SUCCESS"),
        ("COMPLETED", "SUCCESS"), ("SUCCESS", "SUCCESS"),
        ("FAILED", "FAILED"), ("TXN_FAILED", "FAILED"), ("Fail", "FAILED"),
        ("Declined", "FAILED"), ("F", "FAILED"),
        ("PENDING", "PENDING"), ("PROCESSING", "PENDING"), ("Initiated", "PENDING"),
    ],
)
def test_transaction_status_canonicalization(raw, expected):
    assert st.canonicalize(pd.Series([raw]), st.TXN_STATUS).iloc[0] == expected


def test_kyc_status_and_risk_segment_canonicalization():
    assert st.canonicalize(pd.Series(["KYC_DONE"]), st.KYC_STATUS).iloc[0] == "VERIFIED"
    assert st.canonicalize(pd.Series(["Under Review"]), st.KYC_STATUS).iloc[0] == "PENDING"
    assert st.canonicalize(pd.Series(["Reject"]), st.KYC_STATUS).iloc[0] == "REJECTED"
    for v in ["low", "LOW", "Low"]:
        assert st.canonicalize(pd.Series([v]), st.RISK_SEGMENT).iloc[0] == "LOW"


def test_severity_maps_both_word_and_priority_scales():
    assert st.canonicalize(pd.Series(["P1"]), st.SEVERITY).iloc[0] == "CRITICAL"
    assert st.canonicalize(pd.Series(["CRIT"]), st.SEVERITY).iloc[0] == "CRITICAL"
    assert st.canonicalize(pd.Series(["P4"]), st.SEVERITY).iloc[0] == "LOW"
    assert st.canonicalize(pd.Series(["M"]), st.SEVERITY).iloc[0] == "MEDIUM"


def test_merchant_status_S_is_suspended_not_success():
    """'S' is overloaded across files; each domain has its own map."""
    assert st.canonicalize(pd.Series(["S"]), st.MERCHANT_STATUS).iloc[0] == "SUSPENDED"
    assert st.canonicalize(pd.Series(["S"]), st.TXN_STATUS).iloc[0] == "SUCCESS"


# --------------------------------------------------------------------------
# 8. PAN / Aadhaar
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [("AB0CD1234E", "ABOCD1234E"), ("PQR1S5678T", "PQRIS5678T"), ("0MNOP4321Z", "OMNOP4321Z")],
)
def test_pan_homoglyph_repair(raw, expected):
    pan, status = kycmod.clean_pan(raw)
    assert status == "REPAIRED"
    assert pan == expected


def test_valid_pan_passes_through_untouched():
    assert kycmod.clean_pan("ABCDE1234F") == ("ABCDE1234F", "VALID")


def test_truncated_pan_is_never_fabricated():
    """9-character PANs are missing the check letter — inventing one is not repair."""
    pan, status = kycmod.clean_pan("ABCDE1234")
    assert pan is None
    assert status == "TRUNCATED"


def test_pan_separators_are_stripped_before_validation():
    assert kycmod.clean_pan("PQRST-4567-U")[1] == "VALID"
    assert kycmod.clean_pan("LMNOP 8910 Q")[1] == "VALID"


def test_aadhaar_never_returns_the_full_number():
    last4, status = kycmod.clean_aadhaar("1234 5678 9012")
    assert status == "FULL"
    assert last4 == "9012"
    assert len(last4) == 4


def test_masked_aadhaar_is_not_hashed():
    """Hashing a 4-digit remnant would manufacture false identity matches."""
    assert kycmod.aadhaar_full_hash("XXXX-XXXX-4321") is None
    assert kycmod.aadhaar_full_hash("987654321098") is not None


def test_pan_masking_hides_the_identifier():
    assert kycmod.mask_pan("ABCDE1234F") == "AB*******F"


# --------------------------------------------------------------------------
# 9. City / state / MCC / category
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [("Bombay", "Mumbai"), ("Mumbay", "Mumbai"), ("Poona", "Pune"), ("Calcutta", "Kolkata"),
     ("Madras", "Chennai"), ("LDH", "Ludhiana"), ("BLR", "Bengaluru"), ("Dilli", "Delhi")],
)
def test_city_alias_normalization(raw, expected):
    assert kycmod.normalize_city(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("MCC-5999", "5999"), ("05912", "5912"), ("5311.0", "5311"), ("4131", "4131")],
)
def test_mcc_canonicalization(raw, expected):
    assert mermod.normalize_mcc(raw) == expected


def test_mcc_placeholders_become_none_not_a_fake_code():
    for bad in ["misc", "UNKNOWN", "NA", ""]:
        assert mermod.normalize_mcc(bad) is None


def test_merchant_category_aliases_collapse():
    for raw in ["kirana", "GROCERY_STORE", "groceries", "Grocery Stores"]:
        assert mermod.normalize_category(raw) == "Grocery"


def test_mcc_is_authoritative_over_free_text_category():
    category, source = mermod.resolve_category("7011", "Retail")
    assert category == "Hotel"
    assert source == "MCC"
    assert mermod.category_conflict("7011", "Retail") is True


# --------------------------------------------------------------------------
# 10. UTR
# --------------------------------------------------------------------------
def test_utr_whitespace_is_stripped_and_format_checked():
    assert normalize_utr("UTR 2787678319") == ("UTR2787678319", True)
    assert normalize_utr("UTR6498104698") == ("UTR6498104698", True)


def test_missing_utr_is_never_fabricated():
    value, valid = normalize_utr("")
    assert value is None and valid is False


# --------------------------------------------------------------------------
# 11. Duplicates vs identity collisions
# --------------------------------------------------------------------------
def test_exact_duplicate_collapse_removes_only_identical_rows():
    df = pd.DataFrame({"id": ["A", "A", "B"], "v": [1, 1, 2]})
    out, stats = dup.collapse_exact_duplicates(df)
    assert stats["rows_removed"] == 1
    assert len(out) == 2


def test_key_duplicates_are_verified_identical_before_collapsing():
    identical = pd.DataFrame({"id": ["A", "A"], "v": [1, 1]})
    conflicting = pd.DataFrame({"id": ["A", "A"], "v": [1, 2]})
    assert dup.verify_key_duplicates_are_exact(identical, "id")["all_identical"] is True
    assert dup.verify_key_duplicates_are_exact(conflicting, "id")["all_identical"] is False


def test_identity_collision_is_distinguished_from_a_true_duplicate():
    """Same ID + different names = two people, not one record twice."""
    df = pd.DataFrame(
        {
            "user_key": ["USR1", "USR1", "USR2", "USR2"],
            "name": ["Rehaan Buch", "Inaya Lanka", "Asha Rao", "Asha Rao"],
        }
    )
    groups = dup.classify_identity_groups(df, "user_key", "name").set_index("user_key")
    assert groups.loc["USR1", "identity_class"] == dup.ID_COLLISION
    assert bool(groups.loc["USR1", "identity_ambiguous"]) is True
    assert groups.loc["USR2", "identity_class"] == dup.SAME_ENTITY_CONFLICTING
    assert bool(groups.loc["USR2", "identity_ambiguous"]) is False


def test_survivor_selection_is_deterministic_and_keeps_all_candidates():
    df = pd.DataFrame(
        {
            "user_key": ["USR1", "USR1"],
            "name": ["A", "B"],
            "city": [None, "Pune"],
            "pan": [None, "ABCDE1234F"],
        }
    )
    survivors, ordered = dup.resolve_survivor(df, "user_key", ["city", "pan"])
    assert len(survivors) == 1
    assert survivors.iloc[0]["name"] == "B"        # more complete record wins
    assert len(ordered) == 2                        # nothing discarded
    assert ordered["is_survivor"].sum() == 1


def test_resolution_confidence_grades_ambiguity():
    assert dup.resolution_confidence(dup.ID_COLLISION, 2) == "LOW"
    assert dup.resolution_confidence(dup.SAME_ENTITY_CONFLICTING, 2) == "MEDIUM"
    assert dup.resolution_confidence(dup.UNIQUE, 1) == "HIGH"


# --------------------------------------------------------------------------
# 12. Chargeback delay + text
# --------------------------------------------------------------------------
def test_reporting_delay_uses_the_chargeback_files_own_timestamps():
    txn = pd.Series([pd.Timestamp("2026-01-10"), pd.Timestamp("2026-02-01")])
    rep = pd.Series([pd.Timestamp("2026-01-13"), pd.Timestamp("2026-01-30")])
    out = compute_reporting_delay(txn, rep)
    assert out["reporting_delay_days"].iloc[0] == pytest.approx(3.0)
    assert bool(out["delay_negative"].iloc[1]) is True   # flagged, not clipped


def test_complaint_text_noise_suffixes_are_trimmed():
    assert clean_complaint_text("Customer says amount was debited twice. pls check asap") == \
        "Customer says amount was debited twice."
    assert clean_complaint_text("Merchant service was not delivered after payment. NA") == \
        "Merchant service was not delivered after payment."


def test_complaint_theme_classification():
    assert classify_complaint_text("Customer says amount was debited twice.") == "DUPLICATE_DEBIT"
    assert classify_complaint_text("User claims account was compromised before transaction.") == \
        "ACCOUNT_TAKEOVER"
