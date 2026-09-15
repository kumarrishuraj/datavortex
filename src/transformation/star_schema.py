"""Build the DataVortex analytical star schema.

    FACT_TRANSACTIONS          grain: 1 row = 1 unique txn_id      (20,000)
    FACT_CHARGEBACKS           grain: 1 row = 1 unique complaint_id (2,800)
    DIM_USERS                  1 row per normalized user_id + UNKNOWN
    DIM_MERCHANTS              1 row per normalized merchant_id + UNKNOWN
    DIM_DATE                   1 row per calendar date + UNKNOWN
    BRIDGE_IDENTITY_COLLISION  every candidate row behind a repeated ID

Design rules enforced here:
  * raw data is never modified; all output goes to data/processed/
  * dimensions are collapsed to one row per key BEFORE any join, so the facts
    cannot fan out
  * unresolved foreign keys point at UNKNOWN instead of being dropped
  * original values are retained beside every normalized value
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning import amounts as am
from src.cleaning import duplicates as dup
from src.cleaning import kyc as kycmod
from src.cleaning import merchants as mermod
from src.cleaning import statuses as st
from src.cleaning import timestamps as ts
from src.cleaning.chargebacks import (
    classify_complaint_text,
    clean_complaint_text,
    compute_bank_response_delay,
    compute_reporting_delay,
)
from src.cleaning.ids import normalize_series, normalize_utr
from src.cleaning.validation import (
    QualityLedger,
    apply_unknown_member,
    assert_no_fanout,
    independence_test,
    validate_foreign_key,
)
from src.config import UNKNOWN_KEY

KYC_COMPLETENESS_COLS = [
    "pan", "aadhaar", "date_of_birth", "city", "state",
    "monthly_income", "occupation", "signup_timestamp",
]
MERCHANT_COMPLETENESS_COLS = [
    "merchant_name", "mcc", "merchant_category", "business_type",
    "city", "state", "onboarding_date", "settlement_account",
    "declared_avg_ticket_size",
]


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------
def build_dim_users(kyc_raw: pd.DataFrame, ledger: QualityLedger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DIM_USERS plus the user half of the collision bridge."""
    df = kyc_raw.copy()
    before = len(df)

    df, stats = dup.collapse_exact_duplicates(df)
    ledger.record(
        "KYC repair", "Collapse byte-identical KYC rows", before, len(df), stats["rows_removed"],
        stats["method"], f"{len(df):,} rows remain; no key-level merging performed",
        "Identical across all 12 columns — re-ingested records, lossless to collapse.",
    )

    df["user_id_original"] = df["user_id"]
    df["user_key"] = normalize_series(df["user_id"], "user")
    unparseable = int(df["user_key"].isna().sum())
    ledger.record(
        "ID normalization", "Normalize KYC user_id", f"{df['user_id_original'].nunique():,} distinct raw",
        f"{df['user_key'].nunique():,} distinct normalized",
        int((df["user_id_original"] != df["user_key"]).sum()),
        "upper -> strip separators -> strip leading zeros -> pad to USR+5",
        f"{unparseable} unparseable values", "Five observed formats collapse onto one canonical key.",
    )

    # --- field-level repair -------------------------------------------------
    pan = df["pan"].map(kycmod.clean_pan)
    df["pan_clean"] = [p[0] for p in pan]
    df["pan_status"] = [p[1] for p in pan]
    df["pan_masked"] = df["pan_clean"].map(kycmod.mask_pan)
    df["pan_hash"] = df["pan_clean"].map(kycmod.hash_identifier)
    df["pan_unrepairable"] = df["pan_status"].isin(["TRUNCATED", "INVALID"])
    ledger.record(
        "KYC repair", "PAN normalize + homoglyph repair",
        f"{int((df['pan_status'] == 'VALID').sum()) + int((df['pan_status'] == 'REPAIRED').sum()) - int((df['pan_status'] == 'REPAIRED').sum()):,} valid",
        f"{int(df['pan_status'].isin(['VALID', 'REPAIRED']).sum()):,} valid",
        int((df["pan_status"] == "REPAIRED").sum()),
        "positional homoglyph map 0<->O 1<->I 2<->Z 5<->S 8<->B",
        f"{int((df['pan_status'] == 'TRUNCATED').sum()):,} 9-char PANs left unrepaired (flagged pan_unrepairable)",
        "Repair is deterministic and reversible; fabricating a check letter is not.",
    )

    aad = df["aadhaar"].map(kycmod.clean_aadhaar)
    df["aadhaar_last4"] = [a[0] for a in aad]
    df["aadhaar_status"] = [a[1] for a in aad]
    df["aadhaar_hash"] = df["aadhaar"].map(kycmod.aadhaar_full_hash)
    ledger.record(
        "KYC repair", "Aadhaar normalize + mask", f"{len(df):,} rows",
        f"{int((df['aadhaar_status'] == 'FULL').sum()):,} full / {int((df['aadhaar_status'] == 'MASKED').sum()):,} masked",
        int(df["aadhaar_status"].isin(["FULL", "MASKED"]).sum()),
        "strip separators; retain last 4 digits only — no full value or hash is persisted",
        "full Aadhaar never written to data/processed/",
        "Privacy: equality is the only property analytics needs.",
    )

    income = am.parse_amount_series(df["monthly_income"])
    df["monthly_income_original"] = df["monthly_income"]
    df["monthly_income_inr"] = income["value"]
    df["income_parse_method"] = income["method"]
    df["income_invalid"] = (income["value"].notna() & (income["value"] < 0)) | (
        income["method"] == am.INVALID
    )
    df.loc[df["income_invalid"], "monthly_income_inr"] = np.nan
    ledger.record(
        "KYC repair", "Parse monthly_income",
        f"{len(df):,} raw values", f"{int(df['monthly_income_inr'].notna().sum()):,} numeric",
        int(df["income_parse_method"].isin([am.OK, am.K_SUFFIX]).sum()),
        "strip currency symbols/commas; expand 'k' suffix; negatives flagged invalid",
        f"{int(df['income_invalid'].sum()):,} invalid (negative or 'Not Available')",
        "Negative income has no valid business meaning, so it is nulled and flagged, not abs()'d.",
    )

    dob = ts.parse_timestamp_series(df["date_of_birth"])
    df["date_of_birth_clean"] = dob["timestamp_clean"]
    df["dob_parse_method"] = dob["timestamp_parse_method"]
    age = (pd.Timestamp("2026-01-01") - df["date_of_birth_clean"]).dt.days / 365.25
    df["age_years"] = age.round(1)
    df["age_implausible"] = age.notna() & ((age < 18) | (age > 100))

    signup = ts.parse_timestamp_series(df["signup_timestamp"])
    df["signup_timestamp_clean"] = signup["timestamp_clean"]
    df["signup_parse_method"] = signup["timestamp_parse_method"]
    ledger.record(
        "Timestamp cleaning", "Parse KYC signup_timestamp + date_of_birth",
        f"{len(df) * 2:,} values",
        f"{int(df['signup_timestamp_clean'].notna().sum() + df['date_of_birth_clean'].notna().sum()):,} parsed",
        int(df["signup_timestamp_clean"].notna().sum() + df["date_of_birth_clean"].notna().sum()),
        "separator-dispatched parser (slash=DMY, hyphen=MDY, ISO, d-Mon-Y, Unix incl. negative epochs)",
        f"{int(df['age_implausible'].sum()):,} implausible ages flagged",
        "Negative epochs are legitimate pre-1970 dates of birth.",
    )

    df["kyc_status_original"] = df["kyc_status"]
    df["kyc_status_canonical"] = st.canonicalize(df["kyc_status"], st.KYC_STATUS)
    df["risk_segment_original"] = df["risk_segment"]
    df["risk_segment_canonical"] = st.canonicalize(df["risk_segment"], st.RISK_SEGMENT)
    df["city_clean"] = df["city"].map(kycmod.normalize_city)
    df["state_clean"] = df["state"].map(kycmod.normalize_state)
    df["full_name_clean"] = df["full_name"].map(kycmod.normalize_name)
    df["occupation_clean"] = df["occupation"].astype(str).str.strip().str.title()
    ledger.record(
        "Status normalization", "Canonicalize KYC status / risk segment / city / state",
        "16 kyc_status, 12 risk_segment, 41 city spellings",
        f"{df['kyc_status_canonical'].nunique()} kyc_status, {df['risk_segment_canonical'].nunique()} risk_segment, {df['city_clean'].nunique()} cities",
        len(df), "explicit observed-value maps; originals retained in *_original columns",
        f"{int(st.unmapped_values(df['kyc_status'], st.KYC_STATUS).sum())} unmapped kyc_status values",
        "Every observed variant is enumerated; nothing falls through to UNKNOWN by accident.",
    )

    # --- identity resolution ------------------------------------------------
    valid = df[df["user_key"].notna()].copy()
    groups = dup.classify_identity_groups(valid, "user_key", "full_name_clean")
    survivors, ordered = dup.resolve_survivor(
        valid, "user_key", KYC_COMPLETENESS_COLS, recency_col="signup_timestamp_clean"
    )
    survivors = survivors.merge(groups, on="user_key", how="left")
    survivors["resolution_confidence"] = [
        dup.resolution_confidence(c, n)
        for c, n in zip(survivors["identity_class"], survivors["source_row_count"])
    ]

    collisions = int((groups["identity_class"] == dup.ID_COLLISION).sum())
    same_entity = int((groups["identity_class"] == dup.SAME_ENTITY_CONFLICTING).sum())
    ledger.record(
        "Identity collision", "Resolve repeated KYC user_id",
        f"{len(valid):,} rows / {valid['user_key'].nunique():,} distinct user_key",
        f"{len(survivors):,} DIM_USERS rows",
        int(len(valid) - len(survivors)),
        "survivor = max completeness, then latest signup, then lowest source index",
        f"{collisions:,} IDs flagged identity_ambiguous; all candidates kept in bridge",
        "85% of repeated user_ids are DIFFERENT PEOPLE — drop_duplicates would erase them.",
    )

    bridge_keys = groups.loc[groups["source_row_count"] > 1, "user_key"]
    bridge = ordered[ordered["user_key"].isin(set(bridge_keys))].merge(groups, on="user_key", how="left")
    bridge = bridge.assign(entity_type="USER").rename(columns={"user_key": "entity_key"})
    bridge_cols = [
        "entity_type", "entity_key", "candidate_rank", "is_survivor",
        "completeness_score", "source_row_count", "candidate_count",
        "identity_class", "identity_ambiguous",
        "full_name_clean", "pan_masked", "aadhaar_last4", "city_clean", "state_clean",
        "kyc_status_canonical", "risk_segment_canonical", "monthly_income_inr",
    ]
    bridge = bridge[[c for c in bridge_cols if c in bridge.columns]]

    dim_cols = [
        # Identifier hashes and the exact date of birth are used during resolution but
        # not persisted: the processed layer is published, and a hash with a public
        # salt is not real protection. age_years carries what the analysis needs.
        "user_key", "user_id_original", "full_name_clean", "pan_masked",
        "pan_status", "pan_unrepairable", "aadhaar_last4", "aadhaar_status",
        "age_years", "age_implausible",
        "city_clean", "state_clean", "occupation_clean",
        "monthly_income_inr", "monthly_income_original", "income_invalid", "income_parse_method",
        "signup_timestamp_clean", "signup_parse_method",
        "kyc_status_canonical", "kyc_status_original",
        "risk_segment_canonical", "risk_segment_original",
        "identity_class", "identity_ambiguous", "candidate_count", "source_row_count",
        "completeness_score", "resolution_confidence",
    ]
    dim = survivors[dim_cols].copy()
    dim = _append_unknown_member(dim, "user_key")
    return dim, bridge


def build_dim_merchants(mer_raw: pd.DataFrame, ledger: QualityLedger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DIM_MERCHANTS plus the merchant half of the collision bridge."""
    df = mer_raw.copy()
    before = len(df)

    df, stats = dup.collapse_exact_duplicates(df)
    ledger.record(
        "Merchant normalization", "Collapse byte-identical merchant rows", before, len(df),
        stats["rows_removed"], stats["method"], f"{len(df):,} rows remain",
        "Identical across all 11 columns.",
    )

    df["merchant_id_original"] = df["merchant_id"]
    df["merchant_key"] = normalize_series(df["merchant_id"], "merchant")
    ledger.record(
        "ID normalization", "Normalize merchant_id",
        f"{df['merchant_id_original'].nunique():,} distinct raw",
        f"{df['merchant_key'].nunique():,} distinct normalized",
        int((df["merchant_id_original"] != df["merchant_key"]).sum()),
        "upper -> strip separators -> strip leading zeros -> pad to MCH+4",
        f"{int(df['merchant_key'].isna().sum())} unparseable",
        "Four observed formats collapse onto one canonical key.",
    )

    df["mcc_original"] = df["mcc"]
    df["mcc_clean"] = df["mcc"].map(mermod.normalize_mcc)
    resolved = [mermod.resolve_category(m, c) for m, c in zip(df["mcc_clean"], df["merchant_category"])]
    df["merchant_category_canonical"] = [r[0] for r in resolved]
    df["category_source"] = [r[1] for r in resolved]
    df["category_conflict"] = [
        mermod.category_conflict(m, c) for m, c in zip(df["mcc_clean"], df["merchant_category"])
    ]
    df["merchant_category_original"] = df["merchant_category"]
    ledger.record(
        "Merchant normalization", "Canonicalize MCC + merchant_category",
        f"{df['mcc_original'].nunique()} MCC spellings, {df['merchant_category_original'].nunique()} category spellings",
        f"{df['mcc_clean'].nunique()} MCC codes, {df['merchant_category_canonical'].nunique()} categories",
        int(df["mcc_clean"].notna().sum()),
        "strip MCC- prefix / .0 suffix / leading zeros; MCC wins over free text",
        f"{int(df['category_conflict'].sum()):,} MCC-vs-text conflicts flagged, not overwritten",
        "Merchant master is the authoritative category source; transaction MCC is not.",
    )

    df["business_type_original"] = df["business_type"]
    df["business_type_canonical"] = st.canonicalize(df["business_type"], st.BUSINESS_TYPE)
    df["merchant_status_original"] = df["merchant_status"]
    df["merchant_status_canonical"] = st.canonicalize(df["merchant_status"], st.MERCHANT_STATUS)
    df["merchant_name_clean"] = df["merchant_name"].map(kycmod.normalize_name)
    df["city_clean"] = df["city"].map(kycmod.normalize_city)
    df["state_clean"] = df["state"].map(kycmod.normalize_state)

    settle = df["settlement_account"].map(mermod.normalize_settlement_account)
    df["settlement_account_display"] = [s[0] for s in settle]
    df["settlement_account_status"] = [s[1] for s in settle]

    ticket = am.parse_amount_series(df["declared_avg_ticket_size"])
    df["declared_avg_ticket_original"] = df["declared_avg_ticket_size"]
    df["declared_avg_ticket_inr"] = ticket["value"]
    df["ticket_invalid"] = ticket["value"].notna() & (ticket["value"] < 0)
    df.loc[df["ticket_invalid"], "declared_avg_ticket_inr"] = np.nan
    ledger.record(
        "Amount cleaning", "Parse declared_avg_ticket_size",
        f"{len(df):,} raw values", f"{int(df['declared_avg_ticket_inr'].notna().sum()):,} numeric",
        int(ticket["value"].notna().sum()),
        "shared currency parser; negatives nulled + flagged",
        f"{int(df['ticket_invalid'].sum()):,} negative values flagged ticket_invalid",
        "A declared average ticket size cannot be negative, so this is corruption not a refund.",
    )

    onb = ts.parse_timestamp_series(df["onboarding_date"])
    df["onboarding_date_clean"] = onb["timestamp_clean"]
    df["onboarding_parse_method"] = onb["timestamp_parse_method"]

    valid = df[df["merchant_key"].notna()].copy()
    groups = dup.classify_identity_groups(valid, "merchant_key", "merchant_name_clean")
    survivors, ordered = dup.resolve_survivor(
        valid, "merchant_key", MERCHANT_COMPLETENESS_COLS, recency_col="onboarding_date_clean"
    )
    survivors = survivors.merge(groups, on="merchant_key", how="left")
    survivors["resolution_confidence"] = [
        dup.resolution_confidence(c, n)
        for c, n in zip(survivors["identity_class"], survivors["source_row_count"])
    ]

    collisions = int((groups["identity_class"] == dup.ID_COLLISION).sum())
    ledger.record(
        "Identity collision", "Resolve repeated merchant_id",
        f"{len(valid):,} rows / {valid['merchant_key'].nunique():,} distinct merchant_key",
        f"{len(survivors):,} DIM_MERCHANTS rows", int(len(valid) - len(survivors)),
        "survivor = max completeness, then latest onboarding, then lowest source index",
        f"{collisions:,} IDs flagged identity_ambiguous; all candidates kept in bridge",
        "93% of repeated merchant_ids are DIFFERENT BUSINESSES.",
    )

    bridge_keys = groups.loc[groups["source_row_count"] > 1, "merchant_key"]
    bridge = ordered[ordered["merchant_key"].isin(set(bridge_keys))].merge(groups, on="merchant_key", how="left")
    bridge = bridge.assign(entity_type="MERCHANT").rename(
        columns={"merchant_key": "entity_key", "merchant_name_clean": "full_name_clean"}
    )
    bridge_cols = [
        "entity_type", "entity_key", "candidate_rank", "is_survivor",
        "completeness_score", "source_row_count", "candidate_count",
        "identity_class", "identity_ambiguous", "full_name_clean",
        "merchant_category_canonical", "city_clean", "state_clean",
        "merchant_status_canonical", "settlement_account_display",
    ]
    bridge = bridge[[c for c in bridge_cols if c in bridge.columns]]

    dim_cols = [
        "merchant_key", "merchant_id_original", "merchant_name_clean",
        "mcc_clean", "mcc_original", "merchant_category_canonical",
        "merchant_category_original", "category_source", "category_conflict",
        "business_type_canonical", "business_type_original",
        "merchant_status_canonical", "merchant_status_original",
        "city_clean", "state_clean",
        "onboarding_date_clean", "onboarding_parse_method",
        "settlement_account_display", "settlement_account_status",
        "declared_avg_ticket_inr", "declared_avg_ticket_original", "ticket_invalid",
        "identity_class", "identity_ambiguous", "candidate_count", "source_row_count",
        "completeness_score", "resolution_confidence",
    ]
    dim = survivors[dim_cols].copy()
    dim = _append_unknown_member(dim, "merchant_key")
    return dim, bridge


def _append_unknown_member(dim: pd.DataFrame, key: str) -> pd.DataFrame:
    """Add the UNKNOWN member so unresolved facts have somewhere to point.

    Column dtypes are preserved deliberately: a boolean flag column must stay
    boolean, otherwise the UNKNOWN row's NA turns the whole column into objects
    and every downstream `.sum()` on that flag silently breaks.
    """
    unknown: dict = {}
    for col in dim.columns:
        dtype = dim[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            unknown[col] = False
        elif pd.api.types.is_numeric_dtype(dtype):
            unknown[col] = pd.NA
        else:
            unknown[col] = pd.NA
    unknown[key] = UNKNOWN_KEY
    for col, val in [
        ("identity_class", "UNKNOWN"), ("identity_ambiguous", False),
        ("candidate_count", 0), ("source_row_count", 0),
        ("completeness_score", 0.0), ("resolution_confidence", "NONE"),
        ("full_name_clean", "Unknown"), ("merchant_name_clean", "Unknown"),
        ("kyc_status_canonical", "UNKNOWN"), ("risk_segment_canonical", "UNKNOWN"),
        ("merchant_category_canonical", "Unknown"),
        ("merchant_status_canonical", "UNKNOWN"),
        ("business_type_canonical", "UNKNOWN"),
        ("city_clean", "Unknown"), ("state_clean", "Unknown"),
    ]:
        if col in unknown:
            unknown[col] = val

    out = pd.concat([dim, pd.DataFrame([unknown])], ignore_index=True)
    for col, dtype in dim.dtypes.items():
        if pd.api.types.is_bool_dtype(dtype):
            out[col] = out[col].fillna(False).astype(bool)
        else:
            try:
                out[col] = out[col].astype(dtype)
            except (TypeError, ValueError):
                pass  # mixed sentinel + typed values: leave as object
    return out


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------
def build_fact_transactions(
    tx_raw: pd.DataFrame, user_keys: set, merchant_keys: set, ledger: QualityLedger
) -> pd.DataFrame:
    df = tx_raw.copy()
    before = len(df)

    evidence = dup.verify_key_duplicates_are_exact(df, "txn_id")
    df, stats = dup.collapse_exact_duplicates(df)
    ledger.record(
        "Duplicate rescue", "Collapse duplicate transactions", before, len(df), stats["rows_removed"],
        "byte-identical row removal, after proving every duplicated txn_id is identical on all 8 columns",
        f"{evidence['duplicated_keys']:,} duplicated txn_ids, {evidence['conflicting_keys']} with any conflicting field",
        "Zero conflicts means these are re-ingested rows, so collapsing is lossless.",
    )

    df["txn_id_original"] = df["txn_id"]
    df["txn_key"] = normalize_series(df["txn_id"], "transaction")
    df["user_id_original"] = df["user_id"]
    df["merchant_id_original"] = df["merchant_id"]
    df["user_key_raw"] = normalize_series(df["user_id"], "user")
    df["merchant_key_raw"] = normalize_series(df["merchant_id"], "merchant")

    # --- amounts ------------------------------------------------------------
    parsed = am.parse_amount_series(df["amount"])
    df["amount_original"] = df["amount"]
    df["amount_parse_method"] = parsed["method"]
    sign_stats = am.sign_repair_stats(parsed["value"])
    repaired = am.repair_sign(parsed["value"])
    df["amount_inr"] = repaired["amount_inr"]
    df["amount_sign_invalid"] = repaired["amount_sign_invalid"]
    df["amount_missing"] = parsed["value"].isna()
    ledger.record(
        "Amount cleaning", "Parse + sign-repair transaction amount",
        f"{sign_stats['negative_before']:,} negative, sum Rs {sign_stats['sum_before']:,.0f}",
        f"{sign_stats['negative_after']:,} negative, sum Rs {sign_stats['sum_after']:,.0f}",
        sign_stats["repaired"],
        "strip currency symbols/commas; amount_inr = abs(parsed); amount_original retained",
        f"{int(df['amount_missing'].sum())} unparseable amounts",
        "Sign corruption, not refunds: 0/429 have a positive twin; negative rate is flat across statuses.",
    )

    # --- timestamps ---------------------------------------------------------
    parsed_ts = ts.parse_timestamp_series(df["timestamp"])
    df["timestamp_original"] = df["timestamp"]
    df["timestamp_clean"] = parsed_ts["timestamp_clean"]
    df["timestamp_parse_method"] = parsed_ts["timestamp_parse_method"]
    df["timestamp_invalid"] = parsed_ts["timestamp_invalid"]
    # Date-only sources land at midnight; hour-of-day analysis must exclude them.
    df["timestamp_time_known"] = parsed_ts["timestamp_time_known"]
    calendar = ts.derive_calendar(df["timestamp_clean"])
    df = pd.concat([df, calendar], axis=1)
    method_counts = df["timestamp_parse_method"].value_counts().to_dict()
    ledger.record(
        "Timestamp cleaning", "Parse transaction timestamp",
        f"{df['timestamp_original'].nunique():,} distinct raw values, 5 formats",
        f"{int(df['timestamp_clean'].notna().sum()):,} parsed ({100 * df['timestamp_clean'].notna().mean():.2f}%)",
        int(df["timestamp_clean"].notna().sum()),
        "separator-dispatched: " + ", ".join(f"{k}={v:,}" for k, v in sorted(method_counts.items())),
        f"range {df['timestamp_clean'].min()} .. {df['timestamp_clean'].max()}; {int(df['timestamp_invalid'].sum())} invalid",
        "Slash=DD/MM/YYYY and hyphen=MM-DD-YYYY, proven with zero counterexamples.",
    )

    # --- UTR ----------------------------------------------------------------
    utr = df["utr"].map(normalize_utr)
    df["utr_original"] = df["utr"]
    df["utr_clean"] = [u[0] for u in utr]
    df["utr_valid"] = [u[1] for u in utr]
    df["utr_missing"] = df["utr_clean"].isna()
    ledger.record(
        "ID normalization", "Normalize UTR",
        f"{int(df['utr_original'].astype(str).str.contains(' ').sum()):,} with embedded spaces",
        f"{int(df['utr_valid'].sum()):,} valid UTR+10 digits",
        int(df["utr_original"].astype(str).str.contains(" ").sum()),
        "strip whitespace and separators; format-check only",
        f"{int(df['utr_missing'].sum()):,} missing; 0 malformed among non-blank",
        "No UTR is ever fabricated. Reused UTRs were duplicate rows, so no collision signal is created.",
    )

    # --- status -------------------------------------------------------------
    df["status_original"] = df["status"]
    df["status_canonical"] = st.canonicalize(df["status"], st.TXN_STATUS)
    unmapped = st.unmapped_values(df["status"], st.TXN_STATUS)
    counts = df["status_canonical"].value_counts().to_dict()
    ledger.record(
        "Status normalization", "Canonicalize transaction status",
        f"{df['status_original'].nunique()} observed variants",
        ", ".join(f"{k}={v:,}" for k, v in counts.items()),
        len(df), "explicit variant map; status_original retained",
        f"{len(unmapped)} unmapped values",
        "All 14 observed spellings are enumerated across SUCCESS/FAILED/PENDING.",
    )

    # --- transaction MCC (retained, NOT authoritative) ----------------------
    df["mcc_original"] = df["mcc"]
    df["mcc_txn_clean"] = df["mcc"].map(mermod.normalize_mcc)
    df["mcc_unreliable"] = True
    ledger.record(
        "Merchant normalization", "Canonicalize transaction-level MCC",
        f"{df['mcc_original'].nunique()} raw variants",
        f"{df['mcc_txn_clean'].nunique()} canonical codes",
        int(df["mcc_txn_clean"].notna().sum()),
        "same MCC canonicalizer; retained as an attribute and flagged mcc_unreliable",
        "agreement with the merchant master is no higher than independent draws would give (audit_facts: txn_mcc_agreement_pct vs txn_mcc_expected_agreement_pct)",
        "Transaction MCC is an independent draw, so it must never drive category analytics.",
    )

    # --- foreign keys -> UNKNOWN where unresolved ---------------------------
    df["user_key"] = apply_unknown_member(df["user_key_raw"], user_keys)
    df["merchant_key"] = apply_unknown_member(df["merchant_key_raw"], merchant_keys)
    df["user_unresolved"] = df["user_key"] == UNKNOWN_KEY
    df["merchant_unresolved"] = df["merchant_key"] == UNKNOWN_KEY

    # The normalized keys are retained even when they resolve to no dimension
    # row. `*_key` is the dimension FK (UNKNOWN when unresolved) and is what
    # joins use; `*_id_normalized` is the entity's real identity and is what
    # entity-level aggregation must group by. Without the second pair, the
    # 10,369 transactions whose merchant is absent from the master would all
    # collapse into a single UNKNOWN bucket and their disputes would become
    # invisible at merchant level.
    df["user_id_normalized"] = df["user_key_raw"]
    df["merchant_id_normalized"] = df["merchant_key_raw"]

    keep = [
        "txn_key", "txn_id_original", "timestamp_clean", "timestamp_original",
        "timestamp_parse_method", "timestamp_invalid", "timestamp_time_known",
        "date", "hour", "day_of_week", "week", "month", "month_name", "quarter", "year",
        "user_key", "user_id_normalized", "user_id_original", "user_unresolved",
        "merchant_key", "merchant_id_normalized", "merchant_id_original", "merchant_unresolved",
        "amount_inr", "amount_original", "amount_sign_invalid", "amount_missing",
        "amount_parse_method",
        "utr_clean", "utr_original", "utr_valid", "utr_missing",
        "mcc_txn_clean", "mcc_original", "mcc_unreliable",
        "status_canonical", "status_original",
    ]
    fact = df[keep].reset_index(drop=True)
    assert_no_fanout(fact, "txn_key", len(df), "FACT_TRANSACTIONS")
    return fact


def build_fact_chargebacks(
    cb_raw: pd.DataFrame, fact_tx: pd.DataFrame, ledger: QualityLedger
) -> pd.DataFrame:
    df = cb_raw.copy()
    before = len(df)

    evidence = dup.verify_key_duplicates_are_exact(df, "complaint_id")
    df, stats = dup.collapse_exact_duplicates(df)
    ledger.record(
        "Duplicate rescue", "Collapse duplicate chargebacks", before, len(df), stats["rows_removed"],
        "byte-identical row removal, after proving every duplicated complaint_id is identical on all 13 fields",
        f"{evidence['duplicated_keys']:,} duplicated complaint_ids, {evidence['conflicting_keys']} with any conflicting field",
        "Zero conflicts means these are re-ingested rows.",
    )

    df["complaint_id_original"] = df["complaint_id"]
    df["complaint_key"] = normalize_series(df["complaint_id"], "complaint")
    df["txn_id_original"] = df["txn_id"]
    df["txn_key"] = normalize_series(df["txn_id"], "transaction")

    # 5-digit txn_ids against the 8-digit standard: a different ID space.
    digits = df["txn_id"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    df["txn_id_malformed"] = (digits.str.len() > 0) & (digits.str.len() != 8)

    # --- the authoritative attribution --------------------------------------
    tx_lookup = fact_tx.set_index("txn_key")[
        ["user_key", "merchant_key", "user_id_normalized", "merchant_id_normalized",
         "timestamp_clean", "amount_inr"]
    ]
    df["attributed_user_key"] = df["txn_key"].map(tx_lookup["user_key"])
    df["attributed_merchant_key"] = df["txn_key"].map(tx_lookup["merchant_key"])
    # Entity-level identity, independent of whether a master record exists —
    # merchant and user dispute analytics group by these, not by the FK.
    df["attributed_user_id"] = df["txn_key"].map(tx_lookup["user_id_normalized"])
    df["attributed_merchant_id"] = df["txn_key"].map(tx_lookup["merchant_id_normalized"])
    df["linked_txn_timestamp"] = df["txn_key"].map(tx_lookup["timestamp_clean"])
    df["linked_txn_amount_inr"] = df["txn_key"].map(tx_lookup["amount_inr"])
    df["txn_unlinked"] = df["attributed_user_key"].isna()
    df["attributed_user_key"] = df["attributed_user_key"].fillna(UNKNOWN_KEY)
    df["attributed_merchant_key"] = df["attributed_merchant_key"].fillna(UNKNOWN_KEY)

    # Denormalized columns: normalized and retained, never joined on.
    df["cb_user_key_declared"] = normalize_series(df["user_id"], "user")
    df["cb_merchant_key_declared"] = normalize_series(df["merchant_id"], "merchant")
    linked = ~df["txn_unlinked"]
    df["cb_userid_conflicts_txn"] = linked & (df["cb_user_key_declared"] != df["attributed_user_key"])
    df["cb_merchantid_conflicts_txn"] = linked & (
        df["cb_merchant_key_declared"] != df["attributed_merchant_key"]
    )
    ledger.record(
        "Chargeback cleaning", "Attribute chargebacks to user/merchant",
        f"{len(df):,} complaints",
        f"{int(linked.sum()):,} linked via txn_id ({100 * linked.mean():.2f}%)",
        int(linked.sum()),
        "chargeback -> txn_id -> transaction -> user_key / merchant_key",
        f"declared cb.user_id conflicts with the linked txn in {int(df['cb_userid_conflicts_txn'].sum()):,} of {int(linked.sum()):,} cases; "
        f"cb.merchant_id in {int(df['cb_merchantid_conflicts_txn'].sum()):,}",
        "cb.user_id/cb.merchant_id agree with the linked transaction 0.000% of the time, so they are noise.",
    )

    # --- timestamps ---------------------------------------------------------
    cb_txn_ts = ts.parse_timestamp_series(df["transaction_timestamp"])
    rep_ts = ts.parse_timestamp_series(df["reported_timestamp"])
    bank_ts = ts.parse_timestamp_series(df["bank_response_timestamp"])
    df["cb_transaction_timestamp_clean"] = cb_txn_ts["timestamp_clean"]
    df["reported_timestamp_clean"] = rep_ts["timestamp_clean"]
    df["bank_response_timestamp_clean"] = bank_ts["timestamp_clean"]
    df["reported_parse_method"] = rep_ts["timestamp_parse_method"]

    delay = compute_reporting_delay(
        df["cb_transaction_timestamp_clean"], df["reported_timestamp_clean"]
    )
    df = pd.concat([df, delay], axis=1)
    bank_delay = compute_bank_response_delay(
        df["reported_timestamp_clean"], df["bank_response_timestamp_clean"]
    )
    df = pd.concat([df, bank_delay], axis=1)

    valid_delay = df["reporting_delay_days"].dropna()
    ledger.record(
        "Chargeback cleaning", "Compute reporting_delay_days",
        f"{len(df):,} complaints",
        f"{int(df['reporting_delay_days'].notna().sum()):,} computed, median {valid_delay.median():.2f} d",
        int(df["reporting_delay_days"].notna().sum()),
        "reported_timestamp - transaction_timestamp, both from the chargeback file itself",
        f"{int(df['delay_negative'].sum()):,} negative ({100 * df['delay_negative'].mean():.2f}%), {int(df['delay_missing'].sum()):,} missing",
        "Using the linked transaction's timestamp instead yields impossible negative delays for a large share of complaints (audit_facts: delay_negative_vs_linked_txn_pct).",
    )

    calendar = ts.derive_calendar(df["reported_timestamp_clean"], prefix="reported")
    df = pd.concat([df, calendar], axis=1)

    # --- categoricals -------------------------------------------------------
    df["reason_code_original"] = df["reason_code"]
    df["reason_code_canonical"] = st.canonicalize(df["reason_code"], st.REASON_CODE)
    df["resolution_status_original"] = df["resolution_status"]
    df["resolution_status_canonical"] = st.canonicalize(df["resolution_status"], st.RESOLUTION_STATUS)
    df["severity_original"] = df["severity"]
    df["severity_canonical"] = st.canonicalize(df["severity"], st.SEVERITY)
    df["channel_original"] = df["channel"]
    df["channel_canonical"] = st.canonicalize(df["channel"], st.CHANNEL)
    df["complaint_text_clean"] = df["complaint_text"].map(clean_complaint_text)
    df["complaint_theme"] = df["complaint_text_clean"].map(classify_complaint_text)
    ledger.record(
        "Status normalization", "Canonicalize chargeback categoricals",
        "34 reason_code, 13 resolution_status, 16 severity, 8 channel spellings",
        f"{df['reason_code_canonical'].nunique()} reasons, {df['resolution_status_canonical'].nunique()} statuses, "
        f"{df['severity_canonical'].nunique()} severities, {df['channel_canonical'].nunique()} channels",
        len(df), "explicit observed-value maps; originals retained",
        f"{len(st.unmapped_values(df['reason_code'], st.REASON_CODE))} unmapped reason codes",
        "complaint_theme is reported as its own dimension, never scored against reason_code.",
    )

    # --- disputed amount ----------------------------------------------------
    parsed = am.parse_amount_series(df["disputed_amount"])
    df["disputed_amount_original"] = df["disputed_amount"]
    df["disputed_amount_inr"] = parsed["value"].abs()
    df["disputed_amount_sign_invalid"] = parsed["value"].notna() & (parsed["value"] < 0)
    df["disputed_amount_missing"] = parsed["value"].isna()
    ledger.record(
        "Amount cleaning", "Parse disputed_amount",
        f"{int((parsed['value'].notna() & (parsed['value'] < 0)).sum()):,} negative",
        f"{int(df['disputed_amount_inr'].notna().sum()):,} numeric, sum Rs {df['disputed_amount_inr'].sum():,.0f}",
        int((parsed["value"].notna() & (parsed["value"] < 0)).sum()),
        "shared currency parser; abs() applied consistently with transaction amounts",
        f"{int(df['disputed_amount_missing'].sum()):,} missing",
        "Same sign-corruption pattern as transaction amounts.",
    )

    keep = [
        "complaint_key", "complaint_id_original",
        "txn_key", "txn_id_original", "txn_unlinked", "txn_id_malformed",
        "attributed_user_key", "attributed_merchant_key",
        "attributed_user_id", "attributed_merchant_id",
        "cb_user_key_declared", "cb_merchant_key_declared",
        "cb_userid_conflicts_txn", "cb_merchantid_conflicts_txn",
        "cb_transaction_timestamp_clean", "reported_timestamp_clean",
        "bank_response_timestamp_clean", "reported_parse_method",
        "reported_date", "reported_month", "reported_month_name", "reported_quarter", "reported_week",
        "reporting_delay_days", "delay_negative", "delay_missing",
        "bank_response_days", "bank_response_missing",
        "disputed_amount_inr", "disputed_amount_original",
        "disputed_amount_sign_invalid", "disputed_amount_missing",
        "linked_txn_amount_inr", "linked_txn_timestamp",
        "reason_code_canonical", "reason_code_original",
        "resolution_status_canonical", "resolution_status_original",
        "severity_canonical", "severity_original",
        "channel_canonical", "channel_original",
        "complaint_text_clean", "complaint_theme",
    ]
    fact = df[keep].reset_index(drop=True)
    assert_no_fanout(fact, "complaint_key", len(df), "FACT_CHARGEBACKS")
    return fact


def build_dim_date(*timestamp_series: pd.Series) -> pd.DataFrame:
    """Calendar spanning every date observed in either fact table, plus UNKNOWN."""
    all_ts = pd.concat([s.dropna() for s in timestamp_series if s is not None])
    start, end = all_ts.min().normalize(), all_ts.max().normalize()
    days = pd.date_range(start, end, freq="D")
    dim = pd.DataFrame({"date_key": days.date, "full_date": days})
    dim["year"] = days.year
    dim["quarter"] = days.year.astype(str) + "-Q" + days.quarter.astype(str)
    dim["month"] = days.month
    dim["month_name"] = days.month_name()
    dim["week"] = days.isocalendar().week.values
    dim["day_of_week"] = days.day_name()
    dim["is_weekend"] = days.dayofweek >= 5
    unknown = {c: pd.NA for c in dim.columns}
    unknown["date_key"] = UNKNOWN_KEY
    unknown["quarter"] = UNKNOWN_KEY
    unknown["month_name"] = UNKNOWN_KEY
    unknown["day_of_week"] = UNKNOWN_KEY
    unknown["is_weekend"] = False
    return pd.concat([dim, pd.DataFrame([unknown])], ignore_index=True)


def validate_relationships(
    tx_raw: pd.DataFrame, kyc_raw: pd.DataFrame, mer_raw: pd.DataFrame,
    cb_raw: pd.DataFrame, fact_tx: pd.DataFrame, dim_users: pd.DataFrame,
    dim_merchants: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw vs normalized FK match rates, plus the independence tests."""
    user_keys = set(dim_users["user_key"]) - {UNKNOWN_KEY}
    merchant_keys = set(dim_merchants["merchant_key"]) - {UNKNOWN_KEY}
    txn_keys = set(fact_tx["txn_key"])

    tx = fact_tx.copy()
    tx["user_key_raw"] = normalize_series(tx["user_id_original"], "user")
    tx["merchant_key_raw"] = normalize_series(tx["merchant_id_original"], "merchant")

    cb = cb_raw.drop_duplicates().copy()
    cb["txn_key_raw"] = normalize_series(cb["txn_id"], "transaction")

    checks = [
        validate_foreign_key(tx, "user_key_raw", user_keys, "user_id_original",
                             set(kyc_raw["user_id"].astype(str)), "transactions -> KYC (user_id)"),
        validate_foreign_key(tx, "merchant_key_raw", merchant_keys, "merchant_id_original",
                             set(mer_raw["merchant_id"].astype(str)), "transactions -> merchants (merchant_id)"),
        validate_foreign_key(cb, "txn_key_raw", txn_keys, "txn_id",
                             set(tx_raw["txn_id"].astype(str)), "chargebacks -> transactions (txn_id)"),
    ]

    tests = [
        independence_test(set(tx["user_key_raw"].dropna()), user_keys, 90000, "tx.user x kyc.user"),
        independence_test(set(tx["merchant_key_raw"].dropna()), merchant_keys, 9000, "tx.merchant x mer.merchant"),
        independence_test(set(cb["txn_key_raw"].dropna()), txn_keys, 99999999, "cb.txn x tx.txn"),
    ]
    return pd.DataFrame(checks), pd.DataFrame(tests)
