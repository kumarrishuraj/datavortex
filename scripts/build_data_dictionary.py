"""Generate docs/data_dictionary.md from the built star schema.

Gate 1 of the competition rubric requires a data dictionary; generating it from
the actual Parquet files keeps it from drifting out of sync with the pipeline.

    python scripts/build_data_dictionary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import DOCS_DIR, PARQUET_TABLES, PROCESSED_DIR, UNKNOWN_KEY  # noqa: E402

TABLE_NOTES = {
    "fact_transactions": (
        "Transaction fact. **Grain: one row per unique `txn_id`.** Unresolved foreign keys "
        "point at the `UNKNOWN` dimension member rather than being dropped, so every metric "
        "reconciles to the full row count."
    ),
    "fact_chargebacks": (
        "Dispute fact. **Grain: one row per unique `complaint_id`.** User and merchant are "
        "attributed through `txn_key`, never through the file's own `user_id`/`merchant_id` "
        "columns — those were shown to be independent noise."
    ),
    "dim_users": (
        "Customer dimension, one row per normalized `user_key` plus `UNKNOWN`. Rows flagged "
        "`identity_ambiguous` are IDs shared by different people; their attributes carry "
        "`resolution_confidence = LOW`."
    ),
    "dim_merchants": (
        "Merchant dimension, one row per normalized `merchant_key` plus `UNKNOWN`. "
        "`merchant_category_canonical` is the authoritative category, derived from the master "
        "MCC — never from transaction-level MCC."
    ),
    "dim_date": "Calendar spanning both fact tables, plus an `UNKNOWN` member.",
    "bridge_identity_collision": (
        "Every candidate row behind a repeated ID, survivors and non-survivors alike. This is "
        "what makes identity resolution auditable instead of destructive."
    ),
}

COLUMN_NOTES = {
    "txn_key": "Canonical transaction ID (TXN + 8 digits).",
    "txn_id_original": "Source `txn_id` exactly as supplied.",
    "user_key": "FK to DIM_USERS. `UNKNOWN` when the source ID has no KYC record.",
    "merchant_key": "FK to DIM_MERCHANTS. `UNKNOWN` when the source ID has no master record.",
    "user_unresolved": "True when `user_key` is UNKNOWN.",
    "merchant_unresolved": "True when `merchant_key` is UNKNOWN.",
    "amount_inr": "Transaction value in INR, absolute. Sign corruption repaired; see `amount_sign_invalid`.",
    "amount_original": "Source amount string, currency symbols and all.",
    "amount_sign_invalid": "True where the source amount was negative (sign corruption, not a refund).",
    "amount_missing": "True where the amount could not be parsed.",
    "amount_parse_method": "OK | K_SUFFIX | BLANK | INVALID.",
    "timestamp_clean": "Parsed transaction timestamp.",
    "timestamp_original": "Source timestamp string.",
    "timestamp_parse_method": "ISO | SLASH_DMY | HYPH_MDY | DMON_Y | UNIX | BLANK | UNPARSEABLE.",
    "timestamp_invalid": "True where no parsing family applied.",
    "utr_clean": "UTR with whitespace and separators removed. Never fabricated.",
    "utr_valid": "True where the cleaned UTR matches `UTR` + 10 digits.",
    "utr_missing": "True where the source UTR was blank.",
    "mcc_txn_clean": "Transaction-level MCC, canonicalized.",
    "mcc_unreliable": "Always True — transaction MCC is independent of the merchant's real MCC.",
    "status_canonical": "SUCCESS | FAILED | PENDING.",
    "status_original": "Source status string (14 observed variants).",
    "complaint_key": "Canonical complaint ID (CBK + 7 digits).",
    "txn_unlinked": "True where `txn_id` resolves to no transaction.",
    "txn_id_malformed": "True where the chargeback's `txn_id` has a non-standard digit width.",
    "attributed_user_key": "User of the linked transaction — the authoritative attribution.",
    "attributed_merchant_key": "Merchant of the linked transaction — the authoritative attribution.",
    "cb_user_key_declared": "The chargeback file's own `user_id`, normalized. Retained, never joined on.",
    "cb_merchant_key_declared": "The chargeback file's own `merchant_id`, normalized. Retained, never joined on.",
    "cb_userid_conflicts_txn": "True where the declared user disagrees with the linked transaction.",
    "cb_merchantid_conflicts_txn": "True where the declared merchant disagrees with the linked transaction.",
    "reporting_delay_days": "Reported minus transaction time, both from the chargeback file itself.",
    "delay_negative": "True where the dispute predates its own transaction timestamp.",
    "delay_missing": "True where either timestamp was unparseable.",
    "bank_response_days": "Bank response minus customer report, in days.",
    "disputed_amount_inr": "Disputed value in INR, absolute.",
    "reason_code_canonical": "UNAUTHORIZED | DUPLICATE_DEBIT | SERVICE_NOT_PROVIDED | WRONG_AMOUNT | ACCOUNT_TAKEOVER | FRAUD_SUSPECTED | OTHER.",
    "severity_canonical": "LOW | MEDIUM | HIGH | CRITICAL (word and P1–P4 scales merged).",
    "resolution_status_canonical": "OPEN | IN_PROGRESS | PENDING_BANK | RESOLVED | CLOSED | REJECTED.",
    "channel_canonical": "IVR | CHATBOT | APP | EMAIL | BRANCH | CALL_CENTER.",
    "complaint_theme": "Theme of the free text. Reported alongside `reason_code_canonical`, never scored against it.",
    "pan_masked": "PAN display form, first 2 + asterisks + last 1. Full PAN is never stored.",
    "pan_hash": "Salted SHA-256 prefix of the PAN, for equality comparison only.",
    "pan_status": "VALID | REPAIRED | TRUNCATED | INVALID | MISSING.",
    "pan_unrepairable": "True for truncated or structurally invalid PANs.",
    "aadhaar_last4": "Last 4 digits only. Full Aadhaar is never stored.",
    "aadhaar_status": "FULL | MASKED | INVALID | MISSING.",
    "aadhaar_hash": "Hash of the full Aadhaar where available; None for masked values.",
    "monthly_income_inr": "Parsed monthly income. Negative values nulled and flagged.",
    "income_invalid": "True for negative income or 'Not Available'.",
    "age_years": "Age at 2026-01-01, derived from date of birth.",
    "age_implausible": "True for age under 18 or over 100.",
    "kyc_status_canonical": "VERIFIED | PENDING | REJECTED.",
    "risk_segment_canonical": "LOW | MEDIUM | HIGH | UNKNOWN — the customer's *declared* segment, not a computed score.",
    "identity_class": "UNIQUE | SAME_ENTITY_CONFLICTING | ID_COLLISION.",
    "identity_ambiguous": "True where several distinct entities share this ID.",
    "candidate_count": "Distinct entities observed under this ID.",
    "source_row_count": "Source rows carrying this ID.",
    "completeness_score": "Fraction of key attributes populated on the surviving row.",
    "resolution_confidence": "HIGH | MEDIUM | LOW | NONE — how much to trust this row's attributes.",
    "merchant_category_canonical": "Authoritative merchant category (11 values).",
    "category_source": "MCC | CATEGORY_TEXT | UNRESOLVED — provenance of the category.",
    "category_conflict": "True where master MCC and free-text category disagree.",
    "merchant_status_canonical": "ACTIVE | INACTIVE | SUSPENDED | BLOCKED.",
    "business_type_canonical": "INDIVIDUAL | PARTNERSHIP | PRIVATE_LIMITED | SOLE_PROPRIETOR.",
    "settlement_account_status": "FULL | MASKED | MISSING.",
    "declared_avg_ticket_inr": "Declared average ticket size. Negatives nulled and flagged.",
    "ticket_invalid": "True where the declared ticket size was negative.",
    "entity_type": "USER | MERCHANT — which dimension this bridge row belongs to.",
    "entity_key": "The shared identifier that several entities collided on.",
    "candidate_rank": "1 = the survivor written to the dimension.",
    "is_survivor": "True for the row promoted into the dimension.",
}


def main() -> int:
    lines: list[str] = []
    w = lines.append
    w("# DataVortex — Data Dictionary")
    w("_Generated by `scripts/build_data_dictionary.py` from `data/processed/*.parquet`._\n")
    w("Source-column meanings for the raw files are in "
      "[`01_forensic_audit.md`](01_forensic_audit.md); this document describes the **cleaned "
      "analytical layer** the dashboard and agent read.\n")
    w("Naming convention: `*_original` holds the untouched source value, `*_canonical` / "
      "`*_clean` holds the normalized value, and boolean `*_invalid` / `*_missing` / "
      "`*_unresolved` columns are quality flags. Original values are never discarded.\n")
    w("---\n")

    for name in PARQUET_TABLES:
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        w(f"## `{name.upper()}`\n")
        w(f"{TABLE_NOTES.get(name, '')}\n")
        w(f"**{len(df):,} rows × {df.shape[1]} columns**\n")
        w("| Column | Type | Nulls | Distinct | Example | Meaning |")
        w("|---|---|---|---|---|---|")
        for col in df.columns:
            s = df[col]
            nulls = int(s.isna().sum())
            sample = s.dropna()
            sample = sample[sample.astype(str) != UNKNOWN_KEY]
            example = str(sample.iloc[0])[:28] if len(sample) else "—"
            note = COLUMN_NOTES.get(col, "")
            w(f"| `{col}` | {s.dtype} | {nulls:,} | {s.nunique():,} | {example} | {note} |")
        w("")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "data_dictionary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}  ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
