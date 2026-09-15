"""DataVortex — central configuration.

Single source of truth for paths, canonical ID shapes and the analysis window.
Everything downstream imports from here so a path never gets hardcoded twice.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_FILES = {
    "transactions": RAW_DIR / "track1_upi_transactions.csv",
    "kyc": RAW_DIR / "track1_kyc_records.csv",
    "merchants": RAW_DIR / "track1_merchants_master.csv",
    "chargebacks": RAW_DIR / "track1_chargebacks.json",
}

# Canonical ID shapes, established in docs/01_forensic_audit.md section D2.
# Width is the zero-padded digit count observed in the clean transaction file.
ID_SPECS = {
    "user": ("USR", 5),
    "merchant": ("MCH", 4),
    "transaction": ("TXN", 8),
    "complaint": ("CBK", 7),
}

# Sentinel key used for every dimension's UNKNOWN member. Transactions whose
# foreign key does not resolve point here instead of being dropped, so that
# every metric still reconciles to the full 20,000 cleaned transactions.
UNKNOWN_KEY = "UNKNOWN"

# Observed transaction window (forensic audit section C). Used only to validate
# that the pipeline has not invented data outside the period — never to filter.
EXPECTED_TXN_WINDOW = ("2026-01-01", "2026-03-31")

# Row-count assertions. The pipeline fails loudly if these drift, so a silent
# regression cannot reach the dashboard. Update only with a written reason.
EXPECTED_COUNTS = {
    "raw_transactions": 20400,
    "fact_transactions": 20000,
    "raw_chargebacks": 2884,
    "fact_chargebacks": 2800,
    "raw_kyc": 36400,
    "raw_merchants": 6210,
}

PARQUET_TABLES = [
    "fact_transactions",
    "fact_chargebacks",
    "dim_users",
    "dim_merchants",
    "dim_date",
    "bridge_identity_collision",
]
