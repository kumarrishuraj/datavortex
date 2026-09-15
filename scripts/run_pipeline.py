"""DataVortex — Stage 2 data rescue pipeline.

Reads data/raw/ (never modified), writes data/processed/*.parquet and
docs/data_quality_report.md.

    python scripts/run_pipeline.py

Every number in the report is generated here, never hardcoded.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_TXN_WINDOW,
    PROCESSED_DIR,
    UNKNOWN_KEY,
)
from src.ingestion.loaders import load_raw, raw_profile  # noqa: E402
from src.cleaning.validation import QualityLedger, assert_row_count  # noqa: E402
from src.transformation.star_schema import (  # noqa: E402
    build_dim_date,
    build_dim_merchants,
    build_dim_users,
    build_fact_chargebacks,
    build_fact_transactions,
    validate_relationships,
)
from scripts.build_report import write_report  # noqa: E402


def main() -> int:
    t0 = time.time()
    ledger = QualityLedger()

    print("=" * 78)
    print("DataVortex — Stage 2: Data Rescue Pipeline")
    print("=" * 78)

    print("\n[1/7] Loading raw data (immutable)…")
    raw = load_raw()
    profile = raw_profile(raw)
    print(profile.to_string(index=False))
    for name, expected_key in [
        ("transactions", "raw_transactions"), ("kyc", "raw_kyc"),
        ("merchants", "raw_merchants"), ("chargebacks", "raw_chargebacks"),
    ]:
        assert_row_count(len(raw[name]), EXPECTED_COUNTS[expected_key], f"raw {name}")

    print("\n[2/7] Building DIM_USERS (identity-collision aware)…")
    dim_users, bridge_users = build_dim_users(raw["kyc"], ledger)
    print(f"      DIM_USERS {len(dim_users):,} rows "
          f"({int(dim_users['identity_ambiguous'].fillna(False).sum()):,} ambiguous) "
          f"| bridge {len(bridge_users):,} candidate rows")

    print("\n[3/7] Building DIM_MERCHANTS…")
    dim_merchants, bridge_merchants = build_dim_merchants(raw["merchants"], ledger)
    print(f"      DIM_MERCHANTS {len(dim_merchants):,} rows "
          f"({int(dim_merchants['identity_ambiguous'].fillna(False).sum()):,} ambiguous) "
          f"| bridge {len(bridge_merchants):,} candidate rows")

    print("\n[4/7] Building FACT_TRANSACTIONS…")
    user_keys = set(dim_users["user_key"]) - {UNKNOWN_KEY}
    merchant_keys = set(dim_merchants["merchant_key"]) - {UNKNOWN_KEY}
    fact_tx = build_fact_transactions(raw["transactions"], user_keys, merchant_keys, ledger)
    assert_row_count(len(fact_tx), EXPECTED_COUNTS["fact_transactions"], "FACT_TRANSACTIONS")
    print(f"      FACT_TRANSACTIONS {len(fact_tx):,} rows, "
          f"Rs {fact_tx['amount_inr'].sum():,.0f} total value")

    print("\n[5/7] Building FACT_CHARGEBACKS…")
    fact_cb = build_fact_chargebacks(raw["chargebacks"], fact_tx, ledger)
    assert_row_count(len(fact_cb), EXPECTED_COUNTS["fact_chargebacks"], "FACT_CHARGEBACKS")
    print(f"      FACT_CHARGEBACKS {len(fact_cb):,} rows, "
          f"{int((~fact_cb['txn_unlinked']).sum()):,} linked to a transaction")

    print("\n[6/7] Validating relationships…")
    fk_checks, independence = validate_relationships(
        raw["transactions"], raw["kyc"], raw["merchants"], raw["chargebacks"],
        fact_tx, dim_users, dim_merchants,
    )
    print(fk_checks[["relationship", "raw_match_pct", "normalized_match_pct",
                     "gain_pp", "unmatched_rows"]].to_string(index=False))
    print()
    print(independence[["pair", "expected_overlap", "observed_overlap", "z_score", "verdict"]]
          .to_string(index=False))

    dim_date = build_dim_date(fact_tx["timestamp_clean"], fact_cb["reported_timestamp_clean"])
    bridge = pd.concat([bridge_users, bridge_merchants], ignore_index=True)

    # Guard: the pipeline must not have invented data outside the observed window.
    lo, hi = EXPECTED_TXN_WINDOW
    outside = int(
        ((fact_tx["timestamp_clean"] < pd.Timestamp(lo))
         | (fact_tx["timestamp_clean"] > pd.Timestamp(hi) + pd.Timedelta(days=1))).sum()
    )
    if outside:
        raise AssertionError(f"{outside:,} transactions fall outside the observed window {lo}..{hi}")

    print("\n[7/7] Writing Parquet + data quality report…")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "fact_transactions": fact_tx,
        "fact_chargebacks": fact_cb,
        "dim_users": dim_users,
        "dim_merchants": dim_merchants,
        "dim_date": dim_date,
        "bridge_identity_collision": bridge,
    }
    for name, df in tables.items():
        path = PROCESSED_DIR / f"{name}.parquet"
        _write_parquet(df, path)
        print(f"      {name:<28} {len(df):>7,} rows x {df.shape[1]:>3} cols  ->  {path.name}")

    # Persist the reconciliation frames so the dashboard can render them without
    # re-reading raw files or carrying hardcoded counts.
    from scripts.build_report import _reconciliation, _row_accounting, _value_frame
    _write_parquet(_reconciliation(raw, tables), PROCESSED_DIR / "recon_rows.parquet")
    _write_parquet(_row_accounting(raw, tables), PROCESSED_DIR / "recon_accounting.parquet")
    _write_parquet(_value_frame(raw["transactions"], fact_tx),
                   PROCESSED_DIR / "recon_value.parquet")
    print(f"      {'reconciliation tables':<28} {'3':>7} files")

    # Forensic facts quoted on the dashboard and by the agent, computed rather than
    # typed, plus the relationship tests they cite. Counts and rates only.
    from src.transformation.audit_facts import compute as compute_audit_facts
    facts = compute_audit_facts(raw, tables)
    _write_parquet(facts, PROCESSED_DIR / "audit_facts.parquet")
    _write_parquet(fk_checks, PROCESSED_DIR / "recon_fk.parquet")
    _write_parquet(independence, PROCESSED_DIR / "recon_independence.parquet")
    print(f"      {'audit facts':<28} {len(facts):>7,} facts")

    report_path = write_report(raw, profile, tables, ledger, fk_checks, independence)
    print(f"      data quality report            ->  {report_path}")

    print(f"\nDone in {time.time() - t0:.1f}s. All assertions passed.")
    return 0


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write Parquet, coercing mixed-type object columns to string.

    Dimensions carry an UNKNOWN sentinel alongside real dates/numbers, which
    Arrow cannot type on its own; casting those columns to string keeps the
    sentinel visible rather than silently nulling it.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            sample = out[col].dropna()
            if not sample.empty and not sample.map(lambda v: isinstance(v, str)).all():
                out[col] = out[col].astype(str)
    out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")


if __name__ == "__main__":
    raise SystemExit(main())
