"""Integration tests against the built star schema.

These run on data/processed/*.parquet, so they fail if the pipeline has not
been run. They check the guarantees the analytics layer depends on: grain,
reconciliation, no fan-out, UNKNOWN routing and attribution correctness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EXPECTED_COUNTS, PROCESSED_DIR, RAW_FILES, UNKNOWN_KEY  # noqa: E402
from src.ingestion.loaders import load_raw  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "fact_transactions.parquet").exists(),
    reason="run `python scripts/run_pipeline.py` first",
)


@pytest.fixture(scope="module")
def tables():
    names = [
        "fact_transactions", "fact_chargebacks", "dim_users",
        "dim_merchants", "dim_date", "bridge_identity_collision",
    ]
    return {n: pd.read_parquet(PROCESSED_DIR / f"{n}.parquet") for n in names}


@pytest.fixture(scope="module")
def raw():
    # The organisers' files are not redistributed with the repository. Tests that
    # reconcile against them skip in a fresh clone and run in full once the files
    # are copied into data/raw/.
    missing = [p.name for p in RAW_FILES.values() if not p.exists()]
    if missing:
        pytest.skip(f"raw competition files not present ({', '.join(missing)}); see data/raw/README.md")
    return load_raw()


# --------------------------------------------------------------------------
# Grain and row-count reconciliation
# --------------------------------------------------------------------------
def test_fact_transactions_row_count(tables):
    assert len(tables["fact_transactions"]) == EXPECTED_COUNTS["fact_transactions"]


def test_fact_chargebacks_row_count(tables):
    assert len(tables["fact_chargebacks"]) == EXPECTED_COUNTS["fact_chargebacks"]


def test_fact_transaction_grain_is_one_row_per_txn_id(tables):
    tx = tables["fact_transactions"]
    assert tx["txn_key"].is_unique
    assert tx["txn_key"].notna().all()


def test_fact_chargeback_grain_is_one_row_per_complaint_id(tables):
    cb = tables["fact_chargebacks"]
    assert cb["complaint_key"].is_unique
    assert cb["complaint_key"].notna().all()


def test_dimensions_have_unique_keys(tables):
    assert tables["dim_users"]["user_key"].is_unique
    assert tables["dim_merchants"]["merchant_key"].is_unique
    assert tables["dim_date"]["date_key"].is_unique


def test_row_counts_reconcile_to_raw(tables, raw):
    """Exactly the duplicate rows are gone, nothing else."""
    tx_removed = len(raw["transactions"]) - len(tables["fact_transactions"])
    cb_removed = len(raw["chargebacks"]) - len(tables["fact_chargebacks"])
    assert tx_removed == int(raw["transactions"].duplicated().sum())
    assert cb_removed == int(raw["chargebacks"].duplicated().sum())


def test_no_kyc_or_merchant_row_is_lost_to_identity_resolution(tables, raw):
    """Dimension survivors + bridge non-survivors must equal the deduped source.

    This is the guarantee that identity resolution is lossless: a dimension
    carries one row per entity, and every row it does not carry is still
    inspectable in BRIDGE_IDENTITY_COLLISION.
    """
    bridge = tables["bridge_identity_collision"]
    for src, dim, etype in [
        (raw["kyc"], tables["dim_users"], "USER"),
        (raw["merchants"], tables["dim_merchants"], "MERCHANT"),
    ]:
        after_dedup = len(src) - int(src.duplicated().sum())
        survivors = len(dim) - 1  # exclude the UNKNOWN member
        b = bridge[bridge["entity_type"] == etype]
        non_survivors = int((~b["is_survivor"].astype(bool)).sum())
        assert survivors + non_survivors == after_dedup, (
            f"{etype}: {after_dedup - survivors - non_survivors} rows unaccounted for"
        )


def test_every_raw_transaction_id_survives(tables, raw):
    """Deduplication must not lose a single distinct transaction."""
    tx = tables["fact_transactions"]
    assert tx["txn_key"].nunique() == raw["transactions"]["txn_id"].nunique()


def test_transaction_value_changes_only_for_documented_transformations(tables, raw):
    from src.cleaning.amounts import parse_amount_series

    raw_vals = parse_amount_series(raw["transactions"]["amount"])["value"]
    dup_mask = raw["transactions"].duplicated(keep="first")
    expected = raw_vals[~dup_mask.values].abs().sum()
    actual = tables["fact_transactions"]["amount_inr"].sum()
    assert actual == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------
# UNKNOWN member handling — nothing is dropped
# --------------------------------------------------------------------------
def test_every_dimension_has_an_unknown_member(tables):
    assert UNKNOWN_KEY in set(tables["dim_users"]["user_key"])
    assert UNKNOWN_KEY in set(tables["dim_merchants"]["merchant_key"])
    assert UNKNOWN_KEY in set(tables["dim_date"]["date_key"])


def test_all_fact_foreign_keys_resolve_to_a_dimension_row(tables):
    """No orphan keys: unresolved ones point at UNKNOWN, never at nothing."""
    tx = tables["fact_transactions"]
    users = set(tables["dim_users"]["user_key"])
    merchants = set(tables["dim_merchants"]["merchant_key"])
    assert tx["user_key"].isin(users).all()
    assert tx["merchant_key"].isin(merchants).all()
    cb = tables["fact_chargebacks"]
    assert cb["attributed_user_key"].isin(users).all()
    assert cb["attributed_merchant_key"].isin(merchants).all()


def test_unmatched_transactions_are_retained_not_deleted(tables):
    tx = tables["fact_transactions"]
    assert int(tx["user_unresolved"].sum()) > 0
    assert int(tx["merchant_unresolved"].sum()) > 0
    # and they still count toward the total
    assert len(tx) == EXPECTED_COUNTS["fact_transactions"]


def test_unresolved_flag_agrees_with_unknown_key(tables):
    tx = tables["fact_transactions"]
    assert (tx["user_unresolved"] == (tx["user_key"] == UNKNOWN_KEY)).all()
    assert (tx["merchant_unresolved"] == (tx["merchant_key"] == UNKNOWN_KEY)).all()


# --------------------------------------------------------------------------
# Joins: correctness and no fan-out
# --------------------------------------------------------------------------
def test_joining_dimensions_does_not_fan_out(tables):
    tx = tables["fact_transactions"]
    merged = (
        tx.merge(tables["dim_users"], on="user_key", how="left", validate="many_to_one")
          .merge(tables["dim_merchants"], on="merchant_key", how="left", validate="many_to_one")
    )
    assert len(merged) == len(tx)


def test_chargeback_to_transaction_join_does_not_fan_out(tables):
    cb = tables["fact_chargebacks"]
    merged = cb.merge(
        tables["fact_transactions"][["txn_key", "amount_inr"]],
        on="txn_key", how="left", validate="many_to_one",
    )
    assert len(merged) == len(cb)


def test_chargeback_link_rate_matches_the_audit(tables):
    cb = tables["fact_chargebacks"]
    linked = int((~cb["txn_unlinked"]).sum())
    assert linked == 2607
    assert round(100 * linked / len(cb), 2) == pytest.approx(93.11, abs=0.05)


def test_chargeback_attribution_flows_through_the_transaction(tables):
    """Attributed user/merchant must equal the linked transaction's own keys."""
    cb = tables["fact_chargebacks"]
    tx = tables["fact_transactions"].set_index("txn_key")
    linked = cb[~cb["txn_unlinked"]]
    expected_user = linked["txn_key"].map(tx["user_key"])
    expected_merchant = linked["txn_key"].map(tx["merchant_key"])
    assert (linked["attributed_user_key"].values == expected_user.values).all()
    assert (linked["attributed_merchant_key"].values == expected_merchant.values).all()


def test_declared_chargeback_ids_are_flagged_as_conflicting(tables):
    """The audit measured 0.000% agreement; every linked row must be flagged."""
    cb = tables["fact_chargebacks"]
    linked = cb[~cb["txn_unlinked"]]
    assert int(linked["cb_userid_conflicts_txn"].sum()) == len(linked)
    assert int(linked["cb_merchantid_conflicts_txn"].sum()) == len(linked)


# --------------------------------------------------------------------------
# Identity collisions
# --------------------------------------------------------------------------
def test_identity_collisions_are_flagged_not_merged(tables):
    users = tables["dim_users"]
    real = users[users["user_key"] != UNKNOWN_KEY]
    assert int((real["identity_class"] == "ID_COLLISION").sum()) == 5341
    assert int(real["identity_ambiguous"].sum()) == 5341


def test_merchant_identity_collisions_are_flagged(tables):
    mer = tables["dim_merchants"]
    real = mer[mer["merchant_key"] != UNKNOWN_KEY]
    assert int((real["identity_class"] == "ID_COLLISION").sum()) == 1310


def test_bridge_preserves_every_candidate_row(tables):
    """Each bridge group must hold all its candidates and exactly one survivor."""
    bridge = tables["bridge_identity_collision"]
    assert len(bridge) > 0
    per_entity = bridge.groupby(["entity_type", "entity_key"]).agg(
        rows=("candidate_rank", "size"), survivors=("is_survivor", "sum")
    )
    assert (per_entity["rows"] > 1).all()
    assert (per_entity["survivors"] == 1).all()


def test_bridge_covers_every_ambiguous_dimension_row(tables):
    bridge = tables["bridge_identity_collision"]
    users = tables["dim_users"]
    ambiguous = set(users.loc[users["identity_ambiguous"] == True, "user_key"])  # noqa: E712
    bridged = set(bridge.loc[bridge["entity_type"] == "USER", "entity_key"])
    assert ambiguous.issubset(bridged)


def test_collision_survivors_carry_low_confidence(tables):
    users = tables["dim_users"]
    collided = users[users["identity_class"] == "ID_COLLISION"]
    assert (collided["resolution_confidence"] == "LOW").all()


# --------------------------------------------------------------------------
# Cleaned-value guarantees
# --------------------------------------------------------------------------
def test_no_timestamp_is_left_unparsed_in_the_fact_table(tables):
    tx = tables["fact_transactions"]
    assert int(tx["timestamp_invalid"].sum()) == 0
    assert tx["timestamp_clean"].notna().all()


def test_transactions_stay_inside_the_observed_window(tables):
    tx = tables["fact_transactions"]
    assert tx["timestamp_clean"].min() >= pd.Timestamp("2026-01-01")
    assert tx["timestamp_clean"].max() < pd.Timestamp("2026-04-01")


def test_no_negative_amounts_remain_but_the_flag_survives(tables):
    tx = tables["fact_transactions"]
    assert (tx["amount_inr"].dropna() >= 0).all()
    assert int(tx["amount_sign_invalid"].sum()) > 0
    assert tx["amount_original"].notna().all()


def test_status_is_fully_canonicalized(tables):
    tx = tables["fact_transactions"]
    assert set(tx["status_canonical"]) == {"SUCCESS", "FAILED", "PENDING"}
    assert tx["status_original"].nunique() == 14   # originals retained


def test_original_values_are_preserved_alongside_cleaned_ones(tables):
    tx = tables["fact_transactions"]
    for col in ["amount_original", "timestamp_original", "utr_original",
                "mcc_original", "status_original", "txn_id_original"]:
        assert col in tx.columns


def test_sensitive_identifiers_are_never_written_to_the_processed_layer(tables):
    users = tables["dim_users"]
    assert "aadhaar" not in users.columns
    assert "pan" not in users.columns
    assert "pan_clean" not in users.columns
    masked = users["pan_masked"].dropna()
    assert masked.str.contains(r"\*").all()


def test_transaction_mcc_is_flagged_unreliable(tables):
    """It must never be used as the category authority."""
    tx = tables["fact_transactions"]
    assert tx["mcc_unreliable"].all()


def test_merchant_category_comes_from_the_master_not_the_transaction(tables):
    mer = tables["dim_merchants"]
    real = mer[mer["merchant_key"] != UNKNOWN_KEY]
    assert set(real["category_source"].dropna()) <= {"MCC", "CATEGORY_TEXT", "UNRESOLVED"}
    assert int((real["category_source"] == "MCC").sum()) > 0


def test_reporting_delay_is_present_and_flagged(tables):
    cb = tables["fact_chargebacks"]
    assert "reporting_delay_days" in cb.columns
    assert int(cb["delay_negative"].sum()) > 0     # flagged, not silently fixed
    assert int(cb["delay_missing"].sum()) > 0
    valid = cb["reporting_delay_days"].dropna()
    assert valid.median() == pytest.approx(3.0, abs=0.5)


def test_dim_date_spans_both_fact_tables(tables):
    dim = tables["dim_date"]
    dates = pd.to_datetime(dim.loc[dim["date_key"] != UNKNOWN_KEY, "full_date"])
    tx = tables["fact_transactions"]
    assert dates.min() <= tx["timestamp_clean"].min()
    assert dates.max() >= tx["timestamp_clean"].max().normalize()


def test_no_fabricated_signal_columns_exist(tables):
    """Guard against a future commit reintroducing invented fraud signals.

    The audit proved this dataset has no transaction rings, no velocity signal
    and no shared-identity ring. No column may imply otherwise.
    """
    banned = {"fraud_ring_id", "ring_id", "is_money_laundering", "confirmed_fraud",
              "is_fraud", "circular_flow", "velocity_anomaly"}
    for name, df in tables.items():
        assert not (banned & set(df.columns)), f"{name} contains a fabricated signal column"
