"""Audit facts — the forensic evidence the dashboard and agent quote, computed.

Several findings are stated in prose on the dashboard and in agent answers:
how many customer IDs carry more than one PAN, whether masked identifiers collide
more than chance allows, how the day/month order of dates was proven, whether
negative amounts have refund twins. Typing those numbers into page text would
make them silently wrong the moment the data changes.

So they are computed here, from the raw files and the built tables, written to
`data/processed/audit_facts.parquet` by the pipeline, and read wherever they are
quoted. Only counts and rates leave this module — never an identifier.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.cleaning.amounts import parse_amount_series
from src.cleaning.ids import normalize_series
from src.cleaning.kyc import clean_pan
from src.cleaning.merchants import normalize_mcc
from src.cleaning.statuses import TXN_STATUS, canonicalize

MASK_SLOTS = 10_000  # a masked identifier keeps 4 digits

DATE_FIELDS = [
    ("transactions", "timestamp"), ("kyc", "signup_timestamp"), ("kyc", "date_of_birth"),
    ("merchants", "onboarding_date"), ("chargebacks", "transaction_timestamp"),
    ("chargebacks", "reported_timestamp"), ("chargebacks", "bank_response_timestamp"),
]


def expected_shared_keys(draws: int, slots: int = MASK_SLOTS) -> float:
    """Expected number of slots holding two or more draws, under uniform chance.

    This is the correct baseline for "masked values shared by more than one
    entity": slots × P(a slot receives ≥ 2 of the n draws).
    """
    if draws <= 1:
        return 0.0
    p = 1.0 / slots
    return slots * (1 - (1 - p) ** draws - draws * p * (1 - p) ** (draws - 1))


def compute(raw: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    facts: list[dict] = []

    def add(name: str, value, unit: str, description: str) -> None:
        facts.append({"fact": name, "value": float(value), "unit": unit,
                      "description": description})

    # --- identity: PAN -----------------------------------------------------
    kyc = raw["kyc"].drop_duplicates()
    user_key = normalize_series(kyc["user_id"], "user")
    pans = pd.DataFrame({"key": user_key, "pan": kyc["pan"].map(lambda v: clean_pan(v)[0])}).dropna()
    add("kyc_ids_with_multiple_pans", int((pans.groupby("key")["pan"].nunique() > 1).sum()),
        "customer IDs", "Customer IDs carrying more than one distinct valid PAN")
    add("pans_shared_across_ids", int((pans.groupby("pan")["key"].nunique() > 1).sum()),
        "PANs", "Valid PANs appearing under more than one customer ID")

    # --- identity: Aadhaar --------------------------------------------------
    aad = kyc["aadhaar"].astype(str).str.strip().str.upper()
    masked = aad.str.contains("X", regex=False)
    digits = aad.str.replace(r"[^0-9]", "", regex=True)
    full = ~masked & digits.str.len().eq(12)
    full_pairs = pd.DataFrame({"key": user_key[full], "v": digits[full]}).dropna()
    add("full_aadhaar_shared_across_ids",
        int((full_pairs.groupby("v")["key"].nunique() > 1).sum()),
        "Aadhaar numbers", "Full 12-digit Aadhaar numbers under more than one customer ID")
    masked_pairs = pd.DataFrame({"key": user_key[masked], "v": digits[masked].str[-4:]}) \
        .dropna().drop_duplicates()
    add("masked_aadhaar_shared_across_ids",
        int((masked_pairs.groupby("v")["key"].nunique() > 1).sum()),
        "masked values", "Masked Aadhaar last-4 values shared by more than one customer ID")
    add("masked_aadhaar_expected_by_chance", round(expected_shared_keys(len(masked_pairs)), 1),
        "masked values", "Expected shared last-4 values if customers were assigned at random")

    # --- identity: settlement accounts --------------------------------------
    mer = raw["merchants"].drop_duplicates()
    merchant_key = normalize_series(mer["merchant_id"], "merchant")
    acct = mer["settlement_account"].astype(str).str.strip().str.upper()
    present = ~acct.isin(["", "NA", "NAN", "NONE", "N/A"])
    acct_masked = present & acct.str.startswith("XXXX")
    acct_full = present & ~acct_masked
    full_acct = pd.DataFrame({"key": merchant_key[acct_full], "v": acct[acct_full]}).dropna()
    add("full_settlement_shared_across_merchants",
        int((full_acct.groupby("v")["key"].nunique() > 1).sum()),
        "accounts", "Full settlement account numbers under more than one merchant ID")
    masked_acct = pd.DataFrame({"key": merchant_key[acct_masked],
                                "v": acct[acct_masked].str[-4:]}).dropna().drop_duplicates()
    add("masked_settlement_shared_across_merchants",
        int((masked_acct.groupby("v")["key"].nunique() > 1).sum()),
        "masked values", "Masked settlement last-4 values shared by more than one merchant ID")
    add("masked_settlement_expected_by_chance", round(expected_shared_keys(len(masked_acct)), 1),
        "masked values", "Expected shared last-4 values if merchants were assigned at random")

    # --- dates: day/month order evidence -------------------------------------
    evidence = {"slash_first_gt12": 0, "slash_second_gt12": 0,
                "hyphen_first_gt12": 0, "hyphen_second_gt12": 0}
    for source, column in DATE_FIELDS:
        s = raw[source][column].astype(str).str.strip()
        for sep, name in (("/", "slash"), ("-", "hyphen")):
            parts = s.str.extract(rf"^(\d{{1,2}}){re.escape(sep)}(\d{{1,2}}){re.escape(sep)}(\d{{4}})")
            parts = parts.dropna()
            if parts.empty:
                continue
            first, second = parts[0].astype(int), parts[1].astype(int)
            evidence[f"{name}_first_gt12"] += int((first > 12).sum())
            evidence[f"{name}_second_gt12"] += int((second > 12).sum())
    descriptions = {
        "slash_first_gt12": "Slash dates whose first part exceeds 12 (proves DD/MM order)",
        "slash_second_gt12": "Slash dates whose second part exceeds 12 (would contradict DD/MM)",
        "hyphen_first_gt12": "Hyphen dates whose first part exceeds 12 (would contradict MM-DD)",
        "hyphen_second_gt12": "Hyphen dates whose second part exceeds 12 (proves MM-DD order)",
    }
    for name, value in evidence.items():
        add(f"date_{name}", value, "rows", descriptions[name])

    # --- amounts: negative values are not refunds ---------------------------
    tx_raw = raw["transactions"]
    amounts = parse_amount_series(tx_raw["amount"])["value"]
    negative = amounts < 0
    add("raw_negative_amounts", int(negative.sum()), "rows",
        "Raw transaction rows with a negative amount")
    positives = set(zip(tx_raw.loc[~negative, "user_id"], tx_raw.loc[~negative, "merchant_id"],
                        amounts[~negative].round(2)))
    twins = sum(1 for u, m, a in zip(tx_raw.loc[negative, "user_id"],
                                     tx_raw.loc[negative, "merchant_id"],
                                     (-amounts[negative]).round(2))
                if (u, m, a) in positives)
    add("negative_amounts_with_positive_twin", twins, "rows",
        "Negative amounts with a matching positive transaction (same user, merchant, magnitude)")
    status = canonicalize(tx_raw["status"], TXN_STATUS)
    for value in ("SUCCESS", "FAILED", "PENDING"):
        sel = status == value
        add(f"negative_amount_rate_{value.lower()}",
            round(100.0 * float(negative[sel].mean()), 2) if sel.any() else float("nan"),
            "percent", f"Share of {value} transactions with a negative amount")

    # --- transaction MCC vs merchant master MCC -----------------------------
    fact_tx = tables["fact_transactions"]
    dim_m = tables["dim_merchants"].set_index("merchant_key")
    tx_mcc = fact_tx["mcc_txn_clean"]
    master_mcc = fact_tx["merchant_id_normalized"].map(dim_m["mcc_clean"])
    both = tx_mcc.notna() & master_mcc.notna()
    if both.any():
        agree = float((tx_mcc[both] == master_mcc[both]).mean())
        p_tx = tx_mcc[both].value_counts(normalize=True)
        p_master = master_mcc[both].value_counts(normalize=True)
        expected = float(sum(p_tx.get(k, 0) * p_master.get(k, 0) for k in set(p_tx.index) | set(p_master.index)))
        add("txn_mcc_agreement_pct", round(100 * agree, 2), "percent",
            "Transactions whose own MCC equals their merchant's master MCC")
        add("txn_mcc_expected_agreement_pct", round(100 * expected, 2), "percent",
            "Agreement expected if the two MCCs were independent")

    # --- chargebacks ---------------------------------------------------------
    cb = tables["fact_chargebacks"]
    linked = ~cb["txn_unlinked"].astype(bool)
    add("chargebacks_linked", int(linked.sum()), "complaints",
        "Complaints linked to a transaction through txn_id")
    add("cb_user_id_agrees_with_linked_txn",
        int((linked & ~cb["cb_userid_conflicts_txn"].astype(bool)).sum()), "complaints",
        "Linked complaints whose own user_id equals the linked transaction's customer")
    add("cb_merchant_id_agrees_with_linked_txn",
        int((linked & ~cb["cb_merchantid_conflicts_txn"].astype(bool)).sum()), "complaints",
        "Linked complaints whose own merchant_id equals the linked transaction's merchant")
    reported = pd.to_datetime(cb["reported_timestamp_clean"])
    linked_ts = pd.to_datetime(cb["linked_txn_timestamp"])
    vs_linked = (reported - linked_ts).dt.total_seconds() / 86400
    both_ts = vs_linked.notna()
    add("delay_negative_vs_linked_txn_pct",
        round(100.0 * float((vs_linked[both_ts] < 0).mean()), 2) if both_ts.any() else float("nan"),
        "percent", "Complaints that would show a negative delay if measured from the linked transaction")
    own = cb["reporting_delay_days"].notna()
    add("delay_negative_own_pct",
        round(100.0 * float(cb.loc[own, "delay_negative"].astype(bool).mean()), 2) if own.any() else float("nan"),
        "percent", "Complaints with a negative delay measured from their own transaction timestamp")
    own_ts = pd.to_datetime(cb["cb_transaction_timestamp_clean"])
    gap = (own_ts - linked_ts).abs().dt.total_seconds() / 86400
    add("cb_txn_timestamp_same_day_pct",
        round(100.0 * float((gap.dropna() < 1).mean()), 2) if gap.notna().any() else float("nan"),
        "percent", "Linked complaints whose own transaction timestamp falls within a day of the linked transaction")

    # --- duplicates ----------------------------------------------------------
    for source, label in (("transactions", "transaction"), ("chargebacks", "complaint"),
                          ("kyc", "KYC"), ("merchants", "merchant")):
        add(f"duplicate_{source}_rows", int(raw[source].duplicated().sum()), "rows",
            f"Byte-identical duplicate {label} rows collapsed")

    return pd.DataFrame(facts)


def lookup(facts: pd.DataFrame | None, name: str, default=None):
    """Read one fact; returns `default` when the table or the fact is absent."""
    if facts is None or facts.empty:
        return default
    row = facts.loc[facts["fact"] == name, "value"]
    if row.empty:
        return default
    value = float(row.iloc[0])
    return default if np.isnan(value) else value
