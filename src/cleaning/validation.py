"""Foreign-key validation and the pipeline's quality ledger.

Policy: we never inflate a match rate and we never drop an unmatched row.
Transactions whose user or merchant does not resolve keep their place in the
fact table pointing at the dimension's UNKNOWN member, so every metric still
reconciles to the full 20,000 cleaned transactions.

The residual mismatch is irreducible, not a cleaning failure. Overlap was
tested against the null hypothesis of independent random draws:

    tx.user x kyc.user       expected 5,745 (sd 62)  observed 5,799  z = +0.87
    tx.merchant x mer.merch  expected 3,885 (sd 45)  observed 3,893  z = +0.18
    cb.txn x tx.txn          expected ~1             observed 2,451  z = +3420

The first two are indistinguishable from chance — no normalization can raise
them. The third confirms txn_id is the real foreign key.
"""
from __future__ import annotations

import math

import pandas as pd

from src.config import UNKNOWN_KEY


class QualityLedger:
    """Collects one row per transformation for docs/data_quality_report.md.

    Every entry records BEFORE, AFTER, ROWS AFFECTED, METHOD and VALIDATION, so
    the report is generated from what the pipeline actually did rather than
    written by hand.
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def record(
        self,
        stage: str,
        transformation: str,
        before,
        after,
        rows_affected,
        method: str,
        validation: str,
        reason: str = "",
    ) -> None:
        self.entries.append(
            {
                "stage": stage,
                "transformation": transformation,
                "before": before,
                "after": after,
                "rows_affected": rows_affected,
                "method": method,
                "validation": validation,
                "reason": reason,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def for_stage(self, stage: str) -> list[dict]:
        return [e for e in self.entries if e["stage"] == stage]


def validate_foreign_key(
    left: pd.DataFrame,
    left_key: str,
    right_keys: set,
    raw_left_col: str,
    raw_right_values: set,
    name: str,
) -> dict:
    """Raw vs normalized match rate for one relationship."""
    n = len(left)
    raw_hits = int(left[raw_left_col].astype(str).isin(raw_right_values).sum())
    norm_hits = int(left[left_key].isin(right_keys).sum())
    unmatched = left[~left[left_key].isin(right_keys)]
    return {
        "relationship": name,
        "left_rows": n,
        "raw_match": raw_hits,
        "raw_match_pct": round(100 * raw_hits / n, 2) if n else 0.0,
        "normalized_match": norm_hits,
        "normalized_match_pct": round(100 * norm_hits / n, 2) if n else 0.0,
        "gain_rows": norm_hits - raw_hits,
        "gain_pp": round(100 * (norm_hits - raw_hits) / n, 2) if n else 0.0,
        "unmatched_rows": int(len(unmatched)),
        "unmatched_pct": round(100 * len(unmatched) / n, 2) if n else 0.0,
        "unmatched_distinct_keys": int(unmatched[left_key].nunique()),
    }


def independence_test(set_a: set, set_b: set, id_space: int, label: str) -> dict:
    """Is the observed key overlap what independent random draws would give?

    Returns the expected overlap, the observed overlap and a z score. A z near
    zero means the residual mismatch is structural and cannot be cleaned away.
    """
    a, b = len(set_a), len(set_b)
    observed = len(set_a & set_b)
    expected = a * b / id_space if id_space else 0.0
    variance = expected * (1 - b / id_space) if id_space else 0.0
    sd = math.sqrt(variance) if variance > 0 else 0.0
    z = (observed - expected) / sd if sd else float("nan")
    return {
        "pair": label,
        "id_space": id_space,
        "left_distinct": a,
        "right_distinct": b,
        "expected_overlap": round(expected, 0),
        "observed_overlap": observed,
        "sd": round(sd, 1),
        "z_score": round(z, 2),
        "verdict": "consistent with chance" if abs(z) < 3 else "genuine relationship",
    }


def apply_unknown_member(keys: pd.Series, valid_keys: set) -> pd.Series:
    """Route unresolved foreign keys to the dimension's UNKNOWN member."""
    return keys.where(keys.isin(valid_keys), UNKNOWN_KEY).fillna(UNKNOWN_KEY)


def assert_no_fanout(df: pd.DataFrame, key: str, expected_rows: int, label: str) -> None:
    """Guard that a join did not multiply rows."""
    if len(df) != expected_rows:
        raise AssertionError(
            f"{label}: fan-out detected — {len(df):,} rows after join, expected {expected_rows:,}"
        )
    dup = int(df[key].duplicated().sum())
    if dup:
        raise AssertionError(f"{label}: {dup:,} duplicate values of grain key {key!r}")


def assert_row_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise AssertionError(
            f"{label}: expected {expected:,} rows, got {actual:,}. "
            "If this change is intended, update src/config.EXPECTED_COUNTS with a written reason."
        )
