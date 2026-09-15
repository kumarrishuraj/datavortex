"""Duplicate classification and identity-collision resolution.

The central distinction this module enforces:

  * A DUPLICATE is the same record ingested twice — byte-identical across every
    column. Collapsing it is lossless. 400 transactions and 84 chargebacks
    qualify, verified field by field in the audit.

  * A COLLISION is two *different* entities that happen to share an ID.
    5,341 user_ids and 1,310 merchant_ids qualify. `USR10043` is both
    'rehaan buch' (Ludhiana, Verified) and 'inaya lanka' (Mumbai, Pending) —
    different PAN, Aadhaar, DOB, city and income. Collapsing these would
    destroy 5,341 real identities and fabricate a golden record.

So `drop_duplicates(subset="user_id")` is never used anywhere in this pipeline.
Collisions are resolved to a survivor for dimensional joins, but every candidate
row is preserved in BRIDGE_IDENTITY_COLLISION and the survivor carries
identity_ambiguous = True with a LOW resolution confidence.
"""
from __future__ import annotations

import pandas as pd

from src.cleaning.kyc import name_key

EXACT_DUPLICATE = "EXACT_DUPLICATE"
SAME_ENTITY_CONFLICTING = "SAME_ENTITY_CONFLICTING"
ID_COLLISION = "ID_COLLISION"
UNIQUE = "UNIQUE"


def collapse_exact_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Remove byte-identical rows only. Returns (deduped, stats)."""
    before = len(df)
    dup_mask = df.duplicated(subset=subset, keep="first")
    out = df.loc[~dup_mask].copy()
    return out, {
        "rows_before": before,
        "rows_removed": int(dup_mask.sum()),
        "rows_after": len(out),
        "method": "drop byte-identical rows across all columns (keep first)",
    }


def verify_key_duplicates_are_exact(df: pd.DataFrame, key: str) -> dict:
    """Check that every row sharing `key` is byte-identical on all columns.

    This is the evidence that collapsing the duplicates is lossless — run
    before the collapse, not asserted afterwards.
    """
    dup_keys = df[key][df[key].duplicated(keep=False)].unique()
    sub = df[df[key].isin(dup_keys)]
    if not len(sub):
        return {"duplicated_keys": 0, "all_identical": True, "conflicting_keys": 0}
    other_cols = [c for c in df.columns if c != key]
    nunique = sub.groupby(key)[other_cols].nunique()
    conflicting = int((nunique > 1).any(axis=1).sum())
    return {
        "duplicated_keys": int(len(dup_keys)),
        "rows_involved": int(len(sub)),
        "conflicting_keys": conflicting,
        "all_identical": conflicting == 0,
    }


def completeness_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Fraction of the given columns that are populated, per row."""
    present = pd.DataFrame(
        {c: df[c].notna() & (df[c].astype(str).str.strip() != "") for c in cols},
        index=df.index,
    )
    return present.mean(axis=1).round(4)


def classify_identity_groups(
    df: pd.DataFrame, key: str, name_col: str
) -> pd.DataFrame:
    """Per-ID classification: UNIQUE, SAME_ENTITY_CONFLICTING or ID_COLLISION.

    Two rows are the same entity when their normalized name keys match. That is
    a conservative test: it never merges rows with different names, which is
    exactly the failure mode we are guarding against.
    """
    work = df[[key, name_col]].copy()
    work["_nk"] = work[name_col].map(name_key)
    grouped = work.groupby(key).agg(
        source_row_count=(key, "size"),
        candidate_count=("_nk", "nunique"),
    )
    grouped["identity_class"] = UNIQUE
    multi = grouped["source_row_count"] > 1
    grouped.loc[multi & (grouped["candidate_count"] == 1), "identity_class"] = SAME_ENTITY_CONFLICTING
    grouped.loc[grouped["candidate_count"] > 1, "identity_class"] = ID_COLLISION
    grouped["identity_ambiguous"] = grouped["candidate_count"] > 1
    return grouped.reset_index()


def resolve_survivor(
    df: pd.DataFrame,
    key: str,
    completeness_cols: list[str],
    recency_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick one deterministic survivor row per key.

    Rule, applied in order:
      1. highest completeness score (most populated record wins)
      2. most recent `recency_col` where available
      3. lowest original row index — a deterministic tiebreak, so two runs of
         the pipeline always produce the same dimension

    Returns (survivors, all_rows_with_resolution_columns). Nothing is dropped:
    the second frame still holds every candidate row and feeds the bridge table.
    """
    work = df.copy()
    work["_orig_idx"] = range(len(work))
    work["completeness_score"] = completeness_score(work, completeness_cols)

    sort_cols = [key, "completeness_score"]
    ascending = [True, False]
    if recency_col is not None and recency_col in work.columns:
        sort_cols.append(recency_col)
        ascending.append(False)
    sort_cols.append("_orig_idx")
    ascending.append(True)

    ordered = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    ordered["candidate_rank"] = ordered.groupby(key).cumcount() + 1
    ordered["is_survivor"] = ordered["candidate_rank"] == 1

    survivors = ordered[ordered["is_survivor"]].drop(columns=["_orig_idx"]).copy()
    return survivors, ordered.drop(columns=["_orig_idx"])


def resolution_confidence(identity_class: str, source_row_count: int) -> str:
    """How much to trust the attributes on a resolved dimension row.

    HIGH   single source row, nothing to reconcile
    MEDIUM one entity described by several conflicting rows; most complete won
    LOW    several entities share this ID; attributes are not attributable
    """
    if identity_class == ID_COLLISION:
        return "LOW"
    if source_row_count > 1:
        return "MEDIUM"
    return "HIGH"
