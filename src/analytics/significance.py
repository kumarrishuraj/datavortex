"""Significance testing for group comparisons.

Motivation, concretely: the competition's headline bonus query is *"Which
merchant category has the highest chargeback-to-transaction ratio this
quarter?"* Sorting a table answers it in one line. But the categories in this
dataset span 11.39% to 13.36%, on denominators of ~900 transactions each, where
the standard error of a single category rate is already ~1.1pp. Reporting
"Transport, 13.36%" as a finding would be reporting sampling noise as insight.

So every ranking this project publishes is accompanied by a test of whether the
ranking means anything:

  * `homogeneity_test` — one chi-square across all groups: is there ANY real
    difference between them?
  * `group_vs_rest` — a two-proportion z-test per group against the pooled
    remainder, with a Benjamini-Hochberg correction for testing many groups.

Both are implemented without SciPy to keep the deployment dependency-free.
"""
from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd

from src.analytics.shrinkage import chi_square_sf


def _z_sf(z: float) -> float:
    """Upper-tail probability of the standard normal."""
    return 0.5 * erfc(z / sqrt(2))


def two_sided_z_p(z: float) -> float:
    return min(1.0, 2 * _z_sf(abs(z)))


def homogeneity_test(successes, trials) -> dict:
    """Chi-square test that all groups share one underlying rate.

    A non-significant result means the observed ranking of groups is
    indistinguishable from what random assignment would produce — the ordering
    is real in the sample but carries no information about the population.
    """
    d = np.asarray(pd.to_numeric(successes, errors="coerce").fillna(0), dtype=float)
    n = np.asarray(pd.to_numeric(trials, errors="coerce").fillna(0), dtype=float)
    keep = n > 0
    d, n = d[keep], n[keep]
    k = len(n)
    if k < 2 or n.sum() == 0:
        return {"groups": k, "chi_square": np.nan, "df": 0, "p_value": np.nan,
                "significant": False, "note": "too few groups"}

    p = d.sum() / n.sum()
    if p <= 0 or p >= 1:
        return {"groups": k, "chi_square": np.nan, "df": k - 1, "p_value": np.nan,
                "significant": False, "note": "degenerate pooled rate"}

    expected = n * p
    chi = float(np.sum((d - expected) ** 2 / (expected * (1 - p))))
    df = k - 1
    pval = chi_square_sf(chi, df)
    significant = bool(pval < 0.05)
    return {
        "groups": k,
        "pooled_rate": float(p),
        "chi_square": round(chi, 3),
        "df": df,
        "p_value": float(pval),
        "significant": significant,
        "note": (
            "at least one group differs from the rest"
            if significant else
            "group differences are within sampling noise"
        ),
    }


def group_vs_rest(df: pd.DataFrame, label_col: str, success_col: str, trial_col: str) -> pd.DataFrame:
    """Two-proportion z-test of each group against the pooled remainder.

    Adds a Benjamini-Hochberg adjusted p-value, because testing eleven
    categories at alpha = 0.05 would produce a false positive roughly half the
    time if we read the raw p-values.
    """
    d = pd.to_numeric(df[success_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    n = pd.to_numeric(df[trial_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    total_d, total_n = d.sum(), n.sum()

    rows = []
    for i, label in enumerate(df[label_col]):
        d_i, n_i = d[i], n[i]
        d_r, n_r = total_d - d_i, total_n - n_i
        if n_i == 0 or n_r == 0:
            rows.append((label, np.nan, np.nan, np.nan, np.nan))
            continue
        p_i, p_r = d_i / n_i, d_r / n_r
        p_pool = (d_i + d_r) / (n_i + n_r)
        se = sqrt(p_pool * (1 - p_pool) * (1 / n_i + 1 / n_r)) if 0 < p_pool < 1 else 0.0
        z = (p_i - p_r) / se if se > 0 else 0.0
        rows.append((label, p_i * 100, p_r * 100, z, two_sided_z_p(z)))

    out = pd.DataFrame(rows, columns=[label_col, "rate_pct", "rest_rate_pct", "z_score", "p_value"])
    out["p_adjusted"] = benjamini_hochberg(out["p_value"])
    out["significant"] = out["p_adjusted"] < 0.05
    return out


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """BH step-up adjusted p-values, controlling the false discovery rate."""
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.notna()
    m = int(valid.sum())
    if m == 0:
        return p
    order = p[valid].sort_values()
    ranks = np.arange(1, m + 1)
    adjusted = (order.to_numpy() * m / ranks)
    # enforce monotonicity from the largest p downward
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = p.copy()
    out.loc[order.index] = adjusted
    return out.round(6)


def rank_with_confidence(
    df: pd.DataFrame, label_col: str, success_col: str, trial_col: str
) -> tuple[pd.DataFrame, dict]:
    """Rank groups by rate, with Wilson intervals and a homogeneity verdict.

    Wilson intervals rather than normal approximations: they behave correctly
    on small denominators, which is exactly where a naive interval would claim
    precision it does not have.
    """
    test = homogeneity_test(df[success_col], df[trial_col])
    out = df.copy()
    d = pd.to_numeric(out[success_col], errors="coerce").fillna(0)
    n = pd.to_numeric(out[trial_col], errors="coerce").fillna(0)
    lo, hi = wilson_interval(d, n)
    out["rate_pct"] = (100 * d / n).round(3)
    out["ci_low_pct"] = (100 * lo).round(3)
    out["ci_high_pct"] = (100 * hi).round(3)
    out["ci_width_pp"] = (out["ci_high_pct"] - out["ci_low_pct"]).round(3)

    comparisons = group_vs_rest(out, label_col, success_col, trial_col)
    out = out.merge(
        comparisons[[label_col, "z_score", "p_value", "p_adjusted", "significant"]],
        on=label_col, how="left",
    )
    out = out.sort_values("rate_pct", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["ranking_is_meaningful"] = test["significant"]
    return out, test


def wilson_interval(successes, trials, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    d = np.asarray(pd.to_numeric(successes, errors="coerce").fillna(0), dtype=float)
    n = np.asarray(pd.to_numeric(trials, errors="coerce").fillna(0), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, d / n, np.nan)
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return centre - margin, centre + margin


def overlap_verdict(ranked: pd.DataFrame) -> str:
    """Plain-language reading of whether the top group really leads.

    This string is what the dashboard and the AI agent say out loud, so that a
    non-statistical reader is not left to infer it from a p-value.
    """
    if ranked.empty:
        return "No groups to compare."
    top = ranked.iloc[0]
    if not bool(top["ranking_is_meaningful"]):
        return (
            f"{top[ranked.columns[0]]} ranks highest at {top['rate_pct']:.2f}%, but the "
            f"differences between groups are within sampling noise, so this ordering should "
            f"not be treated as a real difference in behaviour."
        )
    challengers = ranked[(ranked.index > 0) & (ranked["ci_high_pct"] >= top["ci_low_pct"])]
    if len(challengers):
        names = ", ".join(str(v) for v in challengers[ranked.columns[0]].head(3))
        return (
            f"{top[ranked.columns[0]]} ranks highest at {top['rate_pct']:.2f}% "
            f"(95% CI {top['ci_low_pct']:.2f}–{top['ci_high_pct']:.2f}%), but its interval "
            f"overlaps {names}, so the lead is not clear-cut."
        )
    return (
        f"{top[ranked.columns[0]]} leads at {top['rate_pct']:.2f}% "
        f"(95% CI {top['ci_low_pct']:.2f}–{top['ci_high_pct']:.2f}%), separated from every "
        f"other group."
    )
