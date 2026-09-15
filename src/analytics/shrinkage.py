"""Empirical-Bayes shrinkage for small-denominator rates.

The problem this solves, in one number: 349 merchants in this dataset show a
dispute ratio of 1.000 — every transaction disputed — purely because they have
a single transaction that happened to be disputed. A "top risk merchants" table
ranked on the raw ratio is therefore a list of merchants with n=1, which is
noise presented as insight.

Two guards, in order:

1. **Denominator floor.** A rate computed on fewer than `MIN_DENOMINATOR`
   transactions is not reported as a rate at all. The row is kept and shown,
   but marked `below_floor` and excluded from rankings.

2. **Beta-binomial shrinkage toward the peer mean.** Each merchant's rate is
   pulled toward its category's mean by an amount set by how much evidence the
   merchant actually provides:

       shrunk = (disputes + m * k) / (transactions + k)

   where `m` is the category mean and `k` is a prior strength estimated from
   the data by method of moments. A merchant with many transactions barely
   moves; one with three moves most of the way to its peer group.

The estimate of `k` is itself the interesting result. If between-merchant
variation is no larger than binomial sampling noise would produce, the method
of moments returns no finite `k`, and the honest reading is that **this data
contains no detectable merchant-level dispute propensity** — every merchant's
rate is consistent with the same underlying probability. We report that rather
than inventing a ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approved with the team: rates below this denominator are not ranked.
MIN_DENOMINATOR = 3

# Used only when the method of moments finds no overdispersion, i.e. when the
# data says there is nothing to distinguish. A large k shrinks every estimate
# essentially all the way to the peer mean, which is the correct behaviour.
NO_SIGNAL_K = 1e6


def estimate_prior_strength(successes: pd.Series, trials: pd.Series) -> dict:
    """Method-of-moments estimate of the beta-binomial prior strength k.

    For a beta-binomial with mean m and prior strength k:

        Var(p_i) = m(1-m)/n_i * [1 + (n_i - 1)/(k + 1)]

    Summing n_i(p_i - m)^2 over merchants gives

        S / (m(1-m)) = N + (sum(n_i) - N)/(k + 1)

    so k is recoverable in closed form. When the left side is at or below N,
    the observed spread is no wider than pure binomial noise: there is no
    between-merchant signal, and k is reported as infinite.

    Returns the estimate plus the overdispersion diagnostics, so the caller can
    show its working rather than assert a conclusion.
    """
    d = pd.to_numeric(successes, errors="coerce").fillna(0).to_numpy(dtype=float)
    n = pd.to_numeric(trials, errors="coerce").fillna(0).to_numpy(dtype=float)
    keep = n > 0
    d, n = d[keep], n[keep]

    N = len(n)
    total_n = float(n.sum())
    if total_n == 0:
        return _no_signal(N, total_n, np.nan, np.nan, np.nan, "no trials to estimate from")

    # Compute the pooled mean before the group-count guard: even when there are
    # too few groups to estimate overdispersion, the mean is well defined and
    # callers need it as the shrinkage target. Returning NaN here silently
    # propagates into every shrunk rate.
    m = float(d.sum() / total_n)

    if N < 2:
        return _no_signal(N, total_n, m, np.nan, np.nan, "too few groups to estimate")
    if m <= 0 or m >= 1:
        return _no_signal(N, total_n, m, np.nan, np.nan, "degenerate mean (0% or 100%)")

    p = d / n
    S = float(np.sum(n * (p - m) ** 2))
    scaled = S / (m * (1 - m))          # ~chi-square with N-1 df under the null
    pvalue = chi_square_sf(scaled, N - 1)

    if scaled <= N:
        return _no_signal(N, total_n, m, scaled, pvalue,
                          "observed spread is within binomial sampling noise")

    k = (total_n - N) / (scaled - N) - 1
    if not np.isfinite(k) or k <= 0:
        return _no_signal(N, total_n, m, scaled, pvalue, "non-positive prior strength")

    return {
        "groups": N,
        "total_trials": total_n,
        "mean_rate": m,
        "chi_square": scaled,
        "df": N - 1,
        "p_value": pvalue,
        "overdispersed": True,
        "prior_strength_k": float(k),
        "note": "between-group variation exceeds binomial noise",
    }


def chi_square_sf(stat: float, df: int) -> float:
    """Upper-tail probability of a chi-square statistic.

    Wilson-Hilferty normal approximation, accurate to ~1e-3 for df > 30 which
    is far beyond what we need here — this is a diagnostic, not an inference
    we act on. Implemented locally so the pipeline stays free of SciPy.
    """
    from math import erfc, sqrt

    if df <= 0 or not np.isfinite(stat) or stat <= 0:
        return float("nan")
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
    return 0.5 * erfc(z / sqrt(2))


def _no_signal(N, total_n, m, scaled, pvalue, note) -> dict:
    return {
        "groups": int(N),
        "total_trials": float(total_n),
        "mean_rate": float(m) if m == m else np.nan,
        "chi_square": float(scaled) if scaled == scaled else np.nan,
        "df": int(N - 1) if N else 0,
        "p_value": float(pvalue) if pvalue == pvalue else np.nan,
        "overdispersed": False,
        "prior_strength_k": NO_SIGNAL_K,
        "note": note,
    }


def shrink_rates(
    df: pd.DataFrame,
    success_col: str,
    trial_col: str,
    group_col: str | None = None,
    min_denominator: int = MIN_DENOMINATOR,
) -> pd.DataFrame:
    """Add raw rate, shrunk rate and the floor flag.

    `group_col` names the peer group to shrink toward (merchant category). When
    omitted, everything shrinks toward the global mean. Prior strength is
    estimated once on the eligible population — groups at or above the floor —
    so that the n=1 merchants cannot distort the estimate of how much evidence
    a transaction carries.
    """
    out = df.copy()
    d = pd.to_numeric(out[success_col], errors="coerce").fillna(0)
    n = pd.to_numeric(out[trial_col], errors="coerce").fillna(0)

    out["rate_raw"] = np.where(n > 0, d / n, np.nan)
    out["below_floor"] = n < min_denominator

    eligible = out[~out["below_floor"]]
    if eligible.empty:
        # No group clears the floor: fall back to the whole population so the
        # shrinkage target stays defined rather than becoming NaN.
        eligible = out
    prior = estimate_prior_strength(eligible[success_col], eligible[trial_col])
    k = prior["prior_strength_k"]

    # Guaranteed-finite shrinkage target. Without this guard a degenerate prior
    # would NaN out every shrunk rate in the aggregate, silently.
    global_mean = prior["mean_rate"]
    if not np.isfinite(global_mean):
        total = n.sum()
        global_mean = float(d.sum() / total) if total else 0.0

    if group_col is not None and group_col in out.columns:
        peer = (
            eligible.groupby(group_col)
            .apply(lambda g: g[success_col].sum() / max(g[trial_col].sum(), 1), include_groups=False)
            .rename("peer_mean")
        )
        out = out.merge(peer, left_on=group_col, right_index=True, how="left")
        out["peer_mean"] = out["peer_mean"].fillna(global_mean)
    else:
        out["peer_mean"] = global_mean

    out["rate_shrunk"] = (d + out["peer_mean"] * k) / (n + k)
    # How far the estimate moved from its raw value, i.e. how little the
    # merchant's own history was trusted.
    out["shrink_weight"] = np.where(n + k > 0, k / (n + k), 1.0)

    out.attrs["prior"] = prior
    return out


def excess_over_peer(df: pd.DataFrame) -> pd.Series:
    """Shrunk rate minus peer mean, in percentage points.

    This, not the raw ratio, is what a risk ranking should use: it asks whether
    a merchant is worse than comparable merchants, having already discounted
    the thinness of its own evidence.
    """
    return (df["rate_shrunk"] - df["peer_mean"]) * 100


def binomial_pvalue(successes: pd.Series, trials: pd.Series, baseline: float) -> pd.Series:
    """One-sided P(X >= observed) under the baseline rate.

    Reported beside each merchant so a reviewer can see how ordinary an
    apparently high ratio is. Uses the regularized incomplete beta identity for
    the binomial survival function, so no SciPy dependency is needed.
    """
    from math import lgamma, log, exp

    def _sf(d: float, n: float) -> float:
        if n <= 0 or d <= 0:
            return 1.0
        d, n = int(d), int(n)
        # P(X >= d) = sum_{i=d}^{n} C(n,i) p^i (1-p)^(n-i)
        total = 0.0
        for i in range(d, n + 1):
            logc = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
            term = logc + i * log(baseline) + (n - i) * log(1 - baseline)
            total += exp(term)
        return min(total, 1.0)

    d = pd.to_numeric(successes, errors="coerce").fillna(0)
    n = pd.to_numeric(trials, errors="coerce").fillna(0)
    return pd.Series([_sf(a, b) for a, b in zip(d, n)], index=d.index)
