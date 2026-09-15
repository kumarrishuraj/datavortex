"""Hypothesis register.

`track1_dataset_notes.txt` ships a list headed "Example insights students may
discover". Every one of them is a plausible-sounding claim, and every one is
testable against this data. Asserting them because they sound like fraud
analytics would be the easiest way to publish something false.

So each is registered here with an explicit statistical test and reported with
its verdict — SUPPORTED, NOT SUPPORTED, BORDERLINE or CONTRADICTED — whichever
way it falls. A rejected hypothesis is a finding: it tells the business that a
control they might have built would have fired on noise.

No reading in this module carries a typed number, and each reading branches on
its own verdict, so the register stays true if the data changes. Every test is
implemented without SciPy.
"""
from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd

from src.analytics.significance import group_vs_rest, homogeneity_test
from src.analytics.shrinkage import MIN_DENOMINATOR, chi_square_sf, estimate_prior_strength
from src.config import UNKNOWN_KEY

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT SUPPORTED"
BORDERLINE = "BORDERLINE"
CONTRADICTED = "CONTRADICTED"
DESCRIPTIVE = "QUANTIFIED"
UNTESTABLE = "UNTESTABLE"

NOTES = "track1_dataset_notes.txt"
OURS = "DataVortex"


def _verdict(
    p: float,
    alpha: float = 0.05,
    borderline: float = 0.10,
    direction_matches: bool = True,
    corroborated: bool = True,
) -> str:
    """Grade a hypothesis.

    Two guards beyond the p-value, both learned on this dataset:

    `direction_matches` — a significant effect pointing the *opposite* way to the
    claim is not support for it; on the p-value alone it would read as SUPPORTED,
    which would be exactly backwards.

    `corroborated` — an omnibus test that clears 0.05 while no individual group
    survives correction for multiple comparisons is at best suggestive, so it is
    graded BORDERLINE rather than SUPPORTED.
    """
    if p != p:
        return UNTESTABLE
    if p < alpha:
        if not direction_matches:
            return CONTRADICTED
        return SUPPORTED if corroborated else BORDERLINE
    if p < borderline:
        return BORDERLINE
    return NOT_SUPPORTED


def goodness_of_fit_uniform(observed: pd.Series) -> tuple[float, int, float]:
    """Chi-square goodness of fit against a uniform distribution."""
    o = pd.to_numeric(observed, errors="coerce").dropna().to_numpy(dtype=float)
    k = len(o)
    if k < 2 or o.sum() == 0:
        return float("nan"), 0, float("nan")
    e = o.sum() / k
    chi = float(np.sum((o - e) ** 2 / e))
    return chi, k - 1, chi_square_sf(chi, k - 1)


def _row(claim, source, test, statistic, p, verdict, reading) -> dict:
    return {
        "hypothesis": claim,
        "source": source,
        "test": test,
        "statistic": statistic,
        "p_value": None if p != p else round(float(p), 5),
        "verdict": verdict,
        "reading": reading,
    }


def welch_t_p(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Welch's t statistic and two-sided p, normal approximation for large n."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = sqrt(va + vb)
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    return t, min(1.0, erfc(abs(t) / sqrt(2)))


def _pair_z(independence: pd.DataFrame | None, pair: str) -> float | None:
    if independence is None or independence.empty or "pair" not in independence:
        return None
    hit = independence[independence["pair"] == pair]
    if hit.empty:
        return None
    z = float(hit["z_score"].iloc[0])
    return None if np.isnan(z) else z


def build_register(tx: pd.DataFrame, cb: pd.DataFrame,
                   dim_users: pd.DataFrame, dim_merchants: pd.DataFrame,
                   recon: dict | None = None,
                   independence: pd.DataFrame | None = None) -> pd.DataFrame:
    disputed = set(cb.loc[~cb["txn_unlinked"], "txn_key"].dropna())
    t = tx.assign(is_disputed=tx["txn_key"].isin(disputed))
    baseline = float(t["is_disputed"].mean())
    rows: list[dict] = []

    # 1 -- merchant categories
    category = dim_merchants.set_index("merchant_key")["merchant_category_canonical"]
    tc = t.assign(category=t["merchant_id_normalized"].map(category).fillna(UNKNOWN_KEY))
    g = tc.groupby("category").agg(n=("txn_key", "size"), d=("is_disputed", "sum"))
    test = homogeneity_test(g["d"], g["n"])
    verdict = _verdict(test["p_value"])
    rates = 100 * g["d"] / g["n"]
    typical_n = float(g.loc[g.index != UNKNOWN_KEY, "n"].median()) if (g.index != UNKNOWN_KEY).any() else float(g["n"].median())
    span = f"Category rates span {rates.min():.2f}%-{rates.max():.2f}% on denominators of about {typical_n:,.0f} transactions."
    rows.append(_row(
        "Certain merchant categories have disproportionately high chargebacks.",
        NOTES, f"Chi-square homogeneity across {test['groups']} categories",
        f"chi2={test['chi_square']}, df={test['df']}", test["p_value"], verdict,
        span + (" The spread is what chance alone produces, so ranking categories by dispute "
                "rate does not identify a riskier category." if verdict != SUPPORTED else
                " At least one category differs from the rest by more than chance."),
    ))

    # 2 -- repeat disputers, against the binomial expectation
    per_user = t.groupby("user_id_normalized").agg(n=("txn_key", "size"), d=("is_disputed", "sum"))
    observed = int((per_user["d"] >= 2).sum())
    n_arr = per_user["n"].to_numpy(dtype=float)
    p0 = baseline
    expected = float(np.sum(1 - (1 - p0) ** n_arr - n_arr * p0 * (1 - p0) ** (n_arr - 1)))
    sd = sqrt(max(expected * (1 - expected / max(len(n_arr), 1)), 1e-9))
    z = (observed - expected) / sd
    p = min(1.0, erfc(abs(z) / sqrt(2)))
    verdict = _verdict(p, direction_matches=observed > expected)
    rows.append(_row(
        "Some users appear repeatedly in disputes.",
        NOTES, "Observed vs binomial-expected count of users with >=2 disputes",
        f"observed={observed}, expected={expected:.1f}, z={z:+.2f}", p, verdict,
        f"{observed:,} customers have two or more disputed transactions; independent disputes at "
        f"the {100 * p0:.2f}% base rate would produce about {expected:.0f}. "
        + ("Repeat disputers exist because customers transact more than once, not because a "
           "subgroup is dispute-prone." if verdict != SUPPORTED else
           "More customers repeat than chance allows, which points to a dispute-prone subgroup."),
    ))

    # 3 -- merchant transaction spikes
    daily = t.groupby("date").size()
    mean, var = daily.mean(), daily.var(ddof=1)
    idx = var / mean if mean else float("nan")
    chi = idx * (len(daily) - 1)
    p_spike = chi_square_sf(chi, len(daily) - 1)
    per_merchant_day = t.groupby(["merchant_id_normalized", "date"]).size()
    burst = int(per_merchant_day.max())
    verdict = _verdict(p_spike, direction_matches=idx > 1)
    rows.append(_row(
        "Some merchants show sudden transaction spikes followed by disputes.",
        NOTES, "Poisson index of dispersion on daily volume + max merchant-day burst",
        f"var/mean={idx:.3f}, max merchant-day={burst}", p_spike, verdict,
        f"Daily volume is {mean:.0f} +/- {daily.std():.0f} (CV {daily.std() / mean:.3f}) across "
        f"{len(daily)} days, and no merchant exceeds {burst} transactions in a single day. "
        + ("There is no burst to detect; a spike rule would never fire." if verdict != SUPPORTED
           else "Daily volume is more variable than a steady process would give."),
    ))

    # 4 -- missing UTR
    g = t.groupby("utr_missing").agg(n=("txn_key", "size"), d=("is_disputed", "sum"))
    test = homogeneity_test(g["d"], g["n"])
    gf = t.groupby("utr_missing").agg(n=("txn_key", "size"),
                                      f=("status_canonical", lambda s: int((s == "FAILED").sum())))
    test_f = homogeneity_test(gf["f"], gf["n"])
    p_utr = min(test["p_value"], test_f["p_value"])
    verdict = _verdict(p_utr)
    missing_rate = 100 * g.loc[True, "d"] / g.loc[True, "n"] if True in g.index else float("nan")
    present_rate = 100 * g.loc[False, "d"] / g.loc[False, "n"] if False in g.index else float("nan")
    rows.append(_row(
        "Missing UTRs correlate with failed or disputed transactions.",
        NOTES, "Chi-square, missing-UTR vs present-UTR, on dispute and on failure",
        f"dispute chi2={test['chi_square']}, failure chi2={test_f['chi_square']}", p_utr, verdict,
        f"Dispute rate is {missing_rate:.2f}% where the UTR is missing against {present_rate:.2f}% "
        "where it is present. "
        + ("A missing UTR is a data-quality defect worth fixing in ingestion, not a fraud "
           "indicator." if verdict != SUPPORTED else
           "Missing UTRs are associated with a different dispute or failure rate."),
    ))

    # 5 -- KYC status
    status = dim_users.set_index("user_key")["kyc_status_canonical"]
    tk = t.assign(kyc=t["user_id_normalized"].map(status).fillna(UNKNOWN_KEY))
    g = tk.groupby("kyc").agg(n=("txn_key", "size"), d=("is_disputed", "sum"))
    test = homogeneity_test(g["d"], g["n"])
    rates = (100 * g["d"] / g["n"]).round(2).to_dict()
    pairwise = group_vs_rest(
        g.reset_index().rename(columns={"kyc": "label", "d": "succ", "n": "trials"}),
        "label", "succ", "trials")
    any_pairwise = bool(pairwise["significant"].any())
    verdict = _verdict(test["p_value"], corroborated=any_pairwise)
    parts = ["Rates by KYC status: " + ", ".join(f"{k} {v:.2f}%" for k, v in rates.items()) + "."]
    if test["significant"] and not any_pairwise:
        parts.append("The omnibus test clears 0.05, but no individual status survives correction "
                     "for multiple comparisons.")
    elif test["significant"]:
        parts.append("At least one status differs from the rest after correction.")
    else:
        parts.append("Differences between statuses are within sampling noise.")
    verified, pending = rates.get("VERIFIED"), rates.get("PENDING")
    if verified is not None and pending is not None and verified > pending:
        parts.append(f"The direction also cuts against the claim: VERIFIED customers dispute more "
                     f"often ({verified:.2f}%) than PENDING ones ({pending:.2f}%).")
    z_user = _pair_z(independence, "tx.user x kyc.user")
    if z_user is not None and abs(z_user) < 3:
        parts.append(f"And KYC resolution was shown to be random with respect to transactions "
                     f"(z = {z_user:.2f}), so a real effect would be hard to explain.")
    if verdict != SUPPORTED:
        parts.append("Suggestive at most; not a control we would ship.")
    rows.append(_row(
        "Unverified or rejected KYC users show higher dispute risk.",
        NOTES, f"Chi-square homogeneity across {test['groups']} KYC statuses, "
               "plus per-status tests with Benjamini-Hochberg correction",
        f"chi2={test['chi_square']}, df={test['df']}, "
        f"min adjusted pairwise p={pairwise['p_adjusted'].min():.3f}",
        test["p_value"], verdict, " ".join(parts),
    ))

    # 6 -- delayed reporting and account takeover
    ato = cb[cb["reason_code_canonical"] == "ACCOUNT_TAKEOVER"]["reporting_delay_days"]
    rest = cb[cb["reason_code_canonical"] != "ACCOUNT_TAKEOVER"]["reporting_delay_days"]
    tstat, p_ato = welch_t_p(ato, rest)
    claim_holds = bool(ato.mean() > rest.mean())
    verdict = _verdict(p_ato, direction_matches=claim_holds)
    if verdict == CONTRADICTED:
        reading = (f"The effect is real but points the other way: account-takeover disputes are "
                   f"reported {rest.mean() - ato.mean():.2f} days EARLIER than other reasons "
                   f"({ato.mean():.2f} d vs {rest.mean():.2f} d). A rule that escalated "
                   "late-reported disputes as suspected takeover would systematically look in the "
                   "wrong place on this data.")
    elif verdict == SUPPORTED:
        reading = (f"Account-takeover disputes are reported later ({ato.mean():.2f} d vs "
                   f"{rest.mean():.2f} d), consistent with the claim.")
    else:
        reading = (f"Account-takeover disputes are reported after {ato.mean():.2f} days against "
                   f"{rest.mean():.2f} for other reasons — not a difference reporting delay can "
                   "triage on.")
    rows.append(_row(
        "Delayed chargeback reporting indicates account takeover or late fraud detection.",
        NOTES, "Welch t-test on reporting delay, account-takeover vs other reasons (directional)",
        f"t={tstat:.2f}, mean {ato.mean():.2f} d vs {rest.mean():.2f} d", p_ato, verdict, reading,
    ))

    # 7 -- duplicates, quantified from the pipeline's reconciliation tables
    if recon and "recon_rows" in recon and "recon_value" in recon:
        recon_rows = recon["recon_rows"].set_index("Metric")
        recon_value = recon["recon_value"].set_index("step")
        dup_rows = int(-recon_rows.loc["Transaction rows", "Difference"])
        dup_complaints = int(-recon_rows.loc["Chargeback rows", "Difference"])
        dup_value = float(-recon_value.loc["After duplicate removal", "delta"])
        before = float(recon_value.loc["Raw, sign repaired", "total_inr"])
        rows.append(_row(
            "Duplicate transactions inflate revenue and dispute metrics if not removed.",
            NOTES, "Direct quantification against the raw file",
            f"{dup_rows:,} duplicate rows, Rs {dup_value:,.0f}", float("nan"), DESCRIPTIVE,
            f"Left in place, the {dup_rows:,} byte-identical duplicate transactions would overstate "
            f"transaction value by Rs {dup_value:,.0f} ({100 * dup_value / before:.1f}%) and "
            f"double-count {dup_complaints:,} complaints. This one is straightforwardly true and "
            "the pipeline removes them.",
        ))
    else:
        rows.append(_row(
            "Duplicate transactions inflate revenue and dispute metrics if not removed.",
            NOTES, "Direct quantification against the raw file", "reconciliation tables not supplied",
            float("nan"), UNTESTABLE, "Run scripts/run_pipeline.py to quantify duplicate inflation.",
        ))

    # 8 -- our own: merchant-level dispute propensity
    per_m = t.groupby("merchant_id_normalized").agg(n=("txn_key", "size"), d=("is_disputed", "sum"))
    eligible = per_m[per_m["n"] >= MIN_DENOMINATOR]
    prior = estimate_prior_strength(eligible["d"], eligible["n"])
    verdict = _verdict(prior["p_value"])
    rows.append(_row(
        "Individual merchants differ in how often their transactions are disputed.",
        OURS, f"Beta-binomial overdispersion test on merchants with >={MIN_DENOMINATOR} transactions",
        f"chi2/df={prior['chi_square'] / prior['df']:.4f}, N={prior['groups']:,}",
        prior["p_value"], verdict,
        ("The spread of merchant dispute rates is no wider than binomial sampling noise. Every "
         f"merchant is consistent with one shared {100 * prior['mean_rate']:.2f}% probability, so "
         "merchant dispute rate carries no information and must not drive a risk score."
         if verdict != SUPPORTED else
         "Merchant dispute rates vary by more than sampling noise, so a shrunk merchant rate is "
         "informative."),
    ))

    # 9 -- our own: intraday pattern
    known = t[t["timestamp_time_known"]]
    excluded = len(t) - len(known)
    hourly = known.groupby("hour").size()
    chi_h, df_h, p_h = goodness_of_fit_uniform(hourly)
    verdict = _verdict(p_h)
    cv = hourly.std() / hourly.mean()
    rows.append(_row(
        "Transaction volume follows an intraday pattern.",
        OURS, "Chi-square goodness of fit against a uniform 24-hour distribution "
              "(date-only timestamps excluded)",
        f"chi2={chi_h:.1f}, df={df_h}, CV={cv:.4f}", p_h, verdict,
        f"Excluding the {excluded:,} date-only timestamps that parse to midnight, hourly volume "
        + (f"is flat (CV {cv:.3f}). The apparent midnight peak in the raw data is a parsing "
           "artifact, not customer behaviour — worth knowing before anyone builds an off-hours rule."
           if verdict != SUPPORTED else f"varies by hour (CV {cv:.3f}).")
    ))

    # 10 -- our own: reason code vs complaint text
    themes = {"UNAUTHORIZED", "DUPLICATE_DEBIT", "ACCOUNT_TAKEOVER", "SERVICE_NOT_PROVIDED"}
    comparable = cb[cb["complaint_theme"].isin(themes)]
    n = len(comparable)
    if n:
        agree = float((comparable["reason_code_canonical"] == comparable["complaint_theme"]).mean())
        p_reason = cb["reason_code_canonical"].value_counts(normalize=True)
        p_theme = comparable["complaint_theme"].value_counts(normalize=True)
        expected_rate = float(sum(p_reason.get(k, 0) * p_theme.get(k, 0) for k in themes))
        se = sqrt(expected_rate * (1 - expected_rate) / n) if 0 < expected_rate < 1 else 0.0
        z_rt = (agree - expected_rate) / se if se else 0.0
        p_rt = min(1.0, erfc(abs(z_rt) / sqrt(2)))
        verdict = _verdict(p_rt, direction_matches=agree > expected_rate)
        rows.append(_row(
            "A mismatch between reason code and complaint text is an anomaly signal.",
            OURS, "Agreement rate against the independence expectation (one-sample z-test)",
            f"agreement={100 * agree:.2f}%, expected={100 * expected_rate:.2f}%, z={z_rt:+.2f}, n={n:,}",
            p_rt, verdict,
            f"Where the complaint text names a specific cause, the structured reason code agrees "
            f"with it {100 * agree:.2f}% of the time, against {100 * expected_rate:.2f}% expected if "
            "the two fields were unrelated. "
            + ("They carry no mutual information, so a mismatch means nothing and is reported as a "
               "separate dimension rather than scored." if verdict != SUPPORTED else
               "They agree more than chance, so a mismatch is informative."),
        ))

    return pd.DataFrame(rows)
