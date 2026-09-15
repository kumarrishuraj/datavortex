"""Tests for the Stage 3 analytics layer.

Unit tests use synthetic frames with known answers, so a statistical routine is
checked against arithmetic rather than against its own output. Integration
tests then assert the built aggregates reconcile to the facts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics import kpis as K  # noqa: E402
from src.analytics import semantic  # noqa: E402
from src.analytics.hypotheses import CONTRADICTED, NOT_SUPPORTED, SUPPORTED, _verdict, welch_t_p  # noqa: E402
from src.analytics.shrinkage import (  # noqa: E402
    MIN_DENOMINATOR, NO_SIGNAL_K, binomial_pvalue, chi_square_sf,
    estimate_prior_strength, shrink_rates,
)
from src.analytics.significance import (  # noqa: E402
    benjamini_hochberg, group_vs_rest, homogeneity_test, overlap_verdict,
    rank_with_confidence, wilson_interval,
)
from src.config import PROCESSED_DIR, UNKNOWN_KEY  # noqa: E402

ANALYTICS_DIR = PROCESSED_DIR / "analytics"


# --------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------
def test_no_overdispersion_yields_no_signal_k():
    """Rates generated from ONE shared probability must report no signal."""
    rng = np.random.default_rng(0)
    n = pd.Series(rng.integers(3, 12, size=800))
    d = pd.Series(rng.binomial(n, 0.12))
    prior = estimate_prior_strength(d, n)
    assert prior["overdispersed"] is False
    assert prior["prior_strength_k"] == NO_SIGNAL_K


def test_genuine_overdispersion_is_detected():
    """Rates drawn from a spread of probabilities must be flagged."""
    rng = np.random.default_rng(1)
    n = pd.Series(rng.integers(20, 60, size=800))
    p = rng.beta(2, 8, size=800)          # real between-group variation
    d = pd.Series(rng.binomial(n, p))
    prior = estimate_prior_strength(d, n)
    assert prior["overdispersed"] is True
    assert 0 < prior["prior_strength_k"] < NO_SIGNAL_K


def test_shrinkage_pulls_small_denominators_toward_the_mean():
    df = pd.DataFrame({"d": [1, 50], "n": [1, 500], "g": ["A", "A"]})
    out = shrink_rates(df, "d", "n", min_denominator=MIN_DENOMINATOR)
    thin, thick = out.iloc[0], out.iloc[1]
    assert thin["rate_raw"] == 1.0
    # the n=1 merchant must not stay at 100% after shrinkage
    assert thin["rate_shrunk"] < 0.5
    # the well-evidenced one moves less than the thin one
    assert abs(thick["rate_shrunk"] - thick["peer_mean"]) <= abs(thin["rate_shrunk"] - thin["peer_mean"])


def test_denominator_floor_flags_thin_rows_without_dropping_them():
    df = pd.DataFrame({"d": [1, 0, 2], "n": [1, 2, 10], "g": ["A"] * 3})
    out = shrink_rates(df, "d", "n", min_denominator=3)
    assert list(out["below_floor"]) == [True, True, False]
    assert len(out) == 3          # nothing dropped


def test_shrink_weight_is_between_zero_and_one():
    df = pd.DataFrame({"d": [1, 5, 20], "n": [4, 40, 400], "g": ["A"] * 3})
    out = shrink_rates(df, "d", "n")
    assert ((out["shrink_weight"] >= 0) & (out["shrink_weight"] <= 1)).all()


def test_shrunk_rate_is_never_nan_even_with_a_degenerate_prior():
    """Regression: a single eligible group left mean_rate NaN, which silently
    NaN-ed out every shrunk rate in the aggregate."""
    one_group = pd.DataFrame({"d": [1, 50], "n": [1, 500]})
    out = shrink_rates(one_group, "d", "n", min_denominator=3)
    assert out["rate_shrunk"].notna().all()
    assert out["peer_mean"].notna().all()

    nothing_eligible = pd.DataFrame({"d": [1, 0], "n": [1, 2]})
    out2 = shrink_rates(nothing_eligible, "d", "n", min_denominator=3)
    assert out2["rate_shrunk"].notna().all()

    all_zero = pd.DataFrame({"d": [0, 0, 0], "n": [10, 10, 10]})
    out3 = shrink_rates(all_zero, "d", "n", min_denominator=3)
    assert out3["rate_shrunk"].notna().all()
    assert (out3["rate_shrunk"] == 0).all()


def test_binomial_pvalue_matches_hand_computation():
    # P(X >= 2 | n=3, p=0.5) = 3*0.125 + 0.125 = 0.5
    p = binomial_pvalue(pd.Series([2]), pd.Series([3]), 0.5)
    assert p.iloc[0] == pytest.approx(0.5, abs=1e-9)


def test_binomial_pvalue_of_zero_successes_is_one():
    assert binomial_pvalue(pd.Series([0]), pd.Series([10]), 0.2).iloc[0] == 1.0


def test_chi_square_sf_is_sane():
    assert chi_square_sf(100, 100) == pytest.approx(0.5, abs=0.05)
    assert chi_square_sf(200, 100) < 0.001
    assert chi_square_sf(50, 100) > 0.99


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------
def test_homogeneity_detects_no_difference_when_groups_share_a_rate():
    df = pd.DataFrame({"d": [100, 102, 98], "n": [1000, 1000, 1000]})
    test = homogeneity_test(df["d"], df["n"])
    assert test["significant"] is False


def test_homogeneity_detects_a_real_difference():
    df = pd.DataFrame({"d": [100, 300, 90], "n": [1000, 1000, 1000]})
    test = homogeneity_test(df["d"], df["n"])
    assert test["significant"] is True


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(pd.Series([50]), pd.Series([100]))
    assert lo[0] < 0.5 < hi[0]
    assert 0 <= lo[0] and hi[0] <= 1


def test_wilson_interval_is_wider_on_thin_denominators():
    lo_thin, hi_thin = wilson_interval(pd.Series([1]), pd.Series([3]))
    lo_thick, hi_thick = wilson_interval(pd.Series([100]), pd.Series([300]))
    assert (hi_thin[0] - lo_thin[0]) > (hi_thick[0] - lo_thick[0])


def test_benjamini_hochberg_is_monotone_and_bounded():
    p = pd.Series([0.001, 0.01, 0.03, 0.2, 0.9])
    adj = benjamini_hochberg(p)
    assert (adj >= p).all()
    assert (adj <= 1).all()
    assert list(adj) == sorted(adj)


def test_group_vs_rest_finds_nothing_in_a_homogeneous_set():
    df = pd.DataFrame({"g": list("ABCD"), "d": [100, 102, 98, 101], "n": [1000] * 4})
    out = group_vs_rest(df, "g", "d", "n")
    assert not out["significant"].any()


def test_group_vs_rest_puts_the_outlier_at_the_top():
    """With one extreme group the pooled 'rest' is contaminated, so several
    groups can differ from it. What must hold is that the outlier is the most
    extreme and is flagged."""
    df = pd.DataFrame({"g": list("ABCDE"), "d": [100, 100, 100, 100, 400], "n": [1000] * 5})
    out = group_vs_rest(df, "g", "d", "n").set_index("g")
    assert bool(out.loc["E", "significant"]) is True
    assert out["z_score"].abs().idxmax() == "E"


def test_rank_with_confidence_marks_a_noise_ranking_as_meaningless():
    df = pd.DataFrame({"g": list("ABCD"), "d": [120, 118, 122, 119], "n": [1000] * 4})
    ranked, test = rank_with_confidence(df, "g", "d", "n")
    assert test["significant"] is False
    assert not ranked["ranking_is_meaningful"].any()
    assert "sampling noise" in overlap_verdict(ranked)


def test_rank_with_confidence_reports_a_real_leader():
    df = pd.DataFrame({"g": list("ABCD"), "d": [400, 100, 105, 95], "n": [1000] * 4})
    ranked, test = rank_with_confidence(df, "g", "d", "n")
    assert test["significant"] is True
    assert ranked.iloc[0]["g"] == "A"
    assert "sampling noise" not in overlap_verdict(ranked)


# --------------------------------------------------------------------------
# Hypothesis grading
# --------------------------------------------------------------------------
def test_significant_effect_in_the_wrong_direction_is_contradicted_not_supported():
    """The guard that stopped us publishing a backwards finding."""
    assert _verdict(0.01, direction_matches=False) == CONTRADICTED
    assert _verdict(0.01, direction_matches=True) == SUPPORTED


def test_uncorroborated_omnibus_result_is_only_borderline():
    assert _verdict(0.04, corroborated=False) == "BORDERLINE"
    assert _verdict(0.04, corroborated=True) == SUPPORTED


def test_large_p_value_is_not_supported():
    assert _verdict(0.9) == NOT_SUPPORTED


def test_welch_t_detects_a_real_mean_difference():
    a = pd.Series(np.random.default_rng(2).normal(10, 1, 500))
    b = pd.Series(np.random.default_rng(3).normal(12, 1, 500))
    t, p = welch_t_p(a, b)
    assert p < 0.001


# --------------------------------------------------------------------------
# Semantic layer
# --------------------------------------------------------------------------
def test_every_metric_declares_its_coverage_basis():
    for name, m in semantic.METRICS.items():
        assert m.coverage_basis, f"{name} has no coverage_basis"
        assert m.definition and m.formula, f"{name} is under-specified"


def test_metric_chart_hints_are_from_the_allowed_list():
    for m in semantic.METRICS.values():
        assert m.chart_hint in semantic.ALLOWED_CHARTS


def test_unknown_metric_raises_a_helpful_error():
    with pytest.raises(KeyError, match="unknown metric"):
        semantic.get("revenue_per_unicorn")


def test_chargeback_ratio_definition_says_distinct_transactions():
    """Guard against someone quietly switching the numerator back to complaints."""
    m = semantic.get("chargeback_to_transaction_ratio")
    assert "DISTINCT" in m.definition.upper()


# --------------------------------------------------------------------------
# Integration against the built layer
# --------------------------------------------------------------------------
pytestmark_files = ANALYTICS_DIR / "agg_daily.parquet"
integration = pytest.mark.skipif(
    not pytestmark_files.exists(),
    reason="run `python scripts/build_analytics.py` first",
)


@pytest.fixture(scope="module")
def agg():
    names = ["agg_daily", "agg_hourly", "agg_merchant", "agg_user", "agg_category",
             "agg_kyc_status", "agg_state", "agg_data_quality", "kpi_headline",
             "kpi_validation", "agg_hypothesis_register"]
    return {n: pd.read_parquet(ANALYTICS_DIR / f"{n}.parquet") for n in names}


@pytest.fixture(scope="module")
def facts():
    return (
        pd.read_parquet(PROCESSED_DIR / "fact_transactions.parquet"),
        pd.read_parquet(PROCESSED_DIR / "fact_chargebacks.parquet"),
        pd.read_parquet(PROCESSED_DIR / "dim_users.parquet"),
    )


@integration
def test_all_kpi_validation_checks_pass(agg):
    assert (agg["kpi_validation"]["status"] == "PASS").all()


@integration
@pytest.mark.parametrize("table", ["agg_daily", "agg_merchant", "agg_user", "agg_category"])
def test_aggregates_reconcile_to_the_fact_table(agg, facts, table):
    tx = facts[0]
    assert int(agg[table]["transactions"].sum()) == len(tx)


@integration
def test_every_aggregate_carries_coverage(agg):
    for name in ["agg_daily", "agg_hourly", "agg_merchant", "agg_user",
                 "agg_category", "agg_kyc_status", "agg_state"]:
        assert "coverage_pct" in agg[name].columns, f"{name} has no coverage_pct"
        assert "coverage_basis" in agg[name].columns


@integration
def test_unknown_is_visible_in_category_and_kyc_breakdowns(agg):
    """Hiding UNKNOWN would drop two thirds of customers from the picture."""
    cats = set(agg["agg_category"]["category"])
    assert any("UNKNOWN" in c for c in cats)
    assert UNKNOWN_KEY in set(agg["agg_kyc_status"]["kyc_status"])


@integration
def test_unknown_is_the_largest_group_and_is_not_suppressed(agg):
    cat = agg["agg_category"]
    unknown = cat[cat["category"].str.contains("UNKNOWN")]
    assert len(unknown) == 1
    assert int(unknown["transactions"].iloc[0]) == int(cat["transactions"].max())


@integration
def test_merchant_aggregate_covers_the_full_transacting_population(agg, facts):
    tx = facts[0]
    assert len(agg["agg_merchant"]) == tx["merchant_id_normalized"].nunique()
    assert len(agg["agg_merchant"]) > agg["agg_merchant"]["in_master"].sum()


@integration
def test_hourly_excludes_date_only_timestamps(agg, facts):
    tx = facts[0]
    hourly = agg["agg_hourly"]
    assert int(hourly["transactions"].sum()) == int(tx["timestamp_time_known"].sum())
    assert int(hourly["excluded_date_only"].iloc[0]) == int((~tx["timestamp_time_known"]).sum())


@integration
def test_hourly_profile_is_flat_once_the_artifact_is_removed(agg):
    h = agg["agg_hourly"]["transactions"]
    assert (h.std() / h.mean()) < 0.10


@integration
def test_merchants_below_the_floor_are_flagged_not_removed(agg):
    mer = agg["agg_merchant"]
    assert int(mer["below_floor"].sum()) > 0
    assert (mer.loc[mer["below_floor"], "transactions"] < MIN_DENOMINATOR).all()
    assert (mer.loc[~mer["below_floor"], "transactions"] >= MIN_DENOMINATOR).all()


@integration
def test_shrunk_rates_are_far_less_dispersed_than_raw_rates(agg):
    mer = agg["agg_merchant"]
    e = mer[~mer["below_floor"]]
    assert e["dispute_rate_shrunk_pct"].std() < e["dispute_rate_raw_pct"].std() / 5


@integration
def test_shrunk_rate_never_exceeds_100_percent(agg):
    mer = agg["agg_merchant"]
    assert (mer["dispute_rate_shrunk_pct"] <= 100).all()
    assert (mer["dispute_rate_shrunk_pct"] >= 0).all()


@integration
def test_category_ranking_carries_its_significance_verdict(agg):
    cat = agg["agg_category"]
    for col in ["rate_pct", "ci_low_pct", "ci_high_pct", "p_adjusted",
                "significant", "ranking_is_meaningful"]:
        assert col in cat.columns, f"agg_category missing {col}"


@integration
def test_hypothesis_register_grades_every_claim(agg):
    reg = agg["agg_hypothesis_register"]
    assert len(reg) >= 9
    allowed = {"SUPPORTED", "NOT SUPPORTED", "BORDERLINE", "CONTRADICTED",
               "QUANTIFIED", "UNTESTABLE"}
    assert set(reg["verdict"]) <= allowed
    assert reg["reading"].str.len().min() > 40      # every verdict is explained


@integration
def test_register_does_not_claim_support_for_the_ring_hypotheses(agg):
    """The brief's fraud-pattern claims must not come back marked SUPPORTED."""
    reg = agg["agg_hypothesis_register"].set_index("hypothesis")
    for claim in reg.index:
        if "spikes" in claim or "categories have disproportionately" in claim:
            assert reg.loc[claim, "verdict"] == "NOT SUPPORTED"


@integration
def test_headline_kpis_agree_with_direct_computation(agg, facts):
    tx, cb, dim_users = facts
    head = agg["kpi_headline"].set_index("metric")
    assert head.loc["total_transaction_count", "value"] == len(tx)
    assert head.loc["total_transaction_amount", "value"] == pytest.approx(
        tx["amount_inr"].sum(), abs=0.01)
    assert head.loc["chargeback_to_transaction_ratio", "value"] == pytest.approx(
        K.chargeback_to_transaction_ratio(tx, cb).value, abs=1e-6)


@integration
def test_status_rates_sum_to_one_hundred(agg):
    head = agg["kpi_headline"].set_index("metric")
    total = sum(head.loc[f"{s}_transaction_rate", "value"]
                for s in ["success", "failed", "pending"])
    assert total == pytest.approx(100.0, abs=0.01)


@integration
def test_daily_aggregate_covers_every_day_in_the_window(agg):
    daily = agg["agg_daily"]
    assert len(daily) == 90        # Q1 2026
    assert (daily["transactions"] > 0).all()
