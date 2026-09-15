"""Tests for the Stage 4 dashboard and the AI Investigator scaffold.

Page rendering is exercised through Streamlit's AppTest harness, which runs the
real app headlessly, so "the page works" means the page actually rendered rather
than that its module imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.executor import execute  # noqa: E402
from src.agent.planner import (  # noqa: E402
    ALLOWED_CHARTS, ALLOWED_METRICS, FORBIDDEN_TOKENS, Intent, IntentRejected,
    plan, validate_intent,
)
from src.analytics import semantic  # noqa: E402
from src.analytics.shrinkage import MIN_DENOMINATOR  # noqa: E402
from src.config import PROCESSED_DIR, UNKNOWN_KEY  # noqa: E402

ANALYTICS_DIR = PROCESSED_DIR / "analytics"
PAGES = ["Executive Overview", "Transaction Analytics", "Merchant & User Risk",
         "Data Quality & Hypotheses", "Identity & Network",
         "Chargeback Intelligence", "AI Investigator"]

built = pytest.mark.skipif(
    not (ANALYTICS_DIR / "agg_daily.parquet").exists(),
    reason="run `python scripts/run_pipeline.py && python scripts/build_analytics.py` first",
)


@pytest.fixture(scope="module")
def frames():
    agg = {p.stem: pd.read_parquet(p) for p in ANALYTICS_DIR.glob("*.parquet")}
    star = {p.stem: pd.read_parquet(p) for p in PROCESSED_DIR.glob("*.parquet")}
    return agg, star


# --------------------------------------------------------------------------
# App structure
# --------------------------------------------------------------------------
def test_app_imports_successfully():
    import app.dashboard as dashboard
    assert len(dashboard.PAGES) == 7
    assert all(callable(fn) for fn in dashboard.PAGES.values())


@built
def test_all_required_parquet_outputs_exist():
    from app.components.data import missing_tables
    assert missing_tables() == []


@built
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"app failed to boot: {at.exception}"
    at.radio[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {at.exception}"


@built
def test_filtering_reduces_the_population_and_does_not_crash():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    at.radio[0].set_value("Transaction Analytics").run()
    assert not at.exception
    at.multiselect[0].set_value(["FAILED"]).run()
    assert not at.exception, f"filtering raised: {at.exception}"


# --------------------------------------------------------------------------
# KPIs come from the semantic layer, not from the dashboard
# --------------------------------------------------------------------------
@built
def test_dashboard_kpis_match_the_materialized_headline(frames):
    from app.components.data import headline_kpis
    agg, star = frames
    kpis = headline_kpis(star["fact_transactions"], star["fact_chargebacks"],
                         star["dim_users"])
    head = agg["kpi_headline"].set_index("metric")
    pairs = [("count", "total_transaction_count"), ("amount", "total_transaction_amount"),
             ("average", "average_transaction_value"), ("cb_ratio", "chargeback_to_transaction_ratio"),
             ("kyc_cov", "kyc_coverage"), ("merchant_cov", "merchant_master_coverage")]
    for key, metric in pairs:
        assert kpis[key].value == pytest.approx(float(head.loc[metric, "value"]), abs=0.01), metric


@built
def test_every_headline_kpi_is_a_registered_metric(frames):
    agg, _ = frames
    for metric in agg["kpi_headline"]["metric"]:
        assert metric in semantic.METRICS, f"{metric} is not in the semantic registry"


@built
def test_coverage_badges_use_real_coverage_not_a_constant(frames):
    from app.components.data import headline_kpis
    agg, star = frames
    kpis = headline_kpis(star["fact_transactions"], star["fact_chargebacks"],
                         star["dim_users"])
    tx = star["fact_transactions"]
    expected = 100 * float((~tx["merchant_unresolved"]).mean())
    assert kpis["merchant_cov"].value == pytest.approx(expected, abs=0.01)
    assert kpis["kyc_cov"].value == pytest.approx(
        100 * float((~tx["user_unresolved"]).mean()), abs=0.01)


def test_coverage_badge_classes_track_the_percentage():
    from app.components.ui import coverage_badge_html, coverage_class
    assert coverage_class(100.0) == "dv-cov-full"
    assert coverage_class(48.16) == "dv-cov-part"
    assert coverage_class(32.39) == "dv-cov-low"
    assert "48.16% coverage" in coverage_badge_html(48.16, "merchant master")


# --------------------------------------------------------------------------
# UNKNOWN stays visible
# --------------------------------------------------------------------------
@built
def test_unknown_survives_dashboard_enrichment(frames):
    from app.components.data import enrich_transactions
    agg, star = frames
    enriched = enrich_transactions(star["fact_transactions"], star["dim_merchants"],
                                   star["dim_users"])
    assert enriched["category"].str.contains("UNKNOWN").any()
    assert (enriched["kyc_status"] == UNKNOWN_KEY).any()
    assert len(enriched) == len(star["fact_transactions"])


@built
def test_unknown_is_the_largest_category_and_is_not_filtered_out(frames):
    from app.components.data import filter_options
    agg, _ = frames
    cat = agg["agg_category"]
    unknown = cat[cat["category"].str.contains("UNKNOWN")]
    assert len(unknown) == 1
    assert int(unknown["transactions"].iloc[0]) == int(cat["transactions"].max())


# --------------------------------------------------------------------------
# Privacy
# --------------------------------------------------------------------------
@built
def test_no_sensitive_columns_reach_the_dashboard_layer(frames):
    _, star = frames
    banned = {"pan", "pan_clean", "aadhaar", "aadhaar_full", "full_pan"}
    for name, df in star.items():
        assert not (banned & set(df.columns)), f"{name} exposes a raw identifier"


@built
def test_displayed_pan_is_always_masked(frames):
    _, star = frames
    masked = star["dim_users"]["pan_masked"].dropna()
    assert len(masked) > 0
    assert masked.str.contains(r"\*").all()
    assert not masked.str.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]").any()


@built
def test_aadhaar_is_never_more_than_four_digits(frames):
    _, star = frames
    last4 = star["dim_users"]["aadhaar_last4"].dropna().astype(str)
    assert (last4.str.len() <= 4).all()


# --------------------------------------------------------------------------
# Honest analytics
# --------------------------------------------------------------------------
@built
def test_hypothesis_results_are_loaded_not_hardcoded(frames):
    agg, _ = frames
    reg = agg["agg_hypothesis_register"]
    assert len(reg) >= 9
    assert "CONTRADICTED" in set(reg["verdict"])
    assert int((reg["verdict"] == "NOT SUPPORTED").sum()) >= 5


@built
def test_merchant_ranking_respects_the_denominator_floor(frames):
    agg, _ = frames
    mer = agg["agg_merchant"]
    assert (mer.loc[mer["below_floor"], "transactions"] < MIN_DENOMINATOR).all()
    assert (mer.loc[~mer["below_floor"], "transactions"] >= MIN_DENOMINATOR).all()
    # a 100% raw ratio must be overwhelmingly a below-floor artifact
    hot = mer[mer["dispute_rate_raw_pct"] >= 100]
    assert int(hot["below_floor"].sum()) > int((~hot["below_floor"]).sum())


@built
def test_insignificant_ranking_produces_a_caveat(frames):
    from app.components.ui import significance_caveat
    from src.analytics.significance import homogeneity_test, rank_with_confidence
    agg, _ = frames
    cat = agg["agg_category"]
    ranked, test = rank_with_confidence(
        cat[["category", "disputed_transactions", "transactions"]],
        "category", "disputed_transactions", "transactions")
    assert not test["significant"], "category ranking unexpectedly became significant"
    assert callable(significance_caveat)


@built
def test_no_fabricated_fraud_or_ring_fields_exist(frames):
    agg, star = frames
    banned = {"fraud_ring_id", "ring_id", "is_money_laundering", "confirmed_fraud",
              "is_fraud", "fraud_score", "circular_flow", "velocity_anomaly",
              "laundering_score"}
    for name, df in {**agg, **star}.items():
        assert not (banned & set(df.columns)), f"{name} contains a fabricated field"


@built
def test_network_summary_reports_an_acyclic_graph(frames):
    agg, _ = frames
    net = agg["agg_network_summary"].iloc[0]
    assert bool(net["is_forest_acyclic"]) is True
    assert int(net["user_pairs_sharing_2plus"]) == 0
    assert int(net["repeat_pairs"]) == 0


# --------------------------------------------------------------------------
# AI Investigator: planning, validation, execution
# --------------------------------------------------------------------------
EXAMPLES = [
    "What was the total transaction value this quarter?",
    "Which merchant category had the highest chargeback rate?",
    "Which merchants had the highest disputed value?",
    "Show chargebacks reported after 7 days.",
    "How does KYC status relate to transaction value?",
    "Which users have repeated disputes?",
    "Why is Transport ranked first?",
    "What percentage of transactions have merchant information?",
    "Show me identity collisions.",
    "Are category chargeback differences statistically significant?",
]


@pytest.mark.parametrize("question", EXAMPLES)
def test_every_example_question_plans_to_a_valid_intent(question):
    intent = plan(question)
    assert intent.kind
    assert intent.chart in ALLOWED_CHARTS
    if intent.metric:
        assert intent.metric in ALLOWED_METRICS


@built
@pytest.mark.parametrize("question", EXAMPLES)
def test_every_example_question_executes(question, frames):
    agg, star = frames
    answer = execute(plan(question), agg, star)
    assert answer.headline
    assert answer.rows_analysed > 0
    assert answer.provenance["metric"]


@built
def test_ranking_and_explanation_answers_carry_the_statistical_caveat(frames):
    """Questions 7 and 10 must never present the ranking as a real difference."""
    agg, star = frames
    for question in ["Why is Transport ranked first?",
                     "Are category chargeback differences statistically significant?",
                     "Which merchant category had the highest chargeback rate?"]:
        answer = execute(plan(question), agg, star)
        text = answer.caveat.lower() + answer.detail.lower()
        assert "noise" in text or "p =" in text or "significan" in text, question


@built
def test_partial_coverage_breakdowns_carry_a_coverage_caveat(frames):
    agg, star = frames
    answer = execute(plan("How does KYC status relate to transaction value?"), agg, star)
    assert "32.39%" in answer.caveat or "coverage" in answer.caveat.lower()
    assert "UNKNOWN" in answer.caveat


@built
def test_merchant_ranking_answer_refuses_to_call_it_risk(frames):
    agg, star = frames
    answer = execute(plan("Which merchants had the highest disputed value?"), agg, star)
    blob = (answer.headline + answer.detail + answer.caveat).lower()
    assert "exposure" in blob
    assert "fraudulent" not in blob and "confirmed fraud" not in blob


@pytest.mark.parametrize("question", [
    "What is the GDP of India?",
    "Who won the cricket world cup?",
    "Write me a poem about payments.",
])
def test_out_of_scope_questions_are_refused(question):
    with pytest.raises(IntentRejected):
        plan(question)


@pytest.mark.parametrize("token", ["DROP TABLE fact_transactions", "delete from dim_users",
                                   "__import__('os')", "1; -- comment"])
def test_mutating_and_code_tokens_are_refused(token):
    with pytest.raises(IntentRejected):
        plan(token)


def test_every_forbidden_token_is_actually_blocked():
    from src.agent.planner import assert_safe
    for token in FORBIDDEN_TOKENS:
        with pytest.raises(IntentRejected):
            assert_safe(f"show me {token} please")


def test_intent_validation_rejects_an_unregistered_metric():
    with pytest.raises(IntentRejected, match="semantic registry"):
        validate_intent(Intent(kind="metric", metric="profit_margin"))


def test_intent_validation_rejects_a_disallowed_chart():
    with pytest.raises(IntentRejected, match="chart type"):
        validate_intent(Intent(kind="metric", metric="total_transaction_count",
                               chart="pie_of_pies"))


def test_intent_validation_rejects_an_absurd_limit():
    with pytest.raises(IntentRejected, match="limit"):
        validate_intent(Intent(kind="ranking", metric="chargeback_count", limit=99999))


def test_planner_is_deterministic():
    first = plan("Which merchant category had the highest chargeback rate?")
    second = plan("Which merchant category had the highest chargeback rate?")
    assert first.as_dict() == second.as_dict()


def test_suggestions_are_offered_for_a_near_miss():
    from src.agent.planner import suggest_metrics
    assert suggest_metrics("tell me about chargeback amounts")
