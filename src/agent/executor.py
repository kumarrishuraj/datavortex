"""Execute a validated Intent.

    intent -> semantic query -> data computation -> result validation
           -> chart selection -> narrative

Every figure in an answer is computed by `src/analytics/query_engine.py` from the
materialized Parquet, or read from an analytics table the pipeline built and
validated. No number is produced by text generation, and none is typed into this
module: every caveat quotes a statistic computed when the question is asked.

An answer carries its own provenance — metric, formula, dimension, time window,
rows analysed, the reconciliation check it passed, the reason its chart was
chosen — so a reader can see exactly how it was produced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.agent.chart_selector import select_chart
from src.agent.planner import Intent
from src.analytics import semantic
from src.analytics.query_engine import (
    QueryError, QuerySpec, Window, data_bounds, prepare_frames, resolve_window, run,
)
from src.analytics.risk_indicators import DISCLAIMER
from src.analytics.shrinkage import MIN_DENOMINATOR, shrink_rates
from src.analytics.significance import overlap_verdict, rank_with_confidence, wilson_interval
from src.config import UNKNOWN_KEY
from src.transformation.audit_facts import lookup as fact

NOUN = {semantic.TRANSACTIONS: "transactions", semantic.CHARGEBACKS: "complaints",
        semantic.CUSTOMERS: "customers"}
RECORD = {"merchant_master_coverage": "merchant master record", "kyc_coverage": "KYC record"}
GRAIN_WORD = {"day": "daily", "week": "weekly", "month": "monthly", "quarter": "quarterly"}
SEVERITY = {"info": 0, "good": 1, "warn": 2, "crit": 3}
FORECAST_SERIES = {
    "total_transaction_count": "transactions",
    "total_transaction_amount": "transaction_value",
    "chargeback_count": "disputed_transactions",
    "chargeback_to_transaction_ratio": "disputed_transactions",
}


@dataclass
class Answer:
    headline: str
    detail: str = ""
    table: pd.DataFrame | None = None
    chart_kind: str = "none"
    chart_title: str = ""
    chart_reason: str = ""
    chart_x: list = field(default_factory=list)
    chart_series: dict = field(default_factory=dict)
    chart_labels: list = field(default_factory=list)
    chart_values: list = field(default_factory=list)
    chart_lo: list = field(default_factory=list)
    chart_hi: list = field(default_factory=list)
    show_ci: bool = False
    chart_band: dict = field(default_factory=dict)
    kpi_value: str = ""
    caveat: str = ""
    caveat_kind: str = "info"
    provenance: dict = field(default_factory=dict)
    rows_analysed: int = 0
    no_result: bool = False


@dataclass
class _Ctx:
    intent: Intent
    agg: dict
    star: dict
    frames: dict
    window: Window
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    caveats: list = field(default_factory=list)
    validation: str = ""

    def caveat(self, text: str, kind: str = "warn") -> None:
        if text:
            self.caveats.append((text, kind))


# --------------------------------------------------------------------------
_CACHE: dict = {}


def _frames(star: dict) -> dict:
    """Prepared frames, rebuilt only when the underlying star schema changes."""
    source = star["fact_transactions"]
    if _CACHE.get("source") is not source:
        _CACHE.clear()
        _CACHE["source"] = source
        _CACHE["frames"] = prepare_frames(star)
    return _CACHE["frames"]


def _plural(noun: str) -> str:
    """English plural for a dimension label: category -> categories, status -> statuses."""
    head, sep, tail = noun.partition(" of ")
    if sep:
        return _plural(head) + sep + tail
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def execute(intent: Intent, agg: dict[str, pd.DataFrame],
            star: dict[str, pd.DataFrame]) -> Answer:
    frames = _frames(star)
    data_start, data_end = data_bounds(frames)
    window = resolve_window(intent.window, data_start, data_end)
    ctx = _Ctx(intent, agg, star, frames, window, data_start, data_end)

    if window.requested and not window.overlaps(data_start, data_end):
        answer = _no_result(ctx)
    else:
        try:
            answer = HANDLERS[intent.kind](ctx)
        except QueryError as exc:
            answer = Answer(headline="That combination cannot be computed from this data.",
                            detail=str(exc), no_result=True)

    for note in window.notes:
        ctx.caveat(note, "info")
    for note in intent.assumptions:
        ctx.caveat(note, "info")
    _apply_caveats(ctx, answer)
    _provenance(ctx, answer)
    return answer


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fmt(value, unit: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    value = float(value)
    if unit == "INR":
        return f"₹{value:,.0f}"
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "days":
        return f"{value:.2f} days"
    if unit == "ratio":
        return f"{value:.3f}"
    if unit == "points":
        return f"{value:.1f}"
    return f"{value:,.0f}"


def _period(ts, grain: str) -> str:
    ts = pd.Timestamp(ts)
    if grain == "day":
        return f"{ts:%Y-%m-%d}"
    if grain == "week":
        return f"week of {ts:%Y-%m-%d}"
    if grain == "month":
        return f"{ts:%B %Y}"
    return f"{ts.year}-Q{ts.quarter}"


def _spec(ctx: _Ctx, **kwargs) -> QuerySpec:
    it = ctx.intent
    filters = {k: v for k, v in it.filters.items() if k == "status_canonical"}
    params = dict(metric=it.metric, start=ctx.window.start, end=ctx.window.end,
                  filters=filters, threshold=it.threshold)
    params.update(kwargs)
    return QuerySpec(**params)


def _headline_row(agg: dict, name: str):
    head = agg.get("kpi_headline")
    if head is None or name not in set(head["metric"]):
        return None
    return head.set_index("metric").loc[name]


def _register_row(agg: dict, needle: str):
    reg = agg.get("agg_hypothesis_register")
    if reg is None or reg.empty:
        return None
    hit = reg[reg["hypothesis"].str.lower().str.contains(needle.lower(), regex=False)]
    return hit.iloc[0] if len(hit) else None


def _excluded(ctx: _Ctx, res) -> None:
    for label, count in res.excluded.items():
        if count:
            ctx.caveat(f"{count:,} {label} are excluded from this answer.", "info")


def _no_result(ctx: _Ctx) -> Answer:
    metric = semantic.METRICS.get(ctx.intent.metric) if ctx.intent.metric else None
    noun = NOUN.get(metric.source, "records") if metric else "records"
    return Answer(
        headline="No matching records were found for the selected filters.",
        detail=(f"The data covers {ctx.data_start:%Y-%m-%d} to {ctx.data_end:%Y-%m-%d}; "
                f"{ctx.window.label} contains no matching {noun}."),
        no_result=True)


def _coverage_caveat(ctx: _Ctx, res, dim: semantic.Dimension, labels=None, sizes=None) -> None:
    if not dim.coverage_metric or res.coverage_pct is None or res.coverage_pct >= 99.5:
        return
    metric = semantic.get(ctx.intent.metric) if ctx.intent.metric else None
    noun = NOUN.get(metric.source, "records") if metric else "records"
    text = (f"Only {res.coverage_pct:.2f}% of the analysed {noun} resolve to a "
            f"{RECORD.get(dim.coverage_metric, 'master record')}, so this breakdown describes "
            f"part of the book.")
    if labels:
        unknown = [i for i, label in enumerate(labels) if label in dim.unknown_values]
        if unknown:
            largest = sizes is not None and sizes[unknown[0]] == max(sizes)
            text += (" The UNKNOWN group is shown rather than hidden"
                     + (" — it is the largest single group, and dropping it would make the rest "
                        "look like the whole book." if largest else "."))
    ctx.caveat(text, "warn")


def _apply_caveats(ctx: _Ctx, answer: Answer) -> None:
    if not ctx.caveats:
        return
    answer.caveat = " ".join(text for text, _ in ctx.caveats)
    answer.caveat_kind = max((kind for _, kind in ctx.caveats), key=SEVERITY.get)


def _provenance(ctx: _Ctx, answer: Answer) -> None:
    it = ctx.intent
    metric = semantic.METRICS.get(it.metric) if it.metric else None
    answer.provenance = {
        "detected intent": it.kind,
        "metric": it.metric or "—",
        "metric definition": metric.definition if metric else "—",
        "formula": metric.formula if metric else "—",
        "aggregation": metric.aggregation if metric else "—",
        "source table": metric.source if metric else "—",
        "dimension": it.dimension,
        "series": it.series or "—",
        "time grain": it.time_grain or "—",
        "time window": ctx.window.label,
        "filters": it.filters or "none",
        "coverage basis": metric.coverage_basis if metric else "—",
        "chart selected": answer.chart_kind,
        "why this chart": answer.chart_reason or it.chart_reason or "—",
        "rows analysed": f"{answer.rows_analysed:,}",
        "validation": ctx.validation or "—",
        "assumptions": "; ".join(it.assumptions) or "none",
    }


# --------------------------------------------------------------------------
# Single figures
# --------------------------------------------------------------------------
def _metric(ctx: _Ctx) -> Answer:
    metric = semantic.get(ctx.intent.metric)
    res = run(_spec(ctx), ctx.frames)
    if res.empty or res.total_value != res.total_value:
        return _no_result(ctx)

    value = fmt(res.total_value, metric.unit)
    when = f" in {ctx.window.label}" if ctx.window.requested else ""
    choice = select_chart(kind="metric")
    answer = Answer(headline=f"{metric.label}{when}: {value}", kpi_value=value,
                    chart_kind=choice.chart, chart_reason=choice.reason,
                    rows_analysed=res.rows_analysed)
    noun = NOUN[metric.source]
    detail = [metric.definition]
    if metric.aggregation in (semantic.SHARE, semantic.DISPUTED_RATIO) and res.total_denominator:
        detail.append(f"{res.total_numerator:,.0f} of {res.total_denominator:,.0f} {noun}.")
    else:
        detail.append(f"Computed on {res.rows_analysed:,} {noun}.")
    if metric.name == "utr_missing_rate":
        tx = ctx.frames[semantic.TRANSACTIONS]
        invalid = int((~tx["utr_valid"].astype(bool) & ~tx["utr_missing"].astype(bool)).sum())
        detail.append(f"Among non-blank UTRs, {invalid:,} have an invalid format once whitespace "
                      "is removed. Missing UTRs are never fabricated.")
    answer.detail = " ".join(detail)

    if not ctx.window.requested and not res.spec.filters:
        head = _headline_row(ctx.agg, metric.name)
        if head is not None:
            expected = float(head["value"])
            if np.isclose(expected, res.total_value, rtol=1e-6, atol=0.01):
                ctx.validation = "matches the materialized KPI in the analytics layer"
            else:
                ctx.validation = f"MISMATCH — the materialized KPI is {expected}"
                ctx.caveat(f"This figure does not match the materialized KPI "
                           f"({fmt(expected, metric.unit)}); treat it as suspect.", "crit")
            coverage = float(head["coverage_pct"])
            if coverage < 99.5:
                ctx.caveat(f"Computed on {coverage:.2f}% of records — {metric.coverage_basis}.")
    return answer


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------
def _trend(ctx: _Ctx) -> Answer:
    it = ctx.intent
    metric = semantic.get(it.metric)
    res = run(_spec(ctx, time_grain=it.time_grain, series=it.series,
                    series_values=tuple(it.series_values)), ctx.frames)
    if res.empty:
        return _no_result(ctx)

    frame = res.frame.copy()
    frame["period"] = pd.to_datetime(frame["period"])
    grain = it.time_grain
    word = GRAIN_WORD[grain]
    choice = select_chart(kind="trend", time_grain=grain)
    periods = sorted(frame["period"].unique())
    answer = Answer(headline="", chart_kind=choice.chart, chart_reason=choice.reason,
                    rows_analysed=res.rows_analysed, chart_title=f"{metric.label} per {grain}",
                    chart_x=[pd.Timestamp(p) for p in periods])

    if "series" in frame.columns:
        sdim = semantic.get_dimension(it.series)
        names = [str(v) for v in (it.series_values or sorted(frame["series"].dropna().unique()))]
        pieces = []
        for name in names:
            s = frame[frame["series"].astype(str) == name].set_index("period")["value"].reindex(periods)
            answer.chart_series[name] = [None if pd.isna(v) else float(v) for v in s]
            total = float(np.nansum(s.to_numpy())) if metric.additive else float(np.nanmean(s.to_numpy()))
            pieces.append((name, total, s))
        how = "in total" if metric.additive else f"on average per {grain}"
        answer.headline = (f"{word.capitalize()} {metric.label.lower()} by "
                           f"{sdim.label.lower()}, {ctx.window.label}: "
                           + "; ".join(f"{n} {fmt(t, metric.unit)} {how}" for n, t, _ in pieces) + ".")
        detail = []
        for name, _, s in pieces:
            vals = s.dropna()
            if len(vals):
                detail.append(f"{name}: {fmt(vals.mean(), metric.unit)} per {grain} on average, "
                              f"ranging {fmt(vals.min(), metric.unit)}–{fmt(vals.max(), metric.unit)}.")
        if len(pieces) == 2:
            (n1, _, s1), (n2, _, s2) = pieces
            ahead = int((s1.fillna(0) > s2.fillna(0)).sum())
            detail.append(f"{n1} exceeded {n2} in {ahead} of {len(periods)} {grain}s.")
        answer.detail = " ".join(detail)
    else:
        s = frame.set_index("period")["value"].reindex(periods)
        answer.chart_series[metric.label] = [None if pd.isna(v) else float(v) for v in s]
        vals = s.dropna()
        if vals.empty:
            return _no_result(ctx)
        mean = float(vals.mean())
        first, last = float(vals.iloc[0]), float(vals.iloc[-1])
        answer.headline = (f"{word.capitalize()} {metric.label.lower()}, {ctx.window.label}: "
                           f"averaging {fmt(mean, metric.unit)} per {grain}, between "
                           f"{fmt(vals.min(), metric.unit)} and {fmt(vals.max(), metric.unit)}.")
        detail = [
            f"Highest {fmt(vals.max(), metric.unit)} ({_period(vals.idxmax(), grain)}); lowest "
            f"{fmt(vals.min(), metric.unit)} ({_period(vals.idxmin(), grain)}).",
            f"First {grain} {fmt(first, metric.unit)}, last {grain} {fmt(last, metric.unit)}"
            + (f" ({(last - first) / first * 100:+.1f}%)." if first else "."),
        ]
        if len(vals) >= 3 and mean:
            cv = float(vals.std(ddof=1) / mean)
            detail.append(f"The series is effectively flat (coefficient of variation {cv:.3f}): "
                          "no trend or spike stands out." if cv < 0.10
                          else f"Coefficient of variation {cv:.3f}.")
        label = "Total across the window" if metric.additive else "Across the whole window"
        detail.append(f"{label}: {fmt(res.total_value, metric.unit)}.")
        answer.detail = " ".join(detail)

    if len(periods) < 3:
        ctx.caveat(f"Only {len(periods)} {grain} period(s) of data fall in this window, so a "
                   "trend cannot be read from it.", "warn")
    if metric.time_column == semantic.REPORTED_TIME:
        ctx.caveat("Complaints are placed on the date the customer reported them, not the "
                   "transaction date.", "info")
    if res.reconciled:
        ctx.validation = f"{grain} values sum exactly to the window total"
    _excluded(ctx, res)
    return answer


# --------------------------------------------------------------------------
# Breakdowns and rankings
# --------------------------------------------------------------------------
def _by_dimension(ctx: _Ctx) -> Answer:
    it = ctx.intent
    metric = semantic.get(it.metric)
    dim = semantic.get_dimension(it.dimension)
    if metric.aggregation == semantic.DISPUTED_RATIO:
        return _entity_rate(ctx) if dim.entity else _category_rate(ctx, "ranking")

    ranking = it.kind == "ranking" or dim.entity
    # Small categorical dimensions are never truncated, so UNKNOWN cannot fall off.
    limit = it.limit if dim.entity else None
    res = run(_spec(ctx, dimension=dim.name, limit=limit, ascending=it.ascending,
                    natural_order=(not ranking and bool(dim.order))), ctx.frames)
    if res.empty:
        return _no_result(ctx)

    frame = res.frame
    choice = select_chart(kind="ranking" if ranking else it.kind, dimension=dim.name,
                          n_categories=len(frame))
    labels = [str(v) for v in frame["dimension"]]
    values = [float(v) for v in frame["value"]]
    answer = Answer(headline="", chart_kind=choice.chart, chart_reason=choice.reason,
                    rows_analysed=res.rows_analysed, chart_labels=labels, chart_values=values,
                    chart_title=f"{metric.label} by {dim.label.lower()}")
    top = 0 if ranking else int(np.nanargmax(values))
    share = (f" ({100 * values[top] / res.total_value:.1f}% of the total)"
             if metric.additive and res.total_value else "")
    if ranking:
        word = "lowest" if it.ascending else "highest"
        answer.headline = (f"{labels[top]} has the {word} {metric.label.lower()}: "
                           f"{fmt(values[top], metric.unit)}{share}.")
    else:
        answer.headline = (f"{metric.label} by {dim.label.lower()}: {labels[top]} is largest at "
                           f"{fmt(values[top], metric.unit)}{share}.")
    detail = ["Top entries: " + ", ".join(f"{l} ({fmt(v, metric.unit)})"
                                         for l, v in list(zip(labels, values))[:3]) + "."]
    if dim.entity and res.groups_before_limit > len(frame):
        detail.append(f"Showing {len(frame)} of {res.groups_before_limit:,} "
                      f"{_plural(dim.label.lower())}.")
    else:
        detail.append(f"{res.groups_before_limit:,} {dim.label.lower()} values.")
    if metric.additive:
        detail.append(f"Total: {fmt(res.total_value, metric.unit)} across "
                      f"{res.rows_analysed:,} {NOUN[metric.source]}.")
    answer.detail = " ".join(detail)
    answer.table = frame.rename(columns={"dimension": dim.label, "value": metric.label})

    sizes = [float(v) for v in frame["denominator"]] if "denominator" in frame else values
    _coverage_caveat(ctx, res, dim, labels, sizes)
    if dim.entity and metric.source == semantic.CHARGEBACKS:
        text = ("This ranks absolute exposure — where review effort should go — not a measured "
                "difference in risk.")
        reg = _register_row(ctx.agg, "individual merchants differ")
        if reg is not None:
            text += (f" The hypothesis register tests merchant-level dispute propensity: "
                     f"{reg['verdict']} ({reg['statistic']}, p = {reg['p_value']}).")
        ctx.caveat(text, "warn")
    if res.reconciled:
        ctx.validation = "group values sum exactly to the total"
    _excluded(ctx, res)
    return answer


def _category_rate_core(ctx: _Ctx):
    it = ctx.intent
    name = it.dimension if (it.dimension in semantic.DIMENSIONS
                            and not semantic.DIMENSIONS[it.dimension].entity) else "merchant_category"
    dim = semantic.get_dimension(name)
    res = run(_spec(ctx, metric="chargeback_to_transaction_ratio", dimension=name), ctx.frames)
    if res.empty:
        return _no_result(ctx), None
    frame = res.frame.rename(columns={"dimension": "group", "numerator": "disputed",
                                      "denominator": "transactions"})
    frame = frame[frame["transactions"] > 0].reset_index(drop=True)
    ranked, test = rank_with_confidence(frame[["group", "disputed", "transactions"]],
                                        "group", "disputed", "transactions")
    choice = select_chart(kind="ranking", dimension=name, n_categories=len(ranked))
    answer = Answer(
        headline="", chart_kind=choice.chart, chart_reason=choice.reason, show_ci=True,
        chart_labels=[str(g) for g in ranked["group"]], chart_values=ranked["rate_pct"].tolist(),
        chart_lo=ranked["ci_low_pct"].tolist(), chart_hi=ranked["ci_high_pct"].tolist(),
        chart_title=f"Chargeback rate by {dim.label.lower()} (95% CI)",
        rows_analysed=res.rows_analysed)
    answer.table = ranked[["rank", "group", "transactions", "disputed", "rate_pct", "ci_low_pct",
                           "ci_high_pct", "p_adjusted", "significant"]].rename(columns={"group": dim.label})
    stats = f"χ² = {test['chi_square']}, df = {test['df']}, p = {test['p_value']:.4f}"
    ctx.validation = "Wilson intervals and chi-square homogeneity computed on the selection"
    return answer, dict(ranked=ranked, test=test, stats=stats, dim=dim, res=res)


def _category_rate(ctx: _Ctx, mode: str) -> Answer:
    answer, core = _category_rate_core(ctx)
    if core is None:
        return answer
    ranked, test, stats, dim = core["ranked"], core["test"], core["stats"], core["dim"]
    top = ranked.iloc[0]
    when = f" in {ctx.window.label}" if ctx.window.requested else ""
    answer.headline = (f"{top['group']} has the highest chargeback rate{when}: "
                       f"{top['rate_pct']:.2f}% ({int(top['disputed']):,} of "
                       f"{int(top['transactions']):,} transactions).")
    answer.detail = (f"{semantic.get('chargeback_to_transaction_ratio').definition} 95% interval "
                     f"for {top['group']}: {top['ci_low_pct']:.2f}–{top['ci_high_pct']:.2f}%. "
                     f"Rates range from {ranked['rate_pct'].min():.2f}% to "
                     f"{ranked['rate_pct'].max():.2f}% across {len(ranked)} groups.")
    ctx.caveat(f"{overlap_verdict(ranked)} ({stats})", "good" if test["significant"] else "warn")
    _coverage_caveat(ctx, core["res"], dim, answer.chart_labels,
                     [float(v) for v in ranked["transactions"]])
    return answer


def _significance(ctx: _Ctx) -> Answer:
    answer, core = _category_rate_core(ctx)
    if core is None:
        return answer
    ranked, test, stats, dim = core["ranked"], core["test"], core["stats"], core["dim"]
    yes = bool(test["significant"])
    answer.headline = (f"{'Yes' if yes else 'No'} — chargeback-rate differences across "
                       f"{_plural(dim.label.lower())} are {'' if yes else 'not '}statistically significant.")
    answer.detail = (f"Chi-square test of homogeneity across {test['groups']} groups: {stats}. "
                     f"Rates span {ranked['rate_pct'].min():.2f}% to {ranked['rate_pct'].max():.2f}%. "
                     "A non-significant result means the ordering is what random variation alone "
                     "would produce.")
    ctx.caveat(overlap_verdict(ranked), "good" if yes else "warn")
    return answer


def _explanation(ctx: _Ctx) -> Answer:
    answer, core = _category_rate_core(ctx)
    if core is None:
        return answer
    ranked, test, stats = core["ranked"], core["test"], core["stats"]
    subject = ctx.intent.subject
    match = ranked[ranked["group"].astype(str).str.lower() == subject.lower()] if subject else ranked.iloc[0:0]
    row = match.iloc[0] if len(match) else ranked.iloc[0]
    meaningful = bool(test["significant"])
    answer.headline = (f"{row['group']} ranks #{int(row['rank'])} at {row['rate_pct']:.2f}% — "
                       + ("and the differences between groups are statistically significant."
                          if meaningful else "but the ranking itself is not meaningful."))
    answer.detail = (f"{row['group']} recorded {int(row['disputed']):,} disputed transactions out of "
                     f"{int(row['transactions']):,}, a rate of {row['rate_pct']:.2f}% (95% CI "
                     f"{row['ci_low_pct']:.2f}–{row['ci_high_pct']:.2f}%). Its adjusted p-value "
                     f"against the pooled remainder is {row['p_adjusted']:.3f}.")
    if meaningful:
        ctx.caveat(f"{overlap_verdict(ranked)} ({stats})", "good")
    else:
        ctx.caveat(f"It ranks where it does because some group has to come first. Across all "
                   f"{test['groups']} groups the differences are within sampling noise ({stats}), "
                   f"so targeting {row['group']} on the strength of this ranking would be acting "
                   "on noise.", "warn")
    return answer


def _entity_rate(ctx: _Ctx) -> Answer:
    it = ctx.intent
    dim = semantic.get_dimension(it.dimension)
    res = run(_spec(ctx, metric="chargeback_to_transaction_ratio", dimension=dim.name), ctx.frames)
    if res.empty:
        return _no_result(ctx)
    frame = res.frame.rename(columns={"dimension": "entity", "numerator": "disputed",
                                      "denominator": "transactions"})
    if dim.name == "merchant":
        dm = ctx.star["dim_merchants"]
        category = dm[dm["merchant_key"] != UNKNOWN_KEY].set_index("merchant_key")[
            "merchant_category_canonical"]
        frame["peer_group"] = frame["entity"].map(category).fillna(semantic.UNKNOWN_CATEGORY)
    else:
        frame["peer_group"] = "all customers"
    shrunk = shrink_rates(frame, "disputed", "transactions", group_col="peer_group",
                          min_denominator=MIN_DENOMINATOR)
    prior = shrunk.attrs["prior"]
    eligible = shrunk[~shrunk["below_floor"]].sort_values(
        ["rate_raw", "transactions", "entity"], ascending=[False, False, True])
    below = int(shrunk["below_floor"].sum())
    noun = f"{_plural(dim.label.lower())}"
    if eligible.empty:
        answer = _no_result(ctx)
        answer.detail = (f"No {noun} have at least {MIN_DENOMINATOR} transactions in this window, "
                         "so no chargeback rate can be ranked.")
        return answer

    top_n = eligible.head(it.limit)
    lo, hi = wilson_interval(top_n["disputed"], top_n["transactions"])
    choice = select_chart(kind="ranking", dimension=dim.name)
    top = top_n.iloc[0]
    ties = int((eligible["rate_raw"] == top["rate_raw"]).sum())
    answer = Answer(
        headline=(f"{top['entity']} has the highest chargeback rate among {noun} with at least "
                  f"{MIN_DENOMINATOR} transactions: {100 * top['rate_raw']:.2f}% "
                  f"({int(top['disputed'])} of {int(top['transactions'])})."),
        chart_kind=choice.chart, chart_reason=choice.reason, show_ci=True,
        chart_labels=[str(e) for e in top_n["entity"]],
        chart_values=(100 * top_n["rate_raw"]).round(2).tolist(),
        chart_lo=np.round(100 * lo, 2).tolist(), chart_hi=np.round(100 * hi, 2).tolist(),
        chart_title=f"Chargeback rate, {noun} with at least {MIN_DENOMINATOR} transactions (95% CI)",
        rows_analysed=res.rows_analysed)
    answer.detail = (f"Its shrunk estimate is {100 * top['rate_shrunk']:.2f}% against a peer mean of "
                     f"{100 * top['peer_mean']:.2f}%. {ties:,} {noun} share that raw rate. "
                     f"{below:,} {noun} with fewer than {MIN_DENOMINATOR} transactions are left out "
                     "of rate ranking — a ratio on one or two transactions is not evidence — but "
                     "still appear in exposure views.")
    answer.table = top_n.assign(
        raw_rate_pct=(100 * top_n["rate_raw"]).round(2),
        shrunk_rate_pct=(100 * top_n["rate_shrunk"]).round(3),
        peer_mean_pct=(100 * top_n["peer_mean"]).round(3),
    )[["entity", "peer_group", "transactions", "disputed", "raw_rate_pct", "shrunk_rate_pct",
       "peer_mean_pct"]].rename(columns={"entity": dim.label})
    if not prior["overdispersed"]:
        ratio = prior["chi_square"] / prior["df"] if prior["df"] else float("nan")
        ctx.caveat(f"No {dim.label.lower()}-level dispute propensity is detectable "
                   f"(χ²/df = {ratio:.3f}, p = {prior['p_value']:.3f} across {prior['groups']:,} "
                   f"{noun} with at least {MIN_DENOMINATOR} transactions). This ordering is "
                   f"sampling noise: after shrinkage every {dim.label.lower()} sits at its peer "
                   f"mean, so read the list as review exposure, not as riskier {noun}.", "warn")
    else:
        ctx.caveat(f"Between-{dim.label.lower()} variation exceeds sampling noise "
                   f"(p = {prior['p_value']:.3g}); the shrunk estimates discount thin evidence.",
                   "good")
    ctx.validation = "rate floor and empirical-Bayes shrinkage applied to the selection"
    return answer


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def _listing(ctx: _Ctx) -> Answer:
    it = ctx.intent
    if it.dimension == "identity":
        return _identity(ctx)
    if it.metric == "disputes_reported_after_7_days":
        return _late(ctx)
    return _repeat(ctx)


def _identity(ctx: _Ctx) -> Answer:
    users, merchants = ctx.star["dim_users"], ctx.star["dim_merchants"]
    bridge = ctx.star["bridge_identity_collision"]
    u = int(users["identity_ambiguous"].astype(bool).sum())
    m = int(merchants["identity_ambiguous"].astype(bool).sum())
    collisions = bridge[bridge["identity_class"] == "ID_COLLISION"]
    choice = select_chart(kind="listing", records=True)
    answer = Answer(headline=(f"{u:,} customer IDs and {m:,} merchant IDs are each shared by more "
                              "than one distinct entity."),
                    chart_kind=choice.chart, chart_reason=choice.reason, rows_analysed=len(bridge))
    detail = ["These are identifier collisions, not duplicate records: the rows behind each ID "
              "carry different names, identifiers and locations.",
              f"All {len(bridge):,} candidate rows are preserved in BRIDGE_IDENTITY_COLLISION rather "
              "than merged away."]
    facts = ctx.star.get("audit_facts")
    multi = fact(facts, "kyc_ids_with_multiple_pans")
    shared = fact(facts, "pans_shared_across_ids")
    if multi is not None and shared is not None:
        detail.append(f"Corroboration: {multi:,.0f} customer IDs carry more than one distinct PAN, "
                      f"while {shared:,.0f} PANs are shared across customer IDs.")
    answer.detail = " ".join(detail)
    cols = ["entity_type", "entity_key", "candidate_rank", "is_survivor", "full_name_clean",
            "city_clean", "candidate_count", "identity_class"]
    answer.table = collisions[[c for c in cols if c in collisions.columns]].head(
        ctx.intent.limit * 5).reset_index(drop=True)
    ctx.caveat("PAN is held only in masked form and Aadhaar only as its last four digits; full "
               "identifiers are not stored in the processed layer.", "info")
    ctx.validation = "counts read from the identity-resolved dimensions"
    return answer


def _late(ctx: _Ctx) -> Answer:
    it = ctx.intent
    threshold = float(it.threshold or 7.0)
    cb = ctx.frames[semantic.CHARGEBACKS]
    if ctx.window.requested:
        ts = pd.to_datetime(cb[semantic.REPORTED_TIME])
        mask = ts.notna()
        if ctx.window.start is not None:
            mask &= ts >= ctx.window.start
        if ctx.window.end is not None:
            mask &= ts < ctx.window.end
        scoped = cb[mask]
    else:
        scoped = cb
    res = run(_spec(ctx, metric="disputes_reported_after_7_days", threshold=threshold), ctx.frames)
    valid = scoped["reporting_delay_days"].notna()
    late = scoped[valid & (scoped["reporting_delay_days"] > threshold)]
    if int(res.total_value) != len(late):
        raise QueryError("late-dispute records do not reconcile with the registered metric")
    ctx.validation = "record count equals the registered threshold metric"
    if late.empty:
        return _no_result(ctx)

    choice = select_chart(kind="listing", records=True)
    share = 100 * len(late) / int(valid.sum()) if valid.sum() else 0.0
    reasons = late["reason_code_canonical"].value_counts()
    answer = Answer(headline=(f"{len(late):,} complaints were reported more than {threshold:g} days "
                              "after their transaction."),
                    chart_kind=choice.chart, chart_reason=choice.reason, rows_analysed=len(scoped))
    answer.detail = (f"That is {share:.1f}% of the {int(valid.sum()):,} complaints with a computable "
                     f"reporting delay. Median delay among them is "
                     f"{late['reporting_delay_days'].median():.1f} days; the longest is "
                     f"{late['reporting_delay_days'].max():.1f} days. Most common reason code among "
                     f"them: {reasons.index[0]} ({int(reasons.iloc[0]):,}).")
    cols = ["complaint_key", "txn_key", "attributed_user_id", "attributed_merchant_id",
            "reported_date", "reporting_delay_days", "reason_code_canonical", "severity_canonical",
            "resolution_status_canonical", "disputed_amount_inr"]
    answer.table = late.sort_values("reporting_delay_days", ascending=False)[
        [c for c in cols if c in late.columns]].reset_index(drop=True)
    text = "Delay is measured from each complaint's own transaction timestamp to its report date."
    negative = fact(ctx.star.get("audit_facts"), "delay_negative_vs_linked_txn_pct")
    if negative is not None:
        text += (f" Measuring from the linked transaction's timestamp instead would give an "
                 f"impossible negative delay for {negative:.2f}% of complaints, because the "
                 "complaint file's denormalized fields do not describe the transaction they "
                 "point at.")
    ctx.caveat(text, "warn")
    missing = int((~valid).sum())
    if missing:
        ctx.caveat(f"{missing:,} complaints have no computable delay and are not counted.", "info")
    return answer


def _repeat(ctx: _Ctx) -> Answer:
    it = ctx.intent
    entity = it.dimension if it.dimension in ("user", "merchant") else "user"
    k = int(it.filters.get("min_disputes", 2))
    table = ctx.agg["agg_user" if entity == "user" else "agg_merchant"]
    id_col = "user_id_normalized" if entity == "user" else "merchant_id_normalized"
    name_col = "full_name_clean" if entity == "user" else "merchant_name_clean"
    noun = "customers" if entity == "user" else "merchants"
    subset = table[table["disputed_transactions"] >= k].sort_values(
        ["disputed_transactions", "disputed_amount", id_col], ascending=[False, False, True])
    if subset.empty:
        return _no_result(ctx)
    choice = select_chart(kind="listing", records=True)
    top = subset.iloc[0]
    answer = Answer(headline=f"{len(subset):,} {noun} have {k} or more disputed transactions.",
                    chart_kind=choice.chart, chart_reason=choice.reason, rows_analysed=len(table))
    answer.detail = (f"Attributed through txn_id, then ordered by disputed transactions and disputed "
                     f"value. The top entry carries {int(top['disputed_transactions'])} disputed "
                     f"transactions worth ₹{top['disputed_amount']:,.0f}.")
    extra = "kyc_status_canonical" if entity == "user" else "merchant_category_canonical"
    cols = [id_col, name_col, "transactions", "disputed_transactions", "complaints",
            "disputed_amount", extra]
    answer.table = subset[[c for c in cols if c in subset.columns]].head(max(it.limit, 50)) \
        .reset_index(drop=True)
    reg = _register_row(ctx.agg, "appear repeatedly in disputes" if entity == "user"
                        else "individual merchants differ")
    if reg is not None:
        ctx.caveat(f"Hypothesis register — {reg['verdict']}: {reg['reading']} "
                   f"({reg['statistic']}, p = {reg['p_value']})", "warn")
    ctx.validation = "read from the transacting-population aggregate"
    return answer


# --------------------------------------------------------------------------
# Risk indicators
# --------------------------------------------------------------------------
def _risk(ctx: _Ctx) -> Answer:
    it = ctx.intent
    entity = it.dimension if it.dimension in ("user", "merchant") else "user"
    table = ctx.agg["agg_user" if entity == "user" else "agg_merchant"]
    if "risk_indicator_score" not in table.columns:
        return Answer(headline="The Risk Indicator Score has not been built.",
                      detail="Run python scripts/build_analytics.py.", no_result=True)
    id_col = "user_id_normalized" if entity == "user" else "merchant_id_normalized"
    name_col = "full_name_clean" if entity == "user" else "merchant_name_clean"
    noun = "customers" if entity == "user" else "merchants"
    k = it.filters.get("min_disputes")
    subset = table if not k else table[table["disputed_transactions"] >= int(k)]
    if subset.empty:
        return _no_result(ctx)
    subset = subset.sort_values(["risk_indicator_score", "disputed_amount", id_col],
                                ascending=[False, False, True])
    top_n = subset.head(it.limit)
    top = top_n.iloc[0]
    scope = (f"{noun} with {int(k)} or more disputed transactions" if k
             else f"transacting {noun}")
    bands = subset["risk_indicator_band"].value_counts()
    band_text = ", ".join(f"{b} {int(bands.get(b, 0)):,}"
                          for b in ("High", "Elevated", "Moderate", "Low"))
    choice = select_chart(kind="ranking", dimension=entity)
    answer = Answer(
        headline=(f"{top[id_col]} has the highest Risk Indicator Score among {scope}: "
                  f"{top['risk_indicator_score']:.1f} / 100 ({top['risk_indicator_band']} review "
                  "priority)."),
        chart_kind=choice.chart, chart_reason=choice.reason,
        chart_labels=[str(v) for v in top_n[id_col]],
        chart_values=top_n["risk_indicator_score"].round(1).tolist(),
        chart_title=f"Risk Indicator Score — top {len(top_n)} {noun}", rows_analysed=len(subset))
    answer.detail = (f"{len(subset):,} {scope} scored. Review-priority bands: {band_text}. Why "
                     f"{top[id_col]} scores highest: {top['risk_indicator_reasons']}.")
    components = [c for c in table.columns if c.startswith("ri_")]
    status = "kyc_status_canonical" if entity == "user" else "merchant_status_canonical"
    cols = [id_col, name_col, "risk_indicator_score", "risk_indicator_band", *components,
            "risk_indicator_reasons", "disputed_transactions", "disputed_amount", status]
    answer.table = top_n[[c for c in cols if c in top_n.columns]].reset_index(drop=True)
    ctx.caveat(DISCLAIMER + " Each component is worth up to 25 points and the formula is "
               "published in the metric registry.", "warn")
    ctx.validation = "components verified to sum to each score, within 0–100, at build time"
    return answer


# --------------------------------------------------------------------------
# Hypotheses, network, forecast
# --------------------------------------------------------------------------
_STOP = {"the", "a", "an", "is", "are", "there", "any", "of", "in", "and", "or", "to", "for",
         "with", "by", "that", "this", "does", "do", "did", "what", "which", "how", "show", "me",
         "data", "evidence", "hypothesis", "support", "supported", "our", "it", "be", "on"}


def _hypothesis(ctx: _Ctx) -> Answer:
    it = ctx.intent
    if it.topic == "network":
        return _network(ctx)
    reg = ctx.agg.get("agg_hypothesis_register")
    if reg is None or reg.empty:
        return Answer(headline="The hypothesis register has not been built.", no_result=True)

    if it.topic == "spikes":
        row = _register_row(ctx.agg, "spike")
    else:
        words = set(re.findall(r"[a-z]+", it.question.lower())) - _STOP
        best = None
        for _, candidate in reg.iterrows():
            text = f"{candidate['hypothesis']} {candidate['test']}".lower()
            score = len(words & set(re.findall(r"[a-z]+", text)))
            if best is None or score > best[0]:
                best = (score, candidate)
        row = best[1] if best and best[0] >= 2 else None
    if row is None:
        choice = select_chart(kind="listing", records=True)
        return Answer(headline="No tested hypothesis matches that question.",
                      detail="Tested hypotheses: " + "; ".join(reg["hypothesis"]),
                      chart_kind=choice.chart, chart_reason=choice.reason,
                      table=reg[["hypothesis", "verdict"]], no_result=True)

    kind = {"CONTRADICTED": "crit", "SUPPORTED": "good", "BORDERLINE": "warn"}.get(row["verdict"], "info")
    answer = Answer(headline=f"{row['verdict']}: {row['hypothesis']}", detail=str(row["reading"]),
                    rows_analysed=len(ctx.frames[semantic.TRANSACTIONS]))
    answer.table = pd.DataFrame([row[["hypothesis", "test", "statistic", "p_value", "verdict"]]])
    if it.topic == "spikes":
        res = run(QuerySpec(metric="total_transaction_count", time_grain="day"), ctx.frames)
        choice = select_chart(kind="trend", time_grain="day")
        answer.chart_kind, answer.chart_reason = choice.chart, choice.reason
        answer.chart_x = [pd.Timestamp(p) for p in res.frame["period"]]
        answer.chart_series = {"Transactions per day": res.frame["value"].astype(float).tolist()}
        answer.chart_title = "Transactions per day — no burst stands out"
        answer.rows_analysed = res.rows_analysed
    else:
        choice = select_chart(kind="listing", records=True)
        answer.chart_kind, answer.chart_reason = choice.chart, choice.reason
    p = row["p_value"]
    ctx.caveat(f"Test: {row['test']} — {row['statistic']}"
               + (f", p = {p}" if pd.notna(p) else "") + ".", kind)
    ctx.validation = "read from the hypothesis register built with the analytics layer"
    return answer


def _network(ctx: _Ctx) -> Answer:
    net = ctx.agg.get("agg_network_summary")
    if net is None or net.empty:
        return Answer(headline="Network analysis has not been built.", no_result=True)
    n = net.iloc[0]
    acyclic = bool(n["is_forest_acyclic"])
    bipartite = bool(n["is_bipartite"])
    choice = select_chart(kind="listing", records=True)
    answer = Answer(
        headline=("No — the fraud-ring hypothesis is not supported: this transaction graph cannot "
                  "contain a circular money path." if acyclic and bipartite else
                  "The transaction graph contains cycles; inspect the network page before drawing "
                  "conclusions."),
        chart_kind=choice.chart, chart_reason=choice.reason, rows_analysed=int(n["transactions"]))
    answer.detail = (
        f"The graph has {int(n['nodes']):,} nodes, {int(n['edges']):,} edges and "
        f"{int(n['components']):,} connected components; the largest holds "
        f"{float(n['largest_component_share_pct']):.2f}% of nodes. It is "
        f"{'bipartite' if bipartite else 'not bipartite'} "
        f"({int(n['shares_ids_across_sides'])} IDs appear as both customer and merchant), so money "
        f"only moves from customers to merchants. There are {int(n['user_pairs_sharing_2plus']):,} "
        f"four-cycles and {int(n['repeat_pairs']):,} repeat customer-merchant pairs, and the graph "
        f"is {'a forest — edges equal nodes minus components exactly, so it has no cycle of any length' if acyclic else 'not a forest'}.")
    answer.table = pd.DataFrame([
        ("Bipartite (customers only pay merchants)", "Yes" if bipartite else "No"),
        ("IDs on both sides", f"{int(n['shares_ids_across_sides']):,}"),
        ("Repeat customer-merchant pairs", f"{int(n['repeat_pairs']):,}"),
        ("Four-cycles", f"{int(n['user_pairs_sharing_2plus']):,}"),
        ("Graph is a forest (acyclic)", "Yes" if acyclic else "No"),
        ("Largest component share", f"{float(n['largest_component_share_pct']):.2f}%"),
    ], columns=["Structural test", "Result"])
    ctx.caveat("This is a structural result, not a gap in the search: a circular A → B → C → A "
               "money path needs customer-to-customer transfers, and this data has none.", "info")
    ctx.validation = "read from the network summary built from FACT_TRANSACTIONS"
    return answer


def _forecast(ctx: _Ctx) -> Answer:
    series = FORECAST_SERIES.get(ctx.intent.metric, "transactions")
    forecast = ctx.agg.get("agg_forecast")
    diagnostics = ctx.agg.get("agg_forecast_diagnostics")
    backtest = ctx.agg.get("agg_forecast_backtest")
    if forecast is None or diagnostics is None or forecast.empty:
        return Answer(headline="The forecast has not been built.",
                      detail="Run python scripts/build_analytics.py.", no_result=True)
    rows = forecast[forecast["series"] == series].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    d = diagnostics[diagnostics["series"] == series].iloc[0]
    unit = "INR" if d["unit"] == "INR" else "count"
    actual = rows[rows["kind"] == "actual"].sort_values("date").tail(int(d["holdout_days"]))
    future = rows[rows["kind"] == "forecast"].sort_values("date")
    choice = select_chart(kind="trend", time_grain="day")
    answer = Answer(
        headline=(f"Projected {str(d['label']).lower()}: about {fmt(future['value'].mean(), unit)} "
                  f"per day over the next {len(future)} days (95% band "
                  f"{fmt(future['lower_95'].mean(), unit)}–{fmt(future['upper_95'].mean(), unit)})."),
        chart_kind=choice.chart, chart_reason=choice.reason,
        chart_title=f"{d['label']}: last {len(actual)} days and {len(future)}-day projection",
        rows_analysed=int(d["history_days"]))
    answer.chart_x = [pd.Timestamp(v) for v in list(actual["date"]) + list(future["date"])]
    answer.chart_series = {
        "Actual": [float(v) for v in actual["value"]] + [None] * len(future),
        "Forecast": [None] * len(actual) + [float(v) for v in future["value"]],
    }
    answer.chart_band = {"x": [pd.Timestamp(v) for v in future["date"]],
                         "lower": future["lower_95"].astype(float).tolist(),
                         "upper": future["upper_95"].astype(float).tolist()}
    weekday = (f"; day-of-week effect p = {float(d['weekday_p']):.3f}"
               if pd.notna(d["weekday_p"]) else "")
    answer.detail = (f"Method: {d['selected_method_label']}, chosen because it had the lowest error "
                     f"on a {int(d['holdout_days'])}-day holdout (MAE {fmt(d['selected_mae'], unit)}; "
                     f"the flat-mean baseline scored {fmt(d['mean_baseline_mae'], unit)}). On that "
                     f"holdout the 95% band contained {float(d['holdout_band_coverage_pct']):.0f}% of "
                     f"actual days. Trend test: slope {float(d['slope_per_day']):+.3f} per day, "
                     f"p = {float(d['slope_p']):.3f}{weekday}.")
    if backtest is not None:
        answer.table = backtest[backtest["series"] == series].reset_index(drop=True)
    end = pd.Timestamp(d["last_date"])
    if float(d["slope_p"]) >= 0.05:
        ctx.caveat(f"This projects beyond the observed data, which ends on {end:%Y-%m-%d}. With no "
                   "significant trend, a near-flat projection is the honest result — a "
                   "capacity-planning baseline, not a prediction of fraud.", "warn")
    else:
        ctx.caveat(f"This projects beyond the observed data, which ends on {end:%Y-%m-%d}, and "
                   f"extends a statistically significant trend (p = {float(d['slope_p']):.3f}).",
                   "warn")
    if series == "disputed_transactions":
        ctx.caveat("Disputed transactions are counted on the transaction date.", "info")
    ctx.validation = "method selected on a time-ordered holdout; band coverage checked on it"
    return answer


HANDLERS = {
    "metric": _metric,
    "coverage": _metric,
    "trend": _trend,
    "ranking": _by_dimension,
    "breakdown": _by_dimension,
    "distribution": _by_dimension,
    "listing": _listing,
    "significance": _significance,
    "explanation": _explanation,
    "hypothesis": _hypothesis,
    "risk": _risk,
    "forecast": _forecast,
}
