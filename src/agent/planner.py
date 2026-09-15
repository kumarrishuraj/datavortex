"""Intent planning for the AI Investigator.

    natural language -> structured Intent -> semantic validation

A question is decomposed into independent *slots*, each resolved against the
semantic registry rather than against a list of known questions:

    measure      which registered metric is being asked about
    dimension    which registered dimension it is broken down by
    time grain   day / week / month / quarter, from words like "daily" or "trend"
    time window  "this quarter", "Q3", "last 6 months", "in February"
    series       e.g. "successful vs failed" splits one measure into lines
    ranking      "which", "top 10", "highest", "lowest"

Measures and dimensions are found by matching the registry's own synonyms,
preferring the longest and most specific phrase, so "chargeback-to-transaction
ratio" beats "transaction" and "disputed amount" beats "amount". Adding a synonym
to `semantic.py` is the only way to teach the planner a new phrasing.

A handful of analysis *types* are recognised directly because they are not a
measure at all — significance tests, "why is X ranked first", hypothesis
checks, forecasts, identity collisions. These are intent kinds, not question
templates: each accepts any measure, dimension or phrasing that fits it.

The planner never touches data and never produces a number. Its only output is
an Intent, and every Intent is validated against the registry before execution.
A language model, when added, would produce the same object and be subject to
the same validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.agent.chart_selector import select_chart
from src.analytics import semantic

ALLOWED_METRICS = set(semantic.METRICS)
ALLOWED_DIMENSIONS = set(semantic.ALLOWED_DIMENSIONS) | {
    "merchant", "user", "none", "identity", "component", "network",
}
ALLOWED_CHARTS = set(semantic.ALLOWED_CHARTS)
ALLOWED_TIME_GRAINS = set(semantic.TIME_GRAINS)
ALLOWED_INTENT_KINDS = {
    "metric", "coverage", "trend", "ranking", "breakdown", "distribution", "listing",
    "significance", "explanation", "hypothesis", "risk", "forecast",
}
ALLOWED_FILTERS = {"status_canonical", "min_disputes", "min_delay_days"}
ALLOWED_STATUSES = {"SUCCESS", "FAILED", "PENDING"}

# Dimensions small enough to draw as separate lines on one trend chart.
SERIES_DIMENSIONS = {"status", "severity", "kyc_status", "reason", "channel", "resolution",
                     "risk_segment", "merchant_status"}

# Tokens that must never appear in anything the agent is asked to run. The agent
# is read-only analytics over pre-materialized frames and has no SQL surface;
# this guard keeps that true if one is ever added.
FORBIDDEN_TOKENS = (
    "drop", "delete", "update", "insert", "alter", "truncate", "create",
    "grant", "revoke", "exec", "execute", "attach", "copy", "pragma",
    "__import__", "eval", "open(", "subprocess", "os.system", ";--", "/*",
)

OUT_OF_SCOPE = (
    "That question is outside the scope of this dataset's analytics layer. The "
    "investigator answers questions about transactions, chargebacks, merchants, "
    "customers, KYC status and data quality — using only the metrics registered in "
    "the semantic layer."
)


class IntentRejected(ValueError):
    """Raised when a question cannot be planned or an intent fails validation."""


@dataclass
class Intent:
    kind: str
    metric: str | None = None
    dimension: str = "none"
    chart: str = "kpi_card"
    filters: dict = field(default_factory=dict)
    limit: int = 10
    subject: str | None = None
    question: str = ""
    time_grain: str | None = None
    series: str | None = None
    series_values: tuple = ()
    window: dict | None = None
    threshold: float | None = None
    topic: str | None = None
    ascending: bool = False
    assumptions: list = field(default_factory=list)
    chart_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "metric": self.metric, "dimension": self.dimension,
            "time_grain": self.time_grain, "series": self.series,
            "series_values": list(self.series_values), "window": self.window,
            "filters": self.filters, "threshold": self.threshold, "limit": self.limit,
            "ascending": self.ascending, "subject": self.subject, "topic": self.topic,
            "chart": self.chart, "assumptions": list(self.assumptions),
        }


# --------------------------------------------------------------------------
# Validation — the security boundary
# --------------------------------------------------------------------------
def assert_safe(text: str) -> None:
    """Refuse any string carrying a mutating or code-execution token."""
    lowered = str(text).lower()
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            raise IntentRejected(
                f"input contains the forbidden token {token!r}. The investigator is "
                f"read-only and cannot run statements that modify or execute anything.")


def validate_intent(intent: Intent) -> Intent:
    """Reject anything outside the registry, or any combination it cannot compute."""
    if intent.kind not in ALLOWED_INTENT_KINDS:
        raise IntentRejected(f"intent kind {intent.kind!r} is not allowed")
    if intent.metric is not None and intent.metric not in ALLOWED_METRICS:
        raise IntentRejected(
            f"metric {intent.metric!r} is not in the semantic registry. "
            f"The agent can only answer using registered metrics.")
    if intent.dimension not in ALLOWED_DIMENSIONS:
        raise IntentRejected(f"dimension {intent.dimension!r} is not allowed")
    if intent.chart not in ALLOWED_CHARTS:
        raise IntentRejected(f"chart type {intent.chart!r} is not allowed")
    if not isinstance(intent.limit, int) or not (1 <= intent.limit <= 200):
        raise IntentRejected("limit must be an integer between 1 and 200")
    if intent.time_grain is not None and intent.time_grain not in ALLOWED_TIME_GRAINS:
        raise IntentRejected(f"time grain {intent.time_grain!r} is not allowed")
    if intent.series is not None and intent.series not in semantic.DIMENSIONS:
        raise IntentRejected(f"series dimension {intent.series!r} is not allowed")
    for key, value in intent.filters.items():
        if key not in ALLOWED_FILTERS and key not in ALLOWED_DIMENSIONS:
            raise IntentRejected(f"filter {key!r} is not an allowed filter")
        if key == "status_canonical" and not set(value) <= ALLOWED_STATUSES:
            raise IntentRejected(f"status filter {value!r} is not allowed")
        assert_safe(str(value))
    for value in intent.series_values:
        assert_safe(str(value))
    assert_safe(intent.question)

    # Semantic compatibility: the metric must be computable the way it is asked.
    if intent.metric and intent.kind in {"metric", "coverage", "trend", "ranking",
                                         "breakdown", "distribution"}:
        metric = semantic.get(intent.metric)
        if intent.dimension in semantic.DIMENSIONS:
            dim = semantic.DIMENSIONS[intent.dimension]
            if metric.source not in dim.columns:
                raise IntentRejected(
                    f"{metric.label} cannot be broken down by {dim.label.lower()}: that "
                    f"dimension does not exist on {metric.source}.")
        if intent.time_grain and not metric.trendable:
            raise IntentRejected(f"{metric.label} has no date, so it cannot be shown as a "
                                 "trend over time.")
        if intent.series:
            sdim = semantic.DIMENSIONS[intent.series]
            if metric.source not in sdim.columns:
                raise IntentRejected(f"{metric.label} cannot be split by {sdim.label.lower()}.")
    return intent


# --------------------------------------------------------------------------
# Language normalisation and phrase matching
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    t = str(text).lower().replace("₹", " ")
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"[^a-z0-9%\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=None)
def _phrase(phrase: str) -> re.Pattern:
    body = r"\s+".join(re.escape(p) for p in phrase.split())
    return re.compile(r"(?<![a-z0-9])" + body + r"(?:s|es)?(?![a-z0-9])")


# Single words that name a measure only loosely. They lose to any more specific
# phrase, so "missing UTR transactions" resolves to the UTR metric, not a count.
GENERIC_WORDS = frozenset({"transactions", "volume", "value", "amount", "chargebacks",
                           "chargeback", "disputes", "complaints"})


def _best_phrase(text: str, phrases, generic=frozenset()):
    best = None
    for phrase in phrases:
        match = _phrase(phrase).search(text)
        if not match:
            continue
        score = len(phrase) - (10 if phrase in generic else 0)
        if best is None or score > best[0]:
            best = (score, phrase, match.start(), match.end())
    return best


def resolve_metric(text: str):
    """Most specific registered metric named in the text, or None."""
    best = None
    for metric in semantic.METRICS.values():
        if not metric.synonyms:
            continue
        hit = _best_phrase(text, metric.synonyms, GENERIC_WORDS)
        if hit and (best is None or hit[0] > best[0][0]):
            best = (hit, metric.name)
    return best


def resolve_dimension(text: str):
    """Most specific registered dimension named in the text, or None."""
    best = None
    for dim in semantic.DIMENSIONS.values():
        hit = _best_phrase(text, dim.synonyms)
        if hit and (best is None or hit[0] > best[0][0]):
            best = (hit, dim.name)
    return best


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


# --------------------------------------------------------------------------
# Slot detectors
# --------------------------------------------------------------------------
_GRAIN = (
    ("day", re.compile(r"\b(daily|day by day|day wise|day on day|per day|each day|every day|"
                       r"by day(?! of)|by date|per date)\b")),
    ("week", re.compile(r"\b(weekly|week by week|week wise|week on week|per week|each week|by week)\b")),
    ("month", re.compile(r"\b(monthly|month by month|month wise|month on month|per month|"
                         r"each month|by month)\b")),
    ("quarter", re.compile(r"\b(quarterly|quarter on quarter|per quarter|each quarter|by quarter)\b")),
)
_TREND = re.compile(
    r"\b(trend|trends|trending|over time|timeline|time series|throughout|through the quarter|"
    r"across the quarter|over the quarter|over the period|evolution|evolved|evolving|"
    r"progression|movement|fluctuat\w*)\b|\bhow\b.*\bchang\w*|\bchang\w* over\b")

_WINDOWS = (
    (re.compile(r"\b(this|current|the current|latest|most recent) (quarter|qtr)\b"),
     lambda m: {"type": "latest_quarter", "phrase": m.group(0)}),
    (re.compile(r"\b(last|previous|prior) (quarter|qtr)\b"),
     lambda m: {"type": "previous_quarter", "phrase": m.group(0)}),
    (re.compile(r"\bq([1-4])(?: (20\d\d))?\b"),
     lambda m: {"type": "quarter", "quarter": int(m.group(1)),
                "year": int(m.group(2)) if m.group(2) else None}),
    (re.compile(r"\b(?:last|past|previous|prior|recent) (\d+) (day|days|week|weeks|month|months)\b"),
     lambda m: {"type": "last_n", "n": int(m.group(1)),
                "unit": m.group(2) if m.group(2).endswith("s") else m.group(2) + "s"}),
    (re.compile(r"\b(?:last|past|previous) (day|week|month)\b"),
     lambda m: {"type": "last_n", "n": 1, "unit": m.group(1) + "s"}),
)
_MONTHS = {name: i for i, names in enumerate(
    [("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
     ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
     ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
     ("december", "dec")], start=1) for name in names}
_MONTH_WINDOW = re.compile(
    r"\b(?:in|during|for) (" + "|".join(sorted(_MONTHS, key=len, reverse=True)) +
    r")(?: (20\d\d))?\b")

_STATUS = (
    ("SUCCESS", re.compile(r"\b(success|successes|successful|succeeded|completed)\b")),
    ("FAILED", re.compile(r"\b(fail|fails|failed|failure|failures|failing|declined)\b")),
    ("PENDING", re.compile(r"\b(pending|processing|initiated)\b")),
)

_RANKING = re.compile(r"\b(which|top|highest|lowest|most|least|largest|smallest|biggest|"
                      r"leading|maximum|minimum|rank|ranking|ranked|best|worst|fewest|greatest)\b")
_ASCENDING = re.compile(r"\b(lowest|least|smallest|minimum|fewest)\b")
_DISTRIBUTION = re.compile(r"\b(distribution|distributions|breakdown|split|mix|composition|share)\b")
_TOP_N = re.compile(r"\btop (\d{1,3})\b")

_SIGNIFICANCE = re.compile(r"\b(statistically|significan\w*|p value|chi square|confidence interval|"
                           r"by chance|due to chance|random chance|sampling noise)\b|"
                           r"\bis (?:the|this|that) (?:difference|ranking|gap) real\b")
_WHY = re.compile(r"\bwhy\b")
_RANK_WORD = re.compile(r"\b(rank|ranks|ranked|first|top|highest|lead|leads|leading|number one)\b")
_FORECAST = re.compile(r"\b(forecast\w*|projection|projected|project|next (?:\d+ )?(?:day|days|week|weeks|"
                       r"fortnight|month|months)|coming (?:days|weeks)|upcoming|future)\b")
_SPIKE = re.compile(r"\b(spike|spikes|spiking|burst|bursts|surge|surges|sudden)\b")
_HYPOTHESIS = re.compile(r"\b(hypothes\w*|evidence|correlat\w*|associated with|indicates?|"
                         r"relationship between|is there (?:a|any) (?:pattern|link))\b")
_IDENTITY = re.compile(r"\b(identity collisions?|collid\w*|collision|shared ids?|same id|"
                       r"ambiguous identit\w*|duplicate identit\w*)\b")
_REPEAT = re.compile(r"\b(repeat\w* disput\w*|repeat\w* chargebacks?|multiple disputes|"
                     r"more than one dispute|disputed more than once|repeat disputers?|"
                     r"(\d+) or more disputes)\b")
_RING = re.compile(r"\b(fraud rings?|rings?|circular|money laundering|launder\w*|round tripping|"
                   r"cycles?)\b")
_FRAUD_LABEL = re.compile(r"\b(fraudsters?|fraudulent|confirmed fraud|scammers?|criminals?|guilty|"
                          r"commit(?:s|ted|ting)? fraud)\b")
_INJECTION = re.compile(r"\b(ignore (?:all |the |any )?(?:previous|prior|above|earlier) "
                        r"(?:instructions?|rules?|prompts?)|system prompt|jailbreak|developer mode|"
                        r"disregard (?:the |your )?(?:rules|instructions))\b")
_SENSITIVE = re.compile(r"\b(pan|pans|aadhaar|aadhar|account numbers?|settlement accounts?|"
                        r"dates? of birth|dob)\b")
_EXPOSE = re.compile(r"\b(full|raw|unmasked|complete|actual|real|reveal|expose|dump|export|print|"
                     r"list|show|give|get|fetch|display|see|view|what|which|all|every|each|"
                     r"numbers?|values?|decrypt|unhash|original)\b")
# A question is words. Statement-shaped SQL is refused outright rather than being
# reinterpreted, so nobody can mistake the investigator for a query console.
_SQL = re.compile(r"\bselect\b[\s\S]*?\bfrom\b|\bunion\s+(?:all\s+)?select\b|\binsert\s+into\b|"
                  r"\bwhere\s+\w+\s*(?:=|<|>|\blike\b)")
_TXN_WORDS = re.compile(r"\b(transaction|transactions|value|amount|revenue|sales|volume)\b")

FORECAST_METRICS = {"total_transaction_count", "total_transaction_amount",
                    "chargeback_count", "chargeback_to_transaction_ratio"}


def _categories() -> tuple[str, ...]:
    from src.cleaning.merchants import MCC_TO_CATEGORY
    return tuple(sorted(set(MCC_TO_CATEGORY.values()) | {"Misc"}, key=len, reverse=True))


def _detect_grain(text: str) -> str | None:
    for grain, pattern in _GRAIN:
        if pattern.search(text):
            return grain
    if _TREND.search(text):
        if re.search(r"\bmonths?\b", text):
            return "month"
        if re.search(r"\bweeks?\b", text):
            return "week"
        return "day"
    return None


def _detect_window(text: str) -> dict | None:
    for pattern, build in _WINDOWS:
        match = pattern.search(text)
        if match:
            return build(match)
    match = _MONTH_WINDOW.search(text)
    if match:
        return {"type": "month", "month": _MONTHS[match.group(1)],
                "year": int(match.group(2)) if match.group(2) else None}
    return None


def _detect_statuses(text: str) -> tuple[str, ...]:
    return tuple(status for status, pattern in _STATUS if pattern.search(text))


def _subject(question: str) -> str | None:
    lowered = question.lower()
    for category in _categories():
        if re.search(r"\b" + re.escape(category.lower().replace(" & ", " ")) + r"\b",
                     lowered.replace("&", " ").replace("  ", " ")):
            return category
    return None


def _finish(intent: Intent, n_categories: int | None = None) -> Intent:
    choice = select_chart(
        kind=intent.kind, time_grain=intent.time_grain,
        dimension=intent.dimension if intent.dimension in semantic.DIMENSIONS else None,
        n_categories=n_categories,
        records=intent.kind == "listing" or (intent.kind == "hypothesis" and intent.topic != "spikes"),
    )
    if intent.kind == "hypothesis" and intent.topic == "spikes":
        choice = select_chart(kind="trend", time_grain="day")
    if intent.kind == "forecast":
        choice = select_chart(kind="trend", time_grain="day")
    intent.chart = choice.chart
    intent.chart_reason = choice.reason
    return validate_intent(intent)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def plan(question: str) -> Intent:
    """Turn a question into a validated Intent, or raise IntentRejected.

    Deterministic: the same question always produces the same intent, and a
    question with no recognisable measure or analysis is refused, not guessed at.
    """
    assert_safe(question)
    if _SQL.search(str(question).lower()):
        raise IntentRejected("The investigator does not accept SQL. Ask the question in words: "
                             "every answer is computed from registered metrics, never from a "
                             "query supplied with the question.")
    raw = str(question).strip()
    if not raw:
        raise IntentRejected("empty question")
    text = normalize(raw)

    if _INJECTION.search(text):
        raise IntentRejected("The investigator's instructions cannot be changed from a question. "
                             "Ask about transactions, chargebacks, merchants, customers or data "
                             "quality instead.")
    if _SENSITIVE.search(text) and _EXPOSE.search(text):
        raise IntentRejected("Full PAN, Aadhaar, account numbers and dates of birth are never "
                             "exposed. The processed layer keeps only masked forms, so there is "
                             "nothing unmasked to show.")

    base = dict(question=raw)

    # --- analysis types that are not a measure --------------------------------
    if _RING.search(text):
        return _finish(Intent(kind="hypothesis", topic="network", dimension="network", **base))
    if _FRAUD_LABEL.search(text):
        raise IntentRejected(
            "The data contains disputes, not confirmed fraud labels, so no merchant or customer "
            "can be called fraudulent. Try 'Show high-risk users with repeated disputes' for the "
            "explainable Risk Indicator Score, or 'Is there evidence of a fraud ring?'.")

    window = _detect_window(text)

    if _SIGNIFICANCE.search(text):
        dim_hit = resolve_dimension(text)
        dimension = dim_hit[1] if dim_hit else "merchant_category"
        if dimension not in semantic.DIMENSIONS or semantic.DIMENSIONS[dimension].entity \
                or semantic.TRANSACTIONS not in semantic.DIMENSIONS[dimension].columns:
            dimension = "merchant_category"
        return _finish(Intent(kind="significance", metric="chargeback_to_transaction_ratio",
                              dimension=dimension, window=window, **base))

    if _WHY.search(text) and _RANK_WORD.search(text):
        dim_hit = resolve_dimension(text)
        dimension = dim_hit[1] if dim_hit and not semantic.DIMENSIONS[dim_hit[1]].entity \
            else "merchant_category"
        return _finish(Intent(kind="explanation", metric="chargeback_to_transaction_ratio",
                              dimension=dimension, subject=_subject(raw), window=window, **base))

    metric_hit = resolve_metric(text)
    metric = metric_hit[1] if metric_hit else None

    if _FORECAST.search(text):
        chosen = metric if metric in FORECAST_METRICS else "total_transaction_count"
        assumptions = [] if metric in FORECAST_METRICS else [
            "No forecastable measure was named, so this projects daily transaction volume."]
        return _finish(Intent(kind="forecast", metric=chosen, assumptions=assumptions, **base))

    if _SPIKE.search(text):
        return _finish(Intent(kind="hypothesis", topic="spikes",
                              metric="total_transaction_count", **base))

    if _IDENTITY.search(text):
        return _finish(Intent(kind="listing", metric="identity_ambiguity_rate",
                              dimension="identity", **base))

    rest = _blank(text, metric_hit[0][2], metric_hit[0][3]) if metric_hit else text
    dim_hit = resolve_dimension(rest)
    dimension = dim_hit[1] if dim_hit else None
    limit_match = _TOP_N.search(text)
    limit = int(limit_match.group(1)) if limit_match else 10

    repeat = _REPEAT.search(text)
    min_disputes = 2
    if repeat and repeat.group(2):
        min_disputes = int(repeat.group(2))

    if metric == "risk_indicator_score":
        entity = dimension if dimension in ("user", "merchant") else "user"
        filters = {"min_disputes": min_disputes} if repeat else {}
        return _finish(Intent(kind="risk", metric=metric, dimension=entity, filters=filters,
                              limit=limit, **base))

    if repeat and dimension not in ("merchant_category", "state", "kyc_status"):
        entity = dimension if dimension in ("user", "merchant") else "user"
        return _finish(Intent(kind="listing", metric="chargeback_count", dimension=entity,
                              filters={"min_disputes": min_disputes}, limit=limit, **base))

    if metric == "disputes_reported_after_7_days" and dimension is None:
        days = re.search(r"\b(?:after|over|more than|beyond) (\d+) ?days?\b", text)
        threshold = float(days.group(1)) if days else 7.0
        assumptions = [] if days else [
            "No day threshold was given, so 'late' is read as more than 7 days."]
        return _finish(Intent(kind="listing", metric=metric, threshold=threshold,
                              filters={"min_delay_days": threshold}, window=window,
                              assumptions=assumptions, **base))

    if _HYPOTHESIS.search(text) and not _detect_grain(text):
        return _finish(Intent(kind="hypothesis", topic="register", metric=metric, **base))

    # --- generic measure / dimension / time composition ------------------------
    grain = _detect_grain(text)
    assumptions: list[str] = []
    statuses = _detect_statuses(text)
    transaction_measures = {None, "total_transaction_count", "total_transaction_amount",
                            "average_transaction_value"}
    use_statuses = bool(statuses) and metric in transaction_measures
    if grain and dimension in ("hour", "day_of_week"):
        assumptions.append(f"Read as a {grain} trend; ask for the measure 'by "
                           f"{semantic.DIMENSIONS[dimension].label.lower()}' for that profile.")
        dimension = None

    if metric is None:
        if dimension and set(semantic.DIMENSIONS[dimension].columns) == {semantic.CHARGEBACKS}:
            metric = "chargeback_count"
        elif dimension in ("kyc_status", "risk_segment") and not _TXN_WORDS.search(rest):
            metric = "customer_count"
        elif dimension or grain or use_statuses:
            metric = "total_transaction_count"
        else:
            nearest = suggest_metrics(raw)
            hint = f" Closest registered metrics: {', '.join(nearest)}." if nearest else ""
            raise IntentRejected(OUT_OF_SCOPE + hint)
        assumptions.append(f"No measure was named, so this is read as "
                           f"{semantic.get(metric).label.lower()}.")
    elif metric_hit:
        second = resolve_metric(rest)
        if second and second[1] != metric and second[0][0] > 0 and grain:
            assumptions.append(
                f"The question names more than one measure. This answer shows "
                f"{semantic.get(metric).label.lower()}; ask separately for "
                f"{semantic.get(second[1]).label.lower()} — measures on different scales are "
                f"never drawn on one chart.")

    if metric in ("merchant_master_coverage", "kyc_coverage") and not dimension and not grain:
        return _finish(Intent(kind="coverage", metric=metric, window=window, **base))

    filters: dict = {}
    series = None
    series_values: tuple = ()
    if use_statuses:
        if grain or len(statuses) >= 2:
            if grain:
                series, series_values = "status", statuses
            else:
                dimension = "status"
                filters["status_canonical"] = list(statuses)
        else:
            filters["status_canonical"] = list(statuses)

    if grain:
        if dimension:
            dim = semantic.DIMENSIONS[dimension]
            if dimension in SERIES_DIMENSIONS and series is None:
                series, dimension = dimension, None
            elif dim.entity:
                raise IntentRejected(
                    f"A trend per individual {dim.label.lower()} would draw thousands of lines. "
                    f"Ask for a ranking of {dim.label.lower()}s, or a trend of the overall measure.")
            else:
                raise IntentRejected(
                    f"{dim.label} has too many values to draw as separate trend lines. Ask for "
                    f"the measure by {dim.label.lower()}, or for its overall trend.")
        intent = Intent(kind="trend", metric=metric, time_grain=grain, series=series,
                        series_values=series_values, window=window, filters=filters,
                        assumptions=assumptions, **base)
        return _finish(intent)

    if dimension:
        dim = semantic.DIMENSIONS[dimension]
        if dim.entity or _RANKING.search(text) or limit_match:
            kind = "ranking"
        elif _DISTRIBUTION.search(text):
            kind = "distribution"
        else:
            kind = "breakdown"
        threshold = None
        if metric == "disputes_reported_after_7_days":
            days = re.search(r"\b(?:after|over|more than|beyond) (\d+) ?days?\b", text)
            threshold = float(days.group(1)) if days else 7.0
        intent = Intent(kind=kind, metric=metric, dimension=dimension, filters=filters,
                        limit=limit, window=window, threshold=threshold,
                        ascending=bool(_ASCENDING.search(text)) and kind == "ranking",
                        assumptions=assumptions, **base)
        return _finish(intent)

    return _finish(Intent(kind="metric", metric=metric, window=window, filters=filters,
                          assumptions=assumptions, **base))


def suggest_metrics(question: str, limit: int = 4) -> list[str]:
    """Nearest registered metrics, offered when a question cannot be planned."""
    words = {w for w in re.findall(r"[a-z]+", str(question).lower()) if len(w) > 3}
    scored = []
    for name, metric in semantic.METRICS.items():
        haystack = f"{name} {metric.label} {metric.definition} {' '.join(metric.synonyms)}".lower()
        score = sum(1 for w in words if w in haystack)
        if score:
            scored.append((score, name))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [name for _, name in scored[:limit]]
