"""Semantic layer — the single definition of every business metric and dimension.

One registry, read by the KPI validation, the query engine, the dashboard and the
AI Investigator. A metric defined here cannot mean one thing in a chart and
another in an agent answer.

Each metric carries two halves:

  * its **business definition** — label, definition, formula, grain and coverage
    basis — which is what a reader sees, and
  * its **computation spec** — source table, aggregation, value column, time
    column — which `src/analytics/query_engine.py` executes directly.

Because the agent can only reach a number by handing a metric name from this
registry to that engine, the registry is the source of truth for computation,
not just documentation.

Coverage bases are written definitionally, without percentages. The percentage
is computed from the data at run time, so the registry cannot go stale when the
data changes.

`UNKNOWN` is a first-class dimension member throughout. Hiding it is how two
thirds of the customer base would quietly disappear from a breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Sources, aggregations, time
# --------------------------------------------------------------------------
TRANSACTIONS = "transactions"
CHARGEBACKS = "chargebacks"
CUSTOMERS = "customers"
PRECOMPUTED = "precomputed"
SOURCES = (TRANSACTIONS, CHARGEBACKS, CUSTOMERS, PRECOMPUTED)

COUNT = "count"
SUM = "sum"
MEAN = "mean"
SHARE = "share"                    # percentage of rows meeting a predicate
DISPUTED_RATIO = "disputed_ratio"  # distinct disputed transactions / transactions
COMPLAINTS_RATIO = "complaints_ratio"
THRESHOLD_COUNT = "threshold_count"
PRECOMPUTED_AGG = "precomputed"
AGGREGATIONS = (COUNT, SUM, MEAN, SHARE, DISPUTED_RATIO, COMPLAINTS_RATIO,
                THRESHOLD_COUNT, PRECOMPUTED_AGG)

# Aggregations whose group values must sum to the total. The query engine
# asserts this on every grouped result, so a fan-out or a dropped group cannot
# pass silently.
ADDITIVE_AGGREGATIONS = frozenset({COUNT, SUM, THRESHOLD_COUNT})

TXN_TIME = "timestamp_clean"
REPORTED_TIME = "reported_timestamp_clean"
TIME_GRAINS = ("day", "week", "month", "quarter")

# Definitional coverage bases — no percentages, see module docstring.
FULL = "all cleaned transactions"
MERCHANT_COVERAGE = "transactions whose merchant resolves to a merchant master record"
KYC_COVERAGE = "transactions whose customer resolves to a KYC record"
LINKED_CB = "complaints linked to a transaction through txn_id"
ALL_CB = "all cleaned complaints"
RESOLVED_CUSTOMERS = "resolved customer identities"


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    definition: str
    formula: str
    grain: str
    coverage_basis: str
    chart_hint: str
    unit: str = "count"
    higher_is_worse: bool = False
    notes: str = ""
    dimensions: tuple[str, ...] = field(default_factory=tuple)
    # --- computation spec --------------------------------------------------
    source: str = TRANSACTIONS
    aggregation: str = COUNT
    column: str | None = None
    predicate: tuple | None = None      # (column, value) for SHARE metrics
    time_column: str | None = TXN_TIME
    threshold: float | None = None      # for THRESHOLD_COUNT metrics
    synonyms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def trendable(self) -> bool:
        """Whether the metric can be plotted over time."""
        return self.time_column is not None and self.aggregation not in {
            COMPLAINTS_RATIO, PRECOMPUTED_AGG}

    @property
    def additive(self) -> bool:
        return self.aggregation in ADDITIVE_AGGREGATIONS


@dataclass(frozen=True)
class Dimension:
    name: str
    label: str
    columns: dict = field(default_factory=dict)   # source -> column
    synonyms: tuple[str, ...] = field(default_factory=tuple)
    coverage_metric: str | None = None
    entity: bool = False
    unknown_values: tuple[str, ...] = field(default_factory=tuple)
    order: tuple[str, ...] = field(default_factory=tuple)
    requires_clock_time: bool = False


METRICS: dict[str, Metric] = {}
DIMENSIONS: dict[str, Dimension] = {}


def _add(m: Metric) -> None:
    if m.source not in SOURCES:
        raise ValueError(f"{m.name}: unknown source {m.source!r}")
    if m.aggregation not in AGGREGATIONS:
        raise ValueError(f"{m.name}: unknown aggregation {m.aggregation!r}")
    METRICS[m.name] = m


def _dim(d: Dimension) -> None:
    DIMENSIONS[d.name] = d


UNKNOWN = "UNKNOWN"
UNKNOWN_CATEGORY = "UNKNOWN (no master record)"

# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------
_dim(Dimension(
    name="merchant_category", label="Merchant category",
    columns={TRANSACTIONS: "category", CHARGEBACKS: "category"},
    synonyms=("merchant category", "merchant categories", "category", "categories",
              "sector", "sectors", "mcc category"),
    coverage_metric="merchant_master_coverage", unknown_values=(UNKNOWN_CATEGORY,),
))
_dim(Dimension(
    name="merchant", label="Merchant",
    columns={TRANSACTIONS: "merchant_id_normalized", CHARGEBACKS: "attributed_merchant_id"},
    synonyms=("merchant", "merchants", "store", "stores", "seller", "sellers", "shop", "shops"),
    entity=True,
))
_dim(Dimension(
    name="user", label="Customer",
    columns={TRANSACTIONS: "user_id_normalized", CHARGEBACKS: "attributed_user_id"},
    synonyms=("user", "users", "customer", "customers", "payer", "payers", "account holder",
              "account holders"),
    entity=True,
))
_dim(Dimension(
    name="kyc_status", label="KYC status",
    columns={TRANSACTIONS: "kyc_status", CHARGEBACKS: "kyc_status",
             CUSTOMERS: "kyc_status_canonical"},
    synonyms=("kyc status", "kyc statuses", "kyc", "verification status"),
    coverage_metric="kyc_coverage", unknown_values=(UNKNOWN,),
))
_dim(Dimension(
    name="risk_segment", label="Declared risk segment",
    columns={TRANSACTIONS: "risk_segment", CUSTOMERS: "risk_segment_canonical"},
    synonyms=("risk segment", "risk segments", "declared risk segment", "customer segment"),
    coverage_metric="kyc_coverage", unknown_values=(UNKNOWN,),
))
_dim(Dimension(
    name="state", label="State",
    columns={TRANSACTIONS: "state", CHARGEBACKS: "state"},
    synonyms=("state", "states", "region", "regions", "geography", "location", "locations"),
    coverage_metric="merchant_master_coverage", unknown_values=(UNKNOWN,),
))
_dim(Dimension(
    name="status", label="Transaction status",
    columns={TRANSACTIONS: "status_canonical"},
    synonyms=("transaction status", "payment status", "status", "statuses", "outcome", "outcomes"),
    order=("SUCCESS", "FAILED", "PENDING"),
))
_dim(Dimension(
    name="merchant_status", label="Merchant status",
    columns={TRANSACTIONS: "merchant_status"},
    synonyms=("merchant status", "merchant statuses"),
    coverage_metric="merchant_master_coverage", unknown_values=(UNKNOWN,),
))
_dim(Dimension(
    name="reason", label="Reason code",
    columns={CHARGEBACKS: "reason_code_canonical"},
    synonyms=("reason code", "reason codes", "reason", "reasons", "dispute reason",
              "chargeback reason", "cause", "causes"),
))
_dim(Dimension(
    name="severity", label="Severity",
    columns={CHARGEBACKS: "severity_canonical"},
    synonyms=("severity level", "severity levels", "severity", "priority level"),
    order=("CRITICAL", "HIGH", "MEDIUM", "LOW"),
))
_dim(Dimension(
    name="resolution", label="Resolution status",
    columns={CHARGEBACKS: "resolution_status_canonical"},
    synonyms=("resolution status", "resolution", "case status", "dispute status"),
))
_dim(Dimension(
    name="channel", label="Intake channel",
    columns={CHARGEBACKS: "channel_canonical"},
    synonyms=("intake channel", "complaint channel", "channel", "channels"),
))
_dim(Dimension(
    name="complaint_theme", label="Complaint theme",
    columns={CHARGEBACKS: "complaint_theme"},
    synonyms=("complaint theme", "complaint themes", "complaint text", "theme", "themes"),
))
_dim(Dimension(
    name="hour", label="Hour of day",
    columns={TRANSACTIONS: "hour"},
    synonyms=("hour of day", "time of day", "hour", "hours", "hourly"),
    requires_clock_time=True,
))
_dim(Dimension(
    name="day_of_week", label="Day of week",
    columns={TRANSACTIONS: "day_of_week"},
    synonyms=("day of week", "day of the week", "weekday", "weekdays"),
    order=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
))

# --------------------------------------------------------------------------
# Metrics — volume and value
# --------------------------------------------------------------------------
_add(Metric(
    name="total_transaction_count", label="Transaction count",
    definition="Count of unique cleaned transactions.",
    formula="COUNT(DISTINCT txn_key)",
    grain="transaction", coverage_basis=FULL, chart_hint="kpi_card",
    source=TRANSACTIONS, aggregation=COUNT, time_column=TXN_TIME,
    synonyms=("transaction volume", "transaction count", "transactions count",
              "number of transactions", "count of transactions", "how many transactions",
              "volume of transactions", "txn count", "txn volume", "volume", "transactions"),
))
_add(Metric(
    name="total_transaction_amount", label="Transaction value",
    definition="Sum of transaction value in INR, using the sign-repaired magnitude.",
    formula="SUM(amount_inr)",
    grain="transaction", coverage_basis=FULL, chart_hint="kpi_card", unit="INR",
    notes="Sign-corrupted amounts are included at their magnitude and flagged.",
    source=TRANSACTIONS, aggregation=SUM, column="amount_inr", time_column=TXN_TIME,
    synonyms=("total transaction value", "total transaction amount", "transaction value",
              "transaction amount", "total value", "total amount", "revenue", "sales",
              "gmv", "payment value", "amount", "value"),
))
_add(Metric(
    name="average_transaction_value", label="Average transaction value",
    definition="Mean transaction value in INR.",
    formula="SUM(amount_inr) / COUNT(*)",
    grain="transaction", coverage_basis=FULL, chart_hint="kpi_card", unit="INR",
    source=TRANSACTIONS, aggregation=MEAN, column="amount_inr", time_column=TXN_TIME,
    synonyms=("average transaction value", "average transaction amount",
              "avg transaction value", "avg transaction amount", "mean transaction value",
              "average value", "average amount", "avg value", "average ticket size",
              "ticket size", "average ticket"),
))

# --- outcome rates ---------------------------------------------------------
for _status, _label, _syn in [
    ("SUCCESS", "Success rate", ("success rate", "successful rate", "success ratio",
                                 "success percentage", "successful transaction rate")),
    ("FAILED", "Failure rate", ("failure rate", "failed rate", "failed transaction rate",
                                "failure ratio", "decline rate")),
    ("PENDING", "Pending rate", ("pending rate", "pending transaction rate")),
]:
    _add(Metric(
        name=f"{_status.lower()}_transaction_rate", label=_label,
        definition=f"Share of transactions with canonical status {_status}.",
        formula=f"COUNT(*) FILTER (WHERE status_canonical = '{_status}') / COUNT(*)",
        grain="transaction", coverage_basis=FULL, chart_hint="kpi_card", unit="percent",
        higher_is_worse=_status != "SUCCESS",
        source=TRANSACTIONS, aggregation=SHARE, predicate=("status_canonical", _status),
        time_column=TXN_TIME, synonyms=_syn,
    ))

# --------------------------------------------------------------------------
# Disputes
# --------------------------------------------------------------------------
_add(Metric(
    name="chargeback_count", label="Chargebacks",
    definition="Count of unique dispute complaints.",
    formula="COUNT(DISTINCT complaint_key)",
    grain="complaint", coverage_basis=ALL_CB, chart_hint="kpi_card", higher_is_worse=True,
    notes="Over time, complaints are placed on the date the customer reported them.",
    source=CHARGEBACKS, aggregation=COUNT, time_column=REPORTED_TIME,
    synonyms=("chargeback count", "number of chargebacks", "count of chargebacks",
              "chargebacks count", "dispute count", "number of disputes", "count of disputes",
              "complaint count", "number of complaints", "chargebacks", "chargeback",
              "disputes", "complaints"),
))
_add(Metric(
    name="chargeback_amount", label="Disputed amount",
    definition="Sum of disputed value in INR.",
    formula="SUM(disputed_amount_inr)",
    grain="complaint", coverage_basis="complaints with a parseable disputed amount",
    chart_hint="kpi_card", unit="INR", higher_is_worse=True,
    source=CHARGEBACKS, aggregation=SUM, column="disputed_amount_inr",
    time_column=REPORTED_TIME,
    synonyms=("disputed amount", "disputed value", "dispute amount", "dispute value",
              "chargeback amount", "chargeback value", "amount disputed", "value disputed",
              "disputed money", "high value chargebacks", "high value chargeback",
              "high value disputes", "high value dispute"),
))
_add(Metric(
    name="chargeback_to_transaction_ratio", label="Chargeback rate",
    definition=(
        "Share of transactions that attracted at least one dispute. Counts DISTINCT "
        "disputed transactions, not complaints: some transactions carry more than one "
        "complaint, and counting complaints would push the ratio above 1 for some "
        "merchants and break every statistical test built on it."
    ),
    formula="COUNT(DISTINCT disputed txn_key) / COUNT(DISTINCT txn_key)",
    grain="transaction", coverage_basis=LINKED_CB, chart_hint="kpi_card",
    unit="percent", higher_is_worse=True,
    source=TRANSACTIONS, aggregation=DISPUTED_RATIO, time_column=TXN_TIME,
    synonyms=("chargeback to transaction ratio", "chargeback to transaction rate",
              "chargeback ratio", "chargeback rate", "dispute rate", "dispute ratio",
              "chargeback percentage", "dispute percentage"),
))
_add(Metric(
    name="complaints_per_transaction", label="Complaints per transaction",
    definition="Complaint volume relative to transaction volume. A rate, not a proportion.",
    formula="COUNT(linked complaint_key) / COUNT(DISTINCT txn_key)",
    grain="transaction", coverage_basis=LINKED_CB, chart_hint="bar", unit="ratio",
    higher_is_worse=True,
    notes="Reported beside the chargeback rate to make repeat complaints visible.",
    source=TRANSACTIONS, aggregation=COMPLAINTS_RATIO, time_column=None,
    synonyms=("complaints per transaction",),
))
_add(Metric(
    name="average_dispute_reporting_delay", label="Average reporting delay",
    definition=(
        "Days between the dispute's own transaction timestamp and the customer's report, "
        "both read from the chargeback file."
    ),
    formula="AVG(reported_timestamp - transaction_timestamp)",
    grain="complaint", coverage_basis="complaints with both timestamps parseable",
    chart_hint="histogram", unit="days", higher_is_worse=True,
    notes=(
        "Deliberately not measured against the linked transaction's timestamp: the "
        "chargeback file's denormalized fields do not describe the transaction they point "
        "at, and doing so produces impossible negative delays."
    ),
    source=CHARGEBACKS, aggregation=MEAN, column="reporting_delay_days",
    time_column=REPORTED_TIME,
    synonyms=("average reporting delay", "reporting delay", "dispute delay", "time to report",
              "days to report", "reporting lag"),
))
_add(Metric(
    name="disputes_reported_after_7_days", label="Disputes reported after N days",
    definition="Count of complaints whose reporting delay exceeds a threshold (default 7 days).",
    formula="COUNT(*) FILTER (WHERE reporting_delay_days > N)",
    grain="complaint", coverage_basis="complaints with a computable delay",
    chart_hint="table", higher_is_worse=True,
    source=CHARGEBACKS, aggregation=THRESHOLD_COUNT, column="reporting_delay_days",
    threshold=7.0, time_column=REPORTED_TIME,
    synonyms=("reported after", "late disputes", "late reported disputes", "reported late",
              "long delays", "long delay", "delayed disputes", "late complaints"),
))

# --------------------------------------------------------------------------
# Customers and KYC
# --------------------------------------------------------------------------
_add(Metric(
    name="customer_count", label="Customers",
    definition="Count of resolved customer identities.",
    formula="COUNT(DISTINCT user_key)",
    grain="customer", coverage_basis=RESOLVED_CUSTOMERS, chart_hint="kpi_card",
    source=CUSTOMERS, aggregation=COUNT, time_column=None,
    synonyms=("customer count", "number of customers", "how many customers", "user count",
              "number of users", "how many users"),
))
_add(Metric(
    name="kyc_completion_rate", label="KYC completion rate",
    definition="Share of resolved customers whose KYC status is VERIFIED.",
    formula="COUNT(*) FILTER (WHERE kyc_status_canonical = 'VERIFIED') / COUNT(*)",
    grain="customer", coverage_basis=RESOLVED_CUSTOMERS, chart_hint="kpi_card",
    unit="percent",
    source=CUSTOMERS, aggregation=SHARE, predicate=("kyc_status_canonical", "VERIFIED"),
    time_column=None,
    synonyms=("kyc completion rate", "kyc completion", "kyc verified rate",
              "kyc verification rate", "verified kyc"),
))
_add(Metric(
    name="kyc_rejection_rate", label="KYC rejection rate",
    definition="Share of resolved customers whose KYC status is REJECTED.",
    formula="COUNT(*) FILTER (WHERE kyc_status_canonical = 'REJECTED') / COUNT(*)",
    grain="customer", coverage_basis=RESOLVED_CUSTOMERS, chart_hint="kpi_card",
    unit="percent", higher_is_worse=True,
    source=CUSTOMERS, aggregation=SHARE, predicate=("kyc_status_canonical", "REJECTED"),
    time_column=None,
    synonyms=("kyc rejection rate", "kyc rejection", "kyc rejected", "rejected kyc"),
))

# --------------------------------------------------------------------------
# Data quality as first-class metrics
# --------------------------------------------------------------------------
_add(Metric(
    name="utr_missing_rate", label="Missing UTR rate",
    definition="Share of transactions with no bank reference number.",
    formula="COUNT(*) FILTER (WHERE utr_missing) / COUNT(*)",
    grain="transaction", coverage_basis=FULL, chart_hint="kpi_card",
    unit="percent", higher_is_worse=True,
    source=TRANSACTIONS, aggregation=SHARE, predicate=("utr_missing", True),
    time_column=TXN_TIME,
    synonyms=("missing utr rate", "missing utrs", "missing utr", "utr missing",
              "invalid utr", "utr"),
))
_add(Metric(
    name="merchant_master_coverage", label="Merchant master coverage",
    definition="Share of transactions whose merchant resolves to a master record.",
    formula="COUNT(*) FILTER (WHERE NOT merchant_unresolved) / COUNT(*)",
    grain="transaction", coverage_basis=FULL, chart_hint="kpi_card", unit="percent",
    source=TRANSACTIONS, aggregation=SHARE, predicate=("merchant_resolved", True),
    time_column=TXN_TIME,
    synonyms=("merchant master coverage", "merchant coverage", "merchant information",
              "merchant info", "merchant record", "merchant records", "merchant details"),
))
_add(Metric(
    name="kyc_coverage", label="KYC coverage",
    definition="Share of transactions whose customer resolves to a KYC record.",
    formula="COUNT(*) FILTER (WHERE NOT user_unresolved) / COUNT(*)",
    grain="transaction", coverage_basis=FULL, chart_hint="kpi_card", unit="percent",
    source=TRANSACTIONS, aggregation=SHARE, predicate=("kyc_resolved", True),
    time_column=TXN_TIME,
    synonyms=("kyc coverage", "kyc information", "kyc info", "kyc record", "kyc records",
              "kyc details"),
))
_add(Metric(
    name="identity_ambiguity_rate", label="Ambiguous identity rate",
    definition="Share of resolved identities whose ID is shared by several distinct entities.",
    formula="COUNT(*) FILTER (WHERE identity_ambiguous) / COUNT(*)",
    grain="entity", coverage_basis="resolved identities in each dimension",
    chart_hint="kpi_card", unit="percent", higher_is_worse=True,
    source=CUSTOMERS, aggregation=SHARE, predicate=("identity_ambiguous", True),
    time_column=None,
    synonyms=("identity collisions", "identity collision", "ambiguous identities",
              "ambiguous identity", "shared ids", "shared id", "colliding ids"),
))

# --------------------------------------------------------------------------
# Legacy compound entries, kept so earlier builds and docs still resolve
# --------------------------------------------------------------------------
_add(Metric(
    name="dispute_rate_by_merchant_category", label="Dispute rate by merchant category",
    definition="Chargeback rate grouped by the merchant master's canonical category.",
    formula="disputed transactions / transactions, grouped by merchant category",
    grain="merchant category", coverage_basis=MERCHANT_COVERAGE,
    chart_hint="bar", unit="percent", higher_is_worse=True,
    notes="UNKNOWN is shown as its own bar; it is the largest single group.",
    dimensions=("merchant_category",),
    source=TRANSACTIONS, aggregation=DISPUTED_RATIO, time_column=TXN_TIME,
))
_add(Metric(
    name="merchant_dispute_exposure", label="Merchant dispute exposure",
    definition=(
        "Absolute disputed value per merchant — operational exposure, not risk propensity."
    ),
    formula="SUM(disputed_amount_inr) grouped by attributed merchant",
    grain="merchant", coverage_basis="all transacting merchants",
    chart_hint="horizontal_bar", unit="INR", higher_is_worse=True,
    notes=(
        "Ranked on absolute exposure because the overdispersion test in the hypothesis "
        "register found no detectable merchant-level dispute propensity. A merchant with "
        "many disputes is a real operational priority; it is not evidence of a riskier merchant."
    ),
    dimensions=("merchant",),
    source=CHARGEBACKS, aggregation=SUM, column="disputed_amount_inr",
    time_column=REPORTED_TIME,
))

# --------------------------------------------------------------------------
# Risk Indicator Score — explainable review priority, NOT a fraud score
# --------------------------------------------------------------------------
_add(Metric(
    name="risk_indicator_score", label="Risk Indicator Score",
    definition=(
        "Explainable 0-100 review-priority score: the sum of four 25-point components "
        "built only from observed conditions (dispute history, disputed value, KYC or "
        "merchant status, identity ambiguity). It is NOT a fraud probability — the data "
        "contains no fraud label, so no score can be validated as a fraud predictor."
    ),
    formula=("customer: 25*min(disputed_txns,2)/2 + 25*pct_rank(disputed_amount) "
             "+ KYC points + 25*identity_ambiguous; merchant: 25*min(disputed_txns,3)/3 "
             "+ 25*pct_rank(disputed_amount) + status points + 25*identity_ambiguous"),
    grain="customer or merchant", coverage_basis="all transacting customers or merchants",
    chart_hint="horizontal_bar", unit="points", higher_is_worse=True,
    notes="Component weights are equal by design; see src/analytics/risk_indicators.py.",
    dimensions=("user", "merchant"),
    source=PRECOMPUTED, aggregation=PRECOMPUTED_AGG, time_column=None,
    synonyms=("risk indicator score", "risk indicator scores", "risk indicators",
              "risk indicator", "risk score", "risk scores", "review priority",
              "high risk", "riskiest", "most risky", "risky"),
))

# --------------------------------------------------------------------------
LEGACY_DIMENSIONS = {
    "date", "week", "month", "quarter", "day_of_week", "status_canonical",
    "merchant_category_canonical", "merchant_status_canonical", "business_type_canonical",
    "kyc_status_canonical", "risk_segment_canonical", "state_clean", "city_clean",
    "reason_code_canonical", "severity_canonical", "resolution_status_canonical",
    "channel_canonical", "complaint_theme", "hour", "occupation_clean",
}

ALLOWED_DIMENSIONS = sorted(set(DIMENSIONS) | LEGACY_DIMENSIONS)

ALLOWED_CHARTS = ("kpi_card", "line", "bar", "horizontal_bar", "stacked_bar",
                  "histogram", "scatter", "table", "network")


def get(name: str) -> Metric:
    if name not in METRICS:
        raise KeyError(
            f"unknown metric {name!r}. Registered: {', '.join(sorted(METRICS))}"
        )
    return METRICS[name]


def get_dimension(name: str) -> Dimension:
    if name not in DIMENSIONS:
        raise KeyError(
            f"unknown dimension {name!r}. Registered: {', '.join(sorted(DIMENSIONS))}"
        )
    return DIMENSIONS[name]


def registry_frame():
    """The metric registry as a table — rendered in docs and in the dashboard."""
    import pandas as pd

    return pd.DataFrame([
        {
            "metric": m.name,
            "label": m.label,
            "unit": m.unit,
            "grain": m.grain,
            "definition": m.definition,
            "formula": m.formula,
            "coverage_basis": m.coverage_basis,
            "chart": m.chart_hint,
            "higher_is_worse": m.higher_is_worse,
            "source": m.source,
            "aggregation": m.aggregation,
            "time_column": m.time_column or "—",
            "trendable": m.trendable,
            "notes": m.notes,
        }
        for m in METRICS.values()
    ])


def dimension_frame():
    """The dimension registry as a table."""
    import pandas as pd

    return pd.DataFrame([
        {
            "dimension": d.name,
            "label": d.label,
            "sources": ", ".join(sorted(d.columns)),
            "entity": d.entity,
            "coverage_metric": d.coverage_metric or "—",
            "synonyms": ", ".join(d.synonyms),
        }
        for d in DIMENSIONS.values()
    ])
