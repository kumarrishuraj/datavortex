"""Semantic query engine.

The only place the AI Investigator obtains a number. A `QuerySpec` names a
metric and optionally a dimension, a time grain, a series split, a time window
and filters — every one of them a name from `semantic.py`. The engine executes
the metric's registered computation spec in pandas and returns the result with
its denominator, coverage, exclusions and a reconciliation check.

Nothing here parses language and nothing here is generated. The planner decides
*what* to ask; this module decides nothing about meaning and only computes.

Guarantees enforced on every result:
  * additive metrics (count, sum, threshold count) must reconcile: the grouped
    values sum exactly to the ungrouped total, or the query raises
  * complaints are attributed to customers, merchants and categories through
    `txn_id` only; unlinked complaints are excluded from those breakdowns and the
    exclusion is reported, never silent
  * hour-of-day breakdowns exclude date-only timestamps, which would otherwise
    pile up at midnight
  * UNKNOWN dimension members are kept, and their share is reported as coverage
  * daily trends are gap-filled only inside the data's observed window, so a
    requested window larger than the data never shows invented zeros
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics import semantic
from src.analytics.kpis import disputed_transaction_keys
from src.config import UNKNOWN_KEY

CATEGORY_UNKNOWN = semantic.UNKNOWN_CATEGORY


class QueryError(ValueError):
    """Raised when a query is not computable as specified."""


@dataclass
class QuerySpec:
    metric: str
    dimension: str | None = None
    time_grain: str | None = None
    series: str | None = None
    series_values: tuple = ()
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None          # exclusive
    filters: dict = field(default_factory=dict)
    threshold: float | None = None
    limit: int | None = None
    ascending: bool = False
    natural_order: bool = False


@dataclass
class QueryResult:
    spec: QuerySpec
    frame: pd.DataFrame
    total_value: float
    total_numerator: float | None
    total_denominator: float | None
    rows_analysed: int
    population: int
    unit: str
    coverage_pct: float | None = None
    excluded: dict = field(default_factory=dict)
    reconciled: bool | None = None
    data_start: pd.Timestamp | None = None
    data_end: pd.Timestamp | None = None
    groups_before_limit: int = 0

    @property
    def empty(self) -> bool:
        return self.rows_analysed == 0 or self.frame.empty


@dataclass
class Window:
    start: pd.Timestamp | None
    end: pd.Timestamp | None                 # exclusive
    label: str
    notes: list = field(default_factory=list)
    requested: bool = False

    def overlaps(self, data_start: pd.Timestamp, data_end: pd.Timestamp) -> bool:
        if self.start is None and self.end is None:
            return True
        lo = self.start if self.start is not None else data_start
        hi = self.end if self.end is not None else data_end + pd.Timedelta(days=1)
        return lo <= data_end and hi > data_start


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
def prepare_frames(star: dict) -> dict[str, pd.DataFrame]:
    """Enrich the star schema into the three sources the registry names."""
    dim_m = star["dim_merchants"]
    dim_m = dim_m[dim_m["merchant_key"] != UNKNOWN_KEY].set_index("merchant_key")
    dim_u = star["dim_users"]
    dim_u = dim_u[dim_u["user_key"] != UNKNOWN_KEY]
    dim_u_idx = dim_u.set_index("user_key")

    cb = star["fact_chargebacks"].copy()
    tx = star["fact_transactions"].copy()

    tx["is_disputed"] = tx["txn_key"].isin(disputed_transaction_keys(cb))
    tx["category"] = tx["merchant_id_normalized"].map(
        dim_m["merchant_category_canonical"]).fillna(CATEGORY_UNKNOWN)
    tx["state"] = tx["merchant_id_normalized"].map(dim_m["state_clean"]).fillna(UNKNOWN_KEY)
    tx["merchant_status"] = tx["merchant_id_normalized"].map(
        dim_m["merchant_status_canonical"]).fillna(UNKNOWN_KEY)
    tx["kyc_status"] = tx["user_id_normalized"].map(
        dim_u_idx["kyc_status_canonical"]).fillna(UNKNOWN_KEY)
    tx["risk_segment"] = tx["user_id_normalized"].map(
        dim_u_idx["risk_segment_canonical"]).fillna(UNKNOWN_KEY)
    tx["merchant_resolved"] = ~tx["merchant_unresolved"].astype(bool)
    tx["kyc_resolved"] = ~tx["user_unresolved"].astype(bool)
    tx["timestamp_time_known"] = tx["timestamp_time_known"].astype(bool)

    # Complaints inherit entity attributes through the linked transaction only.
    linked = ~cb["txn_unlinked"].astype(bool)
    merchant = cb["attributed_merchant_id"].where(linked)
    user = cb["attributed_user_id"].where(linked)
    cb["attributed_merchant_id"] = merchant
    cb["attributed_user_id"] = user
    cb["category"] = merchant.map(dim_m["merchant_category_canonical"]).where(
        merchant.isna(), merchant.map(dim_m["merchant_category_canonical"]).fillna(CATEGORY_UNKNOWN))
    cb["category"] = cb["category"].where(linked)
    cb["state"] = merchant.map(dim_m["state_clean"]).fillna(UNKNOWN_KEY).where(linked)
    cb["kyc_status"] = user.map(dim_u_idx["kyc_status_canonical"]).fillna(UNKNOWN_KEY).where(linked)

    return {
        semantic.TRANSACTIONS: tx,
        semantic.CHARGEBACKS: cb,
        semantic.CUSTOMERS: dim_u.reset_index(drop=True),
    }


def data_bounds(frames: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = pd.to_datetime(frames[semantic.TRANSACTIONS][semantic.TXN_TIME]).dropna()
    return ts.min(), ts.max()


# --------------------------------------------------------------------------
# Time windows
# --------------------------------------------------------------------------
def resolve_window(window: dict | None, data_start: pd.Timestamp,
                   data_end: pd.Timestamp) -> Window:
    """Turn a symbolic window from the planner into dates.

    Relative windows ("this quarter", "last 6 months") are anchored to the last
    date present in the data, not to today's calendar date. That is stated in the
    returned notes, so the anchoring is never a silent assumption.
    """
    first, last = data_start.normalize(), data_end.normalize()
    observed = f"{first:%Y-%m-%d} to {last:%Y-%m-%d}"
    if not window:
        return Window(None, None, f"the full observed window, {observed}")

    kind = window.get("type")
    if kind in ("latest_quarter", "previous_quarter"):
        q = last.to_period("Q")
        if kind == "previous_quarter":
            q = q - 1
        label = f"{q.year}-Q{q.quarter}"
        reading = ("the latest quarter present in the data" if kind == "latest_quarter"
                   else "the quarter before the latest one in the data")
        return Window(q.start_time, (q + 1).start_time, label,
                      [f"'{window.get('phrase', 'this quarter')}' is read as {reading} "
                       f"({label}); the data covers {observed}."], requested=True)
    if kind == "quarter":
        year = int(window.get("year") or last.year)
        q = pd.Period(f"{year}Q{int(window['quarter'])}", freq="Q")
        return Window(q.start_time, (q + 1).start_time, f"{q.year}-Q{q.quarter}",
                      [], requested=True)
    if kind == "month":
        year = int(window.get("year") or last.year)
        m = pd.Period(f"{year}-{int(window['month']):02d}", freq="M")
        return Window(m.start_time, (m + 1).start_time, f"{m.start_time:%B %Y}", [],
                      requested=True)
    if kind == "last_n":
        n, unit = int(window["n"]), window["unit"]
        end = last + pd.Timedelta(days=1)
        start = end - (pd.DateOffset(months=n) if unit == "months"
                       else pd.Timedelta(days=n * (7 if unit == "weeks" else 1)))
        notes = [f"'Last {n} {unit}' is anchored to the last date in the data "
                 f"({last:%Y-%m-%d}), not to today's date."]
        if start < first:
            notes.append(f"The data begins on {first:%Y-%m-%d}, so only {observed} falls "
                         f"inside the requested {n}-{unit[:-1]} window.")
        return Window(start, end, f"the last {n} {unit} of data", notes, requested=True)
    raise QueryError(f"unrecognised time window {window!r}")


def floor_period(ts: pd.Series, grain: str) -> pd.Series:
    ts = pd.to_datetime(ts)
    if grain == "day":
        return ts.dt.floor("D")
    if grain == "week":
        return ts.dt.to_period("W-SUN").dt.start_time
    if grain == "month":
        return ts.dt.to_period("M").dt.start_time
    if grain == "quarter":
        return ts.dt.to_period("Q").dt.start_time
    raise QueryError(f"unsupported time grain {grain!r}")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def run(spec: QuerySpec, frames: dict) -> QueryResult:
    metric = semantic.get(spec.metric)
    if metric.aggregation == semantic.PRECOMPUTED_AGG:
        raise QueryError(f"{metric.label} is precomputed per entity in the analytics layer "
                         "and is read from there, not aggregated by the query engine.")
    if metric.aggregation == semantic.COMPLAINTS_RATIO:
        return _complaints_ratio(spec, frames, metric)

    src = frames[metric.source]
    population = len(src)
    excluded: dict[str, int] = {}
    df = src

    data_start = data_end = None
    if metric.time_column is not None:
        ts_all = pd.to_datetime(src[metric.time_column])
        data_start, data_end = ts_all.min(), ts_all.max()

    if spec.start is not None or spec.end is not None:
        if metric.time_column is None:
            raise QueryError(f"{metric.label} has no date, so it cannot be restricted "
                             "to a time period.")
        ts = pd.to_datetime(df[metric.time_column])
        mask = ts.notna()
        if spec.start is not None:
            mask &= ts >= spec.start
        if spec.end is not None:
            mask &= ts < spec.end
        df = df[mask]

    for column, values in (spec.filters or {}).items():
        if column not in df.columns:
            raise QueryError(f"filter column {column!r} does not exist on {metric.source}")
        df = df[df[column].isin(list(values))]

    keys: list[str] = []
    if spec.time_grain:
        if spec.time_grain not in semantic.TIME_GRAINS:
            raise QueryError(f"unsupported time grain {spec.time_grain!r}")
        if not metric.trendable:
            raise QueryError(f"{metric.label} cannot be shown as a trend over time.")
        before = len(df)
        df = df[pd.to_datetime(df[metric.time_column]).notna()]
        if before - len(df):
            excluded["records without a usable date"] = before - len(df)
        df = df.assign(period=floor_period(df[metric.time_column], spec.time_grain))
        keys.append("period")

    dim = None
    if spec.dimension:
        dim = semantic.get_dimension(spec.dimension)
        column = dim.columns.get(metric.source)
        if column is None:
            raise QueryError(f"{metric.label} cannot be broken down by {dim.label.lower()}.")
        if dim.requires_clock_time:
            before = len(df)
            df = df[df["timestamp_time_known"]]
            if before - len(df):
                excluded["date-only timestamps with no clock time"] = before - len(df)
        before = len(df)
        df = df[df[column].notna()]
        if before - len(df):
            label = ("complaints not linked to any transaction"
                     if metric.source == semantic.CHARGEBACKS
                     else "records with no value for this dimension")
            excluded[label] = before - len(df)
        df = df.assign(dimension=df[column])
        keys.append("dimension")

    if spec.series:
        sdim = semantic.get_dimension(spec.series)
        scol = sdim.columns.get(metric.source)
        if scol is None:
            raise QueryError(f"{metric.label} cannot be split by {sdim.label.lower()}.")
        df = df.assign(series=df[scol])
        if spec.series_values:
            df = df[df["series"].isin(list(spec.series_values))]
        keys.append("series")

    rows = len(df)
    total = _scalar(df, metric, spec)
    frame = _grouped(df, keys, metric, spec) if keys else pd.DataFrame([total])

    reconciled = None
    if keys and metric.additive:
        grouped_sum = float(frame["value"].sum())
        reconciled = bool(np.isclose(grouped_sum, float(total["value"]),
                                     rtol=1e-9, atol=1e-6))
        if not reconciled:
            raise QueryError(f"reconciliation failed for {metric.name}: groups sum to "
                             f"{grouped_sum} but the total is {total['value']}")

    if spec.time_grain and not frame.empty and data_start is not None:
        frame = _complete_periods(frame, spec, metric, data_start, data_end)

    groups = len(frame)
    frame = _order(frame, spec, dim, keys)
    if spec.limit and dim is not None and not spec.time_grain:
        frame = frame.head(spec.limit)

    coverage = None
    if dim is not None and dim.coverage_metric and rows:
        known = ~df["dimension"].astype(str).isin(dim.unknown_values)
        coverage = round(100.0 * float(known.mean()), 2)

    return QueryResult(
        spec=spec, frame=frame.reset_index(drop=True),
        total_value=float(total["value"]) if total["value"] == total["value"] else float("nan"),
        total_numerator=total.get("numerator"), total_denominator=total.get("denominator"),
        rows_analysed=rows, population=population, unit=metric.unit,
        coverage_pct=coverage, excluded=excluded, reconciled=reconciled,
        data_start=data_start, data_end=data_end, groups_before_limit=groups,
    )


def _scalar(df: pd.DataFrame, metric: semantic.Metric, spec: QuerySpec) -> dict:
    agg = metric.aggregation
    n = len(df)
    if agg == semantic.COUNT:
        return {"value": float(n), "numerator": float(n), "denominator": float(n)}
    if agg == semantic.SUM:
        return {"value": float(df[metric.column].sum()), "denominator": float(n)}
    if agg == semantic.MEAN:
        valid = df[metric.column].dropna()
        return {"value": float(valid.mean()) if len(valid) else float("nan"),
                "denominator": float(len(valid))}
    if agg == semantic.SHARE:
        col, val = metric.predicate
        hits = int((df[col] == val).sum())
        return {"value": metric.scale * hits / n if n else float("nan"),
                "numerator": float(hits), "denominator": float(n)}
    if agg == semantic.DISPUTED_RATIO:
        hits = int(df["is_disputed"].sum())
        return {"value": metric.scale * hits / n if n else float("nan"),
                "numerator": float(hits), "denominator": float(n)}
    if agg == semantic.THRESHOLD_COUNT:
        thr = spec.threshold if spec.threshold is not None else metric.threshold
        valid = df[metric.column].notna()
        hits = int((valid & (df[metric.column] > thr)).sum())
        return {"value": float(hits), "numerator": float(hits),
                "denominator": float(valid.sum())}
    raise QueryError(f"aggregation {agg!r} is not supported for {metric.name}")


def _grouped(df: pd.DataFrame, keys: list[str], metric: semantic.Metric,
             spec: QuerySpec) -> pd.DataFrame:
    agg = metric.aggregation
    group = dict(dropna=False, observed=True, sort=True)
    if agg == semantic.COUNT:
        out = df.groupby(keys, **group).size().rename("value").reset_index()
        out["value"] = out["value"].astype(float)
        out["denominator"] = out["value"]
        return out
    if agg == semantic.SUM:
        g = df.groupby(keys, **group)[metric.column]
        out = g.sum().rename("value").reset_index()
        out["denominator"] = g.size().to_numpy().astype(float)
        return out
    if agg == semantic.MEAN:
        g = df.groupby(keys, **group)[metric.column]
        out = g.mean().rename("value").reset_index()
        out["denominator"] = g.count().to_numpy().astype(float)
        return out
    if agg in (semantic.SHARE, semantic.DISPUTED_RATIO, semantic.THRESHOLD_COUNT):
        if agg == semantic.SHARE:
            col, val = metric.predicate
            hit = df[col] == val
            base = pd.Series(True, index=df.index)
        elif agg == semantic.DISPUTED_RATIO:
            hit = df["is_disputed"].astype(bool)
            base = pd.Series(True, index=df.index)
        else:
            thr = spec.threshold if spec.threshold is not None else metric.threshold
            base = df[metric.column].notna()
            hit = base & (df[metric.column] > thr)
        tmp = df[keys].assign(_hit=hit.astype(int), _base=base.astype(int))
        out = tmp.groupby(keys, **group)[["_hit", "_base"]].sum().reset_index()
        out = out.rename(columns={"_hit": "numerator", "_base": "denominator"})
        out["numerator"] = out["numerator"].astype(float)
        out["denominator"] = out["denominator"].astype(float)
        if agg == semantic.THRESHOLD_COUNT:
            out["value"] = out["numerator"]
        else:
            out["value"] = np.where(out["denominator"] > 0,
                                    metric.scale * out["numerator"] / out["denominator"], np.nan)
        return out
    raise QueryError(f"aggregation {agg!r} is not supported for {metric.name}")


def _complete_periods(frame: pd.DataFrame, spec: QuerySpec, metric: semantic.Metric,
                      data_start: pd.Timestamp, data_end: pd.Timestamp) -> pd.DataFrame:
    """Gap-fill periods, but only inside the data's own observed window."""
    lo = data_start if spec.start is None else max(spec.start, data_start)
    hi = data_end if spec.end is None else min(spec.end - pd.Timedelta(seconds=1), data_end)
    if lo > hi:
        return frame
    periods = pd.Series(pd.date_range(lo.normalize(), hi.normalize(), freq="D"))
    periods = floor_period(periods, spec.time_grain).drop_duplicates().sort_values()
    fill = 0.0 if metric.additive else np.nan

    if "series" in frame.columns:
        values = spec.series_values or tuple(sorted(frame["series"].dropna().unique()))
        index = pd.MultiIndex.from_product([periods, values], names=["period", "series"])
        out = frame.set_index(["period", "series"]).reindex(index)
    else:
        out = frame.set_index("period").reindex(pd.Index(periods, name="period"))
    out["value"] = out["value"].fillna(fill)
    for col in ("numerator", "denominator"):
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    return out.reset_index()


def _order(frame: pd.DataFrame, spec: QuerySpec, dim, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "period" in keys:
        return frame.sort_values([k for k in ("period", "series") if k in frame.columns])
    if dim is not None:
        if spec.natural_order and dim.order:
            rank = {v: i for i, v in enumerate(dim.order)}
            return frame.assign(_r=frame["dimension"].map(rank).fillna(len(rank))) \
                        .sort_values(["_r", "dimension"]).drop(columns="_r")
        return frame.sort_values(["value", "dimension"],
                                 ascending=[spec.ascending, True], na_position="last")
    return frame


def _complaints_ratio(spec: QuerySpec, frames: dict, metric: semantic.Metric) -> QueryResult:
    if spec.dimension or spec.time_grain or spec.series:
        raise QueryError(f"{metric.label} is only available as a single overall figure.")
    tx = frames[semantic.TRANSACTIONS]
    cb = frames[semantic.CHARGEBACKS]
    linked = int((~cb["txn_unlinked"].astype(bool)).sum())
    n = len(tx)
    value = linked / n if n else float("nan")
    frame = pd.DataFrame([{"value": value, "numerator": float(linked), "denominator": float(n)}])
    return QueryResult(spec=spec, frame=frame, total_value=value,
                       total_numerator=float(linked), total_denominator=float(n),
                       rows_analysed=n, population=n, unit=metric.unit)
