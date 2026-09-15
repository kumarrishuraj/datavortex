"""Chart selection.

A small, explicit rule set — deliberately not left to a language model, and not
keyed to particular questions. The rules depend only on the *shape* of the
intent and its result:

    detailed records                     -> table
    anything grouped by a time grain     -> line
    a continuous distribution            -> histogram
    an entity ranking, or many categories -> horizontal bar
    a small set of categories            -> bar
    a single figure                      -> KPI card

Time comes first on purpose: "chargebacks by severity over time" is still a
trend, and a trend is read as a line. A second measure on a different scale is
never added to the same chart — there are no dual-axis charts in this project.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.analytics import semantic

LINE = "line"
BAR = "bar"
HORIZONTAL_BAR = "horizontal_bar"
TABLE = "table"
KPI_CARD = "kpi_card"
HISTOGRAM = "histogram"

# Past this many categories, labels need a horizontal layout to stay readable.
MAX_VERTICAL_BAR_CATEGORIES = 8


@dataclass(frozen=True)
class ChartChoice:
    chart: str
    reason: str


def select_chart(*, kind: str, time_grain: str | None = None,
                 dimension: str | None = None, n_categories: int | None = None,
                 records: bool = False, continuous: bool = False) -> ChartChoice:
    """Pick the chart for an intent (and, when available, its result shape)."""
    if records:
        return ChartChoice(TABLE, "the answer is a set of detailed records")
    if time_grain:
        return ChartChoice(LINE, f"the answer is a time series grouped by {time_grain}")
    if continuous:
        return ChartChoice(HISTOGRAM, "the answer is the distribution of a continuous measure")
    if dimension:
        dim = semantic.DIMENSIONS.get(dimension)
        if dim is not None and dim.entity:
            return ChartChoice(HORIZONTAL_BAR,
                               f"the answer ranks individual {dim.label.lower()}s")
        if kind == "ranking":
            return ChartChoice(HORIZONTAL_BAR, "the answer is a ranking")
        if n_categories is not None and n_categories > MAX_VERTICAL_BAR_CATEGORIES:
            return ChartChoice(HORIZONTAL_BAR,
                               f"{n_categories} categories need horizontal labels")
        return ChartChoice(BAR, "the answer compares a small set of categories")
    return ChartChoice(KPI_CARD, "the answer is a single figure")


assert all(c in semantic.ALLOWED_CHARTS
           for c in (LINE, BAR, HORIZONTAL_BAR, TABLE, KPI_CARD, HISTOGRAM))
