"""Controlled timestamp parsing.

A blind `pd.to_datetime(dayfirst=...)` is wrong here: the bundle mixes two
conflicting conventions in the same column. The forensic audit (section C)
resolved them by separator, testing all seven date fields:

    d/m/Y   -> DD/MM/YYYY   18,385 rows prove part1>12;  0 rows prove part2>12
    d-m-Y   -> MM-DD-YYYY    9,133 rows prove part2>12;  0 rows prove part1>12
    Y-m-d / Y/m/d -> ISO     0 rows with month>12
    d-Mon-Y -> textual month, unambiguous
    8-10 bare digits -> Unix epoch (may be NEGATIVE in date_of_birth: pre-1970)

Zero counterexamples in either direction, across every field. That makes the
rule deterministic rather than a guess, which is why we dispatch on shape and
use explicit format strings instead of letting pandas infer.
"""
from __future__ import annotations

import re

import pandas as pd

ISO = "ISO"
SLASH_DMY = "SLASH_DMY"
HYPH_MDY = "HYPH_MDY"
DMON_Y = "DMON_Y"
UNIX = "UNIX"
BLANK = "BLANK"
UNPARSEABLE = "UNPARSEABLE"

_NULLISH = {"", "nan", "none", "null", "na", "n/a", "-"}

_RE_UNIX = re.compile(r"^-?\d{8,10}$")
_RE_ISO = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_RE_DMON = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}")
_RE_SLASH = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}")
_RE_HYPH = re.compile(r"^\d{1,2}-\d{1,2}-\d{4}")

# Time-part variants seen in the data: none, 24-hour H:M:S, 12-hour with AM/PM
# (both with and without seconds), and bare H:M. The 12-hour-with-seconds form
# ('03-10-2026 09:27:31 PM') covers ~3,000 transaction rows on its own.
_TIME_SUFFIXES = ["", " %H:%M:%S", " %I:%M:%S %p", " %I:%M %p", " %H:%M"]

_FORMATS = {
    ISO: ["%Y-%m-%d" + t for t in _TIME_SUFFIXES],
    SLASH_DMY: ["%d/%m/%Y" + t for t in _TIME_SUFFIXES],
    HYPH_MDY: ["%m-%d-%Y" + t for t in _TIME_SUFFIXES],
    DMON_Y: ["%d-%b-%Y" + t for t in _TIME_SUFFIXES],
}


def _classify(raw: str) -> str:
    """Pick the parsing family from the value's shape alone."""
    if _RE_UNIX.fullmatch(raw):
        return UNIX
    if _RE_ISO.match(raw):
        return ISO
    if _RE_DMON.match(raw):
        return DMON_Y
    if _RE_SLASH.match(raw):
        return SLASH_DMY
    if _RE_HYPH.match(raw):
        return HYPH_MDY
    return UNPARSEABLE


def parse_timestamp(value) -> tuple[pd.Timestamp | None, str, bool]:
    """Parse one timestamp. Returns (timestamp, parse_method, time_known).

    `time_known` is False when the source carried a date but no clock time.
    Those values land at 00:00:00, and without this flag they pile up in hour 0
    and masquerade as a midnight activity spike — 1,802 transactions against a
    ~790/hour baseline. Any hour-of-day analysis must exclude them.

    >>> parse_timestamp("25/02/2026 00:53:02")[1]
    'SLASH_DMY'
    >>> parse_timestamp("2026/02/02")[2]
    False
    >>> parse_timestamp("1770063471")[1]
    'UNIX'
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, BLANK, False
    raw = str(value).strip()
    if raw.lower() in _NULLISH:
        return None, BLANK, False

    method = _classify(raw)
    if method == UNPARSEABLE:
        return None, UNPARSEABLE, False

    if method == UNIX:
        try:
            # Negative epochs are legitimate here: pre-1970 dates of birth.
            # An epoch always encodes a real instant, so the time is known.
            return pd.to_datetime(int(raw), unit="s"), UNIX, True
        except (ValueError, OverflowError, OSError):
            return None, UNPARSEABLE, False

    # ISO appears with both separators; normalise to '-' so one format works.
    candidate = raw.replace("/", "-") if method == ISO else raw
    for fmt in _FORMATS[method]:
        try:
            parsed = pd.to_datetime(candidate, format=fmt)
            return parsed, method, "%H" in fmt or "%I" in fmt
        except (ValueError, TypeError):
            continue
    return None, UNPARSEABLE, False


def parse_timestamp_series(series: pd.Series) -> pd.DataFrame:
    """Parse a column into timestamp + method + invalid + time-known frames."""
    parsed = series.map(parse_timestamp)
    ts = pd.Series([p[0] for p in parsed], index=series.index, dtype="datetime64[ns]")
    method = pd.Series([p[1] for p in parsed], index=series.index)
    known = pd.Series([p[2] for p in parsed], index=series.index, dtype=bool)
    return pd.DataFrame(
        {
            "timestamp_clean": ts,
            "timestamp_parse_method": method,
            "timestamp_invalid": ts.isna(),
            "timestamp_time_known": known,
        }
    )


def derive_calendar(ts: pd.Series, prefix: str = "") -> pd.DataFrame:
    """Standard date parts used by every downstream aggregation."""
    p = f"{prefix}_" if prefix else ""
    return pd.DataFrame(
        {
            f"{p}date": ts.dt.date,
            f"{p}hour": ts.dt.hour,
            f"{p}day_of_week": ts.dt.day_name(),
            f"{p}week": ts.dt.isocalendar().week.astype("Int64"),
            f"{p}month": ts.dt.month,
            f"{p}month_name": ts.dt.month_name(),
            f"{p}quarter": ts.dt.year.astype("Int64").astype(str)
            + "-Q"
            + ts.dt.quarter.astype("Int64").astype(str),
            f"{p}year": ts.dt.year.astype("Int64"),
        },
        index=ts.index,
    )
