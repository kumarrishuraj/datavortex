"""Reusable UI primitives.

The important one is `coverage_badge`. In this dataset only about half of the
transactions resolve to a merchant master record and about a third to a KYC
record, so a segmented metric is routinely computed on part of the book. Every such number
carries its coverage beside it — the badge takes the coverage from the metric
itself rather than from a caller-supplied constant, so the two cannot drift.
"""
from __future__ import annotations

import html

import streamlit as st

from app.components.theme import INK_MUTED, SERIES, STATUS, VERDICT_STYLE


def _esc(text) -> str:
    return html.escape(str(text))


def coverage_class(pct: float) -> str:
    if pct >= 99.5:
        return "dv-cov-full"
    if pct >= 40:
        return "dv-cov-part"
    return "dv-cov-low"


def coverage_badge_html(pct: float, basis: str = "") -> str:
    """Inline badge: coverage percentage plus what the denominator is."""
    cls = coverage_class(pct)
    icon = "●" if pct >= 99.5 else ("◐" if pct >= 40 else "○")
    label = "full population" if pct >= 99.5 else f"{pct:.2f}% coverage"
    tail = f" · {_esc(basis)}" if basis else ""
    return f'<div class="dv-cov {cls}" title="{_esc(basis)}">{icon} {label}{tail}</div>'


def coverage_badge(pct: float, basis: str = "") -> None:
    st.markdown(coverage_badge_html(pct, basis), unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "",
             coverage: float | None = None, basis: str = "") -> None:
    """One KPI tile. `coverage` renders the badge when the metric is partial."""
    parts = [
        '<div class="dv-kpi">',
        f'<div class="dv-kpi-label">{_esc(label)}</div>',
        f'<div class="dv-kpi-value">{_esc(value)}</div>',
    ]
    if sub:
        parts.append(f'<div class="dv-kpi-sub">{_esc(sub)}</div>')
    if coverage is not None:
        parts.append(coverage_badge_html(coverage, basis))
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def kpi_row(items: list[dict]) -> None:
    """A row of KPI tiles from `data.headline_kpis`-shaped dicts."""
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        with col:
            kpi_card(**item)


def section(title: str, note: str = "") -> None:
    body = f'<div class="dv-sec"><div class="dv-sec-title">{_esc(title)}</div>'
    if note:
        body += f'<div class="dv-sec-note">{note}</div>'
    st.markdown(body + "</div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "info") -> None:
    """Kinds: info, warn, crit, good. `text` may contain <strong> markup."""
    cls = {"info": "", "warn": " dv-call-warn", "crit": " dv-call-crit",
           "good": " dv-call-good"}.get(kind, "")
    st.markdown(f'<div class="dv-call{cls}">{text}</div>', unsafe_allow_html=True)


def verdict_chip(verdict: str) -> str:
    """Status chips always pair colour with an icon and a word, never colour alone."""
    colour, icon = VERDICT_STYLE.get(verdict, (INK_MUTED, "—"))
    return (f'<span class="dv-chip" style="background:{colour}">'
            f'{icon} {_esc(verdict)}</span>')


def insight_card(heading: str, body: str) -> None:
    st.markdown(
        f'<div class="dv-insight"><div class="dv-insight-h">{_esc(heading)}</div>'
        f'<div class="dv-insight-b">{body}</div></div>',
        unsafe_allow_html=True,
    )


def significance_caveat(ranked, test: dict, label_col: str, what: str) -> None:
    """Render the ranking caveat whenever a group comparison is not significant.

    Called automatically wherever a ranked breakdown is shown, so the caveat
    cannot be forgotten on a chart-by-chart basis.
    """
    if ranked is None or ranked.empty:
        return
    top = ranked.iloc[0]
    name = _esc(top[label_col])
    if not bool(test.get("significant", False)):
        callout(
            f"<strong>{name} ranks highest at {top['rate_pct']:.2f}%</strong>, but "
            f"differences across {what} are not statistically significant "
            f"(χ² = {test['chi_square']}, df = {test['df']}, p = {test['p_value']:.3f}). "
            f"This ordering reflects sampling noise, not a real difference in behaviour, "
            f"and should not be used to target a category.",
            kind="warn",
        )
    else:
        callout(
            f"<strong>{name} ranks highest at {top['rate_pct']:.2f}%</strong> "
            f"(95% CI {top['ci_low_pct']:.2f}–{top['ci_high_pct']:.2f}%). Differences "
            f"across {what} are statistically significant "
            f"(χ² = {test['chi_square']}, df = {test['df']}, p = {test['p_value']:.4f}), "
            f"though overlapping intervals mean individual pairs may not differ.",
            kind="good",
        )


def empty_state(message: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="dv-call dv-call-warn"><strong>No matching records.</strong> '
        f'{_esc(message)}{" " + _esc(hint) if hint else ""}</div>',
        unsafe_allow_html=True,
    )


def fmt_inr(value: float, compact: bool = False) -> str:
    if value is None or value != value:
        return "—"
    if compact:
        if abs(value) >= 1e7:
            return f"₹{value / 1e7:.2f} Cr"
        if abs(value) >= 1e5:
            return f"₹{value / 1e5:.2f} L"
    return f"₹{value:,.0f}"


def fmt_int(value) -> str:
    if value is None or value != value:
        return "—"
    return f"{int(value):,}"


def fmt_pct(value, dp: int = 2) -> str:
    if value is None or value != value:
        return "—"
    return f"{value:.{dp}f}%"
