"""Plotly chart builders.

Every chart in the product is built here, so the mark specs hold everywhere:
2px lines, >=8px markers, thin bars with rounded data-ends, a 2px surface gap
between adjacent fills, recessive grid and axes, and a hover layer by default.

Colour is assigned by the job it does — categorical hues in fixed order for
identity, one hue light-to-dark for magnitude, reserved status colours only
where the colour means good or bad. No dual-axis charts exist in this module,
by construction: two measures of different scale get two charts.
"""
from __future__ import annotations

import plotly.graph_objects as go

from app.components.theme import (
    BASELINE, GRID, INK, INK_MUTED, INK_SECONDARY, SERIES, STATUS,
    STATUS_COLOURS, SEQUENTIAL, SURFACE, plotly_layout,
)

BAR_RADIUS = 4


def line(x, y, title="", name="", colour=SERIES[0], height=320,
         yfmt=",.0f", hovertemplate=None, fill=False):
    """Single-series line. No legend — the title names the series."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name or title,
        line=dict(color=colour, width=2, shape="linear"),
        fill="tozeroy" if fill else None,
        fillcolor=_alpha(colour, 0.10) if fill else None,
        hovertemplate=hovertemplate or f"%{{x}}<br><b>%{{y:{yfmt}}}</b><extra></extra>",
    ))
    fig.update_layout(**plotly_layout(height=height, title_text=title, hovermode="x unified"))
    return fig


def multi_line(x, series: dict, title="", height=340, yfmt=",.0f", colours=None):
    """Several series on ONE axis. A legend is always present for 2+ series."""
    fig = go.Figure()
    palette = colours or [STATUS_COLOURS.get(k, SERIES[i % len(SERIES)])
                          for i, k in enumerate(series)]
    for (name, values), colour in zip(series.items(), palette):
        fig.add_trace(go.Scatter(
            x=x, y=values, mode="lines", name=name,
            line=dict(color=colour, width=2),
            hovertemplate=f"%{{x}}<br>{name}: <b>%{{y:{yfmt}}}</b><extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        height=height, showlegend=True, title_text=title, hovermode="x unified"))
    return fig


def hbar(labels, values, title="", colour=SERIES[0], height=None, xfmt=",.0f",
         text=None, highlight_index=None, highlight_colour=None, xtitle=""):
    """Horizontal bar — the form for ranking.

    One series, one colour. A value ramp across nominal categories would
    double-encode length as hue, so a single hue is used and a specific bar is
    highlighted only when one of them is genuinely the subject.
    """
    height = height or max(220, 30 * len(labels) + 70)
    colours = [colour] * len(labels)
    if highlight_index is not None and 0 <= highlight_index < len(labels):
        colours[highlight_index] = highlight_colour or SERIES[1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=text, textposition="outside" if text is not None else "none",
        textfont=dict(size=11, color=INK_SECONDARY),
        hovertemplate=f"%{{y}}<br><b>%{{x:{xfmt}}}</b><extra></extra>",
        cliponaxis=False,
    ))
    layout = plotly_layout(height=height, title_text=title)
    layout["yaxis"].update(showgrid=False, autorange="reversed",
                           tickfont=dict(size=11, color=INK_SECONDARY))
    layout["xaxis"].update(showgrid=True, gridcolor=GRID, linecolor="rgba(0,0,0,0)",
                           title=dict(text=xtitle, font=dict(size=11, color=INK_MUTED)))
    layout["margin"] = dict(l=8, r=52, t=34, b=8)
    fig.update_layout(**layout)
    fig.update_traces(marker_cornerradius=BAR_RADIUS)
    return fig


def hbar_with_ci(labels, values, lo, hi, baseline=None, title="",
                 colour=SERIES[0], height=None, xtitle=""):
    """Ranking with 95% confidence intervals drawn as error bars.

    The intervals are the point: when they overlap, the ranking is not a real
    ordering, and the reader can see that without reading a p-value.
    """
    height = height or max(240, 30 * len(labels) + 80)
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colour, line=dict(width=0)),
        error_x=dict(
            type="data", symmetric=False,
            array=[h - v for h, v in zip(hi, values)],
            arrayminus=[v - l for l, v in zip(lo, values)],
            color=INK_MUTED, thickness=1.4, width=4,
        ),
        hovertemplate="%{y}<br><b>%{x:.2f}%</b><extra></extra>",
        cliponaxis=False,
    ))
    if baseline is not None:
        fig.add_vline(x=baseline, line=dict(color=BASELINE, width=1.5, dash="dot"),
                      annotation_text="overall", annotation_position="top",
                      annotation_font=dict(size=10, color=INK_MUTED))
    layout = plotly_layout(height=height, title_text=title)
    layout["yaxis"].update(showgrid=False, autorange="reversed",
                           tickfont=dict(size=11, color=INK_SECONDARY))
    layout["xaxis"].update(showgrid=True, gridcolor=GRID, linecolor="rgba(0,0,0,0)",
                           title=dict(text=xtitle, font=dict(size=11, color=INK_MUTED)))
    layout["margin"] = dict(l=8, r=40, t=34, b=8)
    fig.update_layout(**layout)
    fig.update_traces(marker_cornerradius=BAR_RADIUS)
    return fig


def status_composition(counts: dict, title="", height=110):
    """Part-to-whole for a small number of ordered states.

    A 2px surface gap separates the segments, so adjacent fills never merge.
    """
    total = sum(counts.values()) or 1
    fig = go.Figure()
    for name, value in counts.items():
        fig.add_trace(go.Bar(
            x=[value], y=[""], orientation="h", name=name,
            marker=dict(color=STATUS_COLOURS.get(name, SERIES[0]),
                        line=dict(color=SURFACE, width=2)),
            hovertemplate=f"{name}: <b>%{{x:,.0f}}</b> ({value / total:.2%})<extra></extra>",
        ))
    layout = plotly_layout(height=height, showlegend=True, title_text=title, barmode="stack")
    layout["xaxis"].update(showgrid=False, showticklabels=False, linecolor="rgba(0,0,0,0)")
    layout["yaxis"].update(showgrid=False, showticklabels=False)
    layout["margin"] = dict(l=4, r=4, t=34, b=4)
    fig.update_layout(**layout)
    return fig


def histogram(values, title="", colour=SERIES[0], height=300, nbins=40,
              xtitle="", vline=None, vline_label=""):
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=nbins,
        marker=dict(color=colour, line=dict(color=SURFACE, width=1)),
        hovertemplate="%{x}<br><b>%{y:,.0f}</b><extra></extra>",
    ))
    if vline is not None:
        fig.add_vline(x=vline, line=dict(color=STATUS["critical"], width=1.6, dash="dash"),
                      annotation_text=vline_label, annotation_position="top right",
                      annotation_font=dict(size=10, color=STATUS["critical"]))
    layout = plotly_layout(height=height, title_text=title)
    layout["xaxis"].update(title=dict(text=xtitle, font=dict(size=11, color=INK_MUTED)))
    fig.update_layout(**layout)
    fig.update_traces(marker_cornerradius=2)
    return fig


def scatter(x, y, labels=None, title="", height=340, colour=SERIES[0],
            xtitle="", ytitle="", size=None, colour_by=None, colour_map=None,
            diagonal=False):
    """Scatter. Capped at three categorical classes — the all-pairs limit."""
    fig = go.Figure()
    if colour_by is not None and colour_map:
        for name, col in list(colour_map.items())[:3]:
            mask = [c == name for c in colour_by]
            fig.add_trace(go.Scatter(
                x=[v for v, m in zip(x, mask) if m],
                y=[v for v, m in zip(y, mask) if m],
                mode="markers", name=name,
                text=[v for v, m in zip(labels or x, mask) if m],
                marker=dict(color=col, size=8, opacity=0.75,
                            line=dict(color=SURFACE, width=1)),
                hovertemplate="%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", text=labels,
            marker=dict(color=colour, size=size or 8, opacity=0.7,
                        line=dict(color=SURFACE, width=1)),
            hovertemplate="%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>",
        ))
    if diagonal:
        lo = min(min(x), min(y))
        hi = max(max(x), max(y))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(color=BASELINE, width=1, dash="dot"))
    layout = plotly_layout(height=height, showlegend=colour_by is not None,
                           title_text=title)
    layout["xaxis"].update(showgrid=True, gridcolor=GRID,
                           title=dict(text=xtitle, font=dict(size=11, color=INK_MUTED)))
    layout["yaxis"].update(title=dict(text=ytitle, font=dict(size=11, color=INK_MUTED)))
    fig.update_layout(**layout)
    return fig


def network(edges, title="", height=460, seed=7):
    """Bipartite user-merchant network for one component.

    Users and merchants are two identity classes, so they take categorical
    slots 1 and 2; disputed edges take the reserved critical colour because the
    colour means "disputed", and they are also drawn thicker so the distinction
    is not colour-alone.
    """
    import networkx as nx

    g = nx.Graph()
    for row in edges.itertuples():
        g.add_node(row.user_id_normalized, kind="User")
        g.add_node(row.merchant_id_normalized, kind="Merchant")
        g.add_edge(row.user_id_normalized, row.merchant_id_normalized,
                   disputed=bool(row.is_disputed), amount=float(row.amount_inr),
                   status=row.status_canonical)
    if g.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(g, seed=seed, k=0.9, iterations=90)
    fig = go.Figure()

    for disputed, colour, width in [(False, BASELINE, 1.2),
                                    (True, STATUS["critical"], 2.4)]:
        xs, ys = [], []
        for u, v, d in g.edges(data=True):
            if bool(d["disputed"]) != disputed:
                continue
            xs += [pos[u][0], pos[v][0], None]
            ys += [pos[u][1], pos[v][1], None]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", hoverinfo="skip",
                name="Disputed transaction" if disputed else "Transaction",
                line=dict(color=colour, width=width),
            ))

    for kind, colour, symbol in [("User", SERIES[0], "circle"),
                                 ("Merchant", SERIES[1], "square")]:
        nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == kind]
        if not nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers", name=kind, text=nodes,
            marker=dict(color=colour, size=11, symbol=symbol,
                        line=dict(color=SURFACE, width=1.5)),
            hovertemplate=f"{kind}: %{{text}}<extra></extra>",
        ))

    layout = plotly_layout(height=height, showlegend=True, title_text=title)
    for axis in ("xaxis", "yaxis"):
        layout[axis].update(showgrid=False, zeroline=False, showticklabels=False,
                            linecolor="rgba(0,0,0,0)")
    layout["margin"] = dict(l=4, r=4, t=34, b=4)
    fig.update_layout(**layout)
    return fig


def vbar(labels, values, title="", colour=SERIES[0], height=320, yfmt=",.0f",
         text=None, xtitle="", ytitle=""):
    """Vertical bar for a small set of categories."""
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colour, line=dict(width=0)),
        text=text, textposition="outside" if text is not None else "none",
        textfont=dict(size=11, color=INK_SECONDARY),
        hovertemplate=f"%{{x}}<br><b>%{{y:{yfmt}}}</b><extra></extra>",
        cliponaxis=False,
    ))
    layout = plotly_layout(height=height, title_text=title)
    layout["xaxis"].update(tickfont=dict(size=11, color=INK_SECONDARY),
                           title=dict(text=xtitle, font=dict(size=11, color=INK_MUTED)))
    layout["yaxis"].update(title=dict(text=ytitle, font=dict(size=11, color=INK_MUTED)))
    fig.update_layout(**layout)
    fig.update_traces(marker_cornerradius=BAR_RADIUS)
    return fig


def forecast_chart(actual_x, actual_y, forecast_x, forecast_y, lower, upper,
                   title="", height=340, yfmt=",.0f"):
    """Actual series, projection and its 95% band — one axis, one measure."""
    fx, lo, hi = list(forecast_x), list(lower), list(upper)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fx + fx[::-1], y=hi + lo[::-1], fill="toself",
        fillcolor=_alpha(SERIES[1], 0.14), line=dict(width=0),
        hoverinfo="skip", name="95% band"))
    fig.add_trace(go.Scatter(
        x=list(actual_x), y=list(actual_y), mode="lines", name="Actual",
        line=dict(color=SERIES[0], width=2),
        hovertemplate=f"%{{x}}<br>Actual: <b>%{{y:{yfmt}}}</b><extra></extra>"))
    fig.add_trace(go.Scatter(
        x=fx, y=list(forecast_y), mode="lines", name="Forecast",
        line=dict(color=SERIES[1], width=2, dash="dash"),
        hovertemplate=f"%{{x}}<br>Forecast: <b>%{{y:{yfmt}}}</b><extra></extra>"))
    fig.update_layout(**plotly_layout(height=height, showlegend=True, title_text=title,
                                      hovermode="x unified"))
    return fig


def _alpha(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
