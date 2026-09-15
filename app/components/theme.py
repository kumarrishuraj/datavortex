"""Visual system for DataVortex.

Palette is the validated reference instance from the data-viz method: eight
categorical hues in a fixed CVD-safe order, a single-hue sequential ramp, a
warm/cool diverging pair, and four reserved status colors that never double as
series colors.

Two rules this module enforces by construction:
  * categorical hues are assigned in fixed order and never cycled — a ninth
    series folds into "Other" rather than generating a colour
  * status colours always ship with an icon and a label, never colour alone
"""
from __future__ import annotations

# --- categorical slots, fixed order (light surface) -------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Scatter / all-pairs forms are capped at three slots; past that the palette
# cannot clear the colour-vision floors on every pair.
SERIES_ALL_PAIRS = SERIES[:3]

# --- single-hue sequential ramp (blue), light -> dark -----------------------
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]

# --- reserved status palette (never themed, never reused as a series) -------
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# --- chart chrome and ink ---------------------------------------------------
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "#e5e4de"

# Semantic mapping used across pages so a status keeps one colour everywhere.
STATUS_COLOURS = {
    "SUCCESS": SERIES[2],     # aqua
    "FAILED": STATUS["critical"],
    "PENDING": STATUS["warning"],
}

VERDICT_STYLE = {
    "SUPPORTED": (STATUS["good"], "✓"),
    "NOT SUPPORTED": (INK_MUTED, "○"),
    "BORDERLINE": (STATUS["warning"], "◐"),
    "CONTRADICTED": (STATUS["critical"], "✕"),
    "QUANTIFIED": (SERIES[0], "▣"),
    "UNTESTABLE": (INK_MUTED, "—"),
}

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", Arial, sans-serif')


def plotly_layout(height: int = 340, showlegend: bool = False, **kwargs) -> dict:
    """Shared Plotly layout: recessive chrome, thin marks, readable ink."""
    layout = dict(
        height=height,
        showlegend=showlegend,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        margin=dict(l=8, r=12, t=34, b=8),
        title=dict(font=dict(size=13, color=INK), x=0, xanchor="left", y=0.97),
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=BORDER,
            font=dict(family=FONT, size=12, color=INK),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            linecolor=BASELINE, tickcolor=BASELINE,
            tickfont=dict(color=INK_MUTED, size=11),
            title=dict(font=dict(size=11, color=INK_MUTED)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
            linecolor="rgba(0,0,0,0)", tickcolor=BASELINE,
            tickfont=dict(color=INK_MUTED, size=11),
            title=dict(font=dict(size=11, color=INK_MUTED)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=INK_SECONDARY),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
    )
    layout.update(kwargs)
    return layout


CSS = f"""
<style>
  .stApp {{ background: {PLANE}; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}

  h1, h2, h3, h4 {{ font-family: {FONT}; color: {INK}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.75rem !important; font-weight: 640 !important; }}
  h2 {{ font-size: 1.15rem !important; font-weight: 620 !important; margin-top: 1.6rem !important; }}
  h3 {{ font-size: 0.98rem !important; font-weight: 600 !important; }}

  /* --- brand block in the sidebar --- */
  .dv-brand {{ padding: 0 0 0.9rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 0.9rem; }}
  .dv-brand-name {{ font-size: 1.18rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; }}
  .dv-brand-tag {{ font-size: 0.72rem; color: {INK_MUTED}; margin-top: 2px; line-height: 1.35; }}

  /* --- KPI card --- */
  .dv-kpi {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 0.85rem 0.95rem; height: 100%;
  }}
  .dv-kpi-label {{
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.045em;
    text-transform: uppercase; color: {INK_MUTED};
  }}
  .dv-kpi-value {{
    font-size: 1.62rem; font-weight: 660; color: {INK};
    line-height: 1.18; margin-top: 0.22rem; letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
  }}
  .dv-kpi-sub {{ font-size: 0.73rem; color: {INK_SECONDARY}; margin-top: 0.2rem; }}

  /* --- coverage badge --- */
  .dv-cov {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.68rem; font-weight: 600; padding: 2px 7px;
    border-radius: 5px; margin-top: 0.4rem; line-height: 1.5;
  }}
  .dv-cov-full {{ background: #e8f5e9; color: #1b5e20; }}
  .dv-cov-part {{ background: #fff4e0; color: #8a4b00; }}
  .dv-cov-low  {{ background: #fdecea; color: #8c1d18; }}

  /* --- section header --- */
  .dv-sec {{ margin: 1.7rem 0 0.55rem 0; }}
  .dv-sec-title {{ font-size: 1.03rem; font-weight: 650; color: {INK}; letter-spacing: -0.01em; }}
  .dv-sec-note {{ font-size: 0.79rem; color: {INK_SECONDARY}; margin-top: 0.2rem; line-height: 1.5; }}

  /* --- callout --- */
  .dv-call {{
    border-left: 3px solid {SERIES[0]}; background: {SURFACE};
    border-radius: 0 8px 8px 0; padding: 0.7rem 0.9rem; margin: 0.5rem 0;
    font-size: 0.83rem; color: {INK_SECONDARY}; line-height: 1.55;
  }}
  .dv-call strong {{ color: {INK}; font-weight: 620; }}
  .dv-call-warn {{ border-left-color: {STATUS['warning']}; background: #fffdf6; }}
  .dv-call-crit {{ border-left-color: {STATUS['critical']}; background: #fef8f8; }}
  .dv-call-good {{ border-left-color: {STATUS['good']}; background: #f6fdf6; }}

  /* --- verdict chip --- */
  .dv-chip {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.03em;
    padding: 2px 9px; border-radius: 20px; color: #fff;
  }}

  /* --- insight card --- */
  .dv-insight {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 0.8rem 0.95rem; height: 100%;
  }}
  .dv-insight-h {{ font-size: 0.78rem; font-weight: 660; color: {INK}; margin-bottom: 0.28rem; }}
  .dv-insight-b {{ font-size: 0.79rem; color: {INK_SECONDARY}; line-height: 1.55; }}

  /* --- tables --- */
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

  /* --- tighten Streamlit chrome --- */
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
  hr {{ margin: 1.1rem 0; border-color: {BORDER}; }}
  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  .stRadio [role="radiogroup"] {{ gap: 1px; }}
</style>
"""
