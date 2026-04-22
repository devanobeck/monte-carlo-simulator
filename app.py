from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.stats import rayleigh
import streamlit as st

from simulator import run_simulation, SimulationResult

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Dice Drop Simulator",
    page_icon="🎲",
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BG      = "#0f1117"
SURFACE = "#1a1d2e"
PRIMARY = "#7c6ef5"
AMBER   = "#FFB347"
CORAL   = "#FF6B6B"
TEXT    = "#e0e0f0"
MUTED   = "#a0a0b8"

# ---------------------------------------------------------------------------
# Shared Plotly layout — applied to every figure
# ---------------------------------------------------------------------------
CHART_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    margin=dict(l=48, r=20, t=44, b=44),
)
_AXIS = dict(
    gridcolor="rgba(255,255,255,0.06)",
    zerolinecolor="rgba(255,255,255,0.12)",
    linecolor="rgba(255,255,255,0.08)",
)


def _apply_layout(fig: go.Figure, *, height: int = 400,
                  xtitle: str = "", ytitle: str = "",
                  chart_title: str = "",
                  square: bool = False) -> None:
    """Apply shared dark layout then per-chart axis titles."""
    fig.update_layout(**CHART_LAYOUT, height=height)
    if chart_title:
        fig.update_layout(
            title=dict(text=chart_title, font=dict(size=12, color=MUTED), x=0, xanchor="left")
        )
    x_extra = dict(scaleanchor="y", scaleratio=1) if square else {}
    fig.update_xaxes(**_AXIS, title=xtitle, **x_extra)
    fig.update_yaxes(**_AXIS, title=ytitle)


# ---------------------------------------------------------------------------
# Tick helper (centimetres)
# ---------------------------------------------------------------------------
def _in_ticks(table_size_in: float) -> np.ndarray:
    """Compute clean tick positions (inches) for a square table of given side length."""
    nice_steps = [1, 2, 5, 10, 15, 20, 25, 50, 100]
    target = table_size_in / 5.0
    step = min(nice_steps, key=lambda s: abs(s - target))
    step = max(step, 1)
    ticks = np.arange(0, table_size_in + step, step)
    ticks = ticks[ticks <= table_size_in]
    if ticks[-1] != table_size_in:
        ticks = np.append(ticks, table_size_in)
    return ticks


# ---------------------------------------------------------------------------
# KPI card helpers
# ---------------------------------------------------------------------------
def _pct_badge(val: float, hi: float = 90.0, lo: float = 70.0) -> str:
    """Colored pill badge for a percentage value."""
    if val >= hi:
        color, bg = "#22c55e", "rgba(34,197,94,0.15)"
    elif val >= lo:
        color, bg = AMBER, "rgba(255,179,71,0.15)"
    else:
        color, bg = CORAL, "rgba(255,107,107,0.15)"
    return (
        f"<span style='background:{bg};color:{color};border-radius:20px;"
        f"padding:3px 10px;font-size:11px;font-weight:600;display:inline-block;"
        f"margin-top:6px'>{val:.1f}%</span>"
    )


def _card(label: str, value: str, badge: str = "") -> str:
    badge_html = f"<div>{badge}</div>" if badge else ""
    return (
        f"<div style='flex:1;background:{SURFACE};border-radius:12px;"
        f"padding:16px 14px;border:1px solid rgba(124,110,245,0.3);min-width:0'>"
        f"<div style='font-size:11px;color:{MUTED};text-transform:uppercase;"
        f"letter-spacing:0.8px;margin-bottom:6px;white-space:nowrap;"
        f"overflow:hidden;text-overflow:ellipsis'>{label}</div>"
        f"<div style='font-size:26px;font-weight:700;color:{TEXT};"
        f"line-height:1.15;font-family:\"IBM Plex Mono\",monospace'>{value}</div>"
        f"{badge_html}</div>"
    )


def _unit(text: str) -> str:
    """Render a small muted unit label inline."""
    return f"<span style='font-size:13px;color:{MUTED};margin-left:3px'>{text}</span>"


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {TEXT};
    }}

    .stApp {{ background-color: {BG}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid rgba(124,110,245,0.18);
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

    /* Plotly chart frames */
    [data-testid="stPlotlyChart"] > div {{
        background: {SURFACE};
        border-radius: 12px;
        border: 1px solid rgba(124,110,245,0.18);
        padding: 4px;
        overflow: hidden;
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background-color: {PRIMARY};
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.02em;
        padding: 0.45rem 1rem;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #6a5ce0;
        color: #ffffff;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {SURFACE};
        border: 1px solid rgba(124,110,245,0.15) !important;
        border-radius: 10px !important;
    }}

    /* Info alert */
    [data-testid="stAlert"] {{
        background-color: rgba(124,110,245,0.08);
        border: 1px solid rgba(124,110,245,0.3);
        border-radius: 8px;
        color: {TEXT};
    }}

    /* Captions */
    [data-testid="stCaptionContainer"] p, .stCaption {{
        color: {MUTED} !important;
        font-size: 0.81rem !important;
    }}

    /* Dividers */
    hr {{ border-color: rgba(124,110,245,0.15); }}

    /* Markdown H3 in sidebar */
    [data-testid="stSidebar"] h3 {{
        color: {TEXT} !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section header helper
# ---------------------------------------------------------------------------
def _section_header(title: str, caption: str) -> None:
    st.markdown(
        f"<div style='margin-bottom:10px'>"
        f"<span style='font-size:18px;font-weight:700;color:{TEXT}'>{title}</span><br>"
        f"<span style='font-size:13px;color:{MUTED}'>{caption}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    st.markdown("---")

    table_size_in = st.slider(
        "Table Size (in)",
        min_value=3, max_value=60, value=36, step=1,
        help="Physical width of the square table in inches. Try 36 in (standard gaming mat section) or smaller to see dice fly off the edge.",
    )
    st.caption(f"Table: **{table_size_in} × {table_size_in} in** square.")
    with st.expander("ℹ️ About Table Size"):
        st.markdown(
            f"Sets the physical side length of the square table. At **{table_size_in} in** "
            f"the drop zone is a {table_size_in}×{table_size_in} inch square with the drop "
            f"point at center ({table_size_in/2:.0f}, {table_size_in/2:.0f}) in. "
            f"One grid cell = one inch exactly — no unit conversion is applied."
        )

    drop_height_in = st.slider(
        "Drop Height (in)",
        min_value=2, max_value=36, value=4, step=1,
        help="How high above the table the dice are held before release. This is the only physical input — scatter is computed from it.",
    )
    st.caption("Scatter is not set by you — it emerges from the physics of the drop. Adjust height to see how it affects spread.")
    with st.expander("ℹ️ About Drop Height"):
        st.markdown(
            "The only physical input you control. A higher drop means a longer fall time, "
            "so the same small lateral release velocity produces a wider scatter on the table. "
            "The simulation computes this exactly: σ ∝ √h."
        )

    st.markdown("---")

    n_dice = st.slider(
        "Dice Per Drop",
        min_value=1, max_value=20, value=5, step=1,
        help="Number of dice dropped simultaneously in each replicate. Capped at 20.",
    )
    st.caption("Dice dropped simultaneously per replicate.")
    with st.expander("ℹ️ About Dice Per Drop"):
        st.markdown(
            "Each replicate is one simultaneous drop of all dice from the center — "
            "think of it as one throw of the cup. More dice per drop increases the "
            "dataset size without changing the distribution shape."
        )

    n_replicates = st.slider(
        "Number of Replicates",
        min_value=1, max_value=10_000, value=1_000, step=1,
        help="How many times to repeat the full drop. More replicates = smoother distribution.",
    )
    st.caption(f"Total landings: **{n_dice * n_replicates:,}**")
    with st.expander("ℹ️ About Replicates"):
        st.markdown(
            "How many times the full throw is repeated. More replicates smooth out "
            "the distribution, like averaging many experiments. With the caps of "
            "20 dice × 10,000 replicates the simulation is always near-instant."
        )

    st.markdown("---")

    fix_seed = st.checkbox("Fix random seed", value=False)
    seed_val: int | None = None
    if fix_seed:
        seed_val = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1)

    st.markdown("")
    run_clicked = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.caption("Monte Carlo · tetrahedral contact physics · mixture distribution")

# ---------------------------------------------------------------------------
# Run simulation on button press
# ---------------------------------------------------------------------------
if run_clicked:
    with st.spinner("Running simulation…"):
        st.session_state["result"] = run_simulation(
            n_dice=n_dice,
            n_replicates=n_replicates,
            table_size_in=float(table_size_in),
            drop_height_in=float(drop_height_in),
            seed=seed_val,
        )
    st.session_state["has_run"] = True

result: SimulationResult | None = st.session_state["result"]

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    f"<div style='padding:8px 0 4px 0'>"
    f"<h1 style='color:{TEXT};font-family:Inter,sans-serif;font-size:2.1rem;"
    f"font-weight:700;margin:0;line-height:1.15'>🎲 Dice Drop Simulator</h1>"
    f"<p style='color:{MUTED};font-size:0.95rem;margin:6px 0 14px 0'>"
    f"Drop <strong style='color:{TEXT}'>{n_dice}</strong> dice from "
    f"<strong style='color:{TEXT}'>{drop_height_in} in</strong>, "
    f"repeat <strong style='color:{TEXT}'>{n_replicates:,}</strong> times on a "
    f"<strong style='color:{TEXT}'>{table_size_in}×{table_size_in} in</strong> table. "
    f"Scatter emerges from fall &amp; bounce physics · landing distances form a face/edge/vertex mixture · "
    f"stateless · no data stored or transmitted."
    f"</p>"
    f"<hr style='border:none;border-top:1px solid rgba(124,110,245,0.22);margin:0 0 6px 0'>"
    f"</div>",
    unsafe_allow_html=True,
)

if result is None:
    st.stop()

# ---------------------------------------------------------------------------
# KPI metric cards
# ---------------------------------------------------------------------------
off_table_count = int(result.total_landings - result.on_grid_mask.sum())

cards_html = "".join([
    _card("Estimated Scatter (σ̂)", f"{result.sigma_estimated_in:.2f}{_unit('in')}"),
    _card("Theoretical σ",          f"{result.sigma_theoretical_in:.2f}{_unit('in')}"),
    _card("Model Bias",             f"{result.sigma_bias_in:+.3f}{_unit('in')}"),
    _card("Drop Height",            f"{result.drop_height_in:.0f}{_unit('in')}"),
    _card("% On Table",             f"{result.pct_on_grid:.1f}%",
          _pct_badge(result.pct_on_grid)),
    _card("Within 2σ̂",             f"{result.pct_within_2sigma:.1f}%",
          _pct_badge(result.pct_within_2sigma)),
])

st.markdown(
    f"<div style='display:flex;gap:10px;margin:10px 0 8px 0'>{cards_html}</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Absolute values depend on the hand release noise constant in the model (estimated at 0.12 m/s). "
    "What matters is how σ̂ changes as you adjust drop height — that trend is physically grounded."
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3 — Heatmap + Radial distribution (two columns)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

# ── LEFT: Aggregate Landing Heatmap ─────────────────────────────────────────
with col_left:
    _section_header(
        "Aggregate Landing Heatmap",
        "Each cell shows how many dice landed there across all replicates. Brighter = more dice.",
    )

    gs = int(result.table_size_in)
    x_vals = np.arange(gs)
    y_vals = np.arange(gs)
    center = result.table_size_in / 2.0
    theta  = np.linspace(0, 2 * np.pi, 300)
    ticks  = _in_ticks(result.table_size_in)
    tick_labels = [str(int(v)) for v in ticks]

    fig_hm = go.Figure()

    fig_hm.add_trace(go.Heatmap(
        z=result.grid_counts.T,
        x=x_vals,
        y=y_vals,
        colorscale=[
            [0.0, SURFACE],
            [0.2, "#2d2070"],
            [0.55, PRIMARY],
            [1.0, "#e8e4ff"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Dice Count", font=dict(color=MUTED, size=11)),
            tickfont=dict(color=MUTED, size=10),
            thickness=12,
            len=0.85,
            outlinewidth=0,
        ),
        hovertemplate="x: %{x} in<br>y: %{y} in<br>count: %{z}<extra></extra>",
    ))

    # 1σ̂ ring
    fig_hm.add_trace(go.Scatter(
        x=center + result.sigma_estimated_in * np.cos(theta),
        y=center + result.sigma_estimated_in * np.sin(theta),
        mode="lines",
        name="1σ̂",
        line=dict(color=PRIMARY, dash="dash", width=1.8),
        opacity=0.85,
    ))

    # 2σ̂ ring
    fig_hm.add_trace(go.Scatter(
        x=center + 2 * result.sigma_estimated_in * np.cos(theta),
        y=center + 2 * result.sigma_estimated_in * np.sin(theta),
        mode="lines",
        name="2σ̂",
        line=dict(color=AMBER, dash="dash", width=1.8),
        opacity=0.85,
    ))

    # Drop point crosshair
    fig_hm.add_trace(go.Scatter(
        x=[center], y=[center],
        mode="markers",
        name="Drop Point",
        marker=dict(
            symbol="cross",
            size=14,
            color="cyan",
            line=dict(width=2.5, color="cyan"),
        ),
    ))

    _apply_layout(
        fig_hm,
        height=420,
        xtitle="X Position (in)",
        ytitle="Y Position (in)",
        chart_title=f"{result.n_dice} dice × {result.n_replicates:,} replicates · {result.drop_height_in:.0f} in drop",
    )
    fig_hm.update_xaxes(tickvals=ticks, ticktext=tick_labels)
    fig_hm.update_yaxes(tickvals=ticks, ticktext=tick_labels, scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig_hm, use_container_width=True)

# ── RIGHT: Radial Distance Distribution ─────────────────────────────────────
with col_right:
    _section_header(
        "Radial Distance Distribution",
        "Due to the d4's tetrahedral shape, the distribution is a mixture — face impacts cluster near center, edge and vertex impacts produce the heavy tail. The Rayleigh curve is fitted to the full distribution and approximates the central tendency, but expect visible deviation in the tails.",
    )

    x_max = result.distances.max() * 1.05
    x_pdf = np.linspace(0, x_max, 400)
    y_pdf = rayleigh.pdf(x_pdf, scale=result.sigma_estimated_in)

    # 1σ̂ shading — filled polygon under Rayleigh curve
    x_shade = np.linspace(0, result.sigma_estimated_in, 200)
    y_shade = rayleigh.pdf(x_shade, scale=result.sigma_estimated_in)

    fig_dist = go.Figure()

    fig_dist.add_trace(go.Scatter(
        x=np.concatenate([x_shade, x_shade[::-1]]),
        y=np.concatenate([y_shade, np.zeros(len(y_shade))]),
        fill="toself",
        fillcolor="rgba(124,110,245,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="1σ̂ zone",
        hoverinfo="skip",
    ))

    fig_dist.add_trace(go.Histogram(
        x=result.distances,
        histnorm="probability density",
        nbinsx=60,
        name="Observed",
        marker_color=PRIMARY,
        opacity=0.5,
        hovertemplate="dist: %{x:.2f} in<br>density: %{y:.4f}<extra></extra>",
    ))

    fig_dist.add_trace(go.Scatter(
        x=x_pdf,
        y=y_pdf,
        name=f"Rayleigh fit (σ̂ = {result.sigma_estimated_in:.2f} in)",
        line=dict(color=AMBER, width=2.5),
    ))

    fig_dist.add_vline(
        x=result.mean_distance,
        line_dash="dash",
        line_color=TEXT,
        opacity=0.65,
        annotation_text=f"Mean = {result.mean_distance:.2f} in",
        annotation_font=dict(color=TEXT, size=11),
        annotation_position="top right",
    )

    _apply_layout(
        fig_dist,
        height=420,
        xtitle="Distance from Center (in)",
        ytitle="Density",
    )

    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4 — Convergence plot (full width)
# ---------------------------------------------------------------------------
_section_header(
    "Convergence of Estimated σ̂ Across Replicates",
    "When the running σ̂ (amber) flattens, the MLE estimate has stabilized — more replicates won't change it meaningfully. σ̂ approximates the mixture spread, not a true Rayleigh parameter.",
)

repl_indices = np.arange(1, result.n_replicates + 1)

# Per-replicate single-throw σ estimate (noisy raw signal)
per_rep_d = result.distances.reshape(result.n_replicates, result.n_dice)
per_rep_sigma_raw = np.sqrt((per_rep_d ** 2).mean(axis=1) / 2.0)

fig_conv = go.Figure()

fig_conv.add_trace(go.Scatter(
    x=repl_indices,
    y=per_rep_sigma_raw,
    name="Single-replicate σ̂",
    line=dict(color=PRIMARY, width=1),
    opacity=0.3,
))

fig_conv.add_trace(go.Scatter(
    x=repl_indices,
    y=result.per_replicate_sigma,
    name="Running σ̂ (MLE)",
    line=dict(color=AMBER, width=2.5),
))

fig_conv.add_hline(
    y=result.sigma_theoretical_in,
    line_dash="dash",
    line_color=TEXT,
    opacity=0.55,
    annotation_text=f"Theoretical σ = {result.sigma_theoretical_in:.2f} in",
    annotation_font=dict(color=TEXT, size=11),
    annotation_position="right",
)

_apply_layout(
    fig_conv,
    height=300,
    xtitle="Replicate #",
    ytitle="Estimated σ̂ (in)",
)

st.plotly_chart(fig_conv, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 5 — Raw scatter (collapsed expander)
# ---------------------------------------------------------------------------
with st.expander("🔍 Raw Scatter Plot (all individual drops)", expanded=False):
    st.markdown(
        f"<span style='font-size:13px;color:{MUTED}'>"
        f"Each dot is one die landing. Violet = on table, red = off table. "
        f"The dashed rectangle marks the table boundary."
        f"</span>",
        unsafe_allow_html=True,
    )

    total       = result.total_landings
    subsample_n = min(2_000, total)
    rng_sub     = np.random.default_rng(0)
    idx_sub     = rng_sub.choice(total, size=subsample_n, replace=False)

    sub_positions = result.positions[idx_sub]
    sub_on_grid   = result.on_grid_mask[idx_sub]
    on_pts        = sub_positions[sub_on_grid]
    off_pts       = sub_positions[~sub_on_grid]

    fig_sc = go.Figure()

    if on_pts.shape[0] > 0:
        fig_sc.add_trace(go.Scatter(
            x=on_pts[:, 0], y=on_pts[:, 1],
            mode="markers",
            name=f"On-table ({sub_on_grid.sum():,})",
            marker=dict(color=PRIMARY, opacity=0.15, size=4, line=dict(width=0)),
        ))

    if off_pts.shape[0] > 0:
        fig_sc.add_trace(go.Scatter(
            x=off_pts[:, 0], y=off_pts[:, 1],
            mode="markers",
            name=f"Off-table ({(~sub_on_grid).sum():,})",
            marker=dict(color=CORAL, opacity=0.40, size=5, line=dict(width=0)),
        ))

    fig_sc.add_shape(
        type="rect",
        x0=0, y0=0, x1=result.table_size_in, y1=result.table_size_in,
        line=dict(color=MUTED, dash="dash", width=1.5),
        fillcolor="rgba(0,0,0,0)",
    )

    sc_center = result.table_size_in / 2.0
    fig_sc.add_trace(go.Scatter(
        x=[sc_center], y=[sc_center],
        mode="markers",
        name="Drop point",
        marker=dict(symbol="cross", size=14, color=TEXT, line=dict(width=2.5, color=TEXT)),
    ))

    _apply_layout(
        fig_sc,
        height=520,
        xtitle="X Position (in)",
        ytitle="Y Position (in)",
        chart_title=f"Showing {subsample_n:,} of {total:,} total landings",
        square=True,
    )

    st.plotly_chart(fig_sc, use_container_width=True)
