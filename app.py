from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import rayleigh
import streamlit as st

from simulator import run_simulation, SimulationResult


def _inch_ticks(grid_size: float) -> np.ndarray:
    """Compute clean tick positions (in inches) for a grid of given side length.

    Targets ~5 evenly-spaced ticks rounded to the nearest nice interval,
    always including 0 and the exact grid_size endpoint.
    """
    nice_steps = [1, 2, 5, 10, 15, 20, 25, 50, 100]
    target = grid_size / 5.0
    step = min(nice_steps, key=lambda s: abs(s - target))
    step = max(step, 1)
    ticks = np.arange(0, grid_size + step, step)
    ticks = ticks[ticks <= grid_size]
    # Always include the exact endpoint
    if ticks[-1] != grid_size:
        ticks = np.append(ticks, grid_size)
    return ticks

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
BG     = "#0F1117"
SURFACE = "#1C2333"
TEAL   = "#00D4B4"
AMBER  = "#FFB347"
CORAL  = "#FF6B6B"
TEXT   = "#E8EDF4"
MUTED  = "#8B96A8"

# ---------------------------------------------------------------------------
# Global Seaborn / Matplotlib dark theme
# ---------------------------------------------------------------------------
sns.set_theme(
    style="dark",
    rc={
        "axes.facecolor":   SURFACE,
        "figure.facecolor": BG,
        "axes.edgecolor":   "#2E3A4E",
        "grid.color":       "#2E3A4E",
        "text.color":       TEXT,
        "axes.labelcolor":  TEXT,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "axes.titlecolor":  TEXT,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "font.family":      "monospace",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS + Google Fonts
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #E8EDF4;
    }

    /* Main background */
    .stApp {
        background-color: #0F1117;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1C2333;
        border-right: 1px solid rgba(0, 212, 180, 0.18);
    }
    [data-testid="stSidebar"] * {
        color: #E8EDF4 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1C2333;
        border: 1px solid rgba(0, 212, 180, 0.35);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #8B96A8 !important;
        font-size: 0.78rem !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E8EDF4 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.3rem !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #00D4B4;
        color: #0F1117;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #00bfa5;
        color: #0F1117;
    }

    /* Slider track */
    [data-testid="stSlider"] .rc-slider-track {
        background-color: #00D4B4;
    }
    [data-testid="stSlider"] .rc-slider-handle {
        border-color: #00D4B4;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #1C2333;
        border: 1px solid #2E3A4E;
        border-radius: 8px;
    }

    /* Info box */
    [data-testid="stAlert"] {
        background-color: rgba(0, 212, 180, 0.08);
        border: 1px solid rgba(0, 212, 180, 0.3);
        border-radius: 8px;
        color: #E8EDF4;
    }

    /* Caption text */
    [data-testid="stCaptionContainer"] p, .stCaption {
        color: #8B96A8 !important;
        font-size: 0.82rem !important;
    }

    /* Divider */
    hr {
        border-color: #2E3A4E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<p style='font-size:1.1rem;font-weight:600;color:{TEAL};margin-bottom:4px;'>⚙ Parameters</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    grid_size = st.slider("Grid Size (inches)", min_value=10, max_value=200, value=50, step=1)

    n_dice = st.slider(
        "Dice Per Drop",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Number of dice dropped simultaneously in each replicate. Capped at 20.",
    )
    st.caption("Each replicate is one simultaneous drop of all dice from the center. Think of it as one throw of the cup.")

    n_replicates = st.slider(
        "Number of Replicates",
        min_value=1,
        max_value=10_000,
        value=1_000,
        step=1,
        help="How many times to repeat the full drop. More replicates = smoother distribution. Total landings = Dice Per Drop × Replicates.",
    )
    st.caption("How many times the full throw is repeated. More replicates smooth out the distribution, like averaging many experiments.")

    st.caption(f"**Total landings: {n_dice * n_replicates:,}**")

    sigma_max = float(grid_size) / 2.0
    sigma_default = float(grid_size) / 8.0
    # Clamp default into valid range
    sigma_default = max(1.0, min(sigma_default, sigma_max))

    scatter_sigma = st.slider(
        "Scatter Spread (σ)",
        min_value=1.0,
        max_value=sigma_max,
        value=sigma_default,
        step=0.5,
        help="Controls how far dice scatter from the drop point. Distances follow a Rayleigh distribution with this σ.",
    )
    st.caption("How energetically the dice scatter from the drop point. A low σ means dice cluster tightly at center; a high σ means they spread wide.")

    st.markdown("---")
    fix_seed = st.checkbox("Fix random seed", value=False)
    seed_val: int | None = None
    if fix_seed:
        seed_val = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1)

    st.markdown("")
    run_clicked = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run simulation on button press
# ---------------------------------------------------------------------------
if run_clicked:
    with st.spinner("Running simulation…"):
        st.session_state["result"] = run_simulation(
            n_dice=n_dice,
            n_replicates=n_replicates,
            grid_size=float(grid_size),
            scatter_sigma=scatter_sigma,
            seed=seed_val,
        )
    st.session_state["has_run"] = True

result: SimulationResult | None = st.session_state["result"]

# ---------------------------------------------------------------------------
# Section 0 — Info banner (shown before first run)
# ---------------------------------------------------------------------------
if not st.session_state["has_run"]:
    st.info(
        "Adjust the parameters in the sidebar and click **▶ Run Simulation** to begin. "
        "Start with the defaults to get a feel for the controls, then try increasing "
        "Scatter Spread or reducing Replicates to see how the distribution changes."
    )

# ---------------------------------------------------------------------------
# Section 1 — Page header
# ---------------------------------------------------------------------------
st.markdown(
    f"<h1 style='color:{TEAL} !important;font-family:\"IBM Plex Sans\",sans-serif;"
    f"font-size:2rem;margin-bottom:0;'>🎲 Dice Drop Simulator</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:{MUTED};font-size:0.92rem;margin-top:4px;'>"
    f"Drop <strong style='color:{TEXT};'>{n_dice}</strong> dice simultaneously, "
    f"repeat <strong style='color:{TEXT};'>{n_replicates:,}</strong> times. "
    f"Scatter is Gaussian; radial distances follow a Rayleigh distribution. "
    f"All computation runs locally."
    f"</p>",
    unsafe_allow_html=True,
)

if result is None:
    st.stop()

# ---------------------------------------------------------------------------
# Section 2 — Summary metric cards
# ---------------------------------------------------------------------------
off_table_count = int(result.total_landings - result.on_grid_mask.sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Landings", f"{result.total_landings:,}")
c2.metric("Mean Distance", f"{result.mean_distance:.2f} in")
c3.metric("Std Dev", f"{result.std_distance:.2f} in")
c4.metric("% On Table", f"{result.pct_on_grid:.1f}%")
c5.metric("Within 2σ", f"{result.pct_within_2sigma:.1f}%")
c6.metric("Off Table", f"{off_table_count:,} dice")

st.caption("The full dataset size driving all charts below — Dice Per Drop × Replicates.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3 — Primary charts (heatmap + distance distribution)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1])

# --- Left: Aggregate Landing Heatmap ---
with col_left:
    st.markdown(
        f"<p style='color:{TEXT};font-weight:600;font-size:1rem;margin-bottom:2px;'>Aggregate Landing Heatmap</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each cell shows how many dice landed there across all replicates combined. "
        "Brighter = more dice. The rings mark 1 and 2 standard deviations from center."
    )

    fig_hm, ax_hm = plt.subplots(figsize=(6, 5.5))

    sns.heatmap(
        result.grid_counts.T,
        ax=ax_hm,
        cmap="mako",
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Dice Count (all replicates)", "shrink": 0.85},
    )

    # Colorbar label color
    cbar = ax_hm.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT)
    cbar.ax.tick_params(colors=MUTED)

    # Inch tick labels — each cell = 1 inch
    ticks_in = _inch_ticks(result.grid_size)
    ax_hm.set_xticks(ticks_in)
    ax_hm.set_xticklabels([str(int(v)) for v in ticks_in], fontsize=8, color=MUTED, rotation=0)
    ax_hm.set_yticks(ticks_in)
    ax_hm.set_yticklabels([str(int(v)) for v in ticks_in], fontsize=8, color=MUTED, rotation=0)
    ax_hm.set_xlabel("X Position (in)", color=TEXT, labelpad=6)
    ax_hm.set_ylabel("Y Position (in)", color=TEXT, labelpad=6)

    # Crosshair at drop point
    center_cell = result.grid_size / 2.0
    ax_hm.plot(center_cell, center_cell, "+", color="cyan", markersize=12, markeredgewidth=2, label="Drop Point", zorder=5)

    # 1σ circle
    circle_1s = mpatches.Circle(
        (center_cell, center_cell),
        radius=result.scatter_sigma,
        fill=False,
        edgecolor=TEAL,
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="1σ",
    )
    ax_hm.add_patch(circle_1s)

    # 2σ circle
    circle_2s = mpatches.Circle(
        (center_cell, center_cell),
        radius=2.0 * result.scatter_sigma,
        fill=False,
        edgecolor=AMBER,
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="2σ",
    )
    ax_hm.add_patch(circle_2s)

    ax_hm.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.4,
        labelcolor=TEXT,
        facecolor=SURFACE,
        edgecolor="#2E3A4E",
    )
    ax_hm.set_title(
        f"Aggregate Heatmap — {result.n_dice} dice × {result.n_replicates:,} replicates"
    )

    st.pyplot(fig_hm)
    plt.close(fig_hm)

# --- Right: Radial Distance Distribution ---
with col_right:
    st.markdown(
        f"<p style='color:{TEXT};font-weight:600;font-size:1rem;margin-bottom:2px;'>Radial Distance Distribution</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "How far each die traveled from the drop point, regardless of direction. "
        "The amber curve is the theoretical prediction — a close match means the simulation is behaving correctly."
    )

    fig_dist, ax_dist = plt.subplots(figsize=(5.5, 5.5))

    sns.histplot(
        result.distances,
        ax=ax_dist,
        stat="density",
        bins=60,
        color=TEAL,
        alpha=0.5,
        edgecolor="none",
        label="Observed",
    )

    # Theoretical Rayleigh PDF
    x_max = result.distances.max() * 1.05
    x_pdf = np.linspace(0, x_max, 400)
    y_pdf = rayleigh.pdf(x_pdf, scale=result.scatter_sigma)
    ax_dist.plot(
        x_pdf,
        y_pdf,
        color=AMBER,
        linewidth=2.5,
        label=f"Rayleigh PDF (σ={result.scatter_sigma:.1f} in)",
    )

    # Vertical dashed mean line
    ax_dist.axvline(
        result.mean_distance,
        color=TEXT,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Mean = {result.mean_distance:.2f} in",
    )

    # Shaded 1σ zone under Rayleigh curve
    x_shade = np.linspace(0, result.scatter_sigma, 200)
    y_shade = rayleigh.pdf(x_shade, scale=result.scatter_sigma)
    ax_dist.fill_between(x_shade, y_shade, color=TEAL, alpha=0.12, label="1σ zone")

    ax_dist.set_xlabel("Distance from Center (in)")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Radial Distance Distribution (all replicates)")
    ax_dist.legend(
        fontsize=8,
        framealpha=0.4,
        labelcolor=TEXT,
        facecolor=SURFACE,
        edgecolor="#2E3A4E",
    )

    st.pyplot(fig_dist)
    plt.close(fig_dist)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4 — Convergence plot (full width)
# ---------------------------------------------------------------------------
st.markdown(
    f"<p style='color:{TEXT};font-weight:600;font-size:1rem;margin-bottom:2px;'>Convergence of Mean Landing Distance</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Shows whether you ran enough replicates. When the running mean (amber) flattens out, "
    "the result has stabilized and more replicates won't change it meaningfully."
)

fig_conv, ax_conv = plt.subplots(figsize=(11, 3.8))

repl_indices = np.arange(1, result.n_replicates + 1)
running_mean = np.cumsum(result.per_replicate_mean_distances) / repl_indices

ax_conv.plot(
    repl_indices,
    result.per_replicate_mean_distances,
    color=TEAL,
    alpha=0.3,
    linewidth=1,
    label="Per-replicate mean",
)
ax_conv.plot(
    repl_indices,
    running_mean,
    color=AMBER,
    linewidth=2,
    label="Running mean",
)
ax_conv.axhline(
    result.mean_distance,
    color=TEXT,
    linestyle="--",
    linewidth=1.2,
    alpha=0.6,
        label=f"Converged = {result.mean_distance:.2f} in",
    )

ax_conv.set_xlabel("Replicate #")
ax_conv.set_ylabel("Mean Distance from Center (in)")
ax_conv.set_title("Convergence of Mean Landing Distance Across Replicates")
ax_conv.legend(
    fontsize=9,
    framealpha=0.4,
    labelcolor=TEXT,
    facecolor=SURFACE,
    edgecolor="#2E3A4E",
)

st.pyplot(fig_conv)
plt.close(fig_conv)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 5 — Raw scatter plot (collapsed expander)
# ---------------------------------------------------------------------------
with st.expander("🔍 Raw Scatter Plot", expanded=False):
    st.caption(
        "Each dot is one die landing. Teal dots stayed on the table; red dots flew off the edge. "
        "The dashed rectangle is the table boundary."
    )

    total = result.total_landings
    subsample_n = min(2_000, total)
    rng_sub = np.random.default_rng(0)
    idx_sub = rng_sub.choice(total, size=subsample_n, replace=False)

    sub_positions = result.positions[idx_sub]
    sub_on_grid = result.on_grid_mask[idx_sub]

    fig_sc, ax_sc = plt.subplots(figsize=(7, 7))

    # On-grid points
    on_pts = sub_positions[sub_on_grid]
    if on_pts.shape[0] > 0:
        ax_sc.scatter(
            on_pts[:, 0],
            on_pts[:, 1],
            color=TEAL,
            alpha=0.15,
            s=4,
            linewidths=0,
            label=f"On-table ({sub_on_grid.sum():,})",
        )

    # Off-grid points
    off_pts = sub_positions[~sub_on_grid]
    if off_pts.shape[0] > 0:
        ax_sc.scatter(
            off_pts[:, 0],
            off_pts[:, 1],
            color=CORAL,
            alpha=0.35,
            s=5,
            linewidths=0,
            label=f"Off-table ({(~sub_on_grid).sum():,})",
        )

    # Table boundary rectangle
    table_rect = mpatches.Rectangle(
        (0, 0),
        result.grid_size,
        result.grid_size,
        fill=False,
        edgecolor=MUTED,
        linestyle="--",
        linewidth=1.2,
        label="Table boundary",
    )
    ax_sc.add_patch(table_rect)

    # Center drop point
    center = result.grid_size / 2.0
    ax_sc.plot(
        center,
        center,
        "+",
        color=TEXT,
        markersize=14,
        markeredgewidth=2,
        zorder=5,
        label="Drop point",
    )

    ax_sc.set_aspect("equal")
    ax_sc.set_title(
        f"Raw Scatter (showing {subsample_n:,} of {total:,} total landings)"
    )
    ax_sc.set_xlabel("X Position (in)")
    ax_sc.set_ylabel("Y Position (in)")
    ax_sc.legend(
        fontsize=8,
        framealpha=0.4,
        labelcolor=TEXT,
        facecolor=SURFACE,
        edgecolor="#2E3A4E",
    )

    st.pyplot(fig_sc)
    plt.close(fig_sc)
