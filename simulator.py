from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HAND_RELEASE_NOISE_MS = 0.30   # m/s — lateral velocity noise at palm release.
                                # Revised from 0.12 (pure tremor) to 0.30 to reflect
                                # active palm-opening motion during a casual drop.
                                # Reasonable range: 0.20–0.45 m/s.

# --- Tetrahedral contact geometry ---
# A d4 strikes in three distinct contact modes on each bounce.
# Probabilities estimated from solid angle analysis of a regular tetrahedron
# combined with typical tumbling angular velocity distributions at impact.

CONTACT_PROBS = [0.42, 0.40, 0.18]   # face, edge, vertex

COR_FACE   = 0.35   # Low — energy absorbed across triangular face area
COR_EDGE   = 0.58   # Moderate — concentrated along edge line
COR_VERTEX = 0.72   # High — point contact, near-elastic

MU_FACE   = 0.28   # Symmetric friction kick, like isotropic sphere model
MU_EDGE   = 0.42   # Higher — edge pivoting amplifies horizontal impulse
MU_VERTEX = 0.18   # Low — point contact, little horizontal grip

# Edge contact deflection bias: when landing on an edge, the die is
# geometrically forced perpendicular to the edge axis. Model this as a
# weighted sum: (1-EDGE_BIAS) random + EDGE_BIAS directed perpendicular.
# Gives heavier tails without full rigid-body simulation.
EDGE_BIAS = 0.65

CLUMP_RADIUS_M = 0.035          # m — radius of the dice clump held in palm (~3.5 cm).
                                # Dice start at random offsets within this sphere.
                                # Minor contributor relative to bounce scatter.

MIN_BOUNCE_HEIGHT_M = 0.001     # m — stop simulating bounces below 1 mm height.

G = 9.81                        # m/s² — standard gravity. Exact.

M_PER_IN = 0.0254               # m / inch — exact by definition (international inch).

# At h = 20 inches (0.508 m), tetrahedral contact model:
#   ~42% face impacts  (COR 0.35, low scatter)
#   ~40% edge impacts  (COR 0.58, directed heavy-tail scatter)
#   ~18% vertex impacts (COR 0.72, high bounce, moderate scatter)
#
# Distribution shape: mixture — tight cluster from face contacts,
# heavy tails from edge/vertex. NOT a clean Rayleigh distribution.
# The Rayleigh MLE fit will still run but expect visible deviation
# in the tail of the distance histogram. This is physically correct.
#
# Mean scatter increases ~15–25% over sphere model at same drop height.
# EDGE_BIAS = 0.65 is the most uncertain parameter here — it controls
# how strongly edge geometry forces a perpendicular deflection vs.
# a random kick. Range 0.5–0.8 is physically plausible.


@dataclass
class SimulationResult:
    positions: np.ndarray         # (N, 2) landing coordinates in inches, table origin (0,0)
    distances: np.ndarray         # (N,)  displacement from center in inches
    table_size_in: float
    n_dice: int
    n_replicates: int
    drop_height_in: float
    sigma_theoretical_in: float   # RMS combination of clump, fall, and bounce scatter

    # Computed in __post_init__
    grid_counts: np.ndarray = field(init=False)
    on_grid_mask: np.ndarray = field(init=False)
    mean_distance: float = field(init=False)
    std_distance: float = field(init=False)
    median_distance: float = field(init=False)
    max_distance: float = field(init=False)
    total_landings: int = field(init=False)
    pct_on_grid: float = field(init=False)
    sigma_estimated_in: float = field(init=False)
    sigma_bias_in: float = field(init=False)
    pct_within_1sigma: float = field(init=False)
    pct_within_2sigma: float = field(init=False)
    per_replicate_sigma: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # On-grid mask
        self.on_grid_mask = np.all(
            (self.positions >= 0) & (self.positions <= self.table_size_in), axis=1
        )

        # 2D histogram for on-grid landings (1 cell = 1 inch)
        on_grid_positions = self.positions[self.on_grid_mask]
        gs = int(self.table_size_in)
        if on_grid_positions.shape[0] > 0:
            counts, _ = np.histogramdd(
                on_grid_positions,
                bins=[gs, gs],
                range=[[0, self.table_size_in], [0, self.table_size_in]],
            )
            self.grid_counts = counts.astype(np.float64)
        else:
            self.grid_counts = np.zeros((gs, gs), dtype=np.float64)

        # Distance stats — on-grid landings only
        on_grid_distances = self.distances[self.on_grid_mask]
        self.total_landings = self.n_dice * self.n_replicates
        self.pct_on_grid = 100.0 * self.on_grid_mask.sum() / self.total_landings

        if on_grid_distances.size > 0:
            self.mean_distance = float(on_grid_distances.mean())
            self.std_distance = float(on_grid_distances.std())
            self.median_distance = float(np.median(on_grid_distances))
            self.max_distance = float(on_grid_distances.max())
        else:
            self.mean_distance = 0.0
            self.std_distance = 0.0
            self.median_distance = 0.0
            self.max_distance = 0.0

        # Rayleigh MLE: σ̂ = sqrt(Σ(rᵢ²) / 2n) — all distances, on-grid and off
        self.sigma_estimated_in = float(
            np.sqrt(np.sum(self.distances ** 2) / (2 * len(self.distances)))
        )
        self.sigma_bias_in = self.sigma_estimated_in - self.sigma_theoretical_in

        # pct_within uses σ̂, not theoretical
        if on_grid_distances.size > 0:
            self.pct_within_1sigma = 100.0 * (
                on_grid_distances <= self.sigma_estimated_in
            ).sum() / on_grid_distances.size
            self.pct_within_2sigma = 100.0 * (
                on_grid_distances <= 2.0 * self.sigma_estimated_in
            ).sum() / on_grid_distances.size
        else:
            self.pct_within_1sigma = 0.0
            self.pct_within_2sigma = 0.0

        # Running Rayleigh MLE across replicates — vectorized
        # distances is ordered: rep0_die0…rep0_dieN, rep1_die0…rep1_dieN, …
        cum_sq = np.cumsum(self.distances ** 2)
        idx = np.arange(1, self.n_replicates + 1) * self.n_dice
        self.per_replicate_sigma = np.sqrt(cum_sq[idx - 1] / (2 * idx))


def run_simulation(
    n_dice: int,
    n_replicates: int,
    table_size_in: float,
    drop_height_in: float,
    seed: int | None = None,
) -> SimulationResult:
    """Run the physics-based multi-bounce dice-drop Monte Carlo simulation.

    Four sources of scatter are modelled:
      1. Clump spread  — random offset within the palm at release
      2. Free fall     — lateral release velocity × fall time
      3. Multi-bounce  — friction kick at each impact, carried forward
    σ̂ emerges from the landing distances via Rayleigh MLE — it is not an input.
    All spatial outputs are in inches (1 grid cell = 1 inch).
    """
    rng   = np.random.default_rng(seed)
    total = n_replicates * n_dice
    h     = drop_height_in * M_PER_IN           # inches → meters for all physics

    # --- Stage 1: Clump release spread ---
    # Each die starts at a random offset within the held clump (palm sphere)
    clump_x = rng.normal(0, CLUMP_RADIUS_M / 2, size=total)
    clump_y = rng.normal(0, CLUMP_RADIUS_M / 2, size=total)

    # --- Stage 2: Free fall with lateral release velocity ---
    v_x    = rng.normal(0, HAND_RELEASE_NOISE_MS, size=total)
    v_y    = rng.normal(0, HAND_RELEASE_NOISE_MS, size=total)
    t_fall = np.sqrt(2 * h / G)
    fall_dx = v_x * t_fall                      # meters
    fall_dy = v_y * t_fall

    # --- Stage 3: Multi-bounce model with tetrahedral contact geometry ---
    # Effective (probability-weighted) COR drives n_bounces and v_vert decay.
    COR_EFF = float(np.dot(CONTACT_PROBS, [COR_FACE, COR_EDGE, COR_VERTEX]))
    MU_EFF  = float(np.dot(CONTACT_PROBS, [MU_FACE,  MU_EDGE,  MU_VERTEX]))

    # Bounce height after k bounces: COR_EFF^(2k) * h < MIN_BOUNCE_HEIGHT_M
    n_bounces = max(1, int(np.ceil(
        np.log(MIN_BOUNCE_HEIGHT_M / h) / (2 * np.log(COR_EFF))
    )))

    v_impact  = np.sqrt(2 * G * h)
    bounce_dx = np.zeros(total)
    bounce_dy = np.zeros(total)
    vx_carry  = v_x.copy()                      # horizontal velocity into first bounce
    vy_carry  = v_y.copy()

    for k in range(n_bounces):
        # --- Sample contact geometry for each die independently ---
        contact = rng.choice([0, 1, 2], size=total, p=CONTACT_PROBS)

        face_mask   = contact == 0
        edge_mask   = contact == 1
        vertex_mask = contact == 2

        # Geometry-specific COR — vertical speed retained into next bounce
        cor_k = np.where(face_mask, COR_FACE,
                np.where(edge_mask, COR_EDGE, COR_VERTEX))

        # Representative vertical speed at bounce k (scalar, using mean COR)
        v_vert_k = v_impact * (np.mean([COR_FACE, COR_EDGE, COR_VERTEX]) ** k)
        t_air_k  = 2 * v_vert_k * cor_k / G

        # Geometry-specific friction coefficient
        mu_k = np.where(face_mask, MU_FACE,
               np.where(edge_mask, MU_EDGE, MU_VERTEX))

        kick_sigma = mu_k * v_vert_k

        # --- Isotropic kick component (all contact types) ---
        kx_iso = rng.normal(0, kick_sigma, size=total)
        ky_iso = rng.normal(0, kick_sigma, size=total)

        # --- Edge deflection: directed component perpendicular to carried velocity ---
        # The carried velocity direction approximates the edge axis orientation.
        # The perpendicular kick is the physically forced component.
        carried_speed = np.sqrt(vx_carry ** 2 + vy_carry ** 2) + 1e-9
        perp_x = -vy_carry / carried_speed        # unit perpendicular
        perp_y =  vx_carry / carried_speed

        edge_directed_mag = rng.normal(0, kick_sigma, size=total)
        kx_edge_directed  = edge_directed_mag * perp_x
        ky_edge_directed  = edge_directed_mag * perp_y

        # Blend isotropic and directed components for edge contacts only
        kx = np.where(edge_mask,
                      (1 - EDGE_BIAS) * kx_iso + EDGE_BIAS * kx_edge_directed,
                      kx_iso)
        ky = np.where(edge_mask,
                      (1 - EDGE_BIAS) * ky_iso + EDGE_BIAS * ky_edge_directed,
                      ky_iso)

        vx_carry += kx
        vy_carry += ky

        # Horizontal displacement during this bounce
        bounce_dx += vx_carry * t_air_k
        bounce_dy += vy_carry * t_air_k

    # --- Stage 4: Combine all displacement sources (meters → inches) ---
    total_dx_in = (clump_x + fall_dx + bounce_dx) / M_PER_IN
    total_dy_in = (clump_y + fall_dy + bounce_dy) / M_PER_IN

    center    = table_size_in / 2.0
    positions = np.column_stack([center + total_dx_in, center + total_dy_in])
    distances = np.sqrt(total_dx_in ** 2 + total_dy_in ** 2)

    # --- Theoretical σ: RMS combination of all three scatter sources ---
    # Uses probability-weighted effective COR/MU to approximate the mixture.
    sigma_clump_m  = CLUMP_RADIUS_M / 2
    sigma_fall_m   = HAND_RELEASE_NOISE_MS * t_fall
    sigma_bounce_m = MU_EFF * v_impact * (2 * COR_EFF / G) / (1 - COR_EFF)
    sigma_theoretical_in = float(
        np.sqrt(sigma_clump_m ** 2 + sigma_fall_m ** 2 + sigma_bounce_m ** 2) / M_PER_IN
    )

    return SimulationResult(
        positions=positions,
        distances=distances,
        table_size_in=table_size_in,
        n_dice=n_dice,
        n_replicates=n_replicates,
        drop_height_in=drop_height_in,
        sigma_theoretical_in=sigma_theoretical_in,
    )
