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

COR = 0.55                      # Coefficient of restitution — plastic d4 on hard table.
                                # Range: 0.45–0.65 (hard surface), 0.20–0.40 (felt).
                                # Controls bounce height retention between impacts.

MU_FRICTION = 0.28              # Kinetic friction coefficient during bounce impact.
                                # Each impact converts MU × v_vertical into a random
                                # horizontal kick. Range: 0.20–0.35 for plastic on wood.

CLUMP_RADIUS_M = 0.035          # m — radius of the dice clump held in palm (~3.5 cm).
                                # Dice start at random offsets within this sphere.
                                # Minor contributor relative to bounce scatter.

MIN_BOUNCE_HEIGHT_M = 0.001     # m — stop simulating bounces below 1 mm height.

G = 9.81                        # m/s² — standard gravity. Exact.

M_PER_IN = 0.0254               # m / inch — exact by definition (international inch).

# At h = 20 in (0.508 m):
#   v_impact = 3.16 m/s
#   t_fall   = 0.322 s
#   n_bounces ≈ 6
#
#   sigma_clump  ≈ 0.018 m  (0.69 in)  — minor
#   sigma_fall   ≈ 0.097 m  (3.80 in)  — moderate
#   sigma_bounce ≈ 0.221 m  (8.68 in)  — dominant
#   sigma_total  ≈ 0.241 m  (9.50 in)
#
#   Mean landing distance (Rayleigh) = sigma * sqrt(π/2) ≈ 11.9 in
#
# This is consistent with real-world d4 drops from tabletop height onto a hard surface.
# COR and MU_FRICTION are the most sensitive parameters — small changes have large effects.
# HAND_RELEASE_NOISE_MS matters most at low drop heights where bounce is minimal.


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

    # --- Stage 3: Multi-bounce model ---
    # Number of significant bounces until bounce height < MIN_BOUNCE_HEIGHT_M.
    # Bounce height after k bounces: COR^(2k) * h < MIN_BOUNCE_HEIGHT_M
    n_bounces = max(1, int(np.ceil(
        np.log(MIN_BOUNCE_HEIGHT_M / h) / (2 * np.log(COR))
    )))

    v_impact  = np.sqrt(2 * G * h)
    bounce_dx = np.zeros(total)
    bounce_dy = np.zeros(total)
    vx_carry  = v_x.copy()                      # horizontal velocity into first bounce
    vy_carry  = v_y.copy()

    for k in range(n_bounces):
        # Vertical speed at kth impact (reduced by COR each bounce)
        v_vert_k = v_impact * (COR ** k)

        # Time die spends in the air during kth bounce
        t_air_k = 2 * v_vert_k * COR / G

        # Random friction kick — converts vertical momentum into a random horizontal
        # impulse (direction unpredictable due to d4's irregular geometry)
        kick_sigma = MU_FRICTION * v_vert_k
        vx_carry += rng.normal(0, kick_sigma, size=total)
        vy_carry += rng.normal(0, kick_sigma, size=total)

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
    sigma_clump_m  = CLUMP_RADIUS_M / 2
    sigma_fall_m   = HAND_RELEASE_NOISE_MS * t_fall
    # Bounce scatter: dominant first-bounce term scaled by geometric series
    sigma_bounce_m = MU_FRICTION * v_impact * (2 * COR / G) / (1 - COR)
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
