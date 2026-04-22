from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HAND_RELEASE_NOISE_MS = 0.12  # m/s — estimated lateral velocity noise at release.
                               # Based on physiological hand tremor (8–12 Hz, ~0.5–2mm amplitude)
                               # giving ~0.06 m/s, scaled up to 0.12 m/s to account for
                               # slight hand motion during a casual drop (not pure tremor).
                               # Reasonable range: 0.05–0.20 m/s. This is an estimate,
                               # not a measured constant. Absolute inch outputs scale with it;
                               # relative trends with drop height are physically grounded.

BOUNCE_NOISE_M = 0.005        # m — post-landing scatter from first bounce (~0.20 in).
                               # A d4's tetrahedral shape stops quickly (no flat face to roll on).
                               # 5mm is a physically plausible estimate; no measured data exists
                               # for this specific geometry. Treat as approximate.

G = 9.81                      # m/s² — standard gravity. Exact.

M_PER_IN = 0.0254             # m / inch — exact by definition (international inch).

# Validation at default parameters (h = 8 in = 0.203 m):
#   t_fall = sqrt(2 * 0.203 / 9.81) ≈ 0.204 s
#   sigma_theoretical = sqrt((0.12 * 0.204)² + 0.005²) / 0.0254 ≈ 0.98 in
#   On a 12 in table: σ ≈ 8.2% of table width
#   This is consistent with observed d4 scatter from casual tabletop drops.
#   Note: HAND_RELEASE_NOISE_MS is an estimate. Absolute inch outputs scale with it;
#   the functional relationship between drop height and σ̂ is physically correct.


@dataclass
class SimulationResult:
    positions: np.ndarray         # (N, 2) landing coordinates in inches, table origin (0,0)
    distances: np.ndarray         # (N,)  displacement from center in inches
    table_size_in: float
    n_dice: int
    n_replicates: int
    drop_height_in: float
    sigma_theoretical_in: float   # from physics: sqrt((noise*t_fall)² + bounce²) / M_PER_IN

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
    """Run the physics-based dice-drop Monte Carlo simulation.

    The genuinely random variable is the lateral hand velocity at release.
    Each die's landing position is derived by propagating that velocity
    through free-fall physics, then adding a small post-landing bounce term.
    σ̂ emerges from the landing distances via Rayleigh MLE — it is not an input.
    All spatial outputs are in inches (1 grid cell = 1 inch).
    """
    rng = np.random.default_rng(seed)
    total = n_replicates * n_dice

    # Sample lateral release velocities (m/s) and bounce offsets (m)
    v_x = rng.normal(0.0, HAND_RELEASE_NOISE_MS, size=total)
    v_y = rng.normal(0.0, HAND_RELEASE_NOISE_MS, size=total)
    b_x = rng.normal(0.0, BOUNCE_NOISE_M, size=total)
    b_y = rng.normal(0.0, BOUNCE_NOISE_M, size=total)

    # Fall time from drop height (inches → meters for physics)
    h = drop_height_in * M_PER_IN
    t_fall = np.sqrt(2.0 * h / G)

    # Horizontal displacements in inches (1 grid cell = 1 inch)
    dx_in = (v_x * t_fall + b_x) / M_PER_IN
    dy_in = (v_y * t_fall + b_y) / M_PER_IN

    # Landing positions on the table
    center = table_size_in / 2.0
    positions = np.column_stack([center + dx_in, center + dy_in])

    # Displacement from center in inches
    distances = np.sqrt(dx_in ** 2 + dy_in ** 2)

    # Theoretical σ from physics (meters → inches)
    sigma_theoretical_in = float(
        np.sqrt((HAND_RELEASE_NOISE_MS * t_fall) ** 2 + BOUNCE_NOISE_M ** 2) / M_PER_IN
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
