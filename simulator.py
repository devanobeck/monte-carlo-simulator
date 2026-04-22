from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimulationResult:
    positions: np.ndarray               # (N, 2)
    per_replicate_positions: np.ndarray  # (n_replicates, n_dice, 2)
    grid_size: float
    n_dice: int
    n_replicates: int
    scatter_sigma: float

    # Computed in __post_init__
    distances: np.ndarray = field(init=False)
    per_replicate_mean_distances: np.ndarray = field(init=False)
    grid_counts: np.ndarray = field(init=False)
    on_grid_mask: np.ndarray = field(init=False)
    mean_distance: float = field(init=False)
    std_distance: float = field(init=False)
    median_distance: float = field(init=False)
    max_distance: float = field(init=False)
    total_landings: int = field(init=False)
    pct_on_grid: float = field(init=False)
    pct_within_1sigma: float = field(init=False)
    pct_within_2sigma: float = field(init=False)

    def __post_init__(self) -> None:
        center = self.grid_size / 2.0

        # Euclidean distances for all landings
        self.distances = np.sqrt(
            (self.positions[:, 0] - center) ** 2
            + (self.positions[:, 1] - center) ** 2
        )

        # Per-replicate mean distance: compute from per_replicate_positions
        per_rep_dists = np.sqrt(
            (self.per_replicate_positions[..., 0] - center) ** 2
            + (self.per_replicate_positions[..., 1] - center) ** 2
        )  # shape (n_replicates, n_dice)
        self.per_replicate_mean_distances = per_rep_dists.mean(axis=1)  # (n_replicates,)

        # On-grid mask
        self.on_grid_mask = np.all(
            (self.positions >= 0) & (self.positions <= self.grid_size), axis=1
        )

        # 2D histogram for on-grid landings
        on_grid_positions = self.positions[self.on_grid_mask]
        gs = int(self.grid_size)
        if on_grid_positions.shape[0] > 0:
            counts, _ = np.histogramdd(
                on_grid_positions,
                bins=[gs, gs],
                range=[[0, self.grid_size], [0, self.grid_size]],
            )
            self.grid_counts = counts.astype(np.float64)
        else:
            self.grid_counts = np.zeros((gs, gs), dtype=np.float64)

        # Summary statistics (distance stats use only on-grid landings)
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

        self.pct_within_1sigma = 100.0 * (
            on_grid_distances <= self.scatter_sigma
        ).sum() / max(on_grid_distances.size, 1)

        self.pct_within_2sigma = 100.0 * (
            on_grid_distances <= 2.0 * self.scatter_sigma
        ).sum() / max(on_grid_distances.size, 1)


def run_simulation(
    n_dice: int,
    n_replicates: int,
    grid_size: float,
    scatter_sigma: float,
    seed: int | None = None,
) -> SimulationResult:
    """Run the dice-drop Monte Carlo simulation.

    All dice are dropped simultaneously from the center of the grid.
    Each landing position is independently sampled from a 2D isotropic
    Gaussian with std=scatter_sigma. The simulation is repeated
    n_replicates times and all positions are aggregated.
    """
    rng = np.random.default_rng(seed)
    center = grid_size / 2.0

    # Single vectorized sample: (n_replicates, n_dice, 2)
    raw = rng.normal(loc=center, scale=scatter_sigma, size=(n_replicates, n_dice, 2))

    # Keep per-replicate structure for convergence plot
    per_replicate_positions = raw  # (n_replicates, n_dice, 2)

    # Flatten to aggregate positions
    positions = raw.reshape(n_replicates * n_dice, 2)

    return SimulationResult(
        positions=positions,
        per_replicate_positions=per_replicate_positions,
        grid_size=grid_size,
        n_dice=n_dice,
        n_replicates=n_replicates,
        scatter_sigma=scatter_sigma,
    )
