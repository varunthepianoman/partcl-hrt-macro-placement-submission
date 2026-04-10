"""
Minimum-displacement legalization for hard macros.

CPU-based sequential placement with spatial-hash acceleration.
Largest-first ordering, spiral search from analytical position.
"""

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Safety gap to prevent float-precision overlaps
GAP = 0.01


class SpatialHash:
    """Grid-based spatial index for O(1) average-case collision queries."""

    def __init__(self, canvas_w: float, canvas_h: float, cell_size: float):
        self.cell_size = cell_size
        self.cols = max(1, int(np.ceil(canvas_w / cell_size)))
        self.rows = max(1, int(np.ceil(canvas_h / cell_size)))
        # Each cell stores list of (idx, x_lo, y_lo, x_hi, y_hi)
        self.grid: dict[int, list] = {}

    def _key(self, col: int, row: int) -> int:
        return row * self.cols + col

    def _cells_for_rect(self, x_lo: float, y_lo: float, x_hi: float, y_hi: float):
        c0 = max(0, int(x_lo / self.cell_size))
        c1 = min(self.cols - 1, int(x_hi / self.cell_size))
        r0 = max(0, int(y_lo / self.cell_size))
        r1 = min(self.rows - 1, int(y_hi / self.cell_size))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                yield self._key(c, r)

    def insert(self, idx: int, x_lo: float, y_lo: float, x_hi: float, y_hi: float):
        entry = (idx, x_lo, y_lo, x_hi, y_hi)
        for key in self._cells_for_rect(x_lo, y_lo, x_hi, y_hi):
            self.grid.setdefault(key, []).append(entry)

    def query_overlap(self, x_lo: float, y_lo: float, x_hi: float, y_hi: float) -> bool:
        """Return True if the rectangle overlaps any inserted rectangle."""
        for key in self._cells_for_rect(x_lo, y_lo, x_hi, y_hi):
            for _, ox_lo, oy_lo, ox_hi, oy_hi in self.grid.get(key, []):
                if x_lo < ox_hi and x_hi > ox_lo and y_lo < oy_hi and y_hi > oy_lo:
                    return True
        return False


def legalize(benchmark: Benchmark, placement: torch.Tensor) -> torch.Tensor:
    """
    Legalize hard macro positions via minimum-displacement spiral search.

    Args:
        benchmark: Benchmark with macro sizes, fixed flags, canvas bounds.
        placement: [num_macros, 2] tensor of analytical positions.

    Returns:
        [num_macros, 2] tensor with legalized hard macro positions.
        Soft macros are unchanged.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    pos = placement.detach().cpu().numpy().astype(np.float64).copy()
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.numpy()
    movable = ~fixed[:n_hard]

    half_w = sizes[:n_hard, 0] / 2
    half_h = sizes[:n_hard, 1] / 2

    # Spatial hash with cell size = largest macro dimension
    max_dim = max(sizes[:n_hard].max(), 1.0)
    shash = SpatialHash(cw, ch, float(max_dim))

    # Sort hard macros by area descending (largest first)
    areas = sizes[:n_hard, 0] * sizes[:n_hard, 1]
    order = np.argsort(-areas)

    legal = pos.copy()
    total_disp = 0.0

    for idx in order:
        hw, hh = half_w[idx], half_h[idx]

        if not movable[idx]:
            # Fixed macro — insert into spatial hash as-is
            x_lo = legal[idx, 0] - hw - GAP
            y_lo = legal[idx, 1] - hh - GAP
            x_hi = legal[idx, 0] + hw + GAP
            y_hi = legal[idx, 1] + hh + GAP
            shash.insert(idx, x_lo, y_lo, x_hi, y_hi)
            continue

        # Try original position first
        cx, cy = pos[idx, 0], pos[idx, 1]
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25

        best_pos = None
        best_dist = float("inf")

        for ring in range(0, 200):
            if ring == 0:
                candidates = [(cx, cy)]
            else:
                candidates = []
                for d in range(-ring, ring + 1):
                    for e in [-ring, ring]:
                        candidates.append((cx + d * step, cy + e * step))
                    if -ring < d < ring:
                        candidates.append((cx + d * step, cy - ring * step))
                        candidates.append((cx + d * step, cy + ring * step))
                # Deduplicate (the corners get added twice in the naive approach)
                seen = set()
                unique = []
                for c in candidates:
                    key = (round(c[0], 6), round(c[1], 6))
                    if key not in seen:
                        seen.add(key)
                        unique.append(c)
                candidates = unique

            found = False
            for tx, ty in candidates:
                # Clamp to canvas
                tx = max(hw, min(cw - hw, tx))
                ty = max(hh, min(ch - hh, ty))

                # Check overlap with placed macros (including GAP)
                x_lo = tx - hw - GAP
                y_lo = ty - hh - GAP
                x_hi = tx + hw + GAP
                y_hi = ty + hh + GAP

                if not shash.query_overlap(x_lo, y_lo, x_hi, y_hi):
                    dist = (tx - cx) ** 2 + (ty - cy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (tx, ty)
                        found = True

            if found:
                break

        if best_pos is None:
            # Fallback: keep original (shouldn't happen)
            best_pos = (cx, cy)

        legal[idx, 0], legal[idx, 1] = best_pos
        total_disp += np.sqrt(best_dist) if best_dist < float("inf") else 0.0

        # Insert legalized macro into spatial hash
        x_lo = legal[idx, 0] - hw - GAP
        y_lo = legal[idx, 1] - hh - GAP
        x_hi = legal[idx, 0] + hw + GAP
        y_hi = legal[idx, 1] + hh + GAP
        shash.insert(idx, x_lo, y_lo, x_hi, y_hi)

    print(f"  [legalize] total displacement: {total_disp:.2f} microns across {movable.sum()} movable macros")

    result = placement.clone()
    result[:n_hard] = torch.tensor(legal[:n_hard], dtype=placement.dtype)
    return result
