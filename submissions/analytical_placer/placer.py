"""
GPU-Accelerated Analytical Macro Placer

4-phase pipeline:
  1. GPU analytical optimization (LSE-HPWL + density + overlap) -> Top-M candidates
  2. Hard macro legalization (CPU, each candidate)
  3. Soft macro re-optimization (GPU, each candidate)
  4. Select best candidate by smooth proxy score

Usage:
    uv run evaluate submissions/analytical_placer/placer.py
    uv run evaluate submissions/analytical_placer/placer.py --all
"""

import math
import sys
from pathlib import Path

# Ensure project root is on sys.path for evaluator compatibility
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

from macro_place.benchmark import Benchmark
from submissions.analytical_placer.legalize import legalize
from submissions.analytical_placer.nesterov import NesterovSolver
from submissions.analytical_placer.refine import refine_soft
from submissions.analytical_placer.smooth_objectives import (
    lse_hpwl,
    overlap_penalty,
    prepare_net_tensors,
    smooth_density,
)


def _detect_device() -> torch.device:
    """Select best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[AnalyticalPlacer] Using device: {device}")
    return device


class AnalyticalPlacer:
    """GPU-accelerated analytical macro placer."""

    def __init__(
        self,
        seed: int = 42,
        batch_size: int = 32,
        top_m: int = 1,
        nesterov_iters: int = 1000,
        refine_iters: int = 400,
        lr_init: float = 0.5,
        lr_min: float = 0.0,
        verbose: bool = True,
    ):
        self.seed = seed
        self.batch_size = batch_size
        self.top_m = top_m
        self.nesterov_iters = nesterov_iters
        self.refine_iters = refine_iters
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.verbose = verbose
        self.device = _detect_device()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)

        # Phase 1: GPU analytical optimization -> [M, num_macros, 2]
        solver = NesterovSolver(
            num_iters=self.nesterov_iters,
            batch_size=self.batch_size,
            top_m=self.top_m,
            lr_init=self.lr_init,
            lr_min=self.lr_min,
            verbose=self.verbose,
        )
        candidates = solver.solve(benchmark, self.device)  # [M, N, 2]
        M = candidates.size(0)

        # Phase 2 & 3: Legalize + refine each candidate
        refined = []
        for i in range(M):
            if self.verbose:
                print(f"  --- Candidate {i+1}/{M} ---")
                print(f"  CPU: Legalizing candidate {i+1} of {M}...")
            legalized = legalize(benchmark, candidates[i])

            if self.verbose:
                print(f"  GPU: Refining soft macros for candidate {i+1} of {M}...")
            refined_i = refine_soft(
                benchmark, legalized, self.device, num_iters=self.refine_iters
            )
            refined.append(refined_i)

        # Phase 4: Score and select best via smooth objectives
        best_idx = self._select_best(benchmark, refined)
        if self.verbose and M > 1:
            print(f"  Best candidate: {best_idx + 1}/{M}")
        return refined[best_idx]

    def _select_best(self, benchmark: Benchmark, candidates: list) -> int:
        """Score each candidate via smooth objectives; return best index."""
        if len(candidates) == 1:
            return 0

        device = self.device
        net_indices, net_mask = prepare_net_tensors(benchmark, device)
        sizes = benchmark.macro_sizes.to(device)
        hard_sizes = sizes[: benchmark.num_hard_macros]
        port_pos = benchmark.port_positions.to(device)
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        canvas_diag = math.sqrt(cw**2 + ch**2)
        sharp_gamma = 0.01 * canvas_diag

        scores = []
        with torch.no_grad():
            for c in candidates:
                c_dev = c.to(device)
                macro_pos = c_dev.unsqueeze(0)  # [1, num_macros, 2]
                all_pos = torch.cat([c_dev, port_pos], dim=0).unsqueeze(0)
                wl = lse_hpwl(
                    all_pos,
                    net_indices,
                    net_mask,
                    sharp_gamma,
                    cw,
                    ch,
                    benchmark.num_nets,
                )
                den = smooth_density(
                    macro_pos,
                    sizes,
                    benchmark.grid_rows,
                    benchmark.grid_cols,
                    cw,
                    ch,
                )
                hard_pos = c_dev[: benchmark.num_hard_macros].unsqueeze(0)
                ovlp = overlap_penalty(hard_pos, hard_sizes)
                s = (
                    1.0 * wl.item()
                    + 0.5 * den.item()
                    + 0.1 * (ovlp.item() / (cw * ch + 1e-9))
                )
                scores.append(s)

        if self.verbose:
            print(f"  Candidate scores: {[f'{s:.4f}' for s in scores]}")
        return int(min(range(len(scores)), key=lambda i: scores[i]))
