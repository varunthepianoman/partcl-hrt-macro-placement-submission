"""
Nesterov momentum solver for GPU-accelerated analytical placement.

Multi-start batched solver: runs B independent optimizations in parallel
using a batch dimension. All B runs share the same kernel launches.
"""

import math

import torch

from macro_place.benchmark import Benchmark
from submissions.analytical_placer.smooth_objectives import (
    gaussian_repulsion,
    lse_hpwl,
    overlap_penalty,
    prepare_net_tensors,
    prepare_repulsion_tensors,
    smooth_density,
)


class NesterovSolver:
    """
    Multi-start Nesterov momentum optimizer for macro placement.

    Minimizes: WL + lambda_d * Density + lambda_o * Overlap
    with annealed gamma (LSE sharpness), density weight, and overlap ramp-up.
    Runs B independent starts in parallel via batch dimension.
    """

    def __init__(
        self,
        num_iters: int = 1000,
        batch_size: int = 8,
        top_m: int = 1,
        lr_init: float = 0.5,
        lr_min: float = 0.0,
        gamma_init_scale: float = 5.0,
        gamma_final_scale: float = 0.01,
        lam_d_init: float = 0.01,
        lam_d_final: float = 10.0,
        lam_r_init: float = 2.0,
        lam_r_final: float = 20.0,
        verbose: bool = True,
    ):
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.top_m = top_m
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.gamma_init_scale = gamma_init_scale
        self.gamma_final_scale = gamma_final_scale
        self.lam_d_init = lam_d_init
        self.lam_d_final = lam_d_final
        self.lam_r_init = lam_r_init
        self.lam_r_final = lam_r_final
        self.verbose = verbose

    def _init_batch(
        self, base: torch.Tensor, canvas_diag: float, lo: torch.Tensor, hi: torch.Tensor
    ) -> torch.Tensor:
        """
        Create B perturbed copies of base positions.

        Args:
            base: [N, 2] — original positions.
            canvas_diag: canvas diagonal for noise scaling.
            lo: [N, 2] — lower clamp bounds.
            hi: [N, 2] — upper clamp bounds.

        Returns:
            [B, N, 2] — batched initial positions, clamped to bounds.
        """
        B = self.batch_size
        N = base.size(0)
        device = base.device

        batch = base.unsqueeze(0).expand(B, -1, -1).clone()  # [B, N, 2]

        # Batch 0: original (no noise)
        # Batches 1-3: small perturbation
        if B > 1:
            n_small = min(3, B - 1)
            noise_small = torch.randn(n_small, N, 2, device=device) * (0.05 * canvas_diag)
            batch[1 : 1 + n_small] += noise_small

        # Batches 4-7: larger perturbation
        if B > 4:
            n_large = B - 4
            noise_large = torch.randn(n_large, N, 2, device=device) * (0.15 * canvas_diag)
            batch[4 : 4 + n_large] += noise_large

        # Clamp all to canvas bounds
        batch.clamp_(min=lo.unsqueeze(0), max=hi.unsqueeze(0))
        return batch

    def solve(
        self,
        benchmark: Benchmark,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Run multi-start Nesterov optimization.

        Returns:
            [M, num_macros, 2] tensor — top-M placements from B starts.
        """
        B = self.batch_size
        n_hard = benchmark.num_hard_macros
        n_soft = benchmark.num_soft_macros
        n_macros = benchmark.num_macros
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        canvas_diag = math.sqrt(cw**2 + ch**2)

        # Prepare net tensors (shared across batch — broadcast)
        net_indices, net_mask = prepare_net_tensors(benchmark, device)

        # Sizes and fixed mask on device
        sizes = benchmark.macro_sizes.to(device)  # [num_macros, 2]
        hard_sizes = sizes[:n_hard]
        fixed_hard = benchmark.macro_fixed[:n_hard].to(device)  # [N_hard]
        fixed_soft = benchmark.macro_fixed[n_hard:n_macros].to(device) if n_soft > 0 else None

        # Port positions (fixed, no grad) — broadcast: [1, num_ports, 2]
        port_pos = benchmark.port_positions.to(device).unsqueeze(0)  # [1, P, 2]

        # Clamping bounds: [N, 2]
        hard_half = hard_sizes / 2
        hard_lo = hard_half.clone()
        hard_hi = torch.tensor([[cw, ch]], device=device) - hard_half

        # Initialize batched positions: [B, N_hard, 2]
        init_hard = benchmark.macro_positions[:n_hard].to(device)
        hard_pos = self._init_batch(init_hard, canvas_diag, hard_lo, hard_hi)
        hard_pos.requires_grad_(True)
        v_hard = torch.zeros_like(hard_pos)

        # Fixed hard positions for restoration: [B, num_fixed, 2] (same across batch)
        fixed_hard_pos = init_hard[fixed_hard].unsqueeze(0).expand(B, -1, -1) if fixed_hard.any() else None

        if n_soft > 0:
            soft_half = sizes[n_hard:n_macros] / 2
            soft_lo = soft_half.clone()
            soft_hi = torch.tensor([[cw, ch]], device=device) - soft_half
            init_soft = benchmark.macro_positions[n_hard:n_macros].to(device)
            soft_pos = self._init_batch(init_soft, canvas_diag, soft_lo, soft_hi)
            soft_pos.requires_grad_(True)
            v_soft = torch.zeros_like(soft_pos)
            fixed_soft_pos = init_soft[fixed_soft].unsqueeze(0).expand(B, -1, -1) if fixed_soft is not None and fixed_soft.any() else None

        # Schedule parameters
        gamma_init = self.gamma_init_scale * canvas_diag
        gamma_final = self.gamma_final_scale * canvas_diag

        # Precompute static repulsion tensors (same every step, depend only on sizes)
        inv_two_sigma_sq, masked_weight = prepare_repulsion_tensors(
            hard_sizes, cw, ch, device
        )

        for step in range(self.num_iters):
            frac = step / self.num_iters

            # Schedules — gamma anneals (sharp HPWL), density grows, repulsion grows linearly
            gamma = gamma_init * (gamma_final / gamma_init) ** frac
            lam_d = self.lam_d_init * (self.lam_d_final / self.lam_d_init) ** frac
            lam_r = self.lam_r_init + (self.lam_r_final - self.lam_r_init) * frac
            # Cosine decay with floor: lr_min + (lr_init - lr_min) * 0.5 * (1 + cos(π·frac))
            lr = self.lr_min + (self.lr_init - self.lr_min) * 0.5 * (1.0 + math.cos(math.pi * frac))

            # Build all_positions: [B, num_nodes, 2]
            if n_soft > 0:
                all_pos = torch.cat([hard_pos, soft_pos, port_pos.expand(B, -1, -1)], dim=1)
                macro_pos = torch.cat([hard_pos, soft_pos], dim=1)  # [B, num_macros, 2]
            else:
                all_pos = torch.cat([hard_pos, port_pos.expand(B, -1, -1)], dim=1)
                macro_pos = hard_pos  # [B, N_hard, 2]

            # Objectives — all return [B]
            wl = lse_hpwl(all_pos, net_indices, net_mask, gamma, cw, ch, benchmark.num_nets)
            den = smooth_density(macro_pos, sizes, benchmark.grid_rows, benchmark.grid_cols, cw, ch)
            rep = gaussian_repulsion(hard_pos, inv_two_sigma_sq, masked_weight)

            loss = wl + lam_d * den + lam_r * rep  # [B]
            loss.sum().backward()

            # Nesterov update — in-place, batched
            with torch.no_grad():
                v_hard.mul_(0.9).sub_(hard_pos.grad, alpha=lr)
                hard_pos.add_(v_hard)
                hard_pos.clamp_(min=hard_lo.unsqueeze(0), max=hard_hi.unsqueeze(0))
                if fixed_hard_pos is not None:
                    hard_pos[:, fixed_hard] = fixed_hard_pos
                hard_pos.grad.zero_()

                if n_soft > 0:
                    v_soft.mul_(0.9).sub_(soft_pos.grad, alpha=lr)
                    soft_pos.add_(v_soft)
                    soft_pos.clamp_(min=soft_lo.unsqueeze(0), max=soft_hi.unsqueeze(0))
                    if fixed_soft_pos is not None:
                        soft_pos[:, fixed_soft] = fixed_soft_pos
                    soft_pos.grad.zero_()

            if self.verbose and (step % 100 == 0 or step == self.num_iters - 1):
                print(
                    f"  [step {step:4d}] loss(mean)={loss.mean().item():.4f}  "
                    f"wl={wl.mean().item():.4f}  den={den.mean().item():.4f}  "
                    f"rep={rep.mean().item():.6f}  lr={lr:.4f}  γ={gamma:.2f}  λr={lam_r:.2f}"
                )

        # Score all B candidates: 1.0*WL + 0.5*Density + 0.1*Overlap
        with torch.no_grad():
            if n_soft > 0:
                final_all = torch.cat([hard_pos, soft_pos, port_pos.expand(B, -1, -1)], dim=1)
                final_macro = torch.cat([hard_pos, soft_pos], dim=1)
            else:
                final_all = torch.cat([hard_pos, port_pos.expand(B, -1, -1)], dim=1)
                final_macro = hard_pos

            sharp_gamma = gamma_final
            final_wl = lse_hpwl(
                final_all, net_indices, net_mask, sharp_gamma, cw, ch, benchmark.num_nets
            )
            final_den = smooth_density(
                final_macro, sizes, benchmark.grid_rows, benchmark.grid_cols, cw, ch
            )
            final_ovlp = overlap_penalty(hard_pos, hard_sizes)

            # Normalize overlap relative to typical canvas-area scale to keep weights meaningful
            score = 1.0 * final_wl + 0.5 * final_den + 0.1 * (final_ovlp / (cw * ch + 1e-9))

            # Top-M lowest scores
            m = min(self.top_m, B)
            top_scores, top_idx = torch.topk(score, m, largest=False)

            if self.verbose:
                print(f"  Top-{m} scores: {top_scores.tolist()}  (indices: {top_idx.tolist()})")

            # Assemble [M, num_macros, 2]
            results = benchmark.macro_positions.unsqueeze(0).expand(m, -1, -1).clone()
            results[:, :n_hard] = hard_pos[top_idx].detach().cpu()
            if n_soft > 0:
                results[:, n_hard:n_macros] = soft_pos[top_idx].detach().cpu()
        return results
