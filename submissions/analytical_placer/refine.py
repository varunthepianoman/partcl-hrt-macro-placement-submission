"""
Soft macro refinement — post-legalization GPU pass.

Freezes hard macros and ports, optimizes soft macros only via Nesterov
gradient descent on LSE-HPWL + density (no overlap penalty).
"""

import math

import torch

from macro_place.benchmark import Benchmark
from submissions.analytical_placer.smooth_objectives import (
    lse_hpwl,
    prepare_net_tensors,
    smooth_density,
)


def refine_soft(
    benchmark: Benchmark,
    placement: torch.Tensor,
    device: torch.device,
    num_iters: int = 400,
    lr_init: float = 0.5,
    gamma_init_scale: float = 1.0,
    gamma_final_scale: float = 0.01,
    lam_d: float = 1.0,
) -> torch.Tensor:
    """
    Optimize soft macros while hard macros remain fixed.

    Args:
        benchmark: Benchmark with netlist and canvas data.
        placement: [num_macros, 2] — legalized positions (hard macros already placed).
        device: Target device.
        num_iters: Optimizer steps.
        lr_init: Initial learning rate (cosine-decayed).
        gamma_init_scale / gamma_final_scale: LSE smoothness schedule (× canvas_diag).
        lam_d: Density weight.

    Returns:
        [num_macros, 2] — refined placement on CPU, same shape as input.
    """
    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    n_macros = benchmark.num_macros

    if n_soft == 0:
        return placement

    cw = benchmark.canvas_width
    ch = benchmark.canvas_height
    canvas_diag = math.sqrt(cw**2 + ch**2)

    # Net tensors (shared, broadcast over batch dim of 1)
    net_indices, net_mask = prepare_net_tensors(benchmark, device)

    # Sizes on device
    sizes = benchmark.macro_sizes.to(device)

    # Hard macros frozen: no grad, held constant
    hard_pos = placement[:n_hard].to(device)  # no requires_grad

    # Ports frozen
    port_pos = benchmark.port_positions.to(device)

    # Soft macros: the only learnable parameter
    soft_pos = placement[n_hard:n_macros].to(device).clone().requires_grad_(True)
    v_soft = torch.zeros_like(soft_pos)

    # Soft macro clamping bounds
    soft_half = sizes[n_hard:n_macros] / 2
    soft_lo = soft_half.clone()
    soft_hi = torch.tensor([[cw, ch]], device=device) - soft_half

    # Preserve fixed soft macros (if any)
    fixed_soft = benchmark.macro_fixed[n_hard:n_macros].to(device)
    fixed_soft_pos = soft_pos.detach()[fixed_soft].clone() if fixed_soft.any() else None

    gamma_init = gamma_init_scale * canvas_diag
    gamma_final = gamma_final_scale * canvas_diag

    initial_wl = None

    for step in range(num_iters):
        frac = step / num_iters
        gamma = gamma_init * (gamma_final / gamma_init) ** frac
        lr = lr_init * 0.5 * (1.0 + math.cos(math.pi * frac))

        # Reassemble all positions — hard is const, soft is leaf with grad
        # unsqueeze for batch dim (B=1)
        all_pos = torch.cat([hard_pos, soft_pos, port_pos], dim=0).unsqueeze(0)
        macro_pos = torch.cat([hard_pos, soft_pos], dim=0).unsqueeze(0)

        wl = lse_hpwl(all_pos, net_indices, net_mask, gamma, cw, ch, benchmark.num_nets)
        den = smooth_density(macro_pos, sizes, benchmark.grid_rows, benchmark.grid_cols, cw, ch)

        loss = wl + lam_d * den  # [1]
        loss.sum().backward()

        if initial_wl is None:
            initial_wl = wl.item()

        # Nesterov update, in-place, soft only
        with torch.no_grad():
            v_soft.mul_(0.9).sub_(soft_pos.grad, alpha=lr)
            soft_pos.add_(v_soft)
            soft_pos.clamp_(min=soft_lo, max=soft_hi)
            if fixed_soft_pos is not None:
                soft_pos[fixed_soft] = fixed_soft_pos
            soft_pos.grad.zero_()

        if step % 100 == 0 or step == num_iters - 1:
            print(
                f"  [refine {step:4d}] loss={loss.item():.4f}  "
                f"wl={wl.item():.4f}  den={den.item():.4f}  lr={lr:.4f}"
            )

    final_wl = wl.item()
    print(f"  [refine] wl: {initial_wl:.4f} -> {final_wl:.4f} "
          f"(Δ={final_wl - initial_wl:+.4f})")

    # Assemble final result on CPU
    with torch.no_grad():
        result = placement.clone()
        result[n_hard:n_macros] = soft_pos.detach().cpu()
    return result
