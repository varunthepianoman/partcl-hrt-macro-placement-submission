"""
Differentiable objectives for GPU-accelerated analytical placement.

Step 2: Net tensor preparation — convert ragged net_nodes into dense GPU tensors.
Step 3: LSE-HPWL — differentiable wirelength over all nets in parallel.
Step 4: Smooth density — grid-cell density via broadcasting + einsum.
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from macro_place.benchmark import Benchmark


def prepare_net_tensors(
    benchmark: Benchmark, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert ragged benchmark.net_nodes into padded dense tensors on device.

    Node indexing convention (unchanged from Benchmark):
        [0, num_hard_macros)          — hard macros
        [num_hard_macros, num_macros) — soft macros
        [num_macros, num_macros + num_ports) — I/O ports

    Args:
        benchmark: Benchmark with net_nodes (List[Tensor], variable length).
        device: Target device (mps/cuda/cpu).

    Returns:
        net_indices: [num_nets, max_net_size] long tensor, padded with 0.
        net_mask:    [num_nets, max_net_size] bool tensor, False at padding.
    """
    # pad_sequence expects [L_i] tensors; pads to max length with 0
    net_indices = pad_sequence(benchmark.net_nodes, batch_first=True, padding_value=0)
    # net_indices shape: [num_nets, max_net_size]

    # Build mask: True where the original data exists
    lengths = torch.tensor([len(n) for n in benchmark.net_nodes], dtype=torch.long)
    max_len = net_indices.size(1)
    net_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    # net_mask shape: [num_nets, max_net_size]

    return net_indices.to(device=device, dtype=torch.long), net_mask.to(device=device)


def lse_hpwl(
    all_positions: torch.Tensor,
    net_indices: torch.Tensor,
    net_mask: torch.Tensor,
    gamma: float,
    canvas_w: float,
    canvas_h: float,
    net_count: int,
) -> torch.Tensor:
    """
    Differentiable Half-Perimeter Wirelength via Log-Sum-Exp, all nets in parallel.

    Args:
        all_positions: [B, num_nodes, 2] — batched node positions.
        net_indices:   [num_nets, max_net_size] — node indices per net (padded with 0).
        net_mask:      [num_nets, max_net_size] — True for valid entries.
        gamma:         LSE smoothness parameter (smaller = closer to true HPWL).
        canvas_w:      Canvas width (microns).
        canvas_h:      Canvas height (microns).
        net_count:     Number of nets (for normalization).

    Returns:
        [B] tensor of normalized wirelength.
    """
    # Gather net positions: [B, num_nets, max_net_size, 2]
    # net_indices is [N_nets, S] -> expand to [B, N_nets, S] for gather
    B = all_positions.size(0)
    idx = net_indices.unsqueeze(0).expand(B, -1, -1)  # [B, num_nets, S]
    net_pos = torch.gather(
        all_positions, 1, idx.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(B, -1, 2)
    ).reshape(B, net_indices.size(0), net_indices.size(1), 2)
    # net_pos: [B, num_nets, S, 2]

    x = net_pos[..., 0]  # [B, num_nets, S]
    y = net_pos[..., 1]  # [B, num_nets, S]

    # Mask padding: for max use -inf, for min use +inf
    mask = net_mask.unsqueeze(0)  # [1, num_nets, S]
    x_for_max = torch.where(mask, x, torch.tensor(-1e10, device=x.device, dtype=x.dtype))
    x_for_min = torch.where(mask, x, torch.tensor(1e10, device=x.device, dtype=x.dtype))
    y_for_max = torch.where(mask, y, torch.tensor(-1e10, device=y.device, dtype=y.dtype))
    y_for_min = torch.where(mask, y, torch.tensor(1e10, device=y.device, dtype=y.dtype))

    # LSE smooth max and min over the net-size dimension
    inv_g = 1.0 / gamma
    x_max = gamma * torch.logsumexp(x_for_max * inv_g, dim=-1)  # [B, num_nets]
    x_min = -gamma * torch.logsumexp(-x_for_min * inv_g, dim=-1)
    y_max = gamma * torch.logsumexp(y_for_max * inv_g, dim=-1)
    y_min = -gamma * torch.logsumexp(-y_for_min * inv_g, dim=-1)

    # HPWL per net, sum across nets, normalize
    hpwl = (x_max - x_min) + (y_max - y_min)  # [B, num_nets]
    return hpwl.sum(dim=-1) / ((canvas_w + canvas_h) * net_count)  # [B]


def smooth_density(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    grid_rows: int,
    grid_cols: int,
    canvas_w: float,
    canvas_h: float,
) -> torch.Tensor:
    """
    Differentiable grid-cell density, all macros x all cells in parallel.

    Args:
        positions: [B, num_macros, 2] — macro center positions.
        sizes:     [num_macros, 2] — (width, height) per macro.
        grid_rows: Number of grid rows.
        grid_cols: Number of grid columns.
        canvas_w:  Canvas width (microns).
        canvas_h:  Canvas height (microns).

    Returns:
        [B] tensor — 0.5 * mean of top-10% density cells.
    """
    device = positions.device
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    # Grid cell boundaries
    cell_left = torch.arange(grid_cols, device=device, dtype=positions.dtype) * cell_w      # [C]
    cell_right = cell_left + cell_w                                                          # [C]
    cell_bottom = torch.arange(grid_rows, device=device, dtype=positions.dtype) * cell_h    # [R]
    cell_top = cell_bottom + cell_h                                                          # [R]

    # Macro boundaries: [B, N]
    half_w = sizes[:, 0] / 2  # [N]
    half_h = sizes[:, 1] / 2  # [N]
    mx_lo = positions[..., 0] - half_w  # [B, N]
    mx_hi = positions[..., 0] + half_w  # [B, N]
    my_lo = positions[..., 1] - half_h  # [B, N]
    my_hi = positions[..., 1] + half_h  # [B, N]

    # Overlap between each macro and each column: [B, N, C]
    ox = torch.clamp(
        torch.min(mx_hi.unsqueeze(-1), cell_right) - torch.max(mx_lo.unsqueeze(-1), cell_left),
        min=0,
    )
    # Overlap between each macro and each row: [B, N, R]
    oy = torch.clamp(
        torch.min(my_hi.unsqueeze(-1), cell_top) - torch.max(my_lo.unsqueeze(-1), cell_bottom),
        min=0,
    )

    # Contract over macros -> density per cell: [B, R, C]
    density = torch.einsum("...nc,...nr->...rc", ox, oy) / (cell_w * cell_h)

    # Top 10% average, scaled by 0.5
    flat = density.flatten(-2)  # [B, R*C]
    k = max(1, int(grid_rows * grid_cols * 0.1))
    topk = torch.topk(flat, k, dim=-1).values  # [B, k]
    return 0.5 * topk.mean(dim=-1)  # [B]


def overlap_penalty(
    hard_positions: torch.Tensor,
    hard_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable pairwise overlap area for hard macros, all pairs in parallel.

    Args:
        hard_positions: [B, N_hard, 2] — hard macro center positions.
        hard_sizes:     [N_hard, 2] — (width, height) per hard macro.

    Returns:
        [B] tensor — total pairwise overlap area (unnormalized microns²).
    """
    # Pairwise absolute distances: [B, N, N]
    dx = (hard_positions[..., 0:1] - hard_positions[..., 0:1].transpose(-1, -2)).abs()
    dy = (hard_positions[..., 1:2] - hard_positions[..., 1:2].transpose(-1, -2)).abs()

    # Separation thresholds: [N, N]
    sep_x = (hard_sizes[:, 0:1] + hard_sizes[:, 0:1].T) / 2
    sep_y = (hard_sizes[:, 1:2] + hard_sizes[:, 1:2].T) / 2

    # Overlap area per pair: [B, N, N]
    ovlp = torch.clamp(sep_x - dx, min=0) * torch.clamp(sep_y - dy, min=0)

    # Upper triangle mask — no self-overlap, no double-counting
    N = hard_sizes.size(0)
    triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=hard_positions.device), diagonal=1)

    return (ovlp * triu_mask).sum(dim=(-1, -2))  # [B]


def prepare_repulsion_tensors(
    hard_sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute static per-pair constants for gaussian_repulsion so they
    don't get rebuilt every training step.

    Returns:
        inv_two_sigma_sq: [N, N] — 1 / (2 σ_ij²) for Gaussian kernel.
        masked_weight:    [N, N] — normalized area-product weight, zeroed
                          on lower triangle and diagonal (triu mask baked in).
    """
    sizes = hard_sizes.to(device)
    N = sizes.size(0)

    # Per-pair sigma = sum of half-diagonals (so σ ≈ macro radius scale)
    half_diag = 0.5 * torch.sqrt(sizes[:, 0] ** 2 + sizes[:, 1] ** 2)  # [N]
    sigma_pair = half_diag.unsqueeze(0) + half_diag.unsqueeze(1)      # [N, N]
    two_sigma_sq = 2.0 * sigma_pair * sigma_pair + 1e-9               # [N, N]
    inv_two_sigma_sq = 1.0 / two_sigma_sq                             # [N, N]

    # Area-weighted, normalized by canvas_area² so max total is O(1)
    canvas_area = float(canvas_w) * float(canvas_h)
    area = sizes[:, 0] * sizes[:, 1]                                   # [N]
    weight = (area.unsqueeze(0) * area.unsqueeze(1)) / (canvas_area * canvas_area)  # [N, N]

    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
    masked_weight = weight * triu.to(weight.dtype)                    # [N, N]

    return inv_two_sigma_sq, masked_weight


def gaussian_repulsion(
    hard_positions: torch.Tensor,
    inv_two_sigma_sq: torch.Tensor,
    masked_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Smooth pairwise Gaussian repulsion — differentiable everywhere.

    Uses the matmul identity for pairwise squared distance to avoid
    materializing a [B, N, N, 2] difference tensor:
        d² = ‖xᵢ‖² + ‖xⱼ‖² − 2 xᵢ·xⱼ

    Args:
        hard_positions:   [B, N_hard, 2] — hard macro center positions.
        inv_two_sigma_sq: [N, N] — precomputed 1 / (2 σ²), from prepare_repulsion_tensors.
        masked_weight:    [N, N] — precomputed area weight with triu mask baked in.

    Returns:
        [B] tensor — normalized total pairwise repulsion energy, O(1) scale.
    """
    # Efficient pairwise squared distance via ‖a‖² + ‖b‖² − 2 a·b
    sq_norms = (hard_positions * hard_positions).sum(dim=-1)               # [B, N]
    dot = hard_positions @ hard_positions.transpose(-1, -2)                # [B, N, N]
    dist_sq = sq_norms.unsqueeze(-2) + sq_norms.unsqueeze(-1) - 2.0 * dot  # [B, N, N]
    dist_sq = dist_sq.clamp(min=0.0)  # numerical floor (matmul can yield tiny negatives)

    kernel = torch.exp(-dist_sq * inv_two_sigma_sq)                        # [B, N, N]
    return (kernel * masked_weight).sum(dim=(-1, -2))                      # [B]
