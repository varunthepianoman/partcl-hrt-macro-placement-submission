"""
Master sweep script for the GPU Analytical Placer.

Dual-profile experiment runner:
  --mode sprint    (~15 min, ibm01 + ibm07, smaller grid)
  --mode overnight (~10-12 h, all 18 boards, full grid)

Features:
- Live ETA progress bar
- Verbose milestone logging
- Checkpoint resume via sweep_results.csv
- KeyboardInterrupt-safe (saves CSV + plot on Ctrl-C)

Usage:
    uv run python submissions/analytical_placer/run_master_sweep.py --mode sprint
    uv run python submissions/analytical_placer/run_master_sweep.py --mode overnight
"""

import argparse
import csv
import datetime
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


class Tee:
    """Duplicate writes to multiple streams (e.g., stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

# Ensure project root on sys.path
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from submissions.analytical_placer.placer import AnalyticalPlacer


# ── Profiles ────────────────────────────────────────────────────────────────

SPRINT_BENCHMARKS = ["ibm01", "ibm07"]
OVERNIGHT_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

MESO_BENCHMARKS = ["ibm01", "ibm03", "ibm07"]  # ibm05 does not exist in ICCAD04

SPRINT_PROFILE = {
    "benchmarks": SPRINT_BENCHMARKS,
    "batch_sizes": [8, 16],
    "top_ms": [1, 2],
    "nesterov_iters_list": [600],
    "lr_min_fracs": [0.0],
    "refine_iters": 200,
    "csv_path": Path("sweep_results.csv"),
}

OVERNIGHT_PROFILE = {
    "benchmarks": OVERNIGHT_BENCHMARKS,
    "batch_sizes": [1, 4, 8, 16, 32, 64],
    "top_ms": [1, 2, 4],
    "nesterov_iters_list": [1000],
    "lr_min_fracs": [0.0],
    "refine_iters": 400,
    "csv_path": Path("sweep_results.csv"),
}

MESO_PROFILE = {
    # Refined based on v1 meso results (2026-04-09):
    #   - lr_floor=0.1 + iters=2500 was the clear winner on both ibm01 and ibm07
    #   - lr_floor=0.1 + iters=1000 was catastrophic on ibm07 (proxy 2.14 vs 1.76)
    #     → LR floor needs enough runway to settle; drop iters=1000
    #   - iters=2500 + lr_floor=0.0 was worse than 1000+0.0 on ibm01
    #     → long tails with lr→0 are harmful; confirmed lr_floor hypothesis
    # Narrowing: iters ∈ {1500, 2500}, lr_floor ∈ {0.05, 0.1, 0.15}
    "benchmarks": MESO_BENCHMARKS,  # ibm01, ibm03, ibm07
    "batch_sizes": [32],
    "top_ms": [4],
    "nesterov_iters_list": [1500, 2500],
    "lr_min_fracs": [0.05, 0.1, 0.15],
    "refine_iters": 400,
    "csv_path": Path("draft_results.csv"),
}

CSV_COLUMNS = [
    "mode", "benchmark", "batch_size", "top_m", "nesterov_iters", "lr_min_frac",
    "proxy_cost", "wirelength", "density", "congestion", "overlaps", "valid", "runtime_s",
]


# ── Checkpoint state ────────────────────────────────────────────────────────

@dataclass
class SweepState:
    """Checkpoint state keyed by (benchmark, B, M, nesterov_iters, lr_min_frac)."""

    results: list = field(default_factory=list)
    completed: set = field(default_factory=set)

    def key(self, benchmark: str, B: int, M: int, iters: int, lr_min_frac: float) -> tuple:
        return (benchmark, int(B), int(M), int(iters), round(float(lr_min_frac), 6))

    def load(self, path: Path):
        if not path.exists():
            return
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.results.append(row)
                self.completed.add(
                    self.key(
                        row["benchmark"],
                        int(row["batch_size"]),
                        int(row["top_m"]),
                        int(row.get("nesterov_iters", 0)),
                        float(row.get("lr_min_frac", 0.0)),
                    )
                )

    def save(self, path: Path):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self.results)

    def has(self, benchmark: str, B: int, M: int, iters: int, lr_min_frac: float) -> bool:
        return self.key(benchmark, B, M, iters, lr_min_frac) in self.completed

    def add(self, row: dict):
        self.results.append(row)
        self.completed.add(
            self.key(
                row["benchmark"],
                int(row["batch_size"]),
                int(row["top_m"]),
                int(row["nesterov_iters"]),
                float(row["lr_min_frac"]),
            )
        )


# ── Benchmark loading ──────────────────────────────────────────────────────

def _load_bench(name: str):
    """Load an ibm benchmark from ICCAD04 testcases."""
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    return load_benchmark_from_dir(str(root))


# ── Progress display ────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _print_progress(idx: int, total: int, avg_run_time: float, tag: str):
    remaining = max(0, total - idx)
    eta = remaining * avg_run_time
    bar_width = 30
    filled = int(bar_width * idx / max(total, 1))
    bar = "█" * filled + "░" * (bar_width - filled)
    print(
        f"\n{'='*80}\n"
        f"[{bar}] {idx}/{total}  |  avg {_fmt_time(avg_run_time)}/run  |  "
        f"ETA {_fmt_time(eta)}  |  {tag}\n"
        f"{'='*80}"
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def _save_plot(state: SweepState, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    if not state.results:
        return

    by_bench: dict[str, list] = {}
    for r in state.results:
        if not r.get("valid") or r.get("valid") == "False":
            continue
        by_bench.setdefault(r["benchmark"], []).append(r)

    if not by_bench:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for bench, rows in by_bench.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                int(r["batch_size"]),
                int(r["top_m"]),
                int(r.get("nesterov_iters", 0)),
                float(r.get("lr_min_frac", 0.0)),
            ),
        )
        labels = [
            f"B{r['batch_size']}M{r['top_m']}i{r.get('nesterov_iters','?')}lr{r.get('lr_min_frac','?')}"
            for r in rows_sorted
        ]
        costs = [float(r["proxy_cost"]) for r in rows_sorted]
        ax.plot(labels, costs, marker="o", label=bench)

    ax.set_ylabel("Proxy cost")
    ax.set_xlabel("Configuration (B=batch, M=top)")
    ax.set_title("Master Sweep Results")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"  Plot saved to {path}")


# ── Main sweep loop ─────────────────────────────────────────────────────────

PROFILES = {
    "sprint": SPRINT_PROFILE,
    "overnight": OVERNIGHT_PROFILE,
    "meso": MESO_PROFILE,
}


def run_sweep(mode: str):
    profile = PROFILES[mode]
    benchmarks = profile["benchmarks"]
    batch_sizes = profile["batch_sizes"]
    top_ms = profile["top_ms"]
    nesterov_iters_list = profile["nesterov_iters_list"]
    lr_min_fracs = profile["lr_min_fracs"]
    refine_iters = profile["refine_iters"]
    csv_path = profile["csv_path"]

    state = SweepState()
    state.load(csv_path)

    # Build full list of runs across all config axes
    runs = []
    for b in benchmarks:
        for B in batch_sizes:
            for M in top_ms:
                if M > B:
                    continue
                for iters in nesterov_iters_list:
                    for lr_min_frac in lr_min_fracs:
                        runs.append((b, B, M, iters, lr_min_frac))

    total = len(runs)
    skipped = sum(1 for r in runs if state.has(*r))
    print(f"\n{'#'*80}")
    print(f"# MASTER SWEEP: {mode.upper()}")
    print(f"# Benchmarks: {benchmarks}")
    print(f"# Batch sizes: {batch_sizes}  Top-M: {top_ms}")
    print(f"# Nesterov iters: {nesterov_iters_list}  LR-min fracs: {lr_min_fracs}")
    print(f"# Total runs: {total}  (already completed: {skipped}, remaining: {total - skipped})")
    print(f"# Checkpoint: {csv_path}")
    print(f"{'#'*80}\n")

    # Install interrupt handler
    interrupted = {"flag": False}

    def _on_sigint(sig, frame):
        print("\n\n[!] KeyboardInterrupt received — saving results and exiting...")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    run_times = []

    try:
        for run_idx, (bench_name, B, M, iters, lr_min_frac) in enumerate(runs, start=1):
            if interrupted["flag"]:
                break

            if state.has(bench_name, B, M, iters, lr_min_frac):
                print(
                    f"[{run_idx}/{total}] SKIP {bench_name} B={B} M={M} "
                    f"iters={iters} lr_floor={lr_min_frac} (already in CSV)"
                )
                continue

            avg_time = sum(run_times) / len(run_times) if run_times else 120.0
            tag = f"{bench_name}  B={B}  M={M}  iters={iters}  lr_floor={lr_min_frac}"
            _print_progress(run_idx - 1, total, avg_time, f"Starting: {tag}")
            print(f"  → Running {bench_name} | Iters={iters} | LR_Floor={lr_min_frac}...")

            t0 = time.time()
            try:
                benchmark, plc = _load_bench(bench_name)

                lr_init = 0.5
                lr_min = lr_min_frac * lr_init

                placer = AnalyticalPlacer(
                    batch_size=B,
                    top_m=M,
                    nesterov_iters=iters,
                    refine_iters=refine_iters,
                    lr_init=lr_init,
                    lr_min=lr_min,
                    verbose=True,
                )

                print(f"  GPU: Running Nesterov batch (B={B})...")
                placement = placer.place(benchmark)
                print(f"  GPU: Nesterov Batch Completed.")
                print(f"  CPU: Legalizing & Refining {M} Candidates...")
                # (legalize/refine already happened inside placer.place — the log lines from
                # inside the placer have already printed; this line marks the phase.)

                print(f"  Evaluating via PlacementCost...")
                costs = compute_proxy_cost(placement, benchmark, plc)

                runtime = time.time() - t0
                run_times.append(runtime)

                row = {
                    "mode": mode,
                    "benchmark": bench_name,
                    "batch_size": B,
                    "top_m": M,
                    "nesterov_iters": iters,
                    "lr_min_frac": lr_min_frac,
                    "proxy_cost": f"{costs['proxy_cost']:.4f}",
                    "wirelength": f"{costs['wirelength_cost']:.4f}",
                    "density": f"{costs['density_cost']:.4f}",
                    "congestion": f"{costs['congestion_cost']:.4f}",
                    "overlaps": costs["overlap_count"],
                    "valid": costs["overlap_count"] == 0,
                    "runtime_s": f"{runtime:.2f}",
                }
                state.add(row)
                state.save(csv_path)

                print(
                    f"  ✓ Final Proxy Cost: {costs['proxy_cost']:.4f} | Time: {runtime:.1f}s  "
                    f"(wl={costs['wirelength_cost']:.3f} "
                    f"den={costs['density_cost']:.3f} "
                    f"cong={costs['congestion_cost']:.3f} "
                    f"overlaps={costs['overlap_count']})"
                )

            except Exception as e:
                runtime = time.time() - t0
                print(f"  ✗ FAILED after {runtime:.1f}s: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

    finally:
        print(f"\n{'='*80}")
        print(f"Saving final CSV to {csv_path}...")
        state.save(csv_path)
        plot_path = csv_path.with_suffix(".png")
        _save_plot(state, plot_path)
        print(f"Done. {len(state.results)} total results in {csv_path}.")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sprint", "overnight", "meso"], required=True)
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (default: sweep_<mode>_<timestamp>.log)",
    )
    args = parser.parse_args()

    # Set up dual stdout/stderr -> terminal + log file
    log_path = Path(
        args.log_file
        or f"sweep_{args.mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file = log_path.open("w", buffering=1)  # line-buffered
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)

    print(f"[log] Writing output to {log_path}")
    try:
        run_sweep(args.mode)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()
        print(f"[log] Log saved to {log_path}")


if __name__ == "__main__":
    main()
