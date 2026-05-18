"""
alltoall.py — benchmarks dist.all_to_all_single using the mini_nccl backend.

Sweeps a range of per-rank tensor sizes, times the all_to_all collective over
many iterations on each, and reports latency and algorithmic bandwidth.

"Algorithm bandwidth" here follows the NCCL convention for all-to-all:
    bus_bytes = input_bytes * (world_size - 1) / world_size
i.e. each rank sends (world_size-1)/world_size of its send buffer out over
the interconnect (the chunk destined for itself stays local), independent
of how the collective is implemented.

Timing is measured on **rank 0 only** (rank 0's local elapsed time per
timed iteration).  No cross-rank latency reduction is performed: the
mini_nccl backend corrupts any collective issued immediately after the
benchmarked one (a buffer-aliasing bug), and the workload is fully
symmetric anyway, so rank 0's local time is a sound measurement.

Usage:
    uv run python alltoall.py [--backend {mini_nccl,nccl}]

Requires at least 2 CUDA-capable GPUs.  One process is spawned per visible
GPU using torch.multiprocessing.spawn (start method: 'spawn'), which gives
each process a clean CUDA context — a requirement for CUDA IPC.
"""

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import mini_nccl  # noqa: F401 — importing registers the "mini_nccl" backend

# Per-rank input sizes to sweep, in bytes (float32 → 4 bytes/element).
# Ranges from 1 MiB up to 256 MiB per rank.
SIZES_BYTES = [
    1 << 20,   # 1 MiB
    1 << 22,   # 4 MiB
    1 << 24,   # 16 MiB
    1 << 26,   # 64 MiB
    1 << 27,   # 128 MiB
    1 << 28,   # 256 MiB
]
WARMUP_ITERS = 5
TIMED_ITERS = 20


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def worker(rank: int, world_size: int, store_file: str, backend: str) -> None:
    # Bind this process exclusively to one GPU so device IDs map 1-to-1 to ranks.
    torch.cuda.set_device(rank)

    # FileStore rendezvous: all processes synchronise on the same file path.
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{store_file}",
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        print(
            f"all_to_all benchmark — backend={backend}, world_size={world_size}, "
            f"warmup={WARMUP_ITERS}, timed={TIMED_ITERS} iters\n"
        )
        header = (
            f"{'per-rank':>12} {'latency (ms)':>14} {'algbw (GB/s)':>14}"
        )
        print(header)
        print("-" * len(header))

    for nbytes in SIZES_BYTES:
        # all_to_all_single splits the buffer evenly into world_size chunks,
        # so the element count must be divisible by world_size.
        n_elem = (nbytes // 4) // world_size * world_size

        input_tensor = torch.full(
            (n_elem,), float(rank + 1), dtype=torch.float32, device="cuda"
        )
        output_tensor = torch.empty(n_elem, dtype=torch.float32, device="cuda")

        # Warmup — lets the backend establish IPC handles / caches.
        for _ in range(WARMUP_ITERS):
            dist.all_to_all_single(output_tensor, input_tensor)
        torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(TIMED_ITERS):
            dist.all_to_all_single(output_tensor, input_tensor)
        end.record()
        torch.cuda.synchronize()

        # Rank 0's local average latency across timed iterations.  No
        # cross-rank reduction (see module docstring).
        elapsed_ms = start.elapsed_time(end) / TIMED_ITERS

        if rank == 0:
            # NCCL-style algorithm bandwidth for all-to-all: each rank sends
            # (world_size-1)/world_size of its buffer over the wire.
            bus_bytes = nbytes * (world_size - 1) / world_size
            algbw_gbps = bus_bytes / (elapsed_ms / 1e3) / 1e9
            print(
                f"{_human_bytes(nbytes):>12} "
                f"{elapsed_ms:>14.4f} {algbw_gbps:>14.2f}"
            )

        dist.barrier()

    if rank == 0:
        print("\nDone.")

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="mini_nccl",
        choices=["mini_nccl", "nccl"],
        help="torch.distributed backend to benchmark (default: mini_nccl).",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to benchmark all-to-all."
        )

    print(f"Launching {world_size} workers, one per GPU.\n")

    # Create a temporary file path for the FileStore rendezvous.
    # The file must not exist when init_process_group is called, so we
    # create it with mkstemp then immediately remove it to reserve the path.
    fd, store_file = tempfile.mkstemp(prefix="mini_nccl_bench_")
    os.close(fd)
    os.unlink(store_file)

    try:
        mp.spawn(
            worker,
            args=(world_size, store_file, args.backend),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(store_file):
            os.unlink(store_file)


if __name__ == "__main__":
    main()
