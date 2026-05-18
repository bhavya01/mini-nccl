"""
allgather.py — benchmarks dist.all_gather using the mini_nccl backend.

Sweeps a range of per-rank tensor sizes, times the all_gather collective over
many iterations on each, and reports latency and algorithmic bandwidth.

"Algorithm bandwidth" here follows the NCCL convention for all-gather:
    bus_bytes = input_bytes * (world_size - 1)
i.e. the bytes that must cross the interconnect per rank, independent of how
the collective is implemented.

Usage:
    uv run python allgather.py [--backend {mini_nccl,nccl}]

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
            f"all_gather benchmark — backend={backend}, world_size={world_size}, "
            f"warmup={WARMUP_ITERS}, timed={TIMED_ITERS} iters\n"
        )
        header = (
            f"{'per-rank':>12} {'total out':>12} "
            f"{'latency (ms)':>14} {'algbw (GB/s)':>14}"
        )
        print(header)
        print("-" * len(header))

    for nbytes in SIZES_BYTES:
        n_elem = nbytes // 4  # float32

        input_tensor = torch.full(
            (n_elem,), float(rank + 1), dtype=torch.float32, device="cuda"
        )
        output_tensors = [
            torch.empty(n_elem, dtype=torch.float32, device="cuda")
            for _ in range(world_size)
        ]

        # Warmup — lets the backend establish IPC handles / caches.
        for _ in range(WARMUP_ITERS):
            dist.all_gather(output_tensors, input_tensor)
        torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(TIMED_ITERS):
            dist.all_gather(output_tensors, input_tensor)
        end.record()
        torch.cuda.synchronize()

        # Per-rank average latency across timed iterations.
        elapsed_ms = start.elapsed_time(end) / TIMED_ITERS

        if rank == 0:
            # NCCL-style algorithm bandwidth for all-gather.
            bus_bytes = nbytes * (world_size - 1)
            algbw_gbps = bus_bytes / (elapsed_ms / 1e3) / 1e9
            total_out = nbytes * world_size
            print(
                f"{_human_bytes(nbytes):>12} {_human_bytes(total_out):>12} "
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
            f"Found {world_size} GPU(s) — need at least 2 to benchmark all-gather."
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
