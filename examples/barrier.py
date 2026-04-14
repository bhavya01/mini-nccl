"""
barrier_example.py — demonstrates dist.barrier using the mini_nccl backend.

Each rank sleeps for (world_size - rank) seconds to arrive at the barrier at
different times, then calls dist.barrier().  All ranks print a "passed" message
only after every rank has arrived, demonstrating that slower ranks hold up
the faster ones.

Usage:
    python barrier.py

Requires at least 2 CUDA-capable GPUs.
"""

import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import mini_nccl  # noqa: F401 — importing registers the "mini_nccl" backend


def worker(rank: int, world_size: int, store_file: str) -> None:
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="mini_nccl",
        init_method=f"file://{store_file}",
        world_size=world_size,
        rank=rank,
    )

    # Stagger arrivals: rank 0 sleeps longest, rank world_size-1 arrives first.
    delay = world_size - rank
    print(f"[rank {rank}] sleeping {delay}s before barrier")
    time.sleep(delay)
    print(f"[rank {rank}] reached barrier")

    dist.barrier()

    print(f"[rank {rank}] PASS   : passed barrier")

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate barrier."
        )

    print(f"Launching {world_size} workers, one per GPU.\n")

    fd, store_file = tempfile.mkstemp(prefix="mini_nccl_")
    os.close(fd)
    os.unlink(store_file)

    try:
        mp.spawn(
            worker,
            args=(world_size, store_file),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(store_file):
            os.unlink(store_file)


if __name__ == "__main__":
    main()
