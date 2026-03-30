"""
allreduce_example.py — demonstrates dist.all_reduce using the mini_nccl backend.

Each rank fills its tensor with a distinct value (rank + 1), runs all-reduce
with the default SUM op, then verifies that every rank holds the same result:
the sum of all rank values (1 + 2 + ... + world_size).

Usage:
    python allreduce_example.py

Requires at least 2 CUDA-capable GPUs.  The script spawns one process per
visible GPU using torch.multiprocessing.spawn (start method: 'spawn'), which
gives each process a clean CUDA context — a requirement for CUDA IPC.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import mini_nccl  # noqa: F401 — importing registers the "mini_nccl" backend


def worker(rank: int, world_size: int, store_file: str) -> None:
    # Bind this process exclusively to one GPU so device IDs map 1-to-1 to ranks.
    torch.cuda.set_device(rank)

    # FileStore rendezvous: all processes synchronise on the same file path.
    dist.init_process_group(
        backend="mini_nccl",
        init_method=f"file://{store_file}",
        world_size=world_size,
        rank=rank,
    )

    tensor_size = 8

    # Input: rank r sends a tensor filled with float(r + 1).
    #   rank 0 → [1. 1. 1. 1. 1. 1. 1. 1.]
    #   rank 1 → [2. 2. 2. 2. 2. 2. 2. 2.]
    #   ...
    tensor = torch.full(
        (tensor_size,), float(rank + 1), dtype=torch.float32, device="cuda"
    )
    print(f"[rank {rank}] input  : {tensor.tolist()}")

    # All-reduce (SUM): after this call every rank holds the element-wise sum
    # of all ranks' inputs, i.e. 1 + 2 + ... + world_size.
    dist.all_reduce(tensor)

    print(f"[rank {rank}] output : {tensor.tolist()}")

    # Correctness check: every element should equal world_size * (world_size + 1) / 2.
    expected = float(world_size * (world_size + 1) // 2)
    if torch.all(tensor == expected).item():
        print(f"[rank {rank}] PASS   : all elements equal {expected:.0f}")
    else:
        print(
            f"[rank {rank}] FAIL   : expected all {expected:.0f}, got {tensor.tolist()}"
        )

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate all-reduce."
        )

    print(f"Launching {world_size} workers, one per GPU.\n")

    # Create a temporary file path for the FileStore rendezvous.
    # The file must not exist when init_process_group is called, so we
    # create it with mkstemp then immediately remove it to reserve the path.
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
        # Clean up the store file if it was left behind.
        if os.path.exists(store_file):
            os.unlink(store_file)


if __name__ == "__main__":
    main()
