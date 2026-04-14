"""
broadcast_example.py — demonstrates dist.broadcast using the mini_nccl backend.

Rank 0 (root) fills its tensor with 42.0; all other ranks fill theirs with 0.0.
After broadcast, every rank holds 42.0.

Usage:
    python broadcast.py

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

ROOT_RANK = 0


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

    # Root fills with 42.0; all others start at 0.0.
    if rank == ROOT_RANK:
        tensor = torch.full(
            (tensor_size,), 42.0, dtype=torch.float32, device="cuda"
        )
    else:
        tensor = torch.zeros(tensor_size, dtype=torch.float32, device="cuda")

    print(f"[rank {rank}] before: {tensor.tolist()}")

    # Broadcast: root's tensor is sent to every rank.
    dist.broadcast(tensor, src=ROOT_RANK)

    print(f"[rank {rank}] after : {tensor.tolist()}")

    # Correctness check: every element should equal 42.0.
    if torch.all(tensor == 42.0).item():
        print(f"[rank {rank}] PASS   : all elements equal 42.0")
    else:
        print(f"[rank {rank}] FAIL   : expected all 42.0, got {tensor.tolist()}")

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate broadcast."
        )

    print(f"Launching {world_size} workers, one per GPU.\n")

    # Create a temporary file path for the FileStore rendezvous.
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
