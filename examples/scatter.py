"""
scatter_example.py — demonstrates dist.scatter using the mini_nccl backend.

Rank 0 (root) holds world_size input tensors; tensor r is filled with
float(r + 1).  After scatter, every rank r holds a tensor of all (r + 1).

Layout (world_size = 4, tensor_size = 8):
  root input[0]: [1. 1. 1. 1. 1. 1. 1. 1.]   → rank 0
  root input[1]: [2. 2. 2. 2. 2. 2. 2. 2.]   → rank 1
  root input[2]: [3. 3. 3. 3. 3. 3. 3. 3.]   → rank 2
  root input[3]: [4. 4. 4. 4. 4. 4. 4. 4.]   → rank 3

  rank 0 output: [1. 1. 1. 1. 1. 1. 1. 1.]
  rank 1 output: [2. 2. 2. 2. 2. 2. 2. 2.]
  ...

Usage:
    python scatter.py

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

    # Root prepares world_size input tensors; tensor r is filled with float(r + 1).
    if rank == ROOT_RANK:
        scatter_list = [
            torch.full(
                (tensor_size,), float(r + 1), dtype=torch.float32, device="cuda"
            )
            for r in range(world_size)
        ]
        print(f"[rank {rank}] scatter_list: {[t.tolist() for t in scatter_list]}")
    else:
        scatter_list = []

    output = torch.empty(tensor_size, dtype=torch.float32, device="cuda")

    # Scatter: root sends scatter_list[r] to rank r.
    dist.scatter(output, scatter_list, src=ROOT_RANK)

    print(f"[rank {rank}] output: {output.tolist()}")

    # Correctness check: rank r should hold all (r + 1).
    expected = float(rank + 1)
    if torch.all(output == expected).item():
        print(f"[rank {rank}] PASS   : all elements equal {expected:.0f}")
    else:
        print(
            f"[rank {rank}] FAIL   : expected all {expected:.0f}, got {output.tolist()}"
        )

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate scatter."
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
