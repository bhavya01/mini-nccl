"""
reducescatter_example.py — demonstrates dist.reduce_scatter using the mini_nccl backend.

Each rank fills world_size chunks with a distinct value (rank + 1). After
reduce_scatter, rank r holds the SUM of chunk r across all ranks, which equals
world_size * (world_size + 1) / 2.

Layout (world_size = 4, chunk_size = 2):
  rank 0 input: [[1,1], [1,1], [1,1], [1,1]]
  rank 1 input: [[2,2], [2,2], [2,2], [2,2]]
  rank 2 input: [[3,3], [3,3], [3,3], [3,3]]
  rank 3 input: [[4,4], [4,4], [4,4], [4,4]]

  rank 0 output (chunk 0): [10, 10]   (1+2+3+4)
  rank 1 output (chunk 1): [10, 10]
  ...

Usage:
    python reducescatter_example.py

Requires at least 2 CUDA-capable GPUs.
"""

import os
import tempfile

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

    chunk_size = 4

    # Each rank contributes world_size chunks, all filled with float(rank + 1).
    # input_list[r] is this rank's contribution to chunk r.
    input_list = [
        torch.full((chunk_size,), float(rank + 1), dtype=torch.float32, device="cuda")
        for _ in range(world_size)
    ]
    output = torch.zeros(chunk_size, dtype=torch.float32, device="cuda")

    print(f"[rank {rank}] input chunks: {[t.tolist() for t in input_list]}")

    dist.reduce_scatter(output, input_list)

    print(f"[rank {rank}] output (chunk {rank}): {output.tolist()}")

    expected = float(world_size * (world_size + 1) // 2)
    if torch.all(output == expected).item():
        print(f"[rank {rank}] PASS: all elements equal {expected:.0f}")
    else:
        print(f"[rank {rank}] FAIL: expected {expected:.0f}, got {output.tolist()}")

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate reduce-scatter."
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
