"""
allgather_example.py — demonstrates dist.all_gather using the mini_nccl backend.

Each rank fills its input tensor with a distinct value (rank + 1), runs
all-gather, then verifies that every output slot holds the expected data.

Usage:
    python allgather_example.py

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
    input_tensor = torch.full(
        (tensor_size,), float(rank + 1), dtype=torch.float32, device="cuda"
    )
    print(f"[rank {rank}] input  : {input_tensor.tolist()}")

    # Output: one pre-allocated tensor per rank, all on this rank's GPU.
    output_tensors = [
        torch.empty(tensor_size, dtype=torch.float32, device="cuda")
        for _ in range(world_size)
    ]

    # All-gather: after this call output_tensors[r] holds rank r's input.
    dist.all_gather(output_tensors, input_tensor)

    print(f"[rank {rank}] outputs: {[t.tolist() for t in output_tensors]}")

    # Correctness check: output_tensors[r] should be all (r + 1).
    passed = True
    for r in range(world_size):
        expected = float(r + 1)
        if not torch.all(output_tensors[r] == expected).item():
            print(
                f"[rank {rank}] FAIL  : slot {r} — "
                f"expected all {expected}, got {output_tensors[r].tolist()}"
            )
            passed = False

    if passed:
        print(f"[rank {rank}] PASS   : all {world_size} slots correct")

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate all-gather."
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
