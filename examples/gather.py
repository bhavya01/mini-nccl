"""
gather_example.py — demonstrates dist.gather using the mini_nccl backend.

Each rank r fills its input tensor with float(r + 1).  After gather, rank 0
(root) holds all world_size tensors; every other rank's output list is empty.

Layout (world_size = 4, tensor_size = 8):
  rank 0 input: [1. 1. 1. 1. 1. 1. 1. 1.]
  rank 1 input: [2. 2. 2. 2. 2. 2. 2. 2.]
  rank 2 input: [3. 3. 3. 3. 3. 3. 3. 3.]
  rank 3 input: [4. 4. 4. 4. 4. 4. 4. 4.]

  root gather_list[0]: [1. 1. 1. 1. 1. 1. 1. 1.]
  root gather_list[1]: [2. 2. 2. 2. 2. 2. 2. 2.]
  root gather_list[2]: [3. 3. 3. 3. 3. 3. 3. 3.]
  root gather_list[3]: [4. 4. 4. 4. 4. 4. 4. 4.]

Usage:
    python gather.py

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

    # Each rank contributes a tensor filled with float(rank + 1).
    input_tensor = torch.full(
        (tensor_size,), float(rank + 1), dtype=torch.float32, device="cuda"
    )
    print(f"[rank {rank}] input  : {input_tensor.tolist()}")

    # Root pre-allocates output slots; other ranks pass None.
    if rank == ROOT_RANK:
        gather_list = [
            torch.empty(tensor_size, dtype=torch.float32, device="cuda")
            for _ in range(world_size)
        ]
    else:
        gather_list = None

    # Gather: every rank sends input_tensor to root.
    dist.gather(input_tensor, gather_list, dst=ROOT_RANK)

    if rank == ROOT_RANK:
        print(f"[rank {rank}] gather_list: {[t.tolist() for t in gather_list]}")

        # Correctness check: gather_list[r] should be all (r + 1).
        passed = True
        for r in range(world_size):
            expected = float(r + 1)
            if not torch.all(gather_list[r] == expected).item():
                print(
                    f"[rank {rank}] FAIL  : slot {r} — "
                    f"expected all {expected}, got {gather_list[r].tolist()}"
                )
                passed = False

        if passed:
            print(f"[rank {rank}] PASS   : all {world_size} slots correct")

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate gather."
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
