"""
alltoall_example.py — demonstrates dist.all_to_all_single using the mini_nccl backend.

Each rank fills world_size chunks with a distinct pattern: rank r fills chunk s
with the value float(rank * world_size + s + 1).  After all-to-all, rank r holds
chunk r from every other rank — i.e. the data is transposed across ranks.

Layout (world_size = 4, chunk_size = 2):
  rank 0 input:  [ 1, 1,  2, 2,  3, 3,  4, 4]   (chunks 0-3, value = chunk+1)
  rank 1 input:  [ 5, 5,  6, 6,  7, 7,  8, 8]
  rank 2 input:  [ 9, 9, 10,10, 11,11, 12,12]
  rank 3 input:  [13,13, 14,14, 15,15, 16,16]

  rank 0 output (chunk r from rank r): [ 1, 1,  5, 5,  9, 9, 13,13]
  rank 1 output:                       [ 2, 2,  6, 6, 10,10, 14,14]
  rank 2 output:                       [ 3, 3,  7, 7, 11,11, 15,15]
  rank 3 output:                       [ 4, 4,  8, 8, 12,12, 16,16]

Usage:
    python alltoall_example.py

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

    chunk_size = 4

    # Build the input tensor: world_size chunks, each filled with a unique value.
    # Chunk s of rank r gets value float(rank * world_size + s + 1) so every
    # element in the entire world is distinct and easy to verify.
    chunks = [
        torch.full(
            (chunk_size,),
            float(rank * world_size + s + 1),
            dtype=torch.float32,
            device="cuda",
        )
        for s in range(world_size)
    ]
    input_tensor = torch.cat(chunks)
    output_tensor = torch.empty_like(input_tensor)

    print(f"[rank {rank}] input  : {input_tensor.tolist()}")

    # All-to-all: rank r's chunk s → rank s's output chunk r.
    dist.all_to_all_single(output_tensor, input_tensor)

    print(f"[rank {rank}] output : {output_tensor.tolist()}")

    # Correctness check: output chunk s should contain rank s's chunk `rank`,
    # which was filled with float(s * world_size + rank + 1).
    passed = True
    for s in range(world_size):
        expected = float(s * world_size + rank + 1)
        got = output_tensor[s * chunk_size : (s + 1) * chunk_size]
        if not torch.all(got == expected).item():
            print(
                f"[rank {rank}] FAIL  : output chunk {s} — "
                f"expected all {expected:.0f}, got {got.tolist()}"
            )
            passed = False

    if passed:
        print(f"[rank {rank}] PASS   : all {world_size} output chunks correct")

    dist.destroy_process_group()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(
            f"Found {world_size} GPU(s) — need at least 2 to demonstrate all-to-all."
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
        if os.path.exists(store_file):
            os.unlink(store_file)


if __name__ == "__main__":
    main()
