# mini-nccl

A from-scratch implementation of basic collective communication operations for a custom PyTorch distributed backend. This project is intended as an educational reference for understanding how collectives work at a low level, without the complexity of a full NCCL implementation.

## Scope

mini-nccl targets a **single rack of GPUs** connected in an **all-to-all topology** — every GPU can communicate directly with every other GPU over high-bandwidth interconnects. No multi-hop routing or hierarchical reduction is assumed.

This constraint keeps the implementation simple: every collective can be expressed as a direct exchange of data between any pair of ranks without intermediate forwarding.

## How It Works

The backend plugs into PyTorch's `torch.distributed` API by implementing a custom `ProcessGroup`. Each collective is broken down into primitive point-to-point sends and receives between ranks, taking advantage of the all-to-all connectivity to avoid serialized ring or tree passes where possible.

### Design Principles

- **Flat topology assumed.** All ranks are one hop away from each other, so algorithms are written without topology awareness.
- **Readable over optimal.** Algorithms are written for clarity. A production NCCL kernel would fuse, pipeline, and pack these differently.

## Collectives Implemented

| Collective | Description |
|---|---|
| **Broadcast** | Sends a tensor from a root rank to all other ranks. |
| **Reduce** | Aggregates tensors from all ranks to a root using a reduction op (sum, max, etc.). |
| **All-Reduce** | Reduces tensors across all ranks so every rank ends up with the result. |
| **Scatter** | Distributes distinct chunks of a tensor from a root to each rank. |
| **Gather** | Collects tensors from all ranks onto a single root rank. |
| **All-Gather** | Every rank gathers the full concatenated tensor from all other ranks. |
| **Reduce-Scatter** | Reduces tensors across ranks and scatters the result so each rank holds one chunk. |
| **All-to-All** | Every rank sends a distinct chunk to every other rank simultaneously. |
| **Barrier** | Synchronizes all ranks — no rank proceeds until all have arrived. |

## Getting Started

```bash
pip install torch
git clone https://github.com/bhavya01/mini-nccl.git
cd mini-nccl
```

Try out your first collective by running `uv run examples/allreduce.py`.

Register the backend and initialize a process group:

```python
import torch.distributed as dist
import mini_nccl

dist.init_process_group(backend="mini_nccl", ...)
```

## Limitations

- Single-rack, all-to-all topology only — no hierarchical or multi-rack support.
- CPU and GPU tensors supported; GPU paths use basic CUDA P2P transfers.
- Not optimized for throughput; intended for learning and experimentation.


## GPU topology

Use the following command to check how your GPU's are organized
```shell
nvidia-smi topo -m
```