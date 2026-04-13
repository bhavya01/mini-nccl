#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace mini_nccl
{

    // Performs an all-to-all collective using CUDA IPC and peer-to-peer transfers.
    //
    // inputTensor is conceptually split into world_size equal chunks.  The
    // collective transposes these chunks across ranks: rank r's chunk s is
    // delivered to rank s's output chunk r.  Equivalently, after the call:
    //
    //   outputTensor[chunk r] == rank r's inputTensor[chunk rank]
    //
    // A three-phase protocol is used:
    //   1. Each rank copies every input chunk into its own raw-cudaMalloc staging
    //      buffer and publishes world_size IPC handles (one per chunk).  Raw
    //      cudaMalloc is used so that every handle is for a distinct base
    //      pointer; PyTorch's caching allocator may sub-allocate multiple small
    //      tensors in the same cudaMalloc block, producing identical handles.
    //   2. Each rank opens each peer's handle for chunk `rank` (the chunk
    //      destined for this rank) and copies it directly into the appropriate
    //      output slot via an offset-free cudaMemcpyPeer.
    //   3. A completion barrier ensures no rank frees its staging buffers before
    //      every peer has finished reading from them.
    //
    // Parameters:
    //   store        - c10d key-value store used for rendezvous
    //   rank         - this process's rank in [0, world_size)
    //   world_size   - total number of ranks
    //   seq          - monotonically increasing call counter (makes store keys unique)
    //   inputTensor  - contiguous CUDA tensor; total byte size must be divisible
    //                  by world_size
    //   outputTensor - contiguous CUDA tensor of the same shape as inputTensor;
    //                  receives the transposed contributions
    void cuda_alltoall(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        const at::Tensor &inputTensor,
        at::Tensor &outputTensor);

} // namespace mini_nccl
