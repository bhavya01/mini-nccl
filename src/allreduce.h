#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

namespace mini_nccl
{

    // Performs an all-reduce collective using CUDA IPC and peer-to-peer transfers.
    //
    // Each rank publishes an IPC handle for its tensor to the store, then opens
    // every other rank's handle and copies their data into a local scratch buffer.
    // The scratch buffer is reduced into the local tensor using the requested
    // reduction op.  A completion barrier ensures no rank dismantles its buffer
    // before all peers have finished reading from it.
    //
    // Parameters:
    //   store      - c10d key-value store used for rendezvous
    //   rank       - this process's rank in [0, size)
    //   world_size - total number of ranks
    //   seq        - monotonically increasing call counter (makes store keys unique)
    //   tensor     - local tensor; modified in-place to hold the reduced result
    //   reduce_op  - reduction operation (SUM, PRODUCT, MIN, MAX, etc.)
    void cuda_allreduce(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &tensor,
        c10d::ReduceOp reduce_op);

} // namespace mini_nccl
