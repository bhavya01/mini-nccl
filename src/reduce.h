#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

namespace mini_nccl
{

    // Performs a reduce collective using CUDA IPC and peer-to-peer transfers.
    //
    // Every rank contributes its tensor; the root accumulates all contributions
    // using the requested reduction op and stores the result in its own tensor.
    // Non-root ranks' tensors are unchanged after the call.  A three-phase
    // protocol is used:
    //
    //   1. Every rank publishes an IPC handle for its tensor.  All ranks block
    //      until every handle is available.
    //   2. Root seeds its output with its own tensor value, then opens each
    //      non-root peer's handle, copies the data into a scratch buffer, and
    //      reduces it into the output tensor.  Non-root ranks skip this phase.
    //   3. A completion barrier ensures no rank frees its tensor before root
    //      has finished reading from it.
    //
    // Parameters:
    //   store      - c10d key-value store used for rendezvous
    //   rank       - this process's rank in [0, world_size)
    //   world_size - total number of ranks
    //   seq        - monotonically increasing call counter (makes store keys unique)
    //   tensor     - contiguous CUDA tensor; only root's tensor holds the result
    //   root       - rank that collects and reduces all contributions
    //   reduce_op  - reduction operation (SUM, PRODUCT, MIN, MAX, etc.)
    void cuda_reduce(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &tensor,
        int root,
        c10d::ReduceOp reduce_op);

} // namespace mini_nccl
