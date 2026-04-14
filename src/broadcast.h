#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace mini_nccl
{

    // Performs a broadcast collective using CUDA IPC and peer-to-peer transfers.
    //
    // The root rank's tensor is copied into every other rank's tensor so that
    // all ranks end up with identical data.  A two-phase protocol is used:
    //
    //   1. Root publishes an IPC handle for its tensor.  All ranks block until
    //      the handle is available.
    //   2. Every non-root rank opens root's IPC handle and copies the data into
    //      its own tensor via cudaMemcpyPeer.  Root does nothing in this phase
    //      (it already holds the authoritative data).
    //   3. A completion barrier ensures root's tensor is not freed or modified
    //      before every rank has finished reading from it.
    //
    // Parameters:
    //   store      - c10d key-value store used for rendezvous
    //   rank       - this process's rank in [0, world_size)
    //   world_size - total number of ranks
    //   seq        - monotonically increasing call counter (makes store keys unique)
    //   tensor     - contiguous CUDA tensor; root's value is broadcast to all others
    //   root       - rank whose tensor is the source of truth
    void cuda_broadcast(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &tensor,
        int root);

} // namespace mini_nccl
