#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <vector>

namespace mini_nccl
{

    // Performs a scatter collective using CUDA IPC and peer-to-peer transfers.
    //
    // The root rank splits inputTensors (world_size chunks, one per rank) and
    // delivers each chunk to the corresponding rank's outputTensor.  Only the
    // root publishes IPC handles; every other rank simply opens the handle for
    // its own chunk and copies it locally.  A three-phase protocol is used:
    //
    //   1. Root copies each inputTensors[r] into a raw-cudaMalloc staging buffer
    //      and publishes an IPC handle keyed by destination rank.  Raw cudaMalloc
    //      is required so every handle is for a distinct base pointer (the same
    //      reason as in all-to-all).
    //   2. Each rank (including root) copies its chunk out of root's staging into
    //      outputTensor; root does a local device-to-device copy.
    //   3. A completion barrier ensures root does not free its staging buffers
    //      before every rank has finished reading.
    //
    // Parameters:
    //   store        - c10d key-value store used for rendezvous
    //   rank         - this process's rank in [0, world_size)
    //   world_size   - total number of ranks
    //   seq          - monotonically increasing call counter (makes store keys unique)
    //   outputTensor - contiguous CUDA tensor; receives this rank's chunk from root
    //   inputTensors - world_size-length vector on the root (inputTensors[r] is the
    //                  chunk destined for rank r); empty on non-root ranks
    //   root         - rank of the process that scatters data
    void cuda_scatter(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &outputTensor,
        std::vector<at::Tensor> &inputTensors,
        int root);

} // namespace mini_nccl
