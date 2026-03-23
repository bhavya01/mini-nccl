#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <vector>

namespace mini_nccl
{

    // Performs an all-gather collective using CUDA IPC and peer-to-peer transfers.
    //
    // Each rank publishes an IPC handle for its inputTensor to the store, then
    // opens every other rank's handle and copies their data into the corresponding
    // slot of outputTensors.  A two-phase store barrier ensures no rank tears down
    // its input buffer before every other rank has finished reading from it.
    //
    // Parameters:
    //   store        - c10d key-value store used for rendezvous
    //   rank         - this process's rank in [0, size)
    //   size         - total number of ranks
    //   seq          - monotonically increasing call counter (makes store keys unique)
    //   inputTensor  - local contribution; must be a contiguous CUDA tensor
    //   outputTensors - size-length vector; slot r receives rank r's inputTensor
    void cuda_allgather(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        const at::Tensor &inputTensor,
        std::vector<at::Tensor> &outputTensors);

} // namespace mini_nccl
