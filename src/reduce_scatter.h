#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <vector>

namespace mini_nccl
{

    // Performs a reduce-scatter collective using CUDA IPC and peer-to-peer transfers.
    //
    // The input is split into world_size equal chunks (one per rank). Each rank
    // reduces chunk r across all ranks and writes the result to outputTensor.
    // Rank r therefore ends up holding the reduction of inputTensors[r] from
    // every rank.
    //
    // Parameters:
    //   store        - c10d key-value store used for rendezvous
    //   rank         - this process's rank in [0, world_size)
    //   world_size   - total number of ranks
    //   seq          - monotonically increasing call counter (makes store keys unique)
    //   outputTensor - receives the reduced result for this rank's chunk; must be
    //                  a contiguous CUDA tensor of size N/world_size
    //   inputTensors - world_size-length vector of contiguous CUDA tensors, each of
    //                  size N/world_size; inputTensors[r] is this rank's contribution
    //                  to chunk r
    //   reduce_op    - reduction operation (SUM, PRODUCT, MIN, MAX, etc.)
    void cuda_reduce_scatter(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &outputTensor,
        std::vector<at::Tensor> &inputTensors,
        c10d::ReduceOp reduce_op);

} // namespace mini_nccl
