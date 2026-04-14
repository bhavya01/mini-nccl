#pragma once

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <vector>

namespace mini_nccl
{

    // Performs a gather collective using CUDA IPC and peer-to-peer transfers.
    //
    // Every rank publishes an IPC handle for its inputTensor.  The root then
    // opens each handle and copies the data into the corresponding slot of
    // outputTensors.  Non-root ranks participate only in the barrier protocol;
    // they do not copy any data.  A three-phase protocol is used:
    //
    //   1. Every rank serialises a HandlePayload (IPC handle + device ordinal)
    //      and writes it to the store.  All ranks block until every handle is
    //      available.
    //   2. Root copies each peer's inputTensor into outputTensors[r] via
    //      cudaMemcpyPeer; its own contribution is copied device-to-device.
    //   3. A completion barrier ensures no rank dismantles its inputTensor
    //      before root has finished reading from it.
    //
    // Parameters:
    //   store        - c10d key-value store used for rendezvous
    //   rank         - this process's rank in [0, world_size)
    //   world_size   - total number of ranks
    //   seq          - monotonically increasing call counter (makes store keys unique)
    //   outputTensors - world_size-length vector on root (outputTensors[r] receives
    //                   rank r's inputTensor); empty on non-root ranks
    //   inputTensor  - contiguous CUDA tensor; this rank's contribution
    //   root         - rank of the process that collects data
    void cuda_gather(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        std::vector<at::Tensor> &outputTensors,
        const at::Tensor &inputTensor,
        int root);

} // namespace mini_nccl
