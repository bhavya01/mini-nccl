#include "broadcast.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_nccl
{

    void cuda_broadcast(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &tensor,
        int root)
    {
        TORCH_CHECK(tensor.is_cuda(),
                    "cuda_broadcast: tensor must be a CUDA tensor");
        TORCH_CHECK(tensor.is_contiguous(),
                    "cuda_broadcast: tensor must be contiguous");

        const int my_device = tensor.get_device();
        const size_t nbytes = tensor.nbytes();

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "bc_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — root publishes its IPC handle; all ranks wait for it
        // -------------------------------------------------------------------------

        if (rank == root)
        {
            HandlePayload payload;
            payload.device_id = my_device;
            CUDA_CHECK(cudaIpcGetMemHandle(&payload.ipc_handle, tensor.data_ptr()));

            std::vector<uint8_t> bytes(sizeof(HandlePayload));
            std::memcpy(bytes.data(), &payload, sizeof(HandlePayload));
            store->set(pfx + "handle", bytes);
        }

        store->wait({pfx + "handle"});

        // -------------------------------------------------------------------------
        // Phase 2 — every non-root rank copies root's tensor into its own
        // -------------------------------------------------------------------------

        if (rank != root)
        {
            auto remote_bytes = store->get(pfx + "handle");
            TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                        "cuda_broadcast: unexpected handle payload size from root");

            HandlePayload remote_payload;
            std::memcpy(&remote_payload, remote_bytes.data(), sizeof(HandlePayload));

            // IPC handles must be opened on the device they were created on.
            void *remote_ptr = nullptr;
            {
                c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    &remote_ptr,
                    remote_payload.ipc_handle,
                    cudaIpcMemLazyEnablePeerAccess));
            }

            CUDA_CHECK(cudaMemcpyPeer(
                tensor.data_ptr(), my_device,
                remote_ptr, remote_payload.device_id,
                nbytes));

            {
                c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
            }
        }

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Root must not modify or free its tensor until every rank has finished
        // reading from it.
        // -------------------------------------------------------------------------

        std::vector<uint8_t> done_signal = {1};
        store->set(pfx + "done_" + std::to_string(rank), done_signal);

        std::vector<std::string> done_keys;
        done_keys.reserve(world_size);
        for (int r = 0; r < world_size; ++r)
            done_keys.push_back(pfx + "done_" + std::to_string(r));
        store->wait(done_keys);
    }

} // namespace mini_nccl
