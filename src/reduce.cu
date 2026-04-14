#include "reduce.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_nccl
{

    void cuda_reduce(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &tensor,
        int root,
        c10d::ReduceOp reduce_op)
    {
        TORCH_CHECK(tensor.is_cuda(),
                    "cuda_reduce: tensor must be a CUDA tensor");
        TORCH_CHECK(tensor.is_contiguous(),
                    "cuda_reduce: tensor must be contiguous");

        const int my_device = tensor.get_device();
        const size_t nbytes = tensor.nbytes();

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "re_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — every rank publishes its IPC handle
        // -------------------------------------------------------------------------

        HandlePayload my_payload;
        my_payload.device_id = my_device;
        CUDA_CHECK(cudaIpcGetMemHandle(&my_payload.ipc_handle, tensor.data_ptr()));

        std::vector<uint8_t> my_bytes(sizeof(HandlePayload));
        std::memcpy(my_bytes.data(), &my_payload, sizeof(HandlePayload));
        store->set(pfx + "handle_" + std::to_string(rank), my_bytes);

        // Block until every rank has published its handle.
        std::vector<std::string> handle_keys;
        handle_keys.reserve(world_size);
        for (int r = 0; r < world_size; ++r)
            handle_keys.push_back(pfx + "handle_" + std::to_string(r));
        store->wait(handle_keys);

        // -------------------------------------------------------------------------
        // Phase 2 — root accumulates all contributions into its tensor
        //
        // Root's own tensor is the initial value; each peer's data is pulled via
        // peer-to-peer copy into a scratch buffer and then reduced in-place.
        // -------------------------------------------------------------------------

        if (rank == root)
        {
            c10::cuda::CUDAGuard guard(my_device);
            at::Tensor scratch = at::empty_like(tensor);

            for (int r = 0; r < world_size; ++r)
            {
                if (r == root)
                    continue;

                auto remote_bytes = store->get(pfx + "handle_" + std::to_string(r));
                TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                            "cuda_reduce: unexpected handle payload size from rank ", r);

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

                // Copy the peer's tensor into our scratch buffer.
                CUDA_CHECK(cudaMemcpyPeer(
                    scratch.data_ptr(), my_device,
                    remote_ptr, remote_payload.device_id,
                    nbytes));

                // Reduce the scratch buffer into the root's tensor.
                apply_reduce(tensor, scratch, reduce_op);

                {
                    c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                    CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
                }
            }

            // For AVG, divide by world_size after all contributions are summed.
            if (reduce_op == c10d::ReduceOp::AVG)
                tensor.div_(static_cast<double>(world_size));
        }

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Non-root ranks must not dismantle their tensors before root finishes.
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
