#include "allreduce.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_nccl
{

    void cuda_allreduce(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &tensor,
        c10d::ReduceOp reduce_op)
    {
        TORCH_CHECK(tensor.is_cuda(),
                    "cuda_allreduce: tensor must be a CUDA tensor");
        TORCH_CHECK(tensor.is_contiguous(),
                    "cuda_allreduce: tensor must be contiguous");

        const int my_device = tensor.get_device();
        const size_t nbytes = tensor.nbytes();

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "ar_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — publish this rank's IPC handle via the store
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
        // Phase 2 — open each peer's handle, pull its data, and reduce locally
        //
        // A single scratch buffer is reused for each peer to bound memory use.
        // The reduction is applied immediately after each copy.
        // -------------------------------------------------------------------------

        c10::cuda::CUDAGuard guard(my_device);
        at::Tensor scratch = at::empty_like(tensor);

        for (int r = 0; r < world_size; ++r)
        {
            if (r == rank)
                continue;

            // Deserialize the remote rank's handle payload.
            auto remote_bytes = store->get(pfx + "handle_" + std::to_string(r));
            TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                        "cuda_allreduce: unexpected handle payload size from rank ", r);

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

            // Copy the peer's tensor into our local scratch buffer.
            // cudaMemcpyPeer works across device boundaries without requiring the
            // current device to be either source or destination.
            CUDA_CHECK(cudaMemcpyPeer(
                scratch.data_ptr(), my_device,
                remote_ptr, remote_payload.device_id,
                nbytes));

            // Reduce the scratch buffer into our local tensor.
            apply_reduce(tensor, scratch, reduce_op);

            // Close the IPC mapping on the device it was opened on.
            {
                c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
            }
        }

        // For AVG, divide by world_size after all contributions have been summed.
        if (reduce_op == c10d::ReduceOp::AVG)
            tensor.div_(static_cast<double>(world_size));

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Ensure no rank dismantles its tensor before every peer has finished
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
