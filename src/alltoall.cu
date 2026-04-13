#include "alltoall.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_nccl
{

    void cuda_alltoall(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        const at::Tensor &inputTensor,
        at::Tensor &outputTensor)
    {
        TORCH_CHECK(inputTensor.is_cuda(),
                    "cuda_alltoall: inputTensor must be a CUDA tensor");
        TORCH_CHECK(inputTensor.is_contiguous(),
                    "cuda_alltoall: inputTensor must be contiguous");
        TORCH_CHECK(outputTensor.is_cuda(),
                    "cuda_alltoall: outputTensor must be a CUDA tensor");
        TORCH_CHECK(outputTensor.is_contiguous(),
                    "cuda_alltoall: outputTensor must be contiguous");
        TORCH_CHECK(inputTensor.nbytes() == outputTensor.nbytes(),
                    "cuda_alltoall: inputTensor and outputTensor must have the same byte size");
        TORCH_CHECK(inputTensor.nbytes() % static_cast<size_t>(world_size) == 0,
                    "cuda_alltoall: inputTensor byte size must be divisible by world_size");

        const int my_device = inputTensor.get_device();
        const size_t total_bytes = inputTensor.nbytes();
        const size_t chunk_bytes = total_bytes / static_cast<size_t>(world_size);

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "a2a_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — copy each input chunk into a raw-cudaMalloc staging buffer and
        //           publish an IPC handle for each.
        //
        // All-to-all semantics: rank r's input chunk s goes to rank s's output
        // chunk r.  Rank r therefore publishes world_size handles keyed by
        // "handle_r_c" (rank r, chunk c).  Rank s will later fetch "handle_r_s"
        // to retrieve rank r's chunk destined for it.
        //
        // We bypass PyTorch's caching allocator and use raw cudaMalloc so that
        // every staging pointer is the base of its own cudaMalloc allocation.
        // cudaIpcGetMemHandle requires a cudaMalloc base pointer; using PyTorch
        // tensors for small chunks is unreliable because the allocator
        // sub-allocates multiple tensors within a single 2 MB block, causing
        // cudaIpcGetMemHandle to return the same block handle for all of them.
        // -------------------------------------------------------------------------

        const char *in_ptr = static_cast<const char *>(inputTensor.data_ptr());

        c10::cuda::CUDAGuard my_guard(my_device);

        std::vector<void *> staging(world_size, nullptr);
        for (int c = 0; c < world_size; ++c)
        {
            CUDA_CHECK(cudaMalloc(&staging[c], chunk_bytes));

            CUDA_CHECK(cudaMemcpy(
                staging[c],
                in_ptr + static_cast<size_t>(c) * chunk_bytes,
                chunk_bytes,
                cudaMemcpyDeviceToDevice));

            HandlePayload payload;
            payload.device_id = my_device;
            CUDA_CHECK(cudaIpcGetMemHandle(&payload.ipc_handle, staging[c]));

            std::vector<uint8_t> bytes(sizeof(HandlePayload));
            std::memcpy(bytes.data(), &payload, sizeof(HandlePayload));
            store->set(pfx + "handle_" + std::to_string(rank) + "_" + std::to_string(c), bytes);
        }

        // Block until every rank has published all of its handles.
        std::vector<std::string> handle_keys;
        handle_keys.reserve(static_cast<size_t>(world_size) * world_size);
        for (int r = 0; r < world_size; ++r)
            for (int c = 0; c < world_size; ++c)
                handle_keys.push_back(pfx + "handle_" + std::to_string(r) + "_" + std::to_string(c));
        store->wait(handle_keys);

        // -------------------------------------------------------------------------
        // Phase 2 — copy each peer's chunk `rank` into our output
        //
        // Rank s fetches "handle_r_s" (rank r's staging buffer for chunk s) and
        // copies it into output chunk r.  The IPC pointer is always used at
        // offset 0 (it equals the cudaMalloc base), so the copy is well-defined.
        // -------------------------------------------------------------------------

        char *out_ptr = static_cast<char *>(outputTensor.data_ptr());

        // Local copy: our own staging[rank] → output chunk `rank`.
        CUDA_CHECK(cudaMemcpy(
            out_ptr + static_cast<size_t>(rank) * chunk_bytes,
            staging[rank],
            chunk_bytes,
            cudaMemcpyDeviceToDevice));

        for (int r = 0; r < world_size; ++r)
        {
            if (r == rank)
                continue;

            // Fetch rank r's handle for chunk `rank` (the chunk destined for us).
            auto remote_bytes = store->get(
                pfx + "handle_" + std::to_string(r) + "_" + std::to_string(rank));
            TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                        "cuda_alltoall: unexpected handle payload size from rank ", r);

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

            // Offset-free P2P copy: remote_ptr is the cudaMalloc base for rank r's
            // chunk `rank`, so no pointer arithmetic is needed.
            CUDA_CHECK(cudaMemcpyPeer(
                out_ptr + static_cast<size_t>(r) * chunk_bytes, my_device,
                remote_ptr, remote_payload.device_id,
                chunk_bytes));

            {
                c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
            }
        }

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Ensure no rank frees its staging buffers until every peer has finished
        // reading from them.
        // -------------------------------------------------------------------------

        std::vector<uint8_t> done_signal = {1};
        store->set(pfx + "done_" + std::to_string(rank), done_signal);

        std::vector<std::string> done_keys;
        done_keys.reserve(world_size);
        for (int r = 0; r < world_size; ++r)
            done_keys.push_back(pfx + "done_" + std::to_string(r));
        store->wait(done_keys);

        // Free staging buffers only after all peers have confirmed they are done.
        for (int c = 0; c < world_size; ++c)
            CUDA_CHECK(cudaFree(staging[c]));
    }

} // namespace mini_nccl
