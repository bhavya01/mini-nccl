#include "scatter.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_nccl
{

    void cuda_scatter(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &outputTensor,
        std::vector<at::Tensor> &inputTensors,
        int root)
    {
        TORCH_CHECK(outputTensor.is_cuda(),
                    "cuda_scatter: outputTensor must be a CUDA tensor");
        TORCH_CHECK(outputTensor.is_contiguous(),
                    "cuda_scatter: outputTensor must be contiguous");
        if (rank == root)
        {
            TORCH_CHECK(static_cast<int>(inputTensors.size()) == world_size,
                        "cuda_scatter: root's inputTensors.size() (", inputTensors.size(),
                        ") must equal world_size (", world_size, ")");
        }

        const int my_device = outputTensor.get_device();
        const size_t nbytes = outputTensor.nbytes();

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "sc_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — root copies each input chunk into a raw-cudaMalloc staging
        //           buffer and publishes an IPC handle keyed by destination rank.
        //
        // Raw cudaMalloc is used so every handle is for a distinct base pointer.
        // PyTorch's caching allocator may sub-allocate multiple tensors within
        // the same 2 MB block, causing cudaIpcGetMemHandle to return the same
        // block handle for all of them.
        // -------------------------------------------------------------------------

        std::vector<void *> staging;
        if (rank == root)
        {
            staging.resize(world_size, nullptr);
            c10::cuda::CUDAGuard guard(my_device);

            for (int r = 0; r < world_size; ++r)
            {
                TORCH_CHECK(inputTensors[r].is_cuda(),
                            "cuda_scatter: inputTensors[", r, "] must be a CUDA tensor");
                TORCH_CHECK(inputTensors[r].is_contiguous(),
                            "cuda_scatter: inputTensors[", r, "] must be contiguous");
                TORCH_CHECK(inputTensors[r].nbytes() == nbytes,
                            "cuda_scatter: inputTensors[", r, "] size mismatch with outputTensor");

                CUDA_CHECK(cudaMalloc(&staging[r], nbytes));
                CUDA_CHECK(cudaMemcpy(
                    staging[r],
                    inputTensors[r].data_ptr(),
                    nbytes,
                    cudaMemcpyDeviceToDevice));

                HandlePayload payload;
                payload.device_id = my_device;
                CUDA_CHECK(cudaIpcGetMemHandle(&payload.ipc_handle, staging[r]));

                std::vector<uint8_t> bytes(sizeof(HandlePayload));
                std::memcpy(bytes.data(), &payload, sizeof(HandlePayload));
                store->set(pfx + "handle_" + std::to_string(r), bytes);
            }
        }

        // Block until the root has published all handles.
        std::vector<std::string> handle_keys;
        handle_keys.reserve(world_size);
        for (int r = 0; r < world_size; ++r)
            handle_keys.push_back(pfx + "handle_" + std::to_string(r));
        store->wait(handle_keys);

        // -------------------------------------------------------------------------
        // Phase 2 — each rank fetches its chunk from root's staging buffer
        // -------------------------------------------------------------------------

        if (rank == root)
        {
            // Local copy: staging[root] → outputTensor (same device, no IPC needed).
            c10::cuda::CUDAGuard guard(my_device);
            CUDA_CHECK(cudaMemcpy(
                outputTensor.data_ptr(),
                staging[root],
                nbytes,
                cudaMemcpyDeviceToDevice));
        }
        else
        {
            // Remote copy: open root's staging handle for this rank's chunk.
            auto remote_bytes = store->get(pfx + "handle_" + std::to_string(rank));
            TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                        "cuda_scatter: unexpected handle payload size from root");

            HandlePayload remote_payload;
            std::memcpy(&remote_payload, remote_bytes.data(), sizeof(HandlePayload));

            void *remote_ptr = nullptr;
            {
                c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    &remote_ptr,
                    remote_payload.ipc_handle,
                    cudaIpcMemLazyEnablePeerAccess));
            }

            // The staging pointer is a cudaMalloc base, so no offset arithmetic.
            CUDA_CHECK(cudaMemcpyPeer(
                outputTensor.data_ptr(), my_device,
                remote_ptr, remote_payload.device_id,
                nbytes));

            {
                c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
            }
        }

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Root must not free staging buffers until every rank has finished reading.
        // -------------------------------------------------------------------------

        std::vector<uint8_t> done_signal = {1};
        store->set(pfx + "done_" + std::to_string(rank), done_signal);

        std::vector<std::string> done_keys;
        done_keys.reserve(world_size);
        for (int r = 0; r < world_size; ++r)
            done_keys.push_back(pfx + "done_" + std::to_string(r));
        store->wait(done_keys);

        // Free staging buffers only after all peers have confirmed they are done.
        if (rank == root)
        {
            for (int r = 0; r < world_size; ++r)
                CUDA_CHECK(cudaFree(staging[r]));
        }
    }

} // namespace mini_nccl
