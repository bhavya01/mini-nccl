#include "gather.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_nccl
{

    void cuda_gather(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        std::vector<at::Tensor> &outputTensors,
        const at::Tensor &inputTensor,
        int root)
    {
        TORCH_CHECK(inputTensor.is_cuda(),
                    "cuda_gather: inputTensor must be a CUDA tensor");
        TORCH_CHECK(inputTensor.is_contiguous(),
                    "cuda_gather: inputTensor must be contiguous");
        if (rank == root)
        {
            TORCH_CHECK(static_cast<int>(outputTensors.size()) == world_size,
                        "cuda_gather: root's outputTensors.size() (", outputTensors.size(),
                        ") must equal world_size (", world_size, ")");
        }

        const int my_device = inputTensor.get_device();
        const size_t nbytes = inputTensor.nbytes();

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "ga_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — every rank publishes an IPC handle for its inputTensor
        // -------------------------------------------------------------------------

        HandlePayload my_payload;
        my_payload.device_id = my_device;
        CUDA_CHECK(cudaIpcGetMemHandle(&my_payload.ipc_handle,
                                       inputTensor.data_ptr()));

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
        // Phase 2 — root copies each rank's inputTensor into outputTensors[r]
        // -------------------------------------------------------------------------

        if (rank == root)
        {
            c10::cuda::CUDAGuard guard(my_device);

            // Local copy: root's own inputTensor → outputTensors[root].
            TORCH_CHECK(outputTensors[root].is_cuda(),
                        "cuda_gather: outputTensors[root] must be a CUDA tensor");
            TORCH_CHECK(outputTensors[root].nbytes() == nbytes,
                        "cuda_gather: outputTensors[root] size mismatch");
            CUDA_CHECK(cudaMemcpy(
                outputTensors[root].data_ptr(),
                inputTensor.data_ptr(),
                nbytes,
                cudaMemcpyDeviceToDevice));

            // Remote copies: for each non-root peer, open its IPC handle and pull
            // its inputTensor into the corresponding output slot.
            for (int r = 0; r < world_size; ++r)
            {
                if (r == root)
                    continue;

                TORCH_CHECK(outputTensors[r].is_cuda(),
                            "cuda_gather: outputTensors[", r, "] must be a CUDA tensor");
                TORCH_CHECK(outputTensors[r].nbytes() == nbytes,
                            "cuda_gather: outputTensors[", r, "] size mismatch");

                auto remote_bytes = store->get(pfx + "handle_" + std::to_string(r));
                TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                            "cuda_gather: unexpected handle payload size from rank ", r);

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
                    outputTensors[r].data_ptr(), outputTensors[r].get_device(),
                    remote_ptr, remote_payload.device_id,
                    nbytes));

                {
                    c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                    CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
                }
            }
        }

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Non-root ranks must not dismantle their inputTensors before root has
        // finished reading from them.
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
