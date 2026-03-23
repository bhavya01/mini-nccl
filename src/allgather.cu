#include "allgather.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                                                                   \
    do                                                                                                                     \
    {                                                                                                                      \
        cudaError_t _err = (expr);                                                                                         \
        if (_err != cudaSuccess)                                                                                           \
        {                                                                                                                  \
            throw std::runtime_error(                                                                                      \
                std::string("CUDA error at " __FILE__ ":") + std::to_string(__LINE__) + " — " + cudaGetErrorString(_err)); \
        }                                                                                                                  \
    } while (0)

namespace mini_nccl
{

    // Payload published to the store for each rank: the CUDA IPC handle for that
    // rank's input tensor plus the device ordinal on which it lives.
    struct HandlePayload
    {
        cudaIpcMemHandle_t ipc_handle;
        int device_id;
    };

    void cuda_allgather(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        const at::Tensor &inputTensor,
        std::vector<at::Tensor> &outputTensors)
    {
        TORCH_CHECK(inputTensor.is_cuda(),
                    "cuda_allgather: inputTensor must be a CUDA tensor");
        TORCH_CHECK(inputTensor.is_contiguous(),
                    "cuda_allgather: inputTensor must be contiguous");
        TORCH_CHECK(static_cast<int>(outputTensors.size()) == world_size,
                    "cuda_allgather: outputTensors.size() (", outputTensors.size(),
                    ") must equal world size (", world_size, ")");

        const int my_device = inputTensor.get_device();
        const size_t nbytes = inputTensor.nbytes();

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "ag_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — exchange IPC handles via the store
        // -------------------------------------------------------------------------

        // Obtain a CUDA IPC handle for this rank's input buffer.
        HandlePayload my_payload;
        my_payload.device_id = my_device;
        CUDA_CHECK(cudaIpcGetMemHandle(&my_payload.ipc_handle,
                                       inputTensor.data_ptr()));

        // Serialize the payload and publish it.
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
        // Phase 2 — copy data into outputTensors
        // -------------------------------------------------------------------------

        // Local copy: inputTensor → outputTensors[rank] (same device, no IPC needed).
        {
            TORCH_CHECK(outputTensors[rank].is_cuda(),
                        "cuda_allgather: outputTensors[rank] must be a CUDA tensor");
            TORCH_CHECK(outputTensors[rank].nbytes() == nbytes,
                        "cuda_allgather: outputTensors[rank] size mismatch");

            c10::cuda::CUDAGuard guard(my_device);
            CUDA_CHECK(cudaMemcpy(
                outputTensors[rank].data_ptr(),
                inputTensor.data_ptr(),
                nbytes,
                cudaMemcpyDeviceToDevice));
        }

        // Remote copies: for each peer rank, open its IPC handle and pull its
        // input data into our corresponding output tensor slot via a peer transfer.
        for (int r = 0; r < world_size; ++r)
        {
            if (r == rank)
                continue;

            TORCH_CHECK(outputTensors[r].is_cuda(),
                        "cuda_allgather: outputTensors[", r, "] must be a CUDA tensor");
            TORCH_CHECK(outputTensors[r].nbytes() == nbytes,
                        "cuda_allgather: outputTensors[", r, "] size mismatch");

            // Deserialize the remote rank's handle payload.
            auto remote_bytes = store->get(pfx + "handle_" + std::to_string(r));
            TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                        "cuda_allgather: unexpected handle payload size from rank ", r);

            HandlePayload remote_payload;
            std::memcpy(&remote_payload, remote_bytes.data(), sizeof(HandlePayload));

            // IPC handles must be opened on the device they were created on.
            void *remote_ptr = nullptr;
            {
                c10::cuda::CUDAGuard guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcOpenMemHandle(
                    &remote_ptr,
                    remote_payload.ipc_handle,
                    cudaIpcMemLazyEnablePeerAccess));
            }

            // cudaMemcpyPeer copies across device boundaries without needing the
            // current device to be either source or destination.
            CUDA_CHECK(cudaMemcpyPeer(
                outputTensors[r].data_ptr(), outputTensors[r].get_device(),
                remote_ptr, remote_payload.device_id,
                nbytes));

            // Close the IPC mapping on the device it was opened on.
            {
                c10::cuda::CUDAGuard guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
            }
        }

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Ensure no rank dismantles its input buffer until every rank has finished
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
