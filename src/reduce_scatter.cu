#include "reduce_scatter.h"
#include "cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_nccl
{

    void cuda_reduce_scatter(
        const c10::intrusive_ptr<c10d::Store> &store,
        int rank,
        int world_size,
        uint64_t seq,
        at::Tensor &outputTensor,
        std::vector<at::Tensor> &inputTensors,
        c10d::ReduceOp reduce_op)
    {
        TORCH_CHECK(static_cast<int>(inputTensors.size()) == world_size,
                    "cuda_reducescatter: inputTensors.size() (", inputTensors.size(),
                    ") must equal world_size (", world_size, ")");
        TORCH_CHECK(outputTensor.is_cuda(),
                    "cuda_reducescatter: outputTensor must be a CUDA tensor");
        TORCH_CHECK(outputTensor.is_contiguous(),
                    "cuda_reducescatter: outputTensor must be contiguous");

        const int my_device = outputTensor.get_device();
        const size_t chunk_bytes = outputTensor.nbytes();

        // Unique prefix for all store keys in this collective invocation.
        const std::string pfx = "rs_" + std::to_string(seq) + "_";

        // -------------------------------------------------------------------------
        // Phase 1 — publish IPC handles for all of this rank's input chunks
        //
        // Each chunk inputTensors[c] gets its own handle keyed by rank and chunk
        // index so that every peer can fetch exactly the chunk it needs.
        // -------------------------------------------------------------------------

        for (int c = 0; c < world_size; ++c)
        {
            TORCH_CHECK(inputTensors[c].is_cuda(),
                        "cuda_reducescatter: inputTensors[", c, "] must be a CUDA tensor");
            TORCH_CHECK(inputTensors[c].is_contiguous(),
                        "cuda_reducescatter: inputTensors[", c, "] must be contiguous");
            TORCH_CHECK(inputTensors[c].nbytes() == chunk_bytes,
                        "cuda_reducescatter: inputTensors[", c, "] size mismatch with outputTensor");

            HandlePayload payload;
            payload.device_id = inputTensors[c].get_device();
            CUDA_CHECK(cudaIpcGetMemHandle(&payload.ipc_handle, inputTensors[c].data_ptr()));

            std::vector<uint8_t> bytes(sizeof(HandlePayload));
            std::memcpy(bytes.data(), &payload, sizeof(HandlePayload));
            store->set(pfx + "handle_" + std::to_string(rank) + "_" + std::to_string(c), bytes);
        }

        // Block until every rank has published all of its handles.
        std::vector<std::string> handle_keys;
        handle_keys.reserve(world_size * world_size);
        for (int r = 0; r < world_size; ++r)
            for (int c = 0; c < world_size; ++c)
                handle_keys.push_back(pfx + "handle_" + std::to_string(r) + "_" + std::to_string(c));
        store->wait(handle_keys);

        // -------------------------------------------------------------------------
        // Phase 2 — reduce chunk `rank` from all peers into outputTensor
        //
        // Start with this rank's own contribution, then accumulate each peer's
        // chunk `rank` via peer-to-peer copy into a scratch buffer.
        // -------------------------------------------------------------------------

        c10::cuda::CUDAGuard guard(my_device);

        // Seed the output with our own contribution for chunk `rank`.
        CUDA_CHECK(cudaMemcpy(
            outputTensor.data_ptr(),
            inputTensors[rank].data_ptr(),
            chunk_bytes,
            cudaMemcpyDeviceToDevice));

        at::Tensor scratch = at::empty_like(outputTensor);

        for (int r = 0; r < world_size; ++r)
        {
            if (r == rank)
                continue;

            // Fetch peer r's handle for chunk `rank` (the chunk this rank owns).
            auto remote_bytes = store->get(pfx + "handle_" + std::to_string(r) + "_" + std::to_string(rank));
            TORCH_CHECK(remote_bytes.size() == sizeof(HandlePayload),
                        "cuda_reducescatter: unexpected handle payload size from rank ", r);

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

            // Copy peer r's chunk `rank` into our scratch buffer.
            CUDA_CHECK(cudaMemcpyPeer(
                scratch.data_ptr(), my_device,
                remote_ptr, remote_payload.device_id,
                chunk_bytes));

            // Accumulate into the output.
            apply_reduce(outputTensor, scratch, reduce_op);

            // Close the IPC mapping on the device it was opened on.
            {
                c10::cuda::CUDAGuard dev_guard(remote_payload.device_id);
                CUDA_CHECK(cudaIpcCloseMemHandle(remote_ptr));
            }
        }

        // For AVG, divide by world_size after all contributions have been summed.
        if (reduce_op == c10d::ReduceOp::AVG)
            outputTensor.div_(static_cast<double>(world_size));

        // -------------------------------------------------------------------------
        // Phase 3 — completion barrier
        // Ensure no rank dismantles its input tensors before every peer has
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
