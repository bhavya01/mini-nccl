#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

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
    // rank's tensor plus the device ordinal on which it lives.
    struct HandlePayload
    {
        cudaIpcMemHandle_t ipc_handle;
        int device_id;
    };

    // Applies `src` into `dst` in-place using the given reduction op.
    // Both tensors must be on the same CUDA device.
    inline void apply_reduce(at::Tensor &dst, const at::Tensor &src, c10d::ReduceOp reduce_op)
    {
        if (reduce_op == c10d::ReduceOp::SUM)
        {
            dst.add_(src);
        }
        else if (reduce_op == c10d::ReduceOp::AVG)
        {
            // AVG is accumulated as SUM here; the caller divides by world_size afterward.
            dst.add_(src);
        }
        else if (reduce_op == c10d::ReduceOp::PRODUCT)
        {
            dst.mul_(src);
        }
        else if (reduce_op == c10d::ReduceOp::MAX)
        {
            dst.copy_(at::maximum(dst, src));
        }
        else if (reduce_op == c10d::ReduceOp::MIN)
        {
            dst.copy_(at::minimum(dst, src));
        }
        else if (reduce_op == c10d::ReduceOp::BAND)
        {
            dst.bitwise_and_(src);
        }
        else if (reduce_op == c10d::ReduceOp::BOR)
        {
            dst.bitwise_or_(src);
        }
        else if (reduce_op == c10d::ReduceOp::BXOR)
        {
            dst.bitwise_xor_(src);
        }
        else
        {
            TORCH_CHECK(false, "apply_reduce: unsupported reduce op");
        }
    }

} // namespace mini_nccl
