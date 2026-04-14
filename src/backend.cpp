#include "backend.h"
#include "allgather.h"
#include "allreduce.h"
#include "reduce_scatter.h"
#include "alltoall.h"
#include "scatter.h"
#include "gather.h"
#include "broadcast.h"
#include "reduce.h"

namespace c10d
{

    bool MiniNcclWork::isCompleted()
    {
        return true;
    }

    bool MiniNcclWork::isSuccess() const
    {
        return true;
    }

    bool MiniNcclWork::wait(std::chrono::milliseconds /* unused */)
    {
        return true;
    }

    c10::intrusive_ptr<c10::ivalue::Future> MiniNcclWork::getFuture()
    {
        return future_;
    }

    MiniNcclBackend::MiniNcclBackend(int rank, int size,
                                     c10::intrusive_ptr<::c10d::Store> store)
        : Backend(rank, size), store_(std::move(store)) {}

    c10::intrusive_ptr<Work> MiniNcclBackend::allgather(
        std::vector<std::vector<at::Tensor>> &outputTensors,
        std::vector<at::Tensor> &inputTensors,
        const AllgatherOptions & /* unused */)
    {
        TORCH_CHECK(inputTensors.size() == outputTensors.size(),
            "allgather: inputTensors and outputTensors must have the same length");

        // Each element of inputTensors is one tensor contributed by this rank.
        // The matching outputTensors[i] is a size-length vector that receives
        // every rank's contribution for that tensor.
        for (size_t i = 0; i < inputTensors.size(); ++i)
        {
            mini_nccl::cuda_allgather(
                store_,
                rank_,
                size_,
                seq_.fetch_add(1, std::memory_order_relaxed),
                inputTensors[i],
                outputTensors[i]);
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
        future->markCompleted(c10::IValue(outputTensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::ALLGATHER, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::allreduce(
        std::vector<at::Tensor> &tensors,
        const AllreduceOptions &opts)
    {
        for (auto &tensor : tensors)
        {
            mini_nccl::cuda_allreduce(
                store_,
                rank_,
                size_,
                seq_.fetch_add(1, std::memory_order_relaxed),
                tensor,
                opts.reduceOp);
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::ALLREDUCE, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::reduce_scatter(
        std::vector<at::Tensor> &outputTensors,
        std::vector<std::vector<at::Tensor>> &inputTensors,
        const ReduceScatterOptions &opts)
    {
        TORCH_CHECK(outputTensors.size() == inputTensors.size(),
            "reduce_scatter: outputTensors and inputTensors must have the same length");

        for (size_t i = 0; i < outputTensors.size(); ++i)
        {
            mini_nccl::cuda_reduce_scatter(
                store_,
                rank_,
                size_,
                seq_.fetch_add(1, std::memory_order_relaxed),
                outputTensors[i],
                inputTensors[i],
                opts.reduceOp);
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(outputTensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::REDUCE_SCATTER, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::alltoall_base(
        at::Tensor &outputTensor,
        at::Tensor &inputTensor,
        std::vector<int64_t> &outputSplitSizes,
        std::vector<int64_t> &inputSplitSizes,
        const AllToAllOptions & /* unused */)
    {
        // Only equal-split all-to-all is supported: every rank contributes and
        // receives chunks of the same size.  Non-empty split-size vectors that
        // specify unequal splits are rejected.
        auto is_equal_split = [&](const std::vector<int64_t> &sizes, int64_t total) {
            if (sizes.empty())
                return true;
            int64_t expected = total / size_;
            for (int64_t s : sizes)
            {
                if (s != expected)
                    return false;
            }
            return true;
        };

        TORCH_CHECK(is_equal_split(inputSplitSizes, inputTensor.size(0)),
                    "alltoall_base: only equal input splits are supported");
        TORCH_CHECK(is_equal_split(outputSplitSizes, outputTensor.size(0)),
                    "alltoall_base: only equal output splits are supported");

        mini_nccl::cuda_alltoall(
            store_,
            rank_,
            size_,
            seq_.fetch_add(1, std::memory_order_relaxed),
            inputTensor,
            outputTensor);

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::TensorType::get());
        future->markCompleted(c10::IValue(outputTensor));
        return c10::make_intrusive<MiniNcclWork>(OpType::ALLTOALL_BASE, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::broadcast(
        std::vector<at::Tensor> &tensors,
        const BroadcastOptions &opts)
    {
        for (auto &tensor : tensors)
        {
            mini_nccl::cuda_broadcast(
                store_,
                rank_,
                size_,
                seq_.fetch_add(1, std::memory_order_relaxed),
                tensor,
                opts.rootRank);
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::BROADCAST, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::reduce(
        std::vector<at::Tensor> &tensors,
        const ReduceOptions &opts)
    {
        for (auto &tensor : tensors)
        {
            mini_nccl::cuda_reduce(
                store_,
                rank_,
                size_,
                seq_.fetch_add(1, std::memory_order_relaxed),
                tensor,
                opts.rootRank,
                opts.reduceOp);
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::REDUCE, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::scatter(
        std::vector<at::Tensor> &outputTensors,
        std::vector<std::vector<at::Tensor>> &inputTensors,
        const ScatterOptions &opts)
    {
        // PyTorch always passes outputTensors = [output_tensor] (length 1).
        // inputTensors = [world_size_chunks] on root (length 1), [] on non-root.
        TORCH_CHECK(outputTensors.size() == 1,
            "scatter: expected exactly one output tensor per rank");

        std::vector<at::Tensor> root_inputs;
        if (rank_ == opts.rootRank)
        {
            TORCH_CHECK(inputTensors.size() == 1,
                "scatter: root must provide exactly one input tensor group");
            root_inputs = inputTensors[0];
        }

        mini_nccl::cuda_scatter(
            store_,
            rank_,
            size_,
            seq_.fetch_add(1, std::memory_order_relaxed),
            outputTensors[0],
            root_inputs,
            opts.rootRank);

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(outputTensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::SCATTER, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::gather(
        std::vector<std::vector<at::Tensor>> &outputTensors,
        std::vector<at::Tensor> &inputTensors,
        const GatherOptions &opts)
    {
        // PyTorch always passes inputTensors = [input_tensor] (length 1).
        // outputTensors = [world_size_slots] on root (length 1), [] on non-root.
        TORCH_CHECK(inputTensors.size() == 1,
            "gather: expected exactly one input tensor per rank");

        std::vector<at::Tensor> root_outputs;
        if (rank_ == opts.rootRank)
        {
            TORCH_CHECK(outputTensors.size() == 1,
                "gather: root must provide exactly one output tensor group");
            root_outputs = outputTensors[0];
        }

        mini_nccl::cuda_gather(
            store_,
            rank_,
            size_,
            seq_.fetch_add(1, std::memory_order_relaxed),
            root_outputs,
            inputTensors[0],
            opts.rootRank);

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
        future->markCompleted(c10::IValue(outputTensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::GATHER, std::move(future));
    }

    c10::intrusive_ptr<Work> MiniNcclBackend::barrier(
        const BarrierOptions & /* unused */)
    {
        // Pure store-based barrier: no CUDA work, no data transfer.
        // Each rank writes a signal; all ranks wait until every signal is present.
        const std::string pfx =
            "barrier_" + std::to_string(seq_.fetch_add(1, std::memory_order_relaxed)) + "_";

        const std::vector<uint8_t> signal = {1};
        store_->set(pfx + std::to_string(rank_), signal);

        std::vector<std::string> keys;
        keys.reserve(size_);
        for (int r = 0; r < size_; ++r)
            keys.push_back(pfx + std::to_string(r));
        store_->wait(keys);

        auto future = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
        future->markCompleted(c10::IValue());
        return c10::make_intrusive<MiniNcclWork>(OpType::BARRIER, std::move(future));
    }

    c10::intrusive_ptr<Backend> MiniNcclBackend::createMiniNcclBackend(
        const c10::intrusive_ptr<::c10d::Store> &store,
        int rank,
        int size,
        const std::chrono::duration<float> &timeout){
        return c10::make_intrusive<MiniNcclBackend>(rank, size, store);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        m.def("createMiniNcclBackend", &MiniNcclBackend::createMiniNcclBackend);
    }
} // namespace c10d