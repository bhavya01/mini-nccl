#include "backend.h"
#include "allgather.h"

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

    // This is a dummy allreduce that sets all output tensors to zero
    // Modify the implementation to conduct real communication asynchronously
    c10::intrusive_ptr<Work> MiniNcclBackend::allreduce(
        std::vector<at::Tensor> &tensors,
        const AllreduceOptions &opts)
    {
        for (auto &tensor : tensors)
        {
            tensor.zero_();
        }

        auto future = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future->markCompleted(c10::IValue(tensors));
        return c10::make_intrusive<MiniNcclWork>(OpType::ALLREDUCE, std::move(future));
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