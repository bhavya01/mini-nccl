#pragma once

#include <torch/python.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <pybind11/chrono.h>

#include <atomic>
#include <cstdint>

namespace c10d
{

  // MiniNcclWork represents a single collective operation submitted to the
  // mini-nccl backend. It wraps a Future that resolves when the operation
  // completes, and exposes the standard Work interface that torch.distributed
  // uses to query status and block on completion.
  class MiniNcclWork : public Work
  {
  public:
    // Constructs a work item for the given collective op type.
    // future: resolves to the output tensor(s) once the operation finishes.
    MiniNcclWork(
        OpType opType,
        c10::intrusive_ptr<c10::ivalue::Future> future)
        : Work(
              -1, // rank, only used by recvAnySource — not applicable here
              opType),
          future_(std::move(future))
    {
    }

    // Returns true if the underlying future has been resolved (op finished).
    bool isCompleted() override;

    // Returns true if the op completed without an error.
    bool isSuccess() const override;

    // Blocks the calling thread until the operation completes or timeout
    // elapses. Throws on error or timeout.
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    // Returns the underlying Future, allowing callers to attach callbacks
    // or compose this work with other async operations.
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

  private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
  };

  // MiniNcclBackend is the custom torch.distributed ProcessGroup backend for
  // mini-nccl. It targets a single rack of GPUs connected in an all-to-all
  // topology and implements collectives as direct point-to-point exchanges
  // between ranks — no ring or tree algorithms are needed.
  //
  // Collective APIs that are not yet implemented will throw an error at runtime
  // if invoked by application code.
  class MiniNcclBackend : public Backend
  {
  public:
    // Initializes the backend for a process group of `size` ranks where this
    // process is `rank`. store is used for rendezvous during collectives.
    MiniNcclBackend(int rank, int size,
                    c10::intrusive_ptr<::c10d::Store> store);

    // All-Gather: each rank contributes its inputTensor and receives the full
    // list of tensors from all ranks in outputTensors.
    c10::intrusive_ptr<Work> allgather(
        std::vector<std::vector<at::Tensor>> &outputTensors,
        std::vector<at::Tensor> &inputTensors,
        const AllgatherOptions &opts = AllgatherOptions()) override;

    // All-Reduce: applies a reduction op (e.g. sum) across all ranks' tensors
    // in-place so every rank ends up with the same reduced result.
    c10::intrusive_ptr<Work> allreduce(
        std::vector<at::Tensor> &tensors,
        const AllreduceOptions &opts = AllreduceOptions()) override;

    // Reduce-Scatter: each rank contributes world_size chunks via inputTensors;
    // rank r receives the reduction of chunk r from all ranks in outputTensors.
    c10::intrusive_ptr<Work> reduce_scatter(
        std::vector<at::Tensor> &outputTensors,
        std::vector<std::vector<at::Tensor>> &inputTensors,
        const ReduceScatterOptions &opts = ReduceScatterOptions()) override;

    // All-to-All: each rank splits its inputTensor into world_size equal chunks
    // and sends chunk s to rank s.  Rank s places the received data in the
    // corresponding slot of its outputTensor.  Concretely, after the call:
    //   outputTensor[chunk r] == rank r's inputTensor[chunk this_rank]
    c10::intrusive_ptr<Work> alltoall_base(
        at::Tensor &outputTensor,
        at::Tensor &inputTensor,
        std::vector<int64_t> &outputSplitSizes,
        std::vector<int64_t> &inputSplitSizes,
        const AllToAllOptions &opts = AllToAllOptions()) override;

    static c10::intrusive_ptr<Backend> createMiniNcclBackend(
        const c10::intrusive_ptr<::c10d::Store> &store,
        int rank,
        int size,
        const std::chrono::duration<float> &timeout);

    // __attribute__((constructor)) is a compiler-specific instruction (GCC/Clang).
    // It tells the system to execute this function automatically as soon as the
    // shared library is loaded into memory.
    static void RegisterMiniNcclBackend() __attribute__((constructor))
    {
      py::object module = py::module::import("torch.distributed");
      py::object register_backend =
          module.attr("Backend").attr("register_backend");
      register_backend("mini_nccl", py::cpp_function(createMiniNcclBackend),
                       py::arg("extended_api") = false,
                       py::arg("devices") = "cuda");
    }

  private:
    c10::intrusive_ptr<::c10d::Store> store_;
    // Monotonically increasing counter; gives each collective call a unique
    // set of store keys so concurrent or back-to-back calls don't collide.
    std::atomic<uint64_t> seq_{0};
  };

} // namespace c10d
