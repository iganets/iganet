/**
   @file net/dataparallel.hpp

   @brief Helpers for synchronous data-parallel training with MPI.

   The training loops of IgANet evaluate the loss and its gradient on
   the share of the data that a process owns. DataParallel combines the
   local results into the loss and gradient of the union of all shares
   inside every evaluation of the optimizer closure, so that all
   processes hold the same loss, the same gradient and, after the
   optimizer step, the same parameters.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <iganet/core/core.hpp>

#if defined(IGANET_WITH_MPI) && defined(IGANET_MPI_HAVE_EXT)
#include <mpi-ext.h>
#endif

namespace iganet {

/// @brief Counters and timers of the training loop.
struct TrainingStatistics {
  /// @brief Number of optimizer steps.
  int64_t steps = 0;
  /// @brief Number of evaluations of loss and gradient (closure calls).
  int64_t evaluations = 0;
  /// @brief Number of MPI reductions of the gradients.
  int64_t reductions = 0;
  /// @brief Number of consistency checks.
  int64_t checks = 0;
  /// @brief Payload of one gradient reduction in bytes.
  int64_t bytes_per_reduction = 0;
  /// @brief Seconds spent in the forward and backward pass.
  double compute_seconds = 0;
  /// @brief Seconds spent in MPI reductions, including the copies to and
  /// from the reduction buffer.
  double reduce_seconds = 0;
  /// @brief Seconds spent in the optimizer outside the loss evaluations.
  double update_seconds = 0;
};

} // namespace iganet

namespace iganet::mpi {

/// @brief Returns the rank of the calling process, zero without MPI.
inline int rank() {
#ifdef IGANET_WITH_MPI
  int flag = 0;
  MPI_Initialized(&flag);
  if (flag) {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
  }
#endif
  return 0;
}

/// @brief Returns the number of processes, one without MPI.
inline int size() {
#ifdef IGANET_WITH_MPI
  int flag = 0;
  MPI_Initialized(&flag);
  if (flag) {
    int s = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
  }
#endif
  return 1;
}

/// @brief Returns the number of items owned by each process.
///
/// The items are split into contiguous blocks of equal length so that
/// a mean over the local block, weighted by one over the number of
/// processes, sums to the mean over all items.
///
/// @param n Total number of items.
/// @throws std::runtime_error If `n` is not divisible by the number of
/// processes.
inline int64_t share(int64_t n) {
  const int64_t s = size();
  if (n % s != 0)
    throw std::runtime_error("Number of items (" + std::to_string(n) +
                             ") is not divisible by the number of MPI "
                             "processes (" +
                             std::to_string(s) + ")");
  return n / s;
}

/// @brief Returns the block of rows of a tensor owned by this process.
/// @param tensor Tensor whose first dimension enumerates the items.
inline torch::Tensor share(const torch::Tensor &tensor) {
  const int64_t n = share(tensor.size(0));
  return tensor.narrow(0, rank() * n, n);
}

/// @brief Returns the block of points of each coordinate owned by this
/// process.
/// @param xi Array of coordinate tensors of equal length.
template <std::size_t N>
inline std::array<torch::Tensor, N>
share(const std::array<torch::Tensor, N> &xi) {
  std::array<torch::Tensor, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = share(xi[i]);
  return result;
}

/// @brief Sums a value over all processes, the value itself without MPI.
inline double sum(double value) {
#ifdef IGANET_WITH_MPI
  if (size() > 1) {
    double result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
  }
#endif
  return value;
}

/// @brief Returns true if device buffers can be passed to MPI.
///
/// The MPI library reports CUDA-aware support at run time. The
/// environment variable `IGANET_MPI_CUDA_AWARE` (0 or 1) overrides the
/// report.
inline bool cuda_aware() {
  if (const char *env = std::getenv("IGANET_MPI_CUDA_AWARE"))
    return std::string(env) == "1";
#if defined(IGANET_WITH_MPI) && defined(IGANET_MPI_HAVE_EXT) &&                \
    defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return MPIX_Query_cuda_support() == 1;
#else
  return false;
#endif
}

#ifdef IGANET_WITH_MPI
namespace detail {

/// @brief Returns the MPI datatype of a tensor scalar type.
inline MPI_Datatype datatype(torch::ScalarType type) {
  switch (type) {
  case torch::kDouble:
    return MPI_DOUBLE;
  case torch::kFloat:
    return MPI_FLOAT;
  case torch::kLong:
    return MPI_INT64_T;
  case torch::kInt:
    return MPI_INT32_T;
  default:
    throw std::runtime_error(
        std::string("Unsupported tensor type for MPI communication: ") +
        c10::toString(type));
  }
}

} // namespace detail
#endif

/// @brief Synchronous data-parallel training over MPI processes.
///
/// With \f$R\f$ processes, process \f$r\f$ evaluates the loss
/// \f$L_r\f$ and its gradient \f$g_r\f$ on its share of the data. The
/// evaluation returns \f$\sum_r w_r L_r\f$ and \f$\sum_r w_r g_r\f$,
/// computed by one MPI_Allreduce on a buffer that holds all gradients
/// and the loss. With \f$w_r=n_r/n\f$ for shares of \f$n_r\f$ of
/// \f$n\f$ items and a loss that is a mean over the items, the result
/// equals the loss and gradient on all items.
///
/// Without MPI, with one process or with data parallelism switched off
/// the class only forwards the evaluation and records timings.
class DataParallel {
private:
  using clock = std::chrono::steady_clock;

  /// @brief Parameters of the network.
  std::vector<torch::Tensor> params_;

  /// @brief Buffers of the network.
  std::vector<torch::Tensor> buffers_;

  /// @brief Combine results of all processes.
  bool enabled_ = false;

  /// @brief Reduce all gradients in one buffer.
  bool pack_ = true;

  /// @brief Total number of parameter entries.
  int64_t count_ = 0;

  /// @brief Reduction buffer holding all gradient entries and the loss.
  torch::Tensor buffer_;

  /// @brief Counters and timers.
  TrainingStatistics stats_;

  static double seconds(clock::time_point since) {
    return std::chrono::duration<double>(clock::now() - since).count();
  }

  /// @brief Waits for pending device work so that timings are meaningful.
  void synchronize() const {
    if (!params_.empty() && params_.front().is_cuda())
      torch::cuda::synchronize();
  }

#ifdef IGANET_WITH_MPI
  /// @brief Sums the tensor over all processes in place.
  ///
  /// Device tensors are staged through host memory unless the MPI library
  /// is CUDA-aware.
  static void allreduce(torch::Tensor &tensor, MPI_Op op = MPI_SUM) {
    if (tensor.is_cpu() || cuda_aware()) {
      MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(), tensor.numel(),
                    detail::datatype(tensor.scalar_type()), op, MPI_COMM_WORLD);
    } else {
      auto host = tensor.to(torch::kCPU);
      MPI_Allreduce(MPI_IN_PLACE, host.data_ptr(), host.numel(),
                    detail::datatype(host.scalar_type()), op, MPI_COMM_WORLD);
      tensor.copy_(host);
    }
  }

  /// @brief Broadcasts the tensor from rank zero in place.
  static void broadcast(torch::Tensor &tensor) {
    if (tensor.is_cpu() || cuda_aware()) {
      MPI_Bcast(tensor.data_ptr(), tensor.numel(),
                detail::datatype(tensor.scalar_type()), 0, MPI_COMM_WORLD);
    } else {
      auto host = tensor.to(torch::kCPU);
      MPI_Bcast(host.data_ptr(), host.numel(),
                detail::datatype(host.scalar_type()), 0, MPI_COMM_WORLD);
      tensor.copy_(host);
    }
  }
#endif

public:
  /// @brief Returns true if results are combined over processes.
  bool enabled() const { return enabled_; }

  /// @brief Returns the counters and timers.
  const TrainingStatistics &statistics() const { return stats_; }

  /// @brief Resets the counters and timers.
  void reset() {
    auto bytes = stats_.bytes_per_reduction;
    stats_ = TrainingStatistics{};
    stats_.bytes_per_reduction = bytes;
  }

  /// @brief Prepares the training of a network.
  ///
  /// With data parallelism enabled the parameters and buffers of rank
  /// zero are broadcast to all processes, so the training does not
  /// depend on how the processes initialized the network.
  ///
  /// @param params Parameters of the network.
  /// @param buffers Buffers of the network.
  /// @param data_parallel Combine the results of all processes.
  /// @param pack Reduce all gradients in a single buffer.
  void attach(const std::vector<torch::Tensor> &params,
              const std::vector<torch::Tensor> &buffers, bool data_parallel,
              bool pack) {
    params_ = params;
    buffers_ = buffers;
    pack_ = pack;
    enabled_ = false;
#ifdef IGANET_WITH_MPI
    if (!data_parallel || size() == 1 || params_.empty())
      return;

    enabled_ = true;
    count_ = 0;
    for (const auto &p : params_) {
      if (p.scalar_type() != params_.front().scalar_type() ||
          p.device() != params_.front().device())
        throw std::runtime_error(
            "All parameters must share the data type and device");
      count_ += p.numel();
    }
    // Floating-point parameters: the buffer holds the gradients followed by
    // the loss
    detail::datatype(params_.front().scalar_type());
    buffer_ = torch::zeros({count_ + 1}, params_.front().options());
    stats_.bytes_per_reduction = (count_ + 1) * params_.front().element_size();

    torch::NoGradGuard guard;
    for (auto &p : params_)
      broadcast(p);
    for (auto &b : buffers_)
      broadcast(b);
#else
    (void)data_parallel;
#endif
  }

  /// @brief Evaluates loss and gradient and combines them over all
  /// processes.
  ///
  /// @param local Function that zeroes the gradients, evaluates the local
  /// loss, runs the backward pass and returns the local loss.
  /// @param weight Weight \f$w_r\f$ of the local loss.
  /// @return The weighted sum of the losses of all processes. The
  /// gradients of the parameters hold the weighted sum of the gradients.
  template <typename Local>
  torch::Tensor evaluate(Local &&local, double weight) {
    auto t0 = clock::now();
    torch::Tensor loss = local();
    synchronize();
    stats_.compute_seconds += seconds(t0);
    ++stats_.evaluations;

    if (!enabled_)
      return loss;

#ifdef IGANET_WITH_MPI
    auto t1 = clock::now();
    loss = reduce(loss, weight);
    stats_.reduce_seconds += seconds(t1);
    ++stats_.reductions;
#else
    (void)weight;
#endif
    return loss;
  }

#ifdef IGANET_WITH_MPI
private:
  /// @brief Replaces loss and gradients by their weighted sum over all
  /// processes.
  torch::Tensor reduce(const torch::Tensor &loss, double weight) {
    torch::NoGradGuard guard;

    for (auto &p : params_)
      if (!p.grad().defined())
        p.mutable_grad() = torch::zeros_like(p);

    torch::Tensor global;
    if (pack_) {
      int64_t offset = 0;
      for (auto &p : params_) {
        buffer_.narrow(0, offset, p.numel()).copy_(p.grad().reshape(-1));
        offset += p.numel();
      }
      buffer_.narrow(0, count_, 1).copy_(loss.detach().reshape(1));
      buffer_.mul_(weight);

      allreduce(buffer_);

      offset = 0;
      for (auto &p : params_) {
        p.grad().copy_(buffer_.narrow(0, offset, p.numel()).view_as(p));
        offset += p.numel();
      }
      global = buffer_.narrow(0, count_, 1).clone().reshape(loss.sizes());
    } else {
      for (auto &p : params_) {
        auto g = p.grad().mul(weight).contiguous();
        allreduce(g);
        p.grad().copy_(g);
      }
      auto l = loss.detach().mul(weight).reshape(1).contiguous();
      allreduce(l);
      global = l.reshape(loss.sizes());
    }
    return global;
  }

public:
#endif

  /// @brief Runs one optimizer step with the given closure and records the
  /// time spent outside the closure as update time.
  template <typename Optimizer, typename Closure>
  void step(Optimizer &optimizer, Closure &&closure) {
    auto t0 = clock::now();
    const double inside = stats_.compute_seconds + stats_.reduce_seconds;
    optimizer.step(closure);
    synchronize();
    stats_.update_seconds +=
        seconds(t0) - (stats_.compute_seconds + stats_.reduce_seconds - inside);
    ++stats_.steps;
  }

  /// @brief Checks that all processes hold the same parameters.
  ///
  /// Reduces the sum and the sum of squares of every parameter over the
  /// processes and compares the extreme values.
  ///
  /// @param tolerance Relative difference that is accepted.
  /// @throws std::runtime_error On all processes if the parameters differ.
  void check(double tolerance = 0.0) {
    if (!enabled_)
      return;
#ifdef IGANET_WITH_MPI
    torch::NoGradGuard guard;
    std::vector<double> local(2 * params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
      auto p = params_[i].detach().to(torch::kCPU, torch::kDouble);
      local[2 * i] = p.sum().item<double>();
      local[2 * i + 1] = p.square().sum().item<double>();
    }
    std::vector<double> lo(local.size()), hi(local.size());
    MPI_Allreduce(local.data(), lo.data(), local.size(), MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local.data(), hi.data(), local.size(), MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    ++stats_.checks;
    for (std::size_t i = 0; i < local.size(); ++i) {
      const double scale = std::max(std::abs(lo[i]), std::abs(hi[i]));
      if (hi[i] - lo[i] > tolerance * scale)
        throw std::runtime_error(
            "MPI processes hold different network parameters: statistic " +
            std::to_string(i % 2) + " of parameter tensor " +
            std::to_string(i / 2) + " ranges over [" + std::to_string(lo[i]) +
            ", " + std::to_string(hi[i]) + "]");
    }
#else
    (void)tolerance;
#endif
  }

  /// @brief Checks the parameters if the step is a multiple of the interval.
  /// @param step Number of the completed optimizer step, starting at one.
  /// @param interval Number of steps between checks, zero disables them.
  /// @param tolerance Relative difference that is accepted.
  void after_step(int64_t step, int64_t interval, double tolerance) {
    if (interval > 0 && step % interval == 0)
      check(tolerance);
  }

  /// @brief Verifies that all processes agree on a flag.
  ///
  /// The loops over batches need every process to take the same number of
  /// steps because each step contains collective operations.
  ///
  /// @param flag Value of the calling process.
  /// @throws std::runtime_error On all processes if the values differ.
  /// @return The common value.
  int agree(int flag) const {
#ifdef IGANET_WITH_MPI
    if (enabled_) {
      int lo = 0, hi = 0;
      MPI_Allreduce(&flag, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&flag, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (lo != hi)
        throw std::runtime_error("MPI processes hold different numbers of "
                                 "batches per epoch");
    }
#endif
    return flag;
  }
};

} // namespace iganet::mpi
