/**
   @file net/dataparallel.hpp

   @brief Helpers for synchronous data-parallel training with MPI.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <iganet/core/core.hpp>

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
inline std::array<torch::Tensor, N> share(const std::array<torch::Tensor, N> &xi) {
  std::array<torch::Tensor, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = share(xi[i]);
  return result;
}

} // namespace iganet::mpi
