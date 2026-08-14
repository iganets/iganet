/**
   @file utils/matrix.hpp

   @brief Matrix utility functions.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <iganet/core/core.hpp>
#include <iganet/utils/tensorarray.hpp>

namespace iganet::utils {

  /// @brief Constructs a sparse-CSR matrix from the column indices,
  /// matrix values and the matrix size.
  ///
  /// @param[in] col_indices Column indices in row-major order.
  ///  ///.
  /// @param[in] values Matrix values in row-major order.
  ///
  /// @param[in] size Matrix size.
  /// @return Result of the operation.
  inline torch::Tensor to_sparseCsrTensor(const torch::Tensor& col_indices,
                                          const torch::Tensor& values,
                                          const torch::IntArrayRef& size) {

    TORCH_CHECK(size.size() == 2,
                "Sparse CSR matrix size must contain two entries");
    
    TORCH_CHECK(col_indices.dim() == 2,
                "col_indices must be a two-dimensional tensor");
    
    TORCH_CHECK(values.sizes() == col_indices.sizes(),
                "values and col_indices must have identical shapes");
    
    TORCH_CHECK(
                col_indices.scalar_type() == torch::kInt32 ||
                col_indices.scalar_type() == torch::kInt64,
                "col_indices must have dtype torch::kInt32 or torch::kInt64");
    
    TORCH_CHECK(values.device() == col_indices.device(),
                "values and col_indices must be on the same device");
    
    const int64_t populated_rows = col_indices.size(0);
    const int64_t entries_per_row = col_indices.size(1);
    const int64_t matrix_rows = size[0];
       
    TORCH_CHECK(
                populated_rows <= matrix_rows,
                "col_indices contains ", populated_rows,
                " rows, but the requested matrix has only ", matrix_rows, " rows");

    // CSR requires exactly matrix_rows + 1 row pointers.
    auto crow_indices =
      torch::arange(
                    populated_rows + 1,
                    col_indices.options())
      * entries_per_row;
    
    // Append empty trailing rows. Each empty row repeats the final
    // accumulated number of nonzero entries.
    const int64_t empty_rows = matrix_rows - populated_rows;
    
    if (empty_rows > 0) {
      auto padding = crow_indices[-1].repeat({empty_rows});
      crow_indices = torch::cat({crow_indices, padding});
    }
    
    return torch::sparse_csr_tensor(crow_indices.flatten(),
                                    col_indices.flatten(),
                                    values.flatten(),
                                    size,
                                    values.options().layout(torch::kSparseCsr));    
  }

  /// @brief Constructs a sparse-CSR matrix from the column indices,
  /// matrix values and the matrix size.
  ///
  /// @tparam N Size of the column index array.
  ///
  /// @param[in] col_indices Array of column indices in row-major order.
  ///  ///.
  /// @param[in] values Matrix values in row-major order.
  ///
  /// @param[in] size Matrix size.
  /// @param nbasfuncs Value of `nbasfuncs`.
  /// @return Result of the operation.
  template<std::size_t N>
  inline torch::Tensor to_sparseCsrTensor(const utils::TensorArray<N>& col_indices,
                                          const std::array<int64_t, N>& nbasfuncs,
                                          const torch::Tensor& values,
                                          const torch::IntArrayRef& size) {
    
    // Compute absolut column indices
    torch::Tensor col_indices_;
    if constexpr (N == 1)
      col_indices_ = col_indices[0];
    else if constexpr (N == 2)
      col_indices_ = (col_indices[0].unsqueeze(2) +
                      nbasfuncs[0]*col_indices[1].unsqueeze(1))
        .permute({0, 2, 1}).reshape({col_indices[0].size(0), -1});
    else if constexpr (N == 3)
      col_indices_ = (col_indices[0].unsqueeze(2).unsqueeze(3) +
                      nbasfuncs[0]*col_indices[1].unsqueeze(1).unsqueeze(3) +
                      nbasfuncs[0]*nbasfuncs[1]*col_indices[2].unsqueeze(1).unsqueeze(2))
        .permute({0, 3, 2, 1}).reshape({col_indices[0].size(0), -1});
    else
      throw std::runtime_error("Invalid dimension");  

    return to_sparseCsrTensor(col_indices_, values, size);
  }
  
  /// @brief Constructs a sparse-CSR matrix from the B-spline basis
  /// function values evaluated at discrete points (e.g., the Greville
  /// abscissae), the corresponding knot_indices (i.e. the list of
  /// knot indices that mark the start of the knot span the discrete
  /// points fall into), the B-spline degrees and the matrix size.
  ///
  /// @tparam N Size of the knot index and degree arrays.
  ///
  /// @param[in] knot_indices List of knot indices marking the start
  /// of the knot span the discrete evaluation points fall into.
  ///
  /// @param[in] values Matrix values in row-major order.
  ///
  /// @param[in] size Matrix size.
  /// @param degrees Value of `degrees`.
  /// @param nbasfuncs Value of `nbasfuncs`.
  /// @return Result of the operation.
  template<std::size_t N>
  inline torch::Tensor to_sparseCsrTensor(const utils::TensorArray<N>& knot_indices,
                                          const std::array<short, N>& degrees,
                                          const std::array<int64_t, N>& nbasfuncs,
                                          const torch::Tensor& values,
                                          const torch::IntArrayRef& size) {
    
    // Apply offsets to indices
    std::array<torch::Tensor, N> col_indices;    
    for (std::size_t i = 0; i < N; ++i) {
      col_indices[i] =
        (knot_indices[i].unsqueeze(0) +
         torch::arange(-degrees[i],
                       1,
                       knot_indices[i].options()
                       ).unsqueeze(1)).t();
    }

    return to_sparseCsrTensor(col_indices, nbasfuncs, values, size);
  }
} // namespace iganet::utils
