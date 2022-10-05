/**
   @file include/utils.hpp

   @brief Utility functions

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

namespace iganet {

  /// @brief Computes the dot-product between two tensors with
  /// summation along the given dimension
  ///
  /// @tparam T0 Type of the first argument
  ///
  /// @tparam T1 Type of the second argument
  ///
  /// @param[in] t0  First argument
  ///
  /// @param[in] t1  Second argument
  ///
  /// @param[in] dim Dimension along which the sum is computed
  ///
  /// @result Tensor containing the dot-product
  template<typename T0, typename T1>
  inline auto dotproduct(T0&& t0, T1&& t1, short_t dim)
  {
    return torch::sum(torch::mul(t0, t1), dim);
  }

  /// @brief Computes the Kronecker-product between two tensors along
  /// the given dimension
  ///
  /// @tparam T0 Type of the first argument
  ///
  /// @tparam T1 Type of the second argument
  ///
  /// @param[in] t0  First argument
  ///
  /// @param[in] t1  Second argument
  ///
  /// @param[in] dim Dimension along which the sum is computed
  ///
  /// @result Tensor containing the Kronecker-product
  template<typename T0, typename T1>
  inline auto kronproduct(T0&& t0, T1&& t1, short_t dim)
  {      
    switch (t1.sizes().size()) {
    case 1:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim)}));
    case 2:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim),1}));
    case 3:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim),1,1}));
    case 4:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim),1,1,1}));
    case 5:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim),1,1,1,1}));
    case 6:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim),1,1,1,1,1}));
    case 7:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim),1,1,1,1,1,1}));
    case 8:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim),1,1,1,1,1,1,1}));
    default:
      throw std::runtime_error("Unsupported tensor dimension");
    }     
  }

  /// @brief Vectorized version of `torch::indexing::Slice` (see
  /// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
  ///
  /// Creates a one-dimensional `torch::Tensor` object of
  /// size `index.numel() * (stop_offset-start_offset)` with the
  /// following content
  ///
  /// \code
  /// [ index[0]+start_offset,   ..., index[N-1]+start_offset,
  ///   index[0]+start_offset+1, ..., index[N-1]+start_offset+1,
  ///                            ...
  ///   index[0]+stop_offset-1,  ...  index[N-1]+stop_offset-1 ]
  /// \endcode
  ///
  /// @param[in] index        Vector of indices
  ///
  /// @param[in] start_offset Starting value of the offset
  ///
  /// @param[in] stop_offset  Stopping value of the offset
  inline auto VSlice(torch::Tensor index, int64_t start_offset, int64_t stop_offset)
  {
    return index.repeat(stop_offset-start_offset)
      +    torch::linspace(start_offset,                        
                           stop_offset-1,                              
                           stop_offset-start_offset,             
                           index.options() )                         
      .repeat_interleave(index.numel());
  }
  
} // namespace iganet
