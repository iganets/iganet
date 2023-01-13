/**
   @file include/utils.hpp

   @brief Utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <random>
#include <json.hpp>

#pragma once

namespace iganet {

  class uuid {
  public:
    
    /// @brief Generates a uuid string in the form
    /// b9317db-02a2-4882-9b94-d1e1defe8c56
    ///
    /// @result std::string
    static std::string create() {
      std::stringstream hexstream;
      hexstream << uuid::random_hex(4) << "-" << uuid::random_hex(2) << "-"
                << uuid::random_hex(2) << "-" << uuid::random_hex(2) << "-"
                << uuid::random_hex(6);
      return hexstream.str();
    }

  private:
    
    /// @brief Generates a string of random hex chars
    ///
    /// @param[in] len Length in bytes
    ///
    /// @result std::string  String random hex chars (2x length of the bytes)
    static std::string random_hex(const unsigned int len) {
      std::stringstream ss;
      for (auto i = 0; i < len; i++) {
        const auto rc = random_char();
        std::stringstream hexstream;
        hexstream << std::hex << rc;
        auto hex = hexstream.str();
        ss << (hex.length() < 2 ? '0' + hex : hex);
      }
      return ss.str();
    }    
    
    /// @brief Generates a safe pseudo-random character
    ///
    /// @result unsigned int
    static unsigned int random_char() {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, 255);
      return dis(gen);
    }
  };
  
  /// @brief Computes the directional dot-product between two tensors
  /// with summation along the given dimension
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
  /// @result Tensor containing the directional dot-product
  template<short_t dim = 0, typename T0, typename T1>
  inline auto dotproduct(T0&& t0, T1&& t1)
  {
    return torch::sum(torch::mul(t0, t1), dim);
  }

  /// @brief Computes the directional Kronecker-product between two
  /// tensors along the given dimension
  ///
  /// @tparam T0 Type of the first argument
  ///
  /// @tparam T1 Type of the second argument
  ///
  /// @param[in] t0  First argument
  ///
  /// @param[in] t1  Second argument
  ///
  /// @param[in] dim Dimension along which the Kronecker-product is computed
  ///
  /// @result Tensor containing the dimensional Kronecker-product
  ///
  /// @note This is not the regular Kronecker-product but a
  /// directional variant, that is, the Kronecker-product is computed
  /// along the given direction. All other directions are left
  /// unchanged. For the regular Kronecker-product use `torch::kron`.
  template<short_t dim = 0, typename T0, typename T1>
  inline auto kronproduct(T0&& t0, T1&& t1)
  {
    switch (t1.sizes().size()) {
    case 1:
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim)}));
    case 2:
      if constexpr (dim == 0)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({t0.size(dim),1}));
      else if constexpr (dim == 1)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                          t1.repeat({1,t0.size(dim)}));
    case 3:
      if constexpr (dim == 0)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({t0.size(dim),1,1}));
      else if constexpr (dim == 1)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                          t1.repeat({1,t0.size(dim),1}));
      else if constexpr (dim == 2)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,t0.size(dim)}));
    case 4:
      if constexpr (dim == 0)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({t0.size(dim),1,1,1}));
      else if constexpr (dim == 1)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                          t1.repeat({1,t0.size(dim),1,1}));
      else if constexpr (dim == 2)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,t0.size(dim),1}));
      else if constexpr (dim == 3)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,t0.size(dim)}));
    case 5:
      if constexpr (dim == 0)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({t0.size(dim),1,1,1,1}));
      else if constexpr (dim == 1)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                          t1.repeat({1,t0.size(dim),1,1,1}));
      else if constexpr (dim == 2)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,t0.size(dim),1,1}));
      else if constexpr (dim == 3)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,t0.size(dim),1}));
      else if constexpr (dim == 4)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,t0.size(dim)}));
    case 6:
      if constexpr (dim == 0)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({t0.size(dim),1,1,1,1,1}));
      else if constexpr (dim == 1)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                          t1.repeat({1,t0.size(dim),1,1,1,1}));
      else if constexpr (dim == 2)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,t0.size(dim),1,1,1}));
      else if constexpr (dim == 3)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,t0.size(dim),1,1}));
      else if constexpr (dim == 4)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,t0.size(dim),1}));
      else if constexpr (dim == 5)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,1,t0.size(dim)}));
    case 7:
      if constexpr (dim == 0)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({t0.size(dim),1,1,1,1,1,1}));
      else if constexpr (dim == 1)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                          t1.repeat({1,t0.size(dim),1,1,1,1,1}));
      else if constexpr (dim == 2)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,t0.size(dim),1,1,1,1}));
      else if constexpr (dim == 3)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,t0.size(dim),1,1,1}));
      else if constexpr (dim == 4)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,t0.size(dim),1,1}));
      else if constexpr (dim == 5)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,1,t0.size(dim),1}));
      else if constexpr (dim == 6)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,1,1,t0.size(dim)}));
    case 8:
      if constexpr (dim == 0)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({t0.size(dim),1,1,1,1,1,1,1}));
      else if constexpr (dim == 1)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                          t1.repeat({1,t0.size(dim),1,1,1,1,1,1}));
      else if constexpr (dim == 2)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,t0.size(dim),1,1,1,1,1}));
      else if constexpr (dim == 3)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,t0.size(dim),1,1,1,1}));
      else if constexpr (dim == 4)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,t0.size(dim),1,1,1}));
      else if constexpr (dim == 5)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,1,t0.size(dim),1,1}));
      else if constexpr (dim == 6)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,1,1,t0.size(dim),1}));
      else if constexpr (dim == 7)
        return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                          t1.repeat({1,1,1,1,1,1,1,t0.size(dim)}));
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
  /// @param[in] index        Tensor of indices
  ///
  /// @param[in] start_offset Starting value of the offset
  ///
  /// @param[in] stop_offset  Stopping value of the offset
  template<bool transpose = false>
  inline auto VSlice(torch::Tensor index, int64_t start_offset, int64_t stop_offset)
  {
    if constexpr (transpose)
      return index.repeat_interleave(stop_offset-start_offset)
        +    torch::linspace(start_offset,
                             stop_offset-1,
                             stop_offset-start_offset,
                             index.options()).repeat(index.numel());
    else
      return index.repeat(stop_offset-start_offset)
        +    torch::linspace(start_offset,
                             stop_offset-1,
                             stop_offset-start_offset,
                             index.options()).repeat_interleave(index.numel());
  }

  /// @brief Vectorized version of `torch::indexing::Slice` (see
  /// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
  ///
  /// @param[in] index        2d array of tensors of indices
  ///
  /// @param[in] start_offset 2d array of starting value of the offset
  ///
  /// @param[in] stop_offset  2d array of stopping value of the offset
  ///
  /// @param[in] leading_dim  Leading dimension
  template<bool transpose = false>
  inline auto VSlice(const std::array<torch::Tensor, 2>& index,
                     const std::array<int64_t, 2> start_offset,
                     const std::array<int64_t, 2> stop_offset,
                     int64_t leading_dim=1)
  {
    assert(index[0].numel() == index[1].numel());

    auto dist0   = stop_offset[0]-start_offset[0];
    auto dist1   = stop_offset[1]-start_offset[1];
    auto dist01  = dist0 * dist1;

    if constexpr (transpose)
      return
        (index[1].repeat_interleave(dist01)
         +
         torch::linspace(start_offset[1], stop_offset[1]-1, dist1, index[0].options()
                         ).repeat_interleave(dist0).repeat(index[0].numel())
         ) * leading_dim
        +
        index[0].repeat_interleave(dist0).repeat_interleave(dist1)
        +
        torch::linspace(start_offset[0], stop_offset[0]-1,dist0, index[0].options()
                        ).repeat(index[1].numel()).repeat(dist1);
    else
      return
        (index[1].repeat(dist01)
         +
         torch::linspace(start_offset[1], stop_offset[1]-1, dist1, index[0].options()
                         ).repeat_interleave(index[0].numel()*dist0)
         ) * leading_dim
        +
        (index[0].repeat(dist0)
         +
         torch::linspace(start_offset[0], stop_offset[0]-1,dist0, index[0].options()
                         ).repeat_interleave(index[1].numel())
         ).repeat(dist1);
  }

  /// @brief Vectorized version of `torch::indexing::Slice` (see
  /// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
  ///
  /// @param[in] index        3d array of tensors of indices
  ///
  /// @param[in] start_offset 3d array of starting value of the offset
  ///
  /// @param[in] stop_offset  3d array of stopping value of the offset
  ///
  /// @param[in] leading_dim  2d array of leading dimension
  template<bool transpose = false>
  inline auto VSlice(const std::array<torch::Tensor, 3>& index,
                     const std::array<int64_t, 3> start_offset,
                     const std::array<int64_t, 3> stop_offset,
                     const std::array<int64_t, 2> leading_dim={1,1})
  {
    assert(index[0].numel() == index[1].numel() &&
           index[1].numel() == index[2].numel());

    auto dist0   = stop_offset[0]-start_offset[0];
    auto dist1   = stop_offset[1]-start_offset[1];
    auto dist2   = stop_offset[2]-start_offset[2];
    auto dist01  = dist0 * dist1;
    auto dist12  = dist1 * dist2;
    auto dist012 = dist0 * dist12;

    if constexpr (transpose)
      return
        (index[2].repeat_interleave(dist012)
         +
         torch::linspace(start_offset[2], stop_offset[2]-1, dist2, index[0].options()
                         ).repeat_interleave(dist01).repeat(index[0].numel())
         ) * leading_dim[0] * leading_dim[1]
        +
        (index[1].repeat_interleave(dist01).repeat_interleave(dist2)
         +
         torch::linspace(start_offset[1], stop_offset[1]-1, dist1, index[0].options()
                         ).repeat_interleave(dist0).repeat(index[0].numel()).repeat(dist2)
         ) * leading_dim[0]
        +
        index[0].repeat_interleave(dist0).repeat_interleave(dist12)
        +
        torch::linspace(start_offset[0], stop_offset[0]-1, dist0, index[0].options()
                        ).repeat(index[0].numel()).repeat(dist12);
    else
      return
        (index[2].repeat(dist012)
         +
         torch::linspace(start_offset[2], stop_offset[2]-1, dist2, index[0].options()
                         ).repeat_interleave(index[0].numel()*dist01)
         ) * leading_dim[0] * leading_dim[1]
        +
        (index[1].repeat(dist01)
         +
         torch::linspace(start_offset[1], stop_offset[1]-1, dist1, index[0].options()
                         ).repeat_interleave(index[0].numel()*dist0)
         ).repeat(dist2) * leading_dim[0]
        +
        (index[0].repeat(dist0)
         +
         torch::linspace(start_offset[0], stop_offset[0]-1, dist0, index[0].options()
                         ).repeat_interleave(index[0].numel())
         ).repeat(dist12);
  }

  /// @brief Vectorized version of `torch::indexing::Slice` (see
  /// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
  ///
  /// @param[in] index        4d array of tensors of indices
  ///
  /// @param[in] start_offset 4d array of starting value of the offset
  ///
  /// @param[in] stop_offset  4d array of stopping value of the offset
  ///
  /// @param[in] leading_dim  3d array of leading dimension
  template<bool transpose = false>
  inline auto VSlice(const std::array<torch::Tensor, 4>& index,
                     const std::array<int64_t, 4> start_offset,
                     const std::array<int64_t, 4> stop_offset,
                     const std::array<int64_t, 3> leading_dim={1,1,1})
  {
    assert(index[0].numel() == index[1].numel() &&
           index[1].numel() == index[2].numel() &&
           index[2].numel() == index[3].numel());

    auto dist0    = stop_offset[0]-start_offset[0];
    auto dist1    = stop_offset[1]-start_offset[1];
    auto dist2    = stop_offset[2]-start_offset[2];
    auto dist3    = stop_offset[3]-start_offset[3];
    auto dist01   = dist0 * dist1;
    auto dist12   = dist1 * dist2;
    auto dist23   = dist2 * dist3;
    auto dist012  = dist0 * dist12;
    auto dist123  = dist1 * dist23;
    auto dist0123 = dist01 * dist23;

    if constexpr(transpose)
      return
        (index[3].repeat_interleave(dist0123)
         +
         torch::linspace(start_offset[3], stop_offset[3]-1, dist3, index[0].options()
                         ).repeat_interleave(dist012).repeat(index[0].numel())
         ) * leading_dim[0] * leading_dim[1] * leading_dim[2]
        +
        (index[2].repeat_interleave(dist012).repeat_interleave(dist3)
         +
         torch::linspace(start_offset[2], stop_offset[2]-1, dist2, index[0].options()
                         ).repeat_interleave(dist01).repeat(index[0].numel()).repeat(dist3)
         ) * leading_dim[0] * leading_dim[1]
        +
        (index[1].repeat_interleave(dist01).repeat_interleave(dist23)
         +
         torch::linspace(start_offset[1], stop_offset[1]-1, dist1, index[0].options()
                         ).repeat_interleave(dist0).repeat(index[0].numel()).repeat(dist23)
         ) * leading_dim[0]
        +
        index[0].repeat_interleave(dist0).repeat_interleave(dist123)
        +
        torch::linspace(start_offset[0], stop_offset[0]-1, dist0, index[0].options()
                        ).repeat(index[0].numel()).repeat(dist123);
    else
      return
        (index[3].repeat(dist0123)
         +
         torch::linspace(start_offset[3], stop_offset[3]-1, dist3, index[0].options()
                         ).repeat_interleave(index[0].numel()*dist012)
         ) * leading_dim[0] * leading_dim[1] * leading_dim[2]
        +
        (index[2].repeat(dist012)
         +
         torch::linspace(start_offset[2], stop_offset[2]-1, dist2, index[0].options()
                         ).repeat_interleave(index[0].numel()*dist01)
         ).repeat(dist3) * leading_dim[0] * leading_dim[1]
        +
        (index[1].repeat(dist01)
         +
         torch::linspace(start_offset[1], stop_offset[1]-1, dist1, index[0].options()
                         ).repeat_interleave(index[0].numel()*dist0)
         ).repeat(dist23) * leading_dim[0]
        +
        (index[0].repeat(dist0)
         +
         torch::linspace(start_offset[0], stop_offset[0]-1, dist0, index[0].options()
                         ).repeat_interleave(index[0].numel())
         ).repeat(dist123);
  }

  /// @brief Concatenates multiple std::vector objects
  template<typename... Ts>
  inline auto concat(const std::vector<Ts>&... vectors)
  {
    std::vector<typename std::tuple_element<0, std::tuple<Ts...> >::type> result;

    (result.insert(result.end(), vectors.begin(), vectors.end()), ...);

    return result;
  }

  /// @brief Concatenates multiple std::array objects
  template<typename T, std::size_t... N>
  inline auto concat(const std::array<T, N>&... arrays)
  {
    std::array<T, (N + ...)> result;
    std::size_t index{};

    ((std::copy_n(arrays.begin(), N, result.begin() + index), index += N), ...);

    return result;
  }

  /// @brief Converts an std::vector object into std::array
  template<std::size_t N, typename T>
  inline std::array<T, N> convert(std::vector<T>&& vector)
  {
    std::array<T, N> array;
    std::move(vector.begin(), vector.end(), array.begin());
    return array;
  }

  /// @brief Converts an std::array object into std::vector
  template<typename T, std::size_t N>
  inline std::vector<T> convert(std::array<T, N>&& array)
  {
    std::vector<T> vector;
    std::move(array.begin(), array.end(), vector.begin());
    return vector;
  }

  /// @brief Converts an std::initializer_list to torch::Tensor
  /// @{
  template<typename T>
  inline auto to_tensor(std::initializer_list<T> list,
                        torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                        const torch::TensorOptions& options = iganet::core<T>{}.options())
  {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T*>(std::data(list)),
                            (sizes == torch::IntArrayRef{-1}) ? list.size() : sizes,
                            options).clone();
  else
    return torch::from_blob(const_cast<T*>(std::data(list)),
                            (sizes == torch::IntArrayRef{-1}) ? list.size() : sizes,
                            options.device(torch::kCPU))
      .to(options.device());
  }

  template<typename T>
  inline auto to_tensor(std::initializer_list<T> list,
                        const torch::TensorOptions& options)
  {
    if (options.device() == torch::kCPU)
      return torch::from_blob(const_cast<T*>(std::data(list)),
                              list.size(),
                              options).clone();
    else
      return torch::from_blob(const_cast<T*>(std::data(list)),
                              list.size(),
                              options.device(torch::kCPU))
        .to(options.device());
  }
  /// @}

  /// @brief Converts an std::initializer_list to TensorArray1
  /// @{
  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list,
                             torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                             const torch::TensorOptions& options = iganet::core<T>{}.options())
  {
    return TensorArray1({to_tensor(list, sizes, options)});
  }

  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list,
                             const torch::TensorOptions& options)
  {
    return TensorArray1({to_tensor(list, torch::IntArrayRef{-1}, options)});
  }
  /// @}


  /// @brief Converts two std::initializer_list's to TensorArray2
  /// @{
  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list0,
                             std::initializer_list<T> list1,
                             torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                             const torch::TensorOptions& options = iganet::core<T>{}.options())
  {
    return TensorArray2({to_tensor(list0, sizes, options),
                         to_tensor(list1, sizes, options)});
  }

  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list0,
                             std::initializer_list<T> list1,
                             const torch::TensorOptions& options)
  {
    return TensorArray2({to_tensor(list0, torch::IntArrayRef{-1}, options),
                         to_tensor(list1, torch::IntArrayRef{-1}, options)});
  }
  /// @}

  /// @brief Converts three std::initializer_list's to TensorArray3
  /// @{
  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list0,
                             std::initializer_list<T> list1,
                             std::initializer_list<T> list2,
                             torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                             const torch::TensorOptions& options = iganet::core<T>{}.options())
  {
    return TensorArray3({to_tensor(list0, sizes, options),
                         to_tensor(list1, sizes, options),
                         to_tensor(list2, sizes, options)});
  }

  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list0,
                             std::initializer_list<T> list1,
                             std::initializer_list<T> list2,
                             const torch::TensorOptions& options)
  {
    return TensorArray3({to_tensor(list0, torch::IntArrayRef{-1}, options),
                         to_tensor(list1, torch::IntArrayRef{-1}, options),
                         to_tensor(list2, torch::IntArrayRef{-1}, options)});
  }
  /// @}

  /// @brief Converts four std::initializer_list's to TensorArray4
  /// @{
  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list0,
                             std::initializer_list<T> list1,
                             std::initializer_list<T> list2,
                             std::initializer_list<T> list3,
                             torch::IntArrayRef sizes = {-1},
                             const torch::TensorOptions& options = iganet::core<T>{}.options())
  {
    return TensorArray4({to_tensor(list0, sizes, options),
                         to_tensor(list1, sizes, options),
                         to_tensor(list2, sizes, options),
                         to_tensor(list3, sizes, options)});
  }

  template<typename T>
  inline auto to_tensorArray(std::initializer_list<T> list0,
                             std::initializer_list<T> list1,
                             std::initializer_list<T> list2,
                             std::initializer_list<T> list3,
                             const torch::TensorOptions& options)
  {
    return TensorArray4({to_tensor(list0, torch::IntArrayRef{-1}, options),
                         to_tensor(list1, torch::IntArrayRef{-1}, options),
                         to_tensor(list2, torch::IntArrayRef{-1}, options),
                         to_tensor(list3, torch::IntArrayRef{-1}, options)});
  }
  /// @}

  /// @brief Converts a torch::Tensor object to a
  /// torch::TensorAccessor object
  template<typename T, std::size_t N>
  auto to_tensorAccessor(const torch::Tensor& tensor) {
    return tensor.template accessor<T, N>();
  }

  namespace detail {
    /// @brief Converts an std::array of torch::Tensor objects to an
    /// array of torch::TensorAccessor objects
    template<typename T, std::size_t N, std::size_t... Is>
    auto to_tensorAccessor(const std::array<torch::Tensor, sizeof...(Is)>& tensors,
                           std::index_sequence<Is...>) {
      return std::array<torch::TensorAccessor<T, N>, sizeof...(Is)>{tensors[Is].template accessor<T, N>()...};
    }
  } // namespace detail

  /// @brief Converts an std::array of torch::Tensor objects to an
  /// array of torch::TensorAccessor objects
  template<typename T, std::size_t N, std::size_t M>
  auto to_tensorAccessor(const std::array<torch::Tensor, M>& tensors) {
    return detail::to_tensorAccessor<T, N>(tensors, std::make_index_sequence<M>());
  }

  /// @brief Converts a torch::TensorAccessor object to a JSON object
  template<typename T, std::size_t N>
  auto to_json(const torch::TensorAccessor<T, N>& accessor)
  {
    auto json = nlohmann::json::array();

    if constexpr (N == 1) {
      for (int64_t i = 0; i < accessor.size(0); ++i)
        json.push_back(accessor[i]);
    } else if constexpr (N == 2) {
      for (int64_t i = 0; i < accessor.size(0); ++i)
        for (int64_t j = 0; j < accessor.size(1); ++j)
          json.push_back(accessor[i][j]);
    } else if constexpr (N == 3) {
      for (int64_t i = 0; i < accessor.size(0); ++i)
        for (int64_t j = 0; j < accessor.size(1); ++j)
          for (int64_t k = 0; k < accessor.size(2); ++k)
            json.push_back(accessor[i][j][k]);
    } else if constexpr (N == 4) {
      for (int64_t i = 0; i < accessor.size(0); ++i)
        for (int64_t j = 0; j < accessor.size(1); ++j)
          for (int64_t k = 0; k < accessor.size(2); ++k)
            for (int64_t l = 0; l < accessor.size(3); ++l)
              json.push_back(accessor[i][j][k][l]);
    }

    return json;
  }

  /// @brief Converts a torch::Tensor object to a JSON object
  template<typename T, std::size_t N>
  auto to_json(const torch::Tensor& tensor) {
    auto accessor = to_tensorAccessor<T, N>(tensor);
    return ::iganet::to_json(accessor);
  }

  /// @brief Converts an std::array of torch::Tensor objects to a JSON
  /// object
  template<typename T, std::size_t N, std::size_t M>
  auto to_json(const std::array<torch::Tensor, M>& tensors) {
    auto accessors = to_tensorAccessor<T, N>(tensors);
    auto json = nlohmann::json::array();
    for (std::size_t i = 0; i < M; ++i)
      json.push_back(::iganet::to_json<T, N>(accessors[i]));
    return json;
  }

  /// @brief Computes the power of integer `E` to the `N` at compile time
  /// @{
  template<int E, int N>
  struct integer_pow {
    enum { value = E * integer_pow<E, N - 1>::value };
  };

  template <int E>
  struct integer_pow<E, 0> {
    enum { value = 1 };
  };
  /// @}

} // namespace iganet
