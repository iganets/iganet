/**
   @file include/core.hpp

   @brief Core components
   
   @author Matthias Moller
      
   @copyright This file is part of the IgaNet project
   
   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <array>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <matplot/matplot.h>
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/types.h>

#pragma once

namespace iganet {

#define short_t unsigned short int

  // Determines the LibTorch dtype from template parameter
  template<typename T>
  inline constexpr auto dtype() { return torch::kByte; }

  template<>
  inline constexpr auto dtype<double>() { return torch::kFloat64; }

  template<>
  inline constexpr auto dtype<float>() { return torch::kFloat32; }

  template<>
  inline constexpr auto dtype<long int>() { return torch::kLong; }

  template<>
  inline constexpr auto dtype<int>() { return torch::kInt; };

  template<>
  inline constexpr auto  dtype<short>() { return torch::kShort; }

  template<>
  inline constexpr auto dtype<char>() { return torch::kChar; };

  // LibTorch core object handles the automated determination of dtype
  // from the template argument and the selection of the device
  template<typename real_t>
  class core {
  public:
    core()
      : options_(torch::TensorOptions()
                 .dtype(dtype<real_t>())
                 .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
                 .requires_grad(true))
    {}

    // Tensor options
    const torch::TensorOptions options_;
  };

  // Concatenates multiple std::vector objects
  template<typename... Ts>
  inline auto concat(const std::vector<Ts>&... vectors)
  {
    std::vector<typename std::tuple_element<0, std::tuple<Ts...> >::type> result;

    (result.insert(result.end(), vectors.begin(), vectors.end()), ...);

    return result;
  }

  // Concatenates multiple std::array objects
  template<typename T, std::size_t... N>
  inline auto concat(const std::array<T, N>&... arrays)
  {
    std::array<T, (N + ...)> result;
    std::size_t index{};

    ((std::copy_n(arrays.begin(), N, result.begin() + index), index += N), ...);

    return result;
  }

  template<typename T>
  inline auto to_tensor(std::initializer_list<T> list)
  {
    auto it = list.begin();
    switch (list.size()) {
    case 1:
      return torch::stack({
          torch::full({1}, *it++)
        }).flatten();
    case 2:
      return torch::stack({
          torch::full({1}, *it++),
          torch::full({1}, *it++)
        }).flatten();
    case 3:
      return torch::stack({
          torch::full({1}, *it++),
          torch::full({1}, *it++),
          torch::full({1}, *it++)
        }).flatten();
    case 4:
      return torch::stack({
          torch::full({1}, *it++),
          torch::full({1}, *it++),
          torch::full({1}, *it++),
          torch::full({1}, *it++)
        }).flatten();
    default:
      throw std::runtime_error("Invalid size");
    }
  }

} // namespace iganet

/// Print (as string) an array of torch::Tensor objects
template<std::size_t N>
inline std::ostream& operator<<(std::ostream& os,
                                const std::array<torch::Tensor, N>& obj)
{
  for (auto i : obj)
    os << i << std::endl;
  return os;
}
