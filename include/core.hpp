#include <array>
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
  constexpr auto dtype() { return torch::kByte; }

  template<>
  constexpr auto dtype<double>() { return torch::kFloat64; }

  template<>
  constexpr auto dtype<float>() { return torch::kFloat32; }

  template<>
  constexpr auto dtype<long int>() { return torch::kLong; }

  template<>
  constexpr auto dtype<int>() { return torch::kInt; };

  template<>
  constexpr auto  dtype<short>() { return torch::kShort; }

  template<>
  constexpr auto dtype<char>() { return torch::kChar; };

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
  auto concat(const std::vector<Ts>&... vectors)
  {
    std::vector<typename std::tuple_element<0, std::tuple<Ts...> >::type> result;

    (result.insert(result.end(), vectors.begin(), vectors.end()), ...);

    return result;
  }
  
  // Concatenates multiple std::array objects
  template<typename T, std::size_t... N>
  auto concat(const std::array<T, N>&... arrays)
  {
    std::array<T, (N + ...)> result;
    std::size_t index{};
    
    ((std::copy_n(arrays.begin(), N, result.begin() + index), index += N), ...);
    
    return result;
  }
  
} // namespace iganet
