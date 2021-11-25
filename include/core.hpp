#include <array>
#include <tuple>

#include <torch/torch.h>
#include <torch/csrc/api/include/torch/types.h>

#pragma once

namespace iganet {
  
  #define short_t unsigned short int
  
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
  
  template <typename Type, std::size_t... sizes>
  auto concat(const std::array<Type, sizes>&... arrays)
  {
    std::array<Type, (sizes + ...)> result;
    std::size_t index{};
    
    ((std::copy_n(arrays.begin(), sizes, result.begin() + index), index += sizes), ...);
    
    return result;
  }

  template <typename... Ts>
  auto concat(const std::vector<Ts>&... vectors)
  {
    std::vector<typename std::tuple_element<0, std::tuple<Ts...> >::type> result;

    (result.insert(result.end(), vectors.begin(), vectors.end()), ...);
    
    return result;
  }
  
} // namespace iganet
