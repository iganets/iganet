/**
   @file utils/tuple.hpp

   @brief Tuple utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <tuple>
#include <type_traits>
#include <vector>

namespace iganet::utils {

/// @brief Type trait for std::tuple type
/// @{
template <class T> struct is_tuple : std::false_type {};

template <class... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type {};
/// @}

template <class T> inline constexpr bool is_tuple_v = is_tuple<T>::value;

/// @brief Type trait for std::tuple<std::tuple> type
/// @{
template <class T> struct is_tuple_of_tuples : std::false_type {};

template <class... Ts>
struct is_tuple_of_tuples<std::tuple<Ts...>>
    : std::common_type_t<is_tuple<Ts>...> {};
/// @}

/// @brief Alias for is_tuple_of_tuples::type
template <class T> using is_tuple_of_tuples_t = is_tuple_of_tuples<T>::type;

/// @brief Alias for is_tuple_of_tuples::value
template <class T>
inline constexpr auto is_tuple_of_tuples_v = is_tuple_of_tuples<T>::value;

/// @brief Type trait for concatenating std::tuples
/// @{
template <typename... Tuples> struct tuple_cat;

template <> struct tuple_cat<> {
  using type = std::tuple<>;
};

template <typename... Ts, typename... Tuples>
struct tuple_cat<std::tuple<Ts...>, Tuples...> {
  using type = decltype(std::tuple_cat(
      std::declval<std::tuple<Ts...>>(),
      std::declval<typename tuple_cat<Tuples...>::type>()));
};

template <typename T, typename... Tuples> struct tuple_cat<T, Tuples...> {
  using type = decltype(std::tuple_cat(
      std::declval<std::tuple<T>>(),
      std::declval<typename tuple_cat<Tuples...>::type>()));
};
/// @}

/// @brief Alias for tuple_cat::type
template <typename... Tuples> using tuple_cat_t = tuple_cat<Tuples...>::type;

/// @brief Alias for tuple_cat::value
template <typename... Tuples>
inline constexpr auto tuple_cat_v = tuple_cat<Tuples...>::value;
  
/// @brief Concatenates the entries of a std::tuple object into a
/// single Torch tensor along the given dimension
template <typename... Tensors>
torch::Tensor cat_tuple_into_tensor(const std::tuple<Tensors...> &tensors,
                                    int64_t dim = 0) {
  std::vector<torch::Tensor> vec;
  vec.reserve(sizeof...(Tensors));
  std::apply([&](const auto &...tensor) { (vec.emplace_back(tensor), ...); },
             tensors);

  return torch::cat(vec, dim);
}

/// @brief Concatenates the entries of a std::tuple object into a
/// single Torch tensor along the given dimension after applying the
/// callback function
template <typename... Tensors, typename Func>
torch::Tensor cat_tuple_into_tensor(const std::tuple<Tensors...> &tensors,
                                    Func &&func, int64_t dim = 0) {
  std::vector<torch::Tensor> vec;
  vec.reserve(sizeof...(Tensors));
  std::apply(
      [&](const auto &...tensor) {
        (vec.emplace_back(std::invoke(func, tensor)), ...);
      },
      tensors);

  return torch::cat(vec, dim);
}

/// @brief Returns a std::tuple object with N replications of the given value
template <std::size_t N, typename T>
constexpr auto repeat_tuple(const T &value) {
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::tuple{((void)Is, value)...};
  }(std::make_index_sequence<N>{});
}

/// @brief Slices the given tensor into the objects of the std::tuple
/// @{
template <std::size_t I = 0, typename... Tensors, typename FuncSize,
          typename FuncAssign>
void slice_tensor_into_tuple(std::tuple<Tensors...> &tuple,
                             const torch::Tensor &tensor, FuncSize &&funcSize,
                             FuncAssign &&funcAssign, int64_t &offset,
                             int64_t dim = 0) {
  if constexpr (I < sizeof...(Tensors)) {
    auto &t = std::get<I>(tuple);
    auto size = std::forward<FuncSize>(funcSize)(t);
    std::forward<FuncAssign>(funcAssign)(
        t, tensor.slice(dim, offset, offset + size));
    offset += size;
    slice_tensor_into_tuple<I + 1>(tuple, tensor, funcSize, funcAssign, offset,
                                   dim);
  }
}

template <typename... Tensors, typename FuncSize, typename FuncAssign>
void slice_tensor_into_tuple(std::tuple<Tensors...> &tuple,
                             const torch::Tensor &tensor, FuncSize &&funcSize,
                             FuncAssign &&funcAssign, int64_t dim = 0) {
  int64_t offset = 0;
  slice_tensor_into_tuple(tuple, tensor, funcSize, funcAssign, offset, dim);
}
/// @}

} // namespace iganet::utils
