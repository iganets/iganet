/**
   @file utils/tensorarray.hpp

   @brief TensorArray utility functions.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <initializer_list>

#include <core/core.hpp>
#include <core/options.hpp>
#include <utils/container.hpp>

namespace iganet::utils {

template <std::size_t N> using TensorArray = std::array<torch::Tensor, N>;

using TensorArray0 = TensorArray<0>;
using TensorArray1 = TensorArray<1>;
using TensorArray2 = TensorArray<2>;
using TensorArray3 = TensorArray<3>;
using TensorArray4 = TensorArray<4>;

/// @brief Converts a set of std::initializer_list objects to a TensorArray
/// object
/// @{
/// @tparam Ts Template parameter `Ts`.
/// @return Result of the operation.
template <typename... Ts>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(std::initializer_list<Ts> &&...lists) {
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists),
                    torch::IntArrayRef{-1}, Options<Ts>{})...};
}

/// @brief Provides the `function` operation.
/// @tparam Ts Template parameter `Ts`.
/// @return Result of the operation.
template <typename... Ts>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(torch::IntArrayRef sizes, std::initializer_list<Ts> &&...lists) {
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists), sizes,
                    Options<Ts>{})...};
}

/// @brief Provides the `function` operation.
/// @tparam Ts Template parameter `Ts`.
/// @tparam T Template parameter `T`.
/// @return Result of the operation.
template <typename... Ts, typename T>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(const iganet::Options<T> &options,
               std::initializer_list<Ts> &&...lists) {
  static_assert(
      (std::is_same_v<T, Ts> && ...),
      "Type mismatch between Options<T> and std::initializer_list<Ts>");
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists),
                    torch::IntArrayRef{-1}, options)...};
}

/// @brief Provides the `function` operation.
/// @tparam Ts Template parameter `Ts`.
/// @tparam T Template parameter `T`.
/// @return Result of the operation.
template <typename... Ts, typename T>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(torch::IntArrayRef sizes, const iganet::Options<T> &options,
               std::initializer_list<Ts> &&...lists) {
  static_assert(
      (std::is_same_v<T, Ts> && ...),
      "Type mismatch between Options<T> and std::initializer_list<Ts>");
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists), sizes,
                    options)...};
}
/// @}

/// @brief Converts a torch::Tensor object to a
/// torch::TensorAccessor object
/// @{
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param tensor Tensor to process.
/// @return Result of the operation.
template <typename T, std::size_t N>
auto to_tensorAccessor(const torch::Tensor &tensor) {
  return tensor.accessor<T, N>();
}

/// @brief Provides the `to_tensorAccessor` operation.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param tensor Tensor to process.
/// @param deviceType Value of `deviceType`.
/// @return Result of the operation.
template <typename T, std::size_t N>
auto to_tensorAccessor(const torch::Tensor &tensor,
                       c10::DeviceType deviceType) {

  if (deviceType != tensor.device().type()) {
    auto tensor_device = tensor.to(deviceType);
    auto accessor = tensor_device.accessor<T, N>();
    return std::tuple(tensor_device, accessor);
  } else {
    auto accessor = tensor.accessor<T, N>();
    return std::tuple(tensor, accessor);
  }
}
/// @}

namespace detail {
/// @brief Converts a std::array of torch::Tensor objects to an
/// array of torch::TensorAccessor objects
/// @{
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam Is Template parameter `Is`.
/// @param tensorArray Value of `tensorArray`.
/// @return Result of the operation.
template <typename T, std::size_t N, std::size_t... Is>
auto to_tensorAccessor(const TensorArray<sizeof...(Is)> &tensorArray,
                       std::index_sequence<Is...>) {
  return std::array<torch::TensorAccessor<T, N>, sizeof...(Is)>{
      tensorArray[Is].template accessor<T, N>()...};
}

/// @brief Provides the `to_tensorAccessor` operation.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam Is Template parameter `Is`.
/// @param tensorArray Value of `tensorArray`.
/// @param deviceType Value of `deviceType`.
/// @return Result of the operation.
template <typename T, std::size_t N, std::size_t... Is>
auto to_tensorAccessor(const TensorArray<sizeof...(Is)> &tensorArray,
                       c10::DeviceType deviceType, std::index_sequence<Is...>) {
  std::array<torch::Tensor, sizeof...(Is)> tensorArray_device{
      tensorArray[Is].to(deviceType)...};
  std::array<torch::TensorAccessor<T, N>, sizeof...(Is)> accessors{
      tensorArray_device[Is].template accessor<T, N>()...};
  return std::tuple(tensorArray_device, accessors);
}

/// @brief Provides the `to_tensorAccessor` operation.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam Dims Template parameter `Dims`.
/// @tparam Is Template parameter `Is`.
/// @param blocktensor Value of `blocktensor`.
/// @param deviceType Value of `deviceType`.
/// @return Result of the operation.
template <typename T, std::size_t N, size_t... Dims, std::size_t... Is>
auto to_tensorAccessor(const BlockTensor<torch::Tensor, Dims...> &blocktensor,
                       c10::DeviceType deviceType, std::index_sequence<Is...>) {
  std::array<torch::Tensor, sizeof...(Is)> tensorArray_device{
      blocktensor[Is]->to(deviceType)...};
  std::array<torch::TensorAccessor<T, N>, sizeof...(Is)> accessors{
      tensorArray_device[Is].template accessor<T, N>()...};
  return std::tuple(tensorArray_device, accessors);
}
/// @}
} // namespace detail

/// @brief Converts a std::array of torch::Tensor objects to an
/// array of torch::TensorAccessor objects
/// @{
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam M Template parameter `M`.
/// @param tensorArray Value of `tensorArray`.
/// @return Result of the operation.
template <typename T, std::size_t N, std::size_t M>
auto to_tensorAccessor(const TensorArray<M> &tensorArray) {
  return detail::to_tensorAccessor<T, N>(tensorArray,
                                         std::make_index_sequence<M>());
}

/// @brief Provides the `to_tensorAccessor` operation.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam M Template parameter `M`.
/// @param tensorArray Value of `tensorArray`.
/// @param deviceType Value of `deviceType`.
/// @return Result of the operation.
template <typename T, std::size_t N, std::size_t M>
auto to_tensorAccessor(const TensorArray<M> &tensorArray,
                       c10::DeviceType deviceType) {
  return detail::to_tensorAccessor<T, N>(tensorArray, deviceType,
                                         std::make_index_sequence<M>());
}

/// @brief Provides the `to_tensorAccessor` operation.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam Dims Template parameter `Dims`.
/// @param blocktensor Value of `blocktensor`.
/// @param deviceType Value of `deviceType`.
/// @return Result of the operation.
template <typename T, std::size_t N, std::size_t... Dims>
auto to_tensorAccessor(const BlockTensor<torch::Tensor, Dims...> &blocktensor,
                       c10::DeviceType deviceType) {
  return detail::to_tensorAccessor<T, N, Dims...>(
      blocktensor, deviceType, std::make_index_sequence<(Dims * ...)>());
}
/// @}

} // namespace iganet::utils

#define TENSORARRAY_FORALL(obj, func, ...)                                     \
  []<std::size_t N>(const ::iganet::utils::TensorArray<N> &tensorArray) {      \
    ::iganet::utils::TensorArray<N> result;                                    \
    for (std::size_t i = 0; i < N; ++i)                                        \
      result[i] = tensorArray[i].func(__VA_ARGS__);                            \
    return result;                                                             \
  }(obj)

namespace std {

/// @brief Print (as string) a TensorArray object.
/// @tparam N Template parameter `N`.
/// @param os Output stream.
/// @param obj Object to process.
/// @return Result of the operation.
template <std::size_t N>
inline std::ostream &operator<<(std::ostream &os,
                                const std::array<torch::Tensor, N> &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\n";
  for (std::size_t i = 0; i < N; ++i) {
    os << obj[i] << "\n";

    if (iganet::is_verbose(os))
      os << "[ " << obj[i].options() << " ]\n";
  }

  os << ")";

  return os;
}

} // namespace std
