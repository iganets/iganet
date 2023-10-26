/**
   @file include/utils/tensorarray.hpp

   @brief TensorArray utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <initializer_list>

#include <options.hpp>
#include <utils/convert.hpp>

#include <torch/torch.h>

namespace iganet {
namespace utils {

using TensorArray0 = std::array<torch::Tensor, 0>;
using TensorArray1 = std::array<torch::Tensor, 1>;
using TensorArray2 = std::array<torch::Tensor, 2>;
using TensorArray3 = std::array<torch::Tensor, 3>;
using TensorArray4 = std::array<torch::Tensor, 4>;

/// @brief Converts an std::initializer_list to TensorArray1
/// @{
template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list,
                           torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                           const iganet::Options<T> &options = Options<T>{}) {
  return TensorArray1({to_tensor(list, sizes, options)});
}

template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list,
                           const iganet::Options<T> &options) {
  return TensorArray1({to_tensor(list, torch::IntArrayRef{-1}, options)});
}
/// @}

/// @brief Converts two std::initializer_list's to TensorArray2
/// @{
template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list0,
                           std::initializer_list<T> &&list1,
                           torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                           const iganet::Options<T> &options = Options<T>{}) {
  return TensorArray2(
      {to_tensor(list0, sizes, options), to_tensor(list1, sizes, options)});
}

template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list0,
                           std::initializer_list<T> &&list1,
                           const iganet::Options<T> &options) {
  return TensorArray2({to_tensor(list0, torch::IntArrayRef{-1}, options),
                       to_tensor(list1, torch::IntArrayRef{-1}, options)});
}
/// @}

/// @brief Converts three std::initializer_list's to TensorArray3
/// @{
template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list0,
                           std::initializer_list<T> &&list1,
                           std::initializer_list<T> &&list2,
                           torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                           const iganet::Options<T> &options = Options<T>{}) {
  return TensorArray3({to_tensor(list0, sizes, options),
                       to_tensor(list1, sizes, options),
                       to_tensor(list2, sizes, options)});
}

template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list0,
                           std::initializer_list<T> &&list1,
                           std::initializer_list<T> &&list2,
                           const iganet::Options<T> &options) {
  return TensorArray3({to_tensor(list0, torch::IntArrayRef{-1}, options),
                       to_tensor(list1, torch::IntArrayRef{-1}, options),
                       to_tensor(list2, torch::IntArrayRef{-1}, options)});
}
/// @}

/// @brief Converts four std::initializer_list's to TensorArray4
/// @{
template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list0,
                           std::initializer_list<T> &&list1,
                           std::initializer_list<T> &&list2,
                           std::initializer_list<T> &&list3,
                           torch::IntArrayRef sizes = {-1},
                           const iganet::Options<T> &options = Options<T>{}) {
  return TensorArray4(
      {to_tensor(list0, sizes, options), to_tensor(list1, sizes, options),
       to_tensor(list2, sizes, options), to_tensor(list3, sizes, options)});
}

template <typename T>
inline auto to_tensorArray(std::initializer_list<T> &&list0,
                           std::initializer_list<T> &&list1,
                           std::initializer_list<T> &&list2,
                           std::initializer_list<T> &&list3,
                           const iganet::Options<T> &options) {
  return TensorArray4({to_tensor(list0, torch::IntArrayRef{-1}, options),
                       to_tensor(list1, torch::IntArrayRef{-1}, options),
                       to_tensor(list2, torch::IntArrayRef{-1}, options),
                       to_tensor(list3, torch::IntArrayRef{-1}, options)});
}
/// @}

/// @brief Converts a torch::Tensor object to a
/// torch::TensorAccessor object
/// @{
template <typename T, std::size_t N>
auto to_tensorAccessor(const torch::Tensor &tensor) {
  return tensor.template accessor<T, N>();
}

template <typename T, std::size_t N>
auto to_tensorAccessor(const torch::Tensor &tensor,
                       c10::DeviceType deviceType) {

  if (deviceType != tensor.device().type()) {
    auto tensor_device = tensor.to(deviceType);
    auto accessor = tensor_device.template accessor<T, N>();
    return std::tuple(tensor_device, accessor);
  } else {
    auto accessor = tensor.template accessor<T, N>();
    return std::tuple(tensor, accessor);    
  }
      
}
/// @}

namespace detail {
/// @brief Converts an std::array of torch::Tensor objects to an
/// array of torch::TensorAccessor objects
/// @{
template <typename T, std::size_t N, std::size_t... Is>
auto to_tensorAccessor(const std::array<torch::Tensor, sizeof...(Is)> &tensors,
                       std::index_sequence<Is...>) {
  return std::array<torch::TensorAccessor<T, N>, sizeof...(Is)>{
      tensors[Is].template accessor<T, N>()...};
}

template <typename T, std::size_t N, std::size_t... Is>
auto to_tensorAccessor(const std::array<torch::Tensor, sizeof...(Is)> &tensors,
                       c10::DeviceType deviceType, std::index_sequence<Is...>) {
  std::array<torch::TensorBase, sizeof...(Is)> tensors_device{
      tensors[Is].to(deviceType)...};
  std::array<torch::TensorAccessor<T, N>, sizeof...(Is)> accessors{
      tensors_device[Is].template accessor<T, N>()...};
  return std::tuple(tensors_device, accessors);
}

template <typename T, std::size_t N, size_t... Dims, std::size_t... Is>
auto to_tensorAccessor(const BlockTensor<torch::Tensor, Dims...> &blocktensor,
                       c10::DeviceType deviceType, std::index_sequence<Is...>) {
  std::array<torch::TensorBase, sizeof...(Is)> tensors_device{
      blocktensor[Is]->to(deviceType)...};
  std::array<torch::TensorAccessor<T, N>, sizeof...(Is)> accessors{
      tensors_device[Is].template accessor<T, N>()...};
  return std::tuple(tensors_device, accessors);
}
/// @}
} // namespace detail

/// @brief Converts an std::array of torch::Tensor objects to an
/// array of torch::TensorAccessor objects
/// @{
template <typename T, std::size_t N, std::size_t M>
auto to_tensorAccessor(const std::array<torch::Tensor, M> &tensors) {
  return detail::to_tensorAccessor<T, N>(tensors,
                                         std::make_index_sequence<M>());
}

template <typename T, std::size_t N, std::size_t M>
auto to_tensorAccessor(const std::array<torch::Tensor, M> &tensors,
                       c10::DeviceType deviceType) {
  return detail::to_tensorAccessor<T, N>(tensors, deviceType,
                                         std::make_index_sequence<M>());
}

template <typename T, std::size_t N, std::size_t... Dims>
auto to_tensorAccessor(const BlockTensor<torch::Tensor, Dims...> &blocktensor,
                       c10::DeviceType deviceType) {
  return detail::to_tensorAccessor<T, N, Dims...>(
      blocktensor, deviceType, std::make_index_sequence<(Dims * ...)>());
}
/// @}

} // namespace utils
} // namespace iganet
