/**
   @file utils/container.hpp

   @brief Container utility functions.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <initializer_list>
#include <vector>

#include <iganet/core/options.hpp>

namespace iganet::utils {

/// @brief Converts a std::vector object into std::array.
/// @tparam N Template parameter `N`.
/// @tparam T Template parameter `T`.
/// @param vector Value of `vector`.
/// @return Result of the operation.
template <std::size_t N, typename T>
inline std::array<T, N> to_array(std::vector<T> &&vector) {
  if (vector.size() != N)
    throw std::invalid_argument("Cannot convert std::vector to std::array: size mismatch");
  
  std::array<T, N> array;
  std::move(vector.begin(), vector.end(), array.begin());
  return array;
}

/// @brief Converts a std::array object into std::vector.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline std::vector<T> to_vector(std::array<T, N> &&array) {
  std::vector<T> vector(N);
  std::move(array.begin(), array.end(), vector.begin());
  return vector;
}

/// @brief Converts a list of arguments into std::array.
/// @tparam Args Template parameter `Args`.
/// @param args Value of `args`.
/// @return Result of the operation.
template <typename... Args> inline auto to_array(Args &&...args) {
  return std::array<std::common_type_t<Args...>, sizeof...(Args)>{
      std::move(args)...};
}

/// @brief Converts a list of arguments into std::vector.
/// @tparam Args Template parameter `Args`.
/// @param args Value of `args`.
/// @return Result of the operation.
template <typename... Args> inline auto to_vector(Args &&...args) {
  return std::vector<std::common_type_t<Args...>>{std::move(args)...};
}

/// @brief Converts a std::array to torch::Tensor
/// @{
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param array Value of `array`.
/// @param sizes Value of `sizes`.
/// @param options Configuration options.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline auto
to_tensor(const std::array<T, N> &array,
          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
          const iganet::Options<T> &options = iganet::Options<T>{}) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(array)),
                            (sizes == torch::IntArrayRef{-1}) ? array.size()
                                                              : sizes,
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(array)),
                            (sizes == torch::IntArrayRef{-1}) ? array.size()
                                                              : sizes,
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}

/// @brief Provides the `to_tensor` operation.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param array Value of `array`.
/// @param options Configuration options.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline auto to_tensor(const std::array<T, N> &array,
                      const iganet::Options<T> &options) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(array)), array.size(),
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(array)), array.size(),
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}
/// @}

/// @brief Converts a std::initializer_list to torch::Tensor
/// @{
/// @tparam T Template parameter `T`.
/// @param list Value of `list`.
/// @param sizes Value of `sizes`.
/// @param options Configuration options.
/// @return Result of the operation.
template <typename T>
inline auto
to_tensor(std::initializer_list<T> list,
          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
          const iganet::Options<T> &options = iganet::Options<T>{}) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(
               const_cast<T *>(std::data(list)),
               (sizes == torch::IntArrayRef{-1}) ? list.size() : sizes, options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(list)),
                            (sizes == torch::IntArrayRef{-1}) ? list.size()
                                                              : sizes,
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}

/// @brief Provides the `to_tensor` operation.
/// @tparam T Template parameter `T`.
/// @param list Value of `list`.
/// @param options Configuration options.
/// @return Result of the operation.
template <typename T>
inline auto to_tensor(std::initializer_list<T> &list,
                      const iganet::Options<T> &options) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(list)), list.size(),
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(list)), list.size(),
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}
/// @}

/// @brief Converts a std::vector to torch::Tensor
/// @{
/// @tparam T Template parameter `T`.
/// @param vector Value of `vector`.
/// @param sizes Value of `sizes`.
/// @param options Configuration options.
/// @return Result of the operation.
template <typename T>
inline auto
to_tensor(const std::vector<T> &vector,
          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
          const iganet::Options<T> &options = iganet::Options<T>{}) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(vector)),
                            (sizes == torch::IntArrayRef{-1}) ? vector.size()
                                                              : sizes,
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(vector)),
                            (sizes == torch::IntArrayRef{-1}) ? vector.size()
                                                              : sizes,
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}

/// @brief Provides the `to_tensor` operation.
/// @tparam T Template parameter `T`.
/// @param vector Value of `vector`.
/// @param options Configuration options.
/// @return Result of the operation.
template <typename T>
inline auto to_tensor(const std::vector<T> &vector,
                      const iganet::Options<T> &options) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(vector)), vector.size(),
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(vector)), vector.size(),
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}
/// @}

/// @brief Converts a std::array<int64_t, N> to an at::IntArrayRef object.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline auto to_ArrayRef(const std::array<T, N> &array) {
  return at::ArrayRef<T>{array};
}

/// @brief Concatenates multiple std::array objects
/// @{
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param arrays Value of `arrays`.
/// @return Result of the operation.
template <typename T, std::size_t... N>
inline auto concat(const std::array<T, N> &...arrays) {
  std::array<T, (N + ...)> result;
  std::size_t index{};

  ((std::copy_n(arrays.begin(), N, result.begin() + index), index += N), ...);

  return result;
}

/// @brief Provides the `concat` operation.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param arrays Value of `arrays`.
/// @return Result of the operation.
template <typename T, std::size_t... N>
inline auto concat(std::array<T, N> &&...arrays) {
  std::array<T, (N + ...)> result;
  std::size_t index{};

  ((std::copy_n(std::make_move_iterator(arrays.begin()), N,
                result.begin() + index),
    index += N),
   ...);

  return result;
}
/// @}

/// @brief Concatenates multiple std::vector objects
/// @{
/// @tparam Ts Template parameter `Ts`.
/// @param vectors Value of `vectors`.
/// @return Result of the operation.
template <typename... Ts>
inline auto concat(const std::vector<Ts> &...vectors) {
  std::vector<std::common_type_t<Ts...>> result;

  (result.insert(result.end(), vectors.begin(), vectors.end()), ...);

  return result;
}

/// @brief Provides the `concat` operation.
/// @tparam Ts Template parameter `Ts`.
/// @param vectors Value of `vectors`.
/// @return Result of the operation.
template <typename... Ts> inline auto concat(std::vector<Ts> &&...vectors) {
  std::vector<std::common_type_t<Ts...>> result;

  (result.insert(result.end(), std::make_move_iterator(vectors.begin()),
                 std::make_move_iterator(vectors.end())),
   ...);

  return result;
}
/// @}

/// @brief Appends data to a torch::ArrayRef object.
/// @tparam T Template parameter `T`.
/// @param array Value of `array`.
/// @param data Value of `data`.
/// @return Result of the operation.
template <typename T>
inline constexpr auto operator+(torch::ArrayRef<T> array, T data) {
  std::vector<T> result{array.vec()};
  result.push_back(data);
  return result;
}

/// @brief Appends data to a std::array object.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param array Value of `array`.
/// @param data Value of `data`.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline constexpr auto operator+(std::array<T, N> array, T data) {
  std::array<T, N + 1> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = array[i];
  result[N] = data;
  return result;
}

/// @brief Appends data to a std::vector object.
/// @tparam T Template parameter `T`.
/// @param vector Value of `vector`.
/// @param data Value of `data`.
/// @return Result of the operation.
template <typename T>
inline constexpr auto operator+(std::vector<T> vector, T data) {
  std::vector<T> result{vector};
  result.push_back(data);
  return result;
}

/// @brief Prepends data to a torch::ArrayRef object.
/// @tparam T Template parameter `T`.
/// @param data Value of `data`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T>
inline constexpr auto operator+(T data, torch::ArrayRef<T> array) {
  std::vector<T> result{array.vec()};
  result.insert(result.begin(), data);
  return result;
}

/// @brief Prepends data to a std::array object.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param data Value of `data`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline constexpr auto operator+(T data, std::array<T, N> array) {
  std::array<T, N + 1> result;
  result[0] = data;
  for (std::size_t i = 0; i < N; ++i)
    result[i + 1] = array[i];
  return result;
}

/// @brief Prepends data to a std::vector object.
/// @tparam T Template parameter `T`.
/// @param data Value of `data`.
/// @param vector Value of `vector`.
/// @return Result of the operation.
template <typename T>
inline constexpr auto operator+(T data, std::vector<T> vector) {
  std::vector<T> result{vector};
  result.insert(result.begin(), data);
  return result;
}

/// @brief Creates a std::array object filled with a constant.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param value Value to process.
/// @return Result of the operation.
template <typename T, std::size_t N> inline constexpr auto make_array(T value) {
  std::array<T, N> result;
  result.fill(value);
  return result;
}

/// @brief Creates a std::array object from another std::array object.
/// @tparam T Template parameter `T`.
/// @tparam U Template parameter `U`.
/// @tparam N Template parameter `N`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T, typename U, std::size_t N>
inline constexpr std::array<T, N> make_array(std::array<U, N> array) {
  std::array<T, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = static_cast<T>(array[i]);
  return result;
}

/// @brief Negates all entries of a std::array.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline constexpr std::array<T, N> operator-(std::array<T, N> array) {
  std::array<T, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = -array[i];
  return result;
}

/// @brief Adds two std::arrays.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param lhs Left-hand operand.
/// @param rhs Right-hand operand.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline constexpr std::array<T, N> operator+(std::array<T, N> lhs,
                                            std::array<T, N> rhs) {
  std::array<T, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = lhs[i] + rhs[i];
  return result;
}

/// @brief Subtracts one std::array from another std::array.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param lhs Left-hand operand.
/// @param rhs Right-hand operand.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline constexpr std::array<T, N> operator-(std::array<T, N> lhs,
                                            std::array<T, N> rhs) {
  std::array<T, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = lhs[i] - rhs[i];
  return result;
}

/// @brief Multiplies two std::arrays.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param lhs Left-hand operand.
/// @param rhs Right-hand operand.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline constexpr std::array<T, N> operator*(std::array<T, N> lhs,
                                            std::array<T, N> rhs) {
  std::array<T, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = lhs[i] * rhs[i];
  return result;
}

/// @brief Divides one std::array by another std::array.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @param lhs Left-hand operand.
/// @param rhs Right-hand operand.
/// @return Result of the operation.
template <typename T, std::size_t N>
inline constexpr std::array<T, N> operator/(std::array<T, N> lhs,
                                            std::array<T, N> rhs) {
  std::array<T, N> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = lhs[i] / rhs[i];
  return result;
}

/// @brief Derives a std::array object from a given std::array object dropping
/// the first M entries.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam M Template parameter `M`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T, std::size_t N, std::size_t M = 1>
inline constexpr std::array<T, N - M>
remove_from_front(std::array<T, N> array) {

  std::array<T, N - M> result;
  for (std::size_t i = 0; i < N - M; ++i)
    result[i] = array[i + M];
  return result;
}

/// @brief Derives a std::array object from a given std::array object dropping
/// the last M entries.
/// @tparam T Template parameter `T`.
/// @tparam N Template parameter `N`.
/// @tparam M Template parameter `M`.
/// @param array Value of `array`.
/// @return Result of the operation.
template <typename T, std::size_t N, std::size_t M = 1>
inline constexpr std::array<T, N - M> remove_from_back(std::array<T, N> array) {

  std::array<T, N - M> result;
  for (std::size_t i = 0; i < N - M; ++i)
    result[i] = array[i];
  return result;
}

} // namespace iganet::utils
