/**
   @file core/dtype.hpp

   @brief DType traits

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <complex>
#include <cstddef>
#include <string_view>
#include <type_traits>

#include <core/core.hpp>

namespace iganet {

struct half {};

template <typename T> using normalized_type_t = std::remove_cv_t<T>;

/// @brief Type trait that maps C++ types to LibTorch dtypes
template <typename T> struct dtype_traits; // Unsupported by default.

#define DEFINE_DTYPE(type, torch_dtype)                                       \
  template <> struct dtype_traits<type> {                                     \
    static constexpr torch::Dtype value = torch_dtype;                        \
  }

DEFINE_DTYPE(bool, torch::kBool);
DEFINE_DTYPE(char, torch::kChar);
DEFINE_DTYPE(short, torch::kShort);
DEFINE_DTYPE(int, torch::kInt);
DEFINE_DTYPE(long, torch::kLong);
DEFINE_DTYPE(long long, torch::kLong);
DEFINE_DTYPE(half, torch::kHalf);
DEFINE_DTYPE(float, torch::kFloat);
DEFINE_DTYPE(double, torch::kDouble);
DEFINE_DTYPE(std::complex<half>, at::kComplexHalf);
DEFINE_DTYPE(std::complex<float>, at::kComplexFloat);
DEFINE_DTYPE(std::complex<double>, at::kComplexDouble);

#undef DEFINE_DTYPE

/// @brief Concept to identify template parameters that are acceptable as DTypes
template <typename T>
concept DType = requires { dtype_traits<normalized_type_t<T>>::value; };

/// Determines the LibTorch dtype from template parameter
///
/// @tparam T C++ type
///
/// @result Torch type corresponding to the C++ type
/// @{
template <DType T>
inline constexpr torch::Dtype dtype_v =
    dtype_traits<normalized_type_t<T>>::value;
/// @}

/// @brief Type trait to obtain the name of a fundamental type as std::string_view  
template <typename T>
struct type_name; // Intentionally undefined for unsupported types.

#define DEFINE_TYPE_NAME(type)                 \
  template <>                                  \
  struct type_name<type> {         \
    static constexpr std::string_view value = #type; \
  }

DEFINE_TYPE_NAME(void);
DEFINE_TYPE_NAME(bool);

DEFINE_TYPE_NAME(char);
DEFINE_TYPE_NAME(signed char);
DEFINE_TYPE_NAME(unsigned char);
DEFINE_TYPE_NAME(wchar_t);
DEFINE_TYPE_NAME(char8_t);
DEFINE_TYPE_NAME(char16_t);
DEFINE_TYPE_NAME(char32_t);

DEFINE_TYPE_NAME(short);
DEFINE_TYPE_NAME(unsigned short);
DEFINE_TYPE_NAME(int);
DEFINE_TYPE_NAME(unsigned int);
DEFINE_TYPE_NAME(long);
DEFINE_TYPE_NAME(unsigned long);
DEFINE_TYPE_NAME(long long);
DEFINE_TYPE_NAME(unsigned long long);

DEFINE_TYPE_NAME(half);  
DEFINE_TYPE_NAME(float);
DEFINE_TYPE_NAME(double);
DEFINE_TYPE_NAME(long double);

#undef DEFINE_TYPE_NAME

template <>
struct type_name<std::complex<half>> {
  static constexpr std::string_view value =
      "std::complex<half>";
};

template <>
struct type_name<std::complex<float>> {
  static constexpr std::string_view value =
      "std::complex<float>";
};

template <>
struct type_name<std::complex<double>> {
  static constexpr std::string_view value =
      "std::complex<double>";
};

template <>
struct type_name<std::complex<long double>> {
  static constexpr std::string_view value =
      "std::complex<long double>";
};

template <>
struct type_name<std::nullptr_t> {
  static constexpr std::string_view value = "std::nullptr_t";
};

template <typename T>
inline constexpr std::string_view type_name_v =
    type_name<normalized_type_t<T>>::value;
/// @}
  
} // namespace iganet
