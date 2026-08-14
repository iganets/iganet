/**
   @file core/dtype.hpp

   @brief DType traits.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <complex>
#include <cstddef>
#include <string_view>
#include <type_traits>

#include <iganet/core/core.hpp>

namespace iganet {

/// @brief Tag type representing IEEE 754 half-precision floating-point data.
struct half {};

/// @brief Removes top-level `const` and `volatile` qualifiers from a type.
/// @tparam T Type to normalize.
template <typename T> using normalized_type_t = std::remove_cv_t<T>;

/// @brief Type trait that maps C++ types to LibTorch dtypes.
/// @tparam T C++ type to map; unsupported primary-template instantiations are
/// intentionally undefined.
template <typename T> struct dtype_traits; // Unsupported by default.

/// @brief Defines a `dtype_traits` specialization.
/// @param type C++ type to specialize.
/// @param torch_dtype Corresponding LibTorch dtype constant.
#define DEFINE_DTYPE(type, torch_dtype)                                       \
  /** @brief LibTorch dtype mapping for `type`. */                            \
  template <> struct dtype_traits<type> {                                     \
    /** @brief LibTorch dtype associated with `type`. */                      \
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

/// @brief Concept to identify template parameters that are acceptable as DTypes.
/// @tparam T Type to test for a `dtype_traits` mapping.
template <typename T>
concept DType = requires { dtype_traits<normalized_type_t<T>>::value; };

/// @brief Determines the LibTorch dtype from a template parameter.
///
/// @tparam T C++ type.
///
/// @result Torch type corresponding to the C++ type.
/// @{
template <DType T>
inline constexpr torch::Dtype dtype_v =
    dtype_traits<normalized_type_t<T>>::value;
/// @}

/// @brief Type trait to obtain the name of a fundamental type as
/// `std::string_view`.
/// @tparam T Type whose human-readable name is requested.
template <typename T>
struct type_name; // Intentionally undefined for unsupported types.

/// @brief Defines a `type_name` specialization.
/// @param type C++ type whose name is exposed.
#define DEFINE_TYPE_NAME(type)                             \
  /** @brief Human-readable type name for `type`. */       \
  template <>                                              \
  struct type_name<type> {                                 \
    /** @brief Name of `type` as a string view. */          \
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

/// @brief Human-readable type name for `std::complex<half>`.
template <>
struct type_name<std::complex<half>> {
  /// @brief Name of the complex half-precision type.
  static constexpr std::string_view value =
      "std::complex<half>";
};

/// @brief Human-readable type name for `std::complex<float>`.
template <>
struct type_name<std::complex<float>> {
  /// @brief Name of the complex single-precision type.
  static constexpr std::string_view value =
      "std::complex<float>";
};

/// @brief Human-readable type name for `std::complex<double>`.
template <>
struct type_name<std::complex<double>> {
  /// @brief Name of the complex double-precision type.
  static constexpr std::string_view value =
      "std::complex<double>";
};

/// @brief Human-readable type name for `std::complex<long double>`.
template <>
struct type_name<std::complex<long double>> {
  /// @brief Name of the complex extended-precision type.
  static constexpr std::string_view value =
      "std::complex<long double>";
};

/// @brief Human-readable type name for `std::nullptr_t`.
template <>
struct type_name<std::nullptr_t> {
  /// @brief Name of the null-pointer type.
  static constexpr std::string_view value = "std::nullptr_t";
};

/// @brief Human-readable name associated with a supported C++ type.
/// @tparam T Type whose name is requested.
/// @result Name supplied by the corresponding `type_name` specialization.
/// @{
template <typename T>
inline constexpr std::string_view type_name_v =
    type_name<normalized_type_t<T>>::value;
/// @}
  
} // namespace iganet
