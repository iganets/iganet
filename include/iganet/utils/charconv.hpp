/**
   @file utils/charconv.hpp

   @brief Locale independent number formatting and parsing utility functions.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <version>

// The floating-point overloads of std::to_chars and std::from_chars are
// available from libstdc++ 11, libc++ 20 (std::to_chars from libc++ 14), and
// MSVC 19.24. Toolchains that do not define __cpp_lib_to_chars use a stream
// that is imbued with the classic locale; define
// IGANET_UTILS_STREAM_NUMBERS to select that path on any toolchain.
#if defined(__cpp_lib_to_chars) && !defined(IGANET_UTILS_STREAM_NUMBERS)
#define IGANET_UTILS_FLOATING_CHARCONV 1
#endif

namespace iganet::utils {

namespace detail {

/// @brief Returns true if `c` is an ASCII whitespace character that separates
/// values (space, tab, newline, carriage return).
/// @param c Character to test.
/// @return True if `c` separates values.
inline constexpr bool is_number_separator(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/// @brief Parses the complete token `[first, last)` as a number.
///
/// A single leading `+` is accepted.
///
/// @tparam T Arithmetic type.
/// @param first Start of the token.
/// @param last End of the token.
/// @param value Parsed value.
/// @return Null on success, otherwise a description of the failure.
template <typename T>
inline const char *parse_token(const char *first, const char *last,
                               T &value) noexcept {
  if (*first == '+' && first + 1 != last && first[1] != '-' && first[1] != '+')
    ++first;

#ifndef IGANET_UTILS_FLOATING_CHARCONV
  if constexpr (std::is_floating_point_v<T>) {
    // Streams do not read "inf" and "nan", which std::from_chars accepts.
    const bool negative = *first == '-';
    std::string token(first + negative, last);
    for (char &c : token)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (token == "inf" || token == "infinity") {
      value = negative ? -std::numeric_limits<T>::infinity()
                       : std::numeric_limits<T>::infinity();
      return nullptr;
    }
    if (token == "nan") {
      value = negative ? -std::numeric_limits<T>::quiet_NaN()
                       : std::numeric_limits<T>::quiet_NaN();
      return nullptr;
    }

    std::istringstream is(std::string(first, last));
    is.imbue(std::locale::classic());
    is >> value;
    return (!is.fail() && is.get() == std::char_traits<char>::eof())
               ? nullptr
               : "not a number";
  } else
#endif
  {
    auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec == std::errc::result_out_of_range)
      return "out of range";
    if (ec != std::errc{} || ptr != last)
      return "not a number";
    return nullptr;
  }
}

} // namespace detail

/// @brief Appends the text form of a number to a string.
///
/// Floating-point values are written in the shortest form that reads back to
/// the same value, integers as integers. The text does not depend on the
/// global C or C++ locale.
///
/// @tparam T Arithmetic type (not `bool`).
/// @param out String to append to.
/// @param value Value to write.
template <typename T> inline void append_number(std::string &out, T value) {
  static_assert(std::is_arithmetic_v<T> &&
                    !std::is_same_v<std::remove_cv_t<T>, bool>,
                "append_number requires an arithmetic type other than bool");

#ifndef IGANET_UTILS_FLOATING_CHARCONV
  if constexpr (std::is_floating_point_v<T>) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os.precision(std::numeric_limits<T>::max_digits10);
    os << value;
    out += os.str();
  } else
#endif
  {
    char buffer[64];
    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (ec != std::errc{})
      throw std::runtime_error("Number cannot be converted to text");
    out.append(buffer, ptr);
  }
}

/// @brief Parses a list of numbers separated by ASCII whitespace.
///
/// Values are separated by one or more spaces, tabs, newlines, or carriage
/// returns; leading and trailing whitespace is ignored. A single leading `+`
/// is accepted. Anything else that is not a complete number (trailing
/// characters, values out of range, empty exponent, ...) is an error.
///
/// @tparam T Type of the values (not `bool`).
/// @tparam OutputIt Output iterator that accepts values of type `T`.
/// @param text Text to parse.
/// @param out Output iterator.
/// @return Number of values parsed.
/// @throws std::runtime_error with the offset and an excerpt of the offending
/// token.
template <typename T, typename OutputIt>
inline std::size_t parse_numbers(std::string_view text, OutputIt out) {
  static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>,
                "parse_numbers requires an arithmetic type other than bool");

  const char *const begin = text.data();
  const char *const end = begin + text.size();
  const char *p = begin;
  std::size_t count = 0;

  while (true) {
    while (p != end && detail::is_number_separator(*p))
      ++p;
    if (p == end)
      return count;

    const char *q = p;
    while (q != end && !detail::is_number_separator(*q))
      ++q;

    T value;
    if (const char *reason = detail::parse_token<T>(p, q, value)) {
      constexpr std::ptrdiff_t max_excerpt = 24;
      const std::ptrdiff_t length = q - p;
      std::string excerpt(p, std::min(length, max_excerpt));
      for (char &c : excerpt)
        if (c < 0x20 || c > 0x7e)
          c = '?';
      throw std::runtime_error(
          std::string("XML value ") + std::to_string(count) + " at offset " +
          std::to_string(p - begin) + " is " + reason + ": \"" +
          excerpt + (length > max_excerpt ? "..." : "") + "\"");
    }

    *out++ = value;
    ++count;
    p = q;
  }
}

} // namespace iganet::utils
