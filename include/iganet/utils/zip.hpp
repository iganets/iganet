/**
   @file utils/zip.hpp

   @brief Zip utility function.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace iganet::utils {

namespace detail {

template <typename... T> class zip_helper {
public:
  class iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::tuple<decltype(*std::declval<T>().begin())...>;
    using difference_type = std::tuple<decltype(*std::declval<T>().begin())...>;
    using pointer = std::tuple<decltype(*std::declval<T>().begin())...> *;
    using reference = std::tuple<decltype(*std::declval<T>().begin())...> &;

  private:
    std::tuple<decltype(std::declval<T>().begin())...> _iterators;

    template <std::size_t... I> auto deref(std::index_sequence<I...>) const {
      return iterator::value_type{*std::get<I>(_iterators)...};
    }

    template <std::size_t... I> void increment(std::index_sequence<I...>) {
      auto l = {(++std::get<I>(_iterators), 0)...};
    }

  public:
    /// @brief Provides the `iterator` operation.
    /// @tparam I Template parameter `I`.
    /// @param iterators Value of `iterators`.
    explicit iterator(decltype(_iterators) iterators)
        : _iterators{std::move(iterators)} {}

    /// @brief Provides the `operator++` operation.
    /// @return Result of the operation.
    iterator &operator++() {
      increment(std::index_sequence_for<T...>{});
      return *this;
    }

    /// @brief Provides the `operator++` operation.
    /// @return Result of the operation.
    iterator operator++(int) {
      auto saved{*this};
      increment(std::index_sequence_for<T...>{});
      return saved;
    }

    /// @brief Provides the `operator!=` operation.
    /// @param other Second input value.
    /// @return Result of the operation.
    bool operator!=(const iterator &other) const {
      return _iterators != other._iterators;
    }

    /// @brief Provides the `operator*` operation.
    /// @return Result of the operation.
    auto operator*() const { return deref(std::index_sequence_for<T...>{}); }
  };

  /// @brief Provides the `zip_helper` operation.
  /// @param seqs Value of `seqs`.
  zip_helper(T &&...seqs)
      : _seqs(seqs...),
        _begin{make_tuple_begin(_seqs,
                                std::make_index_sequence<sizeof...(seqs)>{})},
        _end{make_tuple_end(_seqs,
                            std::make_index_sequence<sizeof...(seqs)>{})} {
    validate_lengths(std::make_index_sequence<sizeof...(seqs)>{});
  }

  /// @brief Provides the `begin` operation.
  /// @return Result of the operation.
  iterator begin() const { return _begin; }
  /// @brief Provides the `end` operation.
  /// @return Result of the operation.
  iterator end() const { return _end; }

private:
  template <std::size_t... Is>
  void validate_lengths(std::index_sequence<Is...>) const {
    if constexpr (sizeof...(Is) > 1) {
      const std::array lengths{
          std::distance(std::get<Is>(_seqs).begin(),
                        std::get<Is>(_seqs).end())...};
      for (std::size_t i = 1; i < lengths.size(); ++i)
        if (lengths[i] != lengths[0])
          throw std::invalid_argument("zip requires equally sized ranges");
    }
  }

  template <typename Tuple, std::size_t... Is>
  auto constexpr make_tuple_begin(Tuple &&t, std::index_sequence<Is...>) {
    return std::make_tuple(std::get<Is>(t).begin()...);
  }

  template <typename Tuple, std::size_t... Is>
  auto constexpr make_tuple_end(Tuple &&t, std::index_sequence<Is...>) {
    return std::make_tuple(std::get<Is>(t).end()...);
  }

private:
  std::tuple<T...> _seqs;
  iterator _begin;
  iterator _end;
};

} // namespace detail

/// @brief Provides the `zip` operation.
/// @param seqs Value of `seqs`.
/// @return Result of the operation.
// Sequences must be the same length.
template <typename... T> auto zip(T &&...seqs) {
  return iganet::utils::detail::zip_helper<T...>(std::forward<T>(seqs)...);
}

} // namespace iganet::utils
