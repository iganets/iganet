/**
   @file include/utils/concat.hpp

   @brief Concatination utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <vector>

namespace iganet {
  namespace utils{
  
    /// @brief Concatenates multiple std::array objects
    /// @{
    template<typename T, std::size_t... N>
    inline auto concat(const std::array<T, N>&... arrays)
    {
      std::array<T, (N + ...)> result;
      std::size_t index{};

      ((std::copy_n(arrays.begin(), N, result.begin() + index), index += N), ...);

      return result;
    }

    template<typename T, std::size_t... N>
    inline auto concat(std::array<T, N>&&... arrays)
    {
      std::array<T, (N + ...)> result;
      std::size_t index{};

      ((std::copy_n(std::make_move_iterator(arrays.begin()), N, result.begin() + index), index += N), ...);

      return result;
    }
    /// @}

    /// @brief Concatenates multiple std::vector objects
    /// @{
    template<typename... Ts>
    inline auto concat(const std::vector<Ts>&... vectors)
    {
      std::vector<typename std::tuple_element<0, std::tuple<Ts...> >::type> result;

      (result.insert(result.end(), vectors.begin(), vectors.end()), ...);

      return result;
    }

    template<typename... Ts>
    inline auto concat(std::vector<Ts>&&... vectors)
    {
      std::vector<typename std::tuple_element<0, std::tuple<Ts...> >::type> result;

      (result.insert(result.end(), std::make_move_iterator(vectors.begin()),
                     std::make_move_iterator(vectors.end())), ...);

      return result;
    }
    /// @}

  } // namespace utils
} // namespace iganet
