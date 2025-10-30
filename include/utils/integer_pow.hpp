/**
   @file include/utils/integer_pow.hpp

   @brief Integer power utility function

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

namespace iganet::utils {

/// @brief Computes the power of integer `E` to the `N` at compile time
/// @{
template <int E, int N> struct integer_pow {
  enum { value = E * integer_pow<E, N - 1>::value };
};

template <int E> struct integer_pow<E, 0> {
  enum { value = 1 };
};
/// @}

} // namespace iganet::utils
