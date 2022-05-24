/**
   @file include/boundary.hpp

   @brief Boundary treatment

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <bspline.hpp>

#pragma once

namespace iganet {

  /**
   * Boundary
   */
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t... Degrees>
  class Boundary : public core<real_t>
  {
  private:
    inline static constexpr auto make_bdr()
    {
      constexpr const std::array<short_t, sizeof...(Degrees)> degrees_ = { Degrees... };
      
      if constexpr (sizeof...(Degrees) == 1) {
        std::tuple<bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>> bdr;
        return bdr;
      }

      else if constexpr (sizeof...(Degrees) == 2) {
        std::tuple<bspline_t<real_t, GeoDim, std::get<1>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<1>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>> bdr;
        return bdr;
      }

      else if constexpr (sizeof...(Degrees) == 3) {
        std::tuple<bspline_t<real_t, GeoDim, std::get<1>(degrees_), std::get<2>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<1>(degrees_), std::get<2>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<2>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<2>(degrees_)>> bdr;
        return bdr;
      }

      else if constexpr (sizeof...(Degrees) == 4) {
        std::tuple<bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
                   bspline_t<real_t, GeoDim, std::get<0>(degrees_)>> bdr;
        return bdr;
      }

      else {
        throw std::runtime_error("Unsupported dimension");
      }
    }
    
    /// Container holding the boundary conditions
    decltype(make_bdr()) bdr_;
    
  public:
    /// Identifiers for topological sides
    enum side { west  = 1, east  = 2, south = 3, north = 4, front = 5, back = 6,
                stime = 7, etime = 8,
                left  = 1, right = 2, down  = 3, up    = 4, none  = 0 };
    
    // Determines the number of boundaries
    inline constexpr short_t sides()
    {
      if constexpr (sizeof...(Degrees) == 1) {
        return side::east;
      }

      else if constexpr (sizeof...(Degrees) == 2) {
        return side::north;
      }

      else if constexpr (sizeof...(Degrees) == 3) {
        return side::back;
      }

      else if constexpr (sizeof...(Degrees) == 4) {
        return side::etime;
      }

      else {
        throw std::runtime_error("Unsupported dimension");
      }      
    }

    /// Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      if constexpr (sizeof...(Degrees) == 1) {
        os << core<real_t>::name()
           << "(\n"
           << "left = " << 1 << "\n"
           << "right = " << 1
           << "\n)";
      }

      else if constexpr (sizeof...(Degrees) == 2) {
        os << core<real_t>::name()
           << "(\n"
           << "west = " << std::get<west-1>(bdr_) << "\n"
           << "east = " << std::get<east-1>(bdr_) << "\n"
           << "south = " << std::get<south-1>(bdr_) << "\n"
           << "north = " << std::get<north-1>(bdr_)
           << "\n)";
      }

      else if constexpr (sizeof...(Degrees) == 3) {
        os << core<real_t>::name()
           << "(\n"
           << "west = " << std::get<west-1>(bdr_) << "\n"
           << "east = " << std::get<east-1>(bdr_) << "\n"
           << "south = " << std::get<south-1>(bdr_) << "\n"
           << "north = " << std::get<north-1>(bdr_) << "\n"
           << "front = " << std::get<front-1>(bdr_) << "\n"
           << "back = " << std::get<back-1>(bdr_)
           << "\n)";
      }

      else if constexpr (sizeof...(Degrees) == 4) {
        os << core<real_t>::name()
           << "(\n"
           << "west = " << std::get<west-1>(bdr_) << "\n"
           << "east = " << std::get<east-1>(bdr_) << "\n"
           << "south = " << std::get<south-1>(bdr_) << "\n"
           << "north = " << std::get<north-1>(bdr_) << "\n"
           << "front = " << std::get<front-1>(bdr_) << "\n"
           << "back = " << std::get<back-1>(bdr_) << "\n"
           << "stime = " << std::get<stime-1>(bdr_) << "\n"
           << "etime = " << std::get<etime-1>(bdr_)
           << "\n)";
      }

      else {
        throw std::runtime_error("Unsupported dimension");
      }      
    }
    
  };

  /// Print (as string) a Boundary object
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const Boundary<bspline_t, real_t, GeoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }
  
} // namespace iganet
