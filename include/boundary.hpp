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

  /// @brief Identifiers for topological sides
  enum side { west  = 1, east  = 2, south = 3, north = 4, front = 5, back = 6,
              stime = 7, etime = 8,
              left  = 1, right = 2, down  = 3, up    = 4, none  = 0 };

  /// @brief BoundaryCore
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t ParDim, short_t... Degrees>
  class BoundaryCore;
  
  /// @brief BoundaryCore (1d specialization)
  ///
  /// This specialization has 2 sides
  /// - west (u=0)
  /// - east (u=1)
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t... Degrees>
  class BoundaryCore<bspline_t, real_t, GeoDim, 1, Degrees...>
    : public core<real_t>
  {
  public:
    /// @brief Default constructor
    BoundaryCore()
      : core<real_t>()
    {}
    
    /// @brief Constructor
    template<typename T>
    BoundaryCore(const std::array<T, 1>& ncoeffs,
                 BSplineInit init = BSplineInit::zeros)
      : core<real_t>() ,
        bdr_(
             {
               bspline_t<real_t, GeoDim>({}, init),
               bspline_t<real_t, GeoDim>({}, init),
             }               
             )
    {}
    
    /// @brief Returns the number of sides
    inline static constexpr short_t sides()
    {
      return side::east;
    }

    /// @brief Returns constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side()
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns a constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs() const
    {
      return bdr_;
    }
    
    /// @brief Returns a non-constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs()
    {
      return bdr_;
    }

    /// @brief Returns the total number of coefficients
    inline int64_t ncoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncoeffs();
      s += std::get<east-1>(bdr_).ncoeffs();    
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<real_t>::name()
         << "(\n"
         << "left  = " << std::get<west-1>(bdr_) << "\n"
         << "right = " << std::get<east-1>(bdr_)
         << "\n)";
    }
    
  private:
    /// @brief Array storing the degrees
    static constexpr const std::array<short_t, sizeof...(Degrees)> degrees_ = { Degrees... };

    /// @brief Tuple of B-Splines
    std::tuple<bspline_t<real_t, GeoDim>,
               bspline_t<real_t, GeoDim>> bdr_;
  };

  /// @brief BoundaryCore (2d specialization)
  ///
  /// This specialization has 4 sides
  /// - west  (u=0, v  )
  /// - east  (u=1, v  )
  /// - south (u,   v=0)
  /// - north (u,   v=1)
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t... Degrees>
  class BoundaryCore<bspline_t, real_t, GeoDim, 2, Degrees...>
    : public core<real_t>
  {
  public:
    /// @brief Constructor
    template<typename T>
    BoundaryCore(const std::array<T, 2>& ncoeffs,
                 BSplineInit init = BSplineInit::zeros)
      : core<real_t>(),
        bdr_(
             {
               bspline_t<real_t, GeoDim, std::get<1>(degrees_)>(std::array<int64_t,1>({ncoeffs[1]}), init),
               bspline_t<real_t, GeoDim, std::get<1>(degrees_)>(std::array<int64_t,1>({ncoeffs[1]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_)>(std::array<int64_t,1>({ncoeffs[0]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_)>(std::array<int64_t,1>({ncoeffs[0]}), init)
             }
             )
    {}
    
    /// @brief Returns the number of sides
    inline static constexpr short_t sides()
    {
      return side::north;
    }

    /// @brief Returns constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side()
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns a constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs() const
    {
      return bdr_;
    }
    
    /// @brief Returns a non-constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs()
    {
      return bdr_;
    }

    /// @brief Returns the total number of coefficients
    inline int64_t ncoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncoeffs();
      s += std::get<east-1>(bdr_).ncoeffs();
      s += std::get<south-1>(bdr_).ncoeffs();
      s += std::get<north-1>(bdr_).ncoeffs();    
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<real_t>::name()
         << "(\n"
         << "west = "  << std::get<west-1>(bdr_) << "\n"
         << "east = "  << std::get<east-1>(bdr_) << "\n"
         << "south = " << std::get<south-1>(bdr_) << "\n"
         << "north = " << std::get<north-1>(bdr_)
         << "\n)";
    }
    
  private:
    /// @brief Array storing the degrees
    static constexpr const std::array<short_t, sizeof...(Degrees)> degrees_ = { Degrees... };
    
    /// @brief Tuple of B-Splines
    std::tuple<bspline_t<real_t, GeoDim, std::get<1>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<1>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_)>> bdr_;
  };

  /// @brief BoundaryCore (3d specialization)
  ///
  /// This specialization has 6 sides
  /// - west  (u=0, v,   w)
  /// - east  (u=1, v,   w)
  /// - south (u,   v=0, w)
  /// - north (u,   v=1, w)
  /// - front (u,   v,   w=0)
  /// - back  (u,   v,   w=1)
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t... Degrees>
  class BoundaryCore<bspline_t, real_t, GeoDim, 3, Degrees...>
    : public core<real_t>
  {
  public:
    /// @brief Constructor
    template<typename T>
    BoundaryCore(const std::array<T, 3>& ncoeffs,
                 BSplineInit init = BSplineInit::zeros)
      : core<real_t>(),
        bdr_(
             {
               bspline_t<real_t, GeoDim, std::get<1>(degrees_),std::get<2>(degrees_)>(std::array<int64_t,2>({ncoeffs[1],ncoeffs[2]}), init),
               bspline_t<real_t, GeoDim, std::get<1>(degrees_),std::get<2>(degrees_)>(std::array<int64_t,2>({ncoeffs[1],ncoeffs[2]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<2>(degrees_)>(std::array<int64_t,2>({ncoeffs[0],ncoeffs[2]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<2>(degrees_)>(std::array<int64_t,2>({ncoeffs[0],ncoeffs[2]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<1>(degrees_)>(std::array<int64_t,2>({ncoeffs[0],ncoeffs[1]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<1>(degrees_)>(std::array<int64_t,2>({ncoeffs[0],ncoeffs[1]}), init)
             }
             )
    {}

    /// @brief Returns constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side()
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }
    
    /// @brief Returns the number of sides
    inline static constexpr short_t sides()
    {
      return side::back;
    }

    /// @brief Returns a constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs() const
    {
      return bdr_;
    }
    
    /// @brief Returns a non-constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs()
    {
      return bdr_;
    }

    /// @brief Returns the total number of coefficients
    inline int64_t ncoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncoeffs();
      s += std::get<east-1>(bdr_).ncoeffs();
      s += std::get<south-1>(bdr_).ncoeffs();
      s += std::get<north-1>(bdr_).ncoeffs();
      s += std::get<front-1>(bdr_).ncoeffs();
      s += std::get<back-1>(bdr_).ncoeffs();
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<real_t>::name()
         << "(\n"
         << "west = "  << std::get<west-1>(bdr_) << "\n"
         << "east = "  << std::get<east-1>(bdr_) << "\n"
         << "south = " << std::get<south-1>(bdr_) << "\n"
         << "north = " << std::get<north-1>(bdr_) << "\n"
         << "front = " << std::get<front-1>(bdr_) << "\n"
         << "back = "  << std::get<back-1>(bdr_)
         << "\n)";
    }
    
  private:
    /// @brief Array storing the degrees
    static constexpr const std::array<short_t, sizeof...(Degrees)> degrees_ = { Degrees... };

    /// @brief Tuple of B-Splines
    std::tuple<bspline_t<real_t, GeoDim, std::get<1>(degrees_), std::get<2>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<1>(degrees_), std::get<2>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<2>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<2>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_)>> bdr_;
  };

  /// @brief BoundaryCore (4d specialization)
  ///   
  /// This specialization has 8 sides
  /// - west  (u=0, v,   w,   t)
  /// - east  (u=1, v,   w,   t)
  /// - south (u,   v=0, w,   t)
  /// - north (u,   v=1, w,   t)
  /// - front (u,   v,   w=0, t)
  /// - back  (u,   v,   w=1, t)
  /// - stime (u,   v,   w,   t=0)
  /// - etime (u,   v,   w,   t=1)
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t... Degrees>
  class BoundaryCore<bspline_t, real_t, GeoDim, 4, Degrees...>
    : public core<real_t>
  {
  public:
    /// @brief Constructor
    template<typename T>
    BoundaryCore(const std::array<T, 4>& ncoeffs,
                 BSplineInit init = BSplineInit::zeros)
      : core<real_t>(),
        bdr_(
             {
               bspline_t<real_t, GeoDim, std::get<1>(degrees_),std::get<2>(degrees_),std::get<3>(degrees_)>(std::array<int64_t,3>({ncoeffs[1],ncoeffs[2],ncoeffs[3]}), init),
               bspline_t<real_t, GeoDim, std::get<1>(degrees_),std::get<2>(degrees_),std::get<3>(degrees_)>(std::array<int64_t,3>({ncoeffs[1],ncoeffs[2],ncoeffs[3]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<2>(degrees_),std::get<3>(degrees_)>(std::array<int64_t,3>({ncoeffs[0],ncoeffs[2],ncoeffs[3]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<2>(degrees_),std::get<3>(degrees_)>(std::array<int64_t,3>({ncoeffs[0],ncoeffs[2],ncoeffs[3]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<1>(degrees_),std::get<3>(degrees_)>(std::array<int64_t,3>({ncoeffs[0],ncoeffs[1],ncoeffs[3]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<1>(degrees_),std::get<3>(degrees_)>(std::array<int64_t,3>({ncoeffs[0],ncoeffs[1],ncoeffs[3]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<1>(degrees_),std::get<2>(degrees_)>(std::array<int64_t,3>({ncoeffs[0],ncoeffs[1],ncoeffs[2]}), init),
               bspline_t<real_t, GeoDim, std::get<0>(degrees_),std::get<1>(degrees_),std::get<2>(degrees_)>(std::array<int64_t,3>({ncoeffs[0],ncoeffs[1],ncoeffs[2]}), init)
             }
             )
    {}
    
    /// @brief Returns the number of sides
    inline static constexpr short_t sides()
    {
      return side::etime;
    }

    /// @brief Returns constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th B-Spline
    template<short_t s>
    inline constexpr auto& side()
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns a constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs() const
    {
      return bdr_;
    }
    
    /// @brief Returns a non-constant reference to the array of
    /// coefficients for all boundary segments.
    inline constexpr auto& coeffs()
    {
      return bdr_;
    }

    /// @brief Returns the total number of coefficients
    inline int64_t ncoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncoeffs();
      s += std::get<east-1>(bdr_).ncoeffs();
      s += std::get<south-1>(bdr_).ncoeffs();
      s += std::get<north-1>(bdr_).ncoeffs();
      s += std::get<front-1>(bdr_).ncoeffs();
      s += std::get<back-1>(bdr_).ncoeffs();
      s += std::get<stime-1>(bdr_).ncoeffs();
      s += std::get<etime-1>(bdr_).ncoeffs();
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<real_t>::name()
         << "(\n"
         << "west = "  << std::get<west-1>(bdr_) << "\n"
         << "east = "  << std::get<east-1>(bdr_) << "\n"
         << "south = " << std::get<south-1>(bdr_) << "\n"
         << "north = " << std::get<north-1>(bdr_) << "\n"
         << "front = " << std::get<front-1>(bdr_) << "\n"
         << "back = "  << std::get<back-1>(bdr_) << "\n"
         << "stime = " << std::get<stime-1>(bdr_) << "\n"
         << "etime = " << std::get<etime-1>(bdr_)
         << "\n)";
    }
    
  private:
    /// @brief Array storing the degrees
    static constexpr const std::array<short_t, sizeof...(Degrees)> degrees_ = { Degrees... };
    
    /// @brief Tuple of B-Splines
    std::tuple<bspline_t<real_t, GeoDim, std::get<1>(degrees_), std::get<2>(degrees_), std::get<3>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<1>(degrees_), std::get<2>(degrees_), std::get<3>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<2>(degrees_), std::get<3>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<2>(degrees_), std::get<3>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_), std::get<3>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_), std::get<3>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_), std::get<2>(degrees_)>,
               bspline_t<real_t, GeoDim, std::get<0>(degrees_), std::get<1>(degrees_), std::get<2>(degrees_)>> bdr_;
  };
  
  /// @brief Boundary
  template<template<typename, short_t, short_t...> class bspline_t,
           typename real_t, short_t GeoDim, short_t... Degrees>
  class Boundary : public BoundaryCore<bspline_t, real_t, GeoDim, sizeof...(Degrees), Degrees...>
  {
  public:
    using BoundaryCore<bspline_t, real_t, GeoDim, sizeof...(Degrees), Degrees...>::BoundaryCore;
  };
  
  /// @brief Print (as string) a Boundary object
template<template<typename, short_t, short_t...> class bspline_t,
         typename real_t, short_t GeoDim, short_t... Degrees>
inline std::ostream& operator<<(std::ostream& os,
                                const Boundary<bspline_t, real_t, GeoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }
  
} // namespace iganet
