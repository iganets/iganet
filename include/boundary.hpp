/**
   @file include/boundary.hpp

   @brief Boundary treatment

   @author Matthias Moller

   @copyright This file is part of the IgANet project

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
  template<typename BSpline_t, short_t>
  class BoundaryCore;
  
  /// @brief BoundaryCore (1d specialization)
  ///
  /// This specialization has 2 sides
  /// - west (u=0)
  /// - east (u=1)
  template<typename BSpline_t>
  class BoundaryCore<BSpline_t, /* parDim */1>
    : public core<typename BSpline_t::value_type>
  {
  private:
    /// @brief Boundary B-spline type
    using BoundaryBSpline_t = typename BSpline_t::template
      derived_self_type_t<typename BSpline_t::value_type,
                          BSpline_t::geoDim()>;
    
    /// @brief Tuple of B-Splines
    std::tuple<BoundaryBSpline_t,
               BoundaryBSpline_t> bdr_;
  public:
    BoundaryCore() = default;

    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 1>&,
                 enum init = init::zeros)
      : core<typename BSpline_t::value_type>() ,
        bdr_(
             {
               BoundaryBSpline_t(std::array<int64_t, 0>{}),
               BoundaryBSpline_t(std::array<int64_t, 0>{}),
             }               
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename BSpline_t::value_type>, 1>&,
                 enum init = init::zeros)
      : core<typename BSpline_t::value_type>() ,
        bdr_(
             {
               BoundaryBSpline_t(std::array<int64_t, 0>{}),
               BoundaryBSpline_t(std::array<int64_t, 0>{}),
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
      os << core<typename BSpline_t::value_type>::name()
         << "(\n"
         << "left  = " << std::get<west-1>(bdr_) << "\n"
         << "right = " << std::get<east-1>(bdr_)
         << "\n)";
    }
  };
  
  /// @brief BoundaryCore (2d specialization)
  ///
  /// This specialization has 4 sides
  /// - west  (u=0, v  )
  /// - east  (u=1, v  )
  /// - south (u,   v=0)
  /// - north (u,   v=1)
  template<typename BSpline_t>
  class BoundaryCore<BSpline_t, /* parDim */2>
    : public core<typename BSpline_t::value_type>
  {
  private:
    /// @brief Boundary B-spline type
    using BoundaryBSpline_t = std::tuple<
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(1)>,
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(0)>>;
    
    /// @brief Tuple of B-Splines
    std::tuple<typename std::tuple_element_t<0,BoundaryBSpline_t>,
               typename std::tuple_element_t<0,BoundaryBSpline_t>,
               typename std::tuple_element_t<1,BoundaryBSpline_t>,
               typename std::tuple_element_t<1,BoundaryBSpline_t>> bdr_;
    
  public:
    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 2>& ncoeffs,
                 enum init init = init::zeros)
      : core<typename BSpline_t::value_type>(),
        bdr_(
             {
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<int64_t,1>({ncoeffs[1]}), init),
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<int64_t,1>({ncoeffs[1]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<int64_t,1>({ncoeffs[0]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<int64_t,1>({ncoeffs[0]}), init)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename BSpline_t::value_type>, 2>& kv,
                 enum init init = init::zeros)
      : core<typename BSpline_t::value_type>(),
        bdr_(
             {
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,1>({kv[1]}), init),
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,1>({kv[1]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,1>({kv[0]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,1>({kv[0]}), init)
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
      os << core<typename BSpline_t::value_type>::name()
         << "(\n"
         << "west = "  << std::get<west-1>(bdr_) << "\n"
         << "east = "  << std::get<east-1>(bdr_) << "\n"
         << "south = " << std::get<south-1>(bdr_) << "\n"
         << "north = " << std::get<north-1>(bdr_)
         << "\n)";
    }
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
  template<typename BSpline_t>
  class BoundaryCore<BSpline_t, /* parDim */3>
    : public core<typename BSpline_t::value_type>
  {
  private:
    /// @brief Boundary B-spline type
    using BoundaryBSpline_t = std::tuple<
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(1), BSpline_t::degree(2)>,
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(0), BSpline_t::degree(2)>,
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(0), BSpline_t::degree(1)>>;


    /// @brief Tuple of B-Splines
    std::tuple<typename std::tuple_element_t<0,BoundaryBSpline_t>,
               typename std::tuple_element_t<0,BoundaryBSpline_t>,
               typename std::tuple_element_t<1,BoundaryBSpline_t>,
               typename std::tuple_element_t<1,BoundaryBSpline_t>,
               typename std::tuple_element_t<2,BoundaryBSpline_t>,
               typename std::tuple_element_t<2,BoundaryBSpline_t>> bdr_;
    
  public:
    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 3>& ncoeffs,
                 enum init init = init::zeros)
      : core<typename BSpline_t::value_type>(),
        bdr_(
             {
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<int64_t,2>({ncoeffs[1], ncoeffs[2]}), init),
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<int64_t,2>({ncoeffs[1], ncoeffs[2]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[2]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[2]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[1]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[1]}), init)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename BSpline_t::value_type>, 3>& kv,
                 enum init init = init::zeros)
      : core<typename BSpline_t::value_type>(),
        bdr_(
             {
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,2>({kv[1], kv[2]}), init),
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,2>({kv[1], kv[2]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,2>({kv[0], kv[2]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,2>({kv[0], kv[2]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,2>({kv[0], kv[1]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,2>({kv[0], kv[1]}), init)
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
      os << core<typename BSpline_t::value_type>::name()
         << "(\n"
         << "west = "  << std::get<west-1>(bdr_) << "\n"
         << "east = "  << std::get<east-1>(bdr_) << "\n"
         << "south = " << std::get<south-1>(bdr_) << "\n"
         << "north = " << std::get<north-1>(bdr_) << "\n"
         << "front = " << std::get<front-1>(bdr_) << "\n"
         << "back = "  << std::get<back-1>(bdr_)
         << "\n)";
    }   
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
  template<typename BSpline_t>
  class BoundaryCore<BSpline_t, /* parDim */4>
    : public core<typename BSpline_t::value_type>
  {
  private:
    /// @brief Array storing the degrees
    using BoundaryBSpline_t = std::tuple<
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(1), BSpline_t::degree(2), BSpline_t::degree(3)>,
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(0), BSpline_t::degree(2), BSpline_t::degree(3)>,
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(0), BSpline_t::degree(1), BSpline_t::degree(3)>,
    typename BSpline_t::template
    derived_self_type_t<typename BSpline_t::value_type,
                        BSpline_t::geoDim(),
                        BSpline_t::degree(0), BSpline_t::degree(1), BSpline_t::degree(2)>>;
    
    
    /// @brief Tuple of B-Splines
    std::tuple<typename std::tuple_element_t<0,BoundaryBSpline_t>,
               typename std::tuple_element_t<0,BoundaryBSpline_t>,
               typename std::tuple_element_t<1,BoundaryBSpline_t>,
               typename std::tuple_element_t<1,BoundaryBSpline_t>,
               typename std::tuple_element_t<2,BoundaryBSpline_t>,
               typename std::tuple_element_t<2,BoundaryBSpline_t>,
               typename std::tuple_element_t<3,BoundaryBSpline_t>,
               typename std::tuple_element_t<3,BoundaryBSpline_t>> bdr_;
    
  public:
    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 4>& ncoeffs,
                 enum init init = init::zeros)
      : core<typename BSpline_t::value_type>(),
        bdr_(
             {
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}), init),
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}), init),
               std::tuple_element_t<3,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}), init),
               std::tuple_element_t<3,BoundaryBSpline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}), init)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename BSpline_t::value_type>, 4>& kv,
                 enum init init = init::zeros)
      : core<typename BSpline_t::value_type>(),
        bdr_(
             {
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[1], kv[2], kv[3]}), init),
               std::tuple_element_t<0,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[1], kv[2], kv[3]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[0], kv[2], kv[3]}), init),
               std::tuple_element_t<1,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[0], kv[2], kv[3]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[0], kv[1], kv[3]}), init),
               std::tuple_element_t<2,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[0], kv[1], kv[3]}), init),
               std::tuple_element_t<3,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[0], kv[1], kv[2]}), init),
               std::tuple_element_t<3,BoundaryBSpline_t>(std::array<std::vector<typename BSpline_t::value_type>,3>({kv[0], kv[1], kv[2]}), init)
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
      os << core<typename BSpline_t::value_type>::name()
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
  };
  
  /// @brief Boundary
  template<typename BSpline_t>
  class Boundary : public BoundaryCore<BSpline_t, BSpline_t::parDim()>
  {
  public:
    using BoundaryCore<BSpline_t, BSpline_t::parDim()>::BoundaryCore;
  };
  
  /// @brief Print (as string) a Boundary object
  template<typename BSpline_t>
  inline std::ostream& operator<<(std::ostream& os,
                                  const Boundary<BSpline_t>& obj)
  {
    obj.pretty_print(os);
    return os;
  }
  
} // namespace iganet
