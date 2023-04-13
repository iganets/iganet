/**
   @file include/boundary.hpp

   @brief Boundary treatment

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <bspline.hpp>

namespace iganet {

  /// @brief Identifiers for topological sides
  enum side { west  = 1, east  = 2, south = 3, north = 4, front = 5, back = 6,
              stime = 7, etime = 8,
              left  = 1, right = 2, down  = 3, up    = 4, none  = 0 };

  /// @brief BoundaryCore
  template<typename spline_t, short_t>
  class BoundaryCore;
  
  /// @brief BoundaryCore (1d specialization)
  ///
  /// This specialization has 2 sides
  /// - west (u=0)
  /// - east (u=1)
  template<typename spline_t>
  class BoundaryCore<spline_t, /* parDim */1>
    : public core<typename spline_t::value_type>
  {
  protected:
    /// @brief Boundary spline type
    using boundaryspline_t = typename spline_t::template
      derived_self_type_t<typename spline_t::value_type,
                          spline_t::geoDim()>;
    
    /// @brief Tuple of splines
    std::tuple<boundaryspline_t,
               boundaryspline_t> bdr_;
  public:
    /// @brief Evaluation type
    using eval_t = std::tuple<torch::Tensor,
                              torch::Tensor>;
    
    /// @brief Default constructor
    BoundaryCore() = default;

    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 1>&,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               boundaryspline_t(std::array<int64_t, 0>{}, init, core),
               boundaryspline_t(std::array<int64_t, 0>{}, init, core),
             }               
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 1>&,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               boundaryspline_t(std::array<int64_t, 0>{}, init, core),
               boundaryspline_t(std::array<int64_t, 0>{}, init, core),
             }               
             )
    {}
    
    /// @brief Returns the number of sides
    inline static constexpr short_t sides()
    {
      return side::east;
    }

    /// @brief Returns constant reference to side-th Spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th Spline
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
    inline int64_t ncumcoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncumcoeffs();
      s += std::get<east-1>(bdr_).ncumcoeffs();    
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<typename spline_t::value_type>::name()
         << "(\n"
         << "west = " << std::get<west-1>(bdr_) << "\n"
         << "east = " << std::get<east-1>(bdr_)
         << "\n)";
    }

    /// @brief Returns the boundary object as JSON object
    inline nlohmann::json to_json() const override
    {
      nlohmann::json json;
      json["west"] = std::get<west-1>(bdr_).to_json();
      json["east"] = std::get<east-1>(bdr_).to_json();

      return json;
    }

    /// @brief Returns the Greville abscissae
    inline eval_t greville() const
    {
      return eval_t{std::get<west-1>(bdr_).greville(),
                    std::get<east-1>(bdr_).greville()};
    }
  };
  
  /// @brief BoundaryCore (2d specialization)
  ///
  /// This specialization has 4 sides
  /// - west  (u=0, v  )
  /// - east  (u=1, v  )
  /// - south (u,   v=0)
  /// - north (u,   v=1)
  template<typename spline_t>
  class BoundaryCore<spline_t, /* parDim */2>
    : public core<typename spline_t::value_type>
  {
  protected:
    /// @brief Boundary spline type
    using boundaryspline_t = std::tuple<
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(1)>,
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(0)>>;
    
    /// @brief Tuple of splines
    std::tuple<typename std::tuple_element_t<0,boundaryspline_t>,
               typename std::tuple_element_t<0,boundaryspline_t>,
               typename std::tuple_element_t<1,boundaryspline_t>,
               typename std::tuple_element_t<1,boundaryspline_t>> bdr_;
    
  public:
    /// @brief Evaluation type
    using eval_t = std::tuple<std::array<torch::Tensor, 1>,
                              std::array<torch::Tensor, 1>,
                              std::array<torch::Tensor, 1>,
                              std::array<torch::Tensor, 1>>;
    
    /// @brief Default constructor
    BoundaryCore() = default;
    
    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 2>& ncoeffs,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[1]}), init, core),
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[1]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[0]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[0]}), init, core)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 2>& kv,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[1]}), init, core),
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[1]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[0]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[0]}), init, core)
             }
             )
    {}
    
    /// @brief Returns the number of sides
    inline static constexpr short_t sides()
    {
      return side::north;
    }

    /// @brief Returns constant reference to side-th spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th spline
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
    inline int64_t ncumcoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncumcoeffs();
      s += std::get<east-1>(bdr_).ncumcoeffs();
      s += std::get<south-1>(bdr_).ncumcoeffs();
      s += std::get<north-1>(bdr_).ncumcoeffs();    
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<typename spline_t::value_type>::name()
         << "(\n"
         << "west = "  << std::get<west-1>(bdr_) << "\n"
         << "east = "  << std::get<east-1>(bdr_) << "\n"
         << "south = " << std::get<south-1>(bdr_) << "\n"
         << "north = " << std::get<north-1>(bdr_)
         << "\n)";
    }

    /// @brief Returns the boundary object as JSON object
    inline nlohmann::json to_json() const override
    {
      nlohmann::json json;
      json["west"]  = std::get<west-1>(bdr_).to_json();
      json["east"]  = std::get<east-1>(bdr_).to_json();
      json["south"] = std::get<south-1>(bdr_).to_json();
      json["north"] = std::get<north-1>(bdr_).to_json();

      return json;
    }

    /// @brief Returns the Greville abscissae
    inline eval_t greville() const
    {
      return eval_t{std::get<west-1>(bdr_).greville(),
                    std::get<east-1>(bdr_).greville(),
                    std::get<south-1>(bdr_).greville(),
                    std::get<north-1>(bdr_).greville()};
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
  template<typename spline_t>
  class BoundaryCore<spline_t, /* parDim */3>
    : public core<typename spline_t::value_type>
  {
  protected:
    /// @brief Boundary spline type
    using boundaryspline_t = std::tuple<
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(1), spline_t::degree(2)>,
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(0), spline_t::degree(2)>,
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(0), spline_t::degree(1)>>;


    /// @brief Tuple of splines
    std::tuple<typename std::tuple_element_t<0,boundaryspline_t>,
               typename std::tuple_element_t<0,boundaryspline_t>,
               typename std::tuple_element_t<1,boundaryspline_t>,
               typename std::tuple_element_t<1,boundaryspline_t>,
               typename std::tuple_element_t<2,boundaryspline_t>,
               typename std::tuple_element_t<2,boundaryspline_t>> bdr_;
    
  public:
    /// @brief Evaluation type
    using eval_t = std::tuple<std::array<torch::Tensor, 2>,
                              std::array<torch::Tensor, 2>,
                              std::array<torch::Tensor, 2>,
                              std::array<torch::Tensor, 2>,
                              std::array<torch::Tensor, 2>,
                              std::array<torch::Tensor, 2>,
                              std::array<torch::Tensor, 2>>;
    
    /// @brief Default constructor
    BoundaryCore() = default;
    
    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 3>& ncoeffs,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[1], ncoeffs[2]}), init, core),
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[1], ncoeffs[2]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[2]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[2]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[1]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[1]}), init, core)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 3>& kv,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[1], kv[2]}), init, core),
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[1], kv[2]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[2]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[2]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[1]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[1]}), init, core)
             }
             )
    {}

    /// @brief Returns constant reference to side-th spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th spline
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
    inline int64_t ncumcoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncumcoeffs();
      s += std::get<east-1>(bdr_).ncumcoeffs();
      s += std::get<south-1>(bdr_).ncumcoeffs();
      s += std::get<north-1>(bdr_).ncumcoeffs();
      s += std::get<front-1>(bdr_).ncumcoeffs();
      s += std::get<back-1>(bdr_).ncumcoeffs();
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<typename spline_t::value_type>::name()
         << "(\n"
         << "west = "  << std::get<west-1>(bdr_) << "\n"
         << "east = "  << std::get<east-1>(bdr_) << "\n"
         << "south = " << std::get<south-1>(bdr_) << "\n"
         << "north = " << std::get<north-1>(bdr_) << "\n"
         << "front = " << std::get<front-1>(bdr_) << "\n"
         << "back = "  << std::get<back-1>(bdr_)
         << "\n)";
    }
    
    /// @brief Returns the boundary object as JSON object
    inline nlohmann::json to_json() const override
    {
      nlohmann::json json;
      json["west"]  = std::get<west-1>(bdr_).to_json();
      json["east"]  = std::get<east-1>(bdr_).to_json();
      json["south"] = std::get<south-1>(bdr_).to_json();
      json["north"] = std::get<north-1>(bdr_).to_json();
      json["front"] = std::get<front-1>(bdr_).to_json();
      json["back"]  = std::get<back-1>(bdr_).to_json();

      return json;
    }

    /// @brief Returns the Greville abscissae
    inline eval_t greville() const
    {
      return eval_t{std::get<west-1>(bdr_).greville(),
                    std::get<east-1>(bdr_).greville(),
                    std::get<south-1>(bdr_).greville(),
                    std::get<north-1>(bdr_).greville(),
                    std::get<front-1>(bdr_).greville(),
                    std::get<back-1>(bdr_).greville()};
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
  template<typename spline_t>
  class BoundaryCore<spline_t, /* parDim */4>
    : public core<typename spline_t::value_type>
  {
  protected:
    /// @brief Array storing the degrees
    using boundaryspline_t = std::tuple<
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(1), spline_t::degree(2), spline_t::degree(3)>,
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(0), spline_t::degree(2), spline_t::degree(3)>,
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(0), spline_t::degree(1), spline_t::degree(3)>,
    typename spline_t::template
    derived_self_type_t<typename spline_t::value_type,
                        spline_t::geoDim(),
                        spline_t::degree(0), spline_t::degree(1), spline_t::degree(2)>>;
    
    
    /// @brief Tuple of splines
    std::tuple<typename std::tuple_element_t<0,boundaryspline_t>,
               typename std::tuple_element_t<0,boundaryspline_t>,
               typename std::tuple_element_t<1,boundaryspline_t>,
               typename std::tuple_element_t<1,boundaryspline_t>,
               typename std::tuple_element_t<2,boundaryspline_t>,
               typename std::tuple_element_t<2,boundaryspline_t>,
               typename std::tuple_element_t<3,boundaryspline_t>,
               typename std::tuple_element_t<3,boundaryspline_t>> bdr_;
    
  public:
    /// @brief Evaluation type
    using eval_t = std::tuple<std::array<torch::Tensor, 3>,
                              std::array<torch::Tensor, 3>,
                              std::array<torch::Tensor, 3>,
                              std::array<torch::Tensor, 3>,
                              std::array<torch::Tensor, 3>,
                              std::array<torch::Tensor, 3>,
                              std::array<torch::Tensor, 3>,
                              std::array<torch::Tensor, 3>>;
    
    /// @brief Default constructor
    BoundaryCore() = default;
    
    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 4>& ncoeffs,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}), init, core),
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}), init, core),
               std::tuple_element_t<3,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}), init, core),
               std::tuple_element_t<3,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}), init, core)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 4>& kv,
                 enum init init = init::zeros,
                 core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : iganet::core<typename spline_t::value_type>(core),
        bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[1], kv[2], kv[3]}), init, core),
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[1], kv[2], kv[3]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[2], kv[3]}), init, core),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[2], kv[3]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[3]}), init, core),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[3]}), init, core),
               std::tuple_element_t<3,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[2]}), init, core),
               std::tuple_element_t<3,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[2]}), init, core)
             }
             )
    {}
    
    /// @brief Returns the number of sides
    inline static constexpr short_t sides()
    {
      return side::etime;
    }

    /// @brief Returns constant reference to side-th spline
    template<short_t s>
    inline constexpr auto& side() const
    {
      static_assert(s>none && s<=sides());
      return std::get<s-1>(bdr_);
    }

    /// @brief Returns non-constant reference to side-th spline
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
    inline int64_t ncumcoeffs() const
    {
      int64_t s=0;
      s += std::get<west-1>(bdr_).ncumcoeffs();
      s += std::get<east-1>(bdr_).ncumcoeffs();
      s += std::get<south-1>(bdr_).ncumcoeffs();
      s += std::get<north-1>(bdr_).ncumcoeffs();
      s += std::get<front-1>(bdr_).ncumcoeffs();
      s += std::get<back-1>(bdr_).ncumcoeffs();
      s += std::get<stime-1>(bdr_).ncumcoeffs();
      s += std::get<etime-1>(bdr_).ncumcoeffs();
      
      return s;
    }
    
    /// @brief Returns a string representation of the Boundary object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<typename spline_t::value_type>::name()
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

    /// @brief Returns the boundary object as JSON object
    inline nlohmann::json to_json() const override
    {
      nlohmann::json json;
      json["west"]  = std::get<west-1>(bdr_).to_json();
      json["east"]  = std::get<east-1>(bdr_).to_json();
      json["south"] = std::get<south-1>(bdr_).to_json();
      json["north"] = std::get<north-1>(bdr_).to_json();
      json["front"] = std::get<front-1>(bdr_).to_json();
      json["back"]  = std::get<back-1>(bdr_).to_json();
      json["stime"] = std::get<etime-1>(bdr_).to_json();
      json["etime"] = std::get<stime-1>(bdr_).to_json();

      return json;
    }

    /// @brief Returns the Greville abscissae
    inline eval_t greville() const
    {
      return eval_t{std::get<west-1>(bdr_).greville(),
                    std::get<east-1>(bdr_).greville(),
                    std::get<south-1>(bdr_).greville(),
                    std::get<north-1>(bdr_).greville(),
                    std::get<front-1>(bdr_).greville(),
                    std::get<back-1>(bdr_).greville(),
                    std::get<stime-1>(bdr_).greville(),
                    std::get<etime-1>(bdr_).greville()};
    }
  };

  /// @brief 
  
  /// @brief Boundary (common high-level functionality)
  template<typename BoundaryCore>
  class BoundaryCommon : public BoundaryCore
  {
  public:
    /// @brief Constructors from the base class
    using BoundaryCore::BoundaryCore;

  private:
    /// @brief Returns the values of the spline objects in the points `xi`
    /// @{
    template<deriv deriv = deriv::func,
             size_t... Is, typename... Xi>
    inline auto eval_(std::index_sequence<Is...>,
                      const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval<deriv>(std::get<Is>(xi))...);
    }
    
    template<deriv deriv = deriv::func,
             size_t... Is, typename... Xi, typename... Idx>
    inline auto eval_(std::index_sequence<Is...>,
                      const std::tuple<Xi...>& xi,
                      const std::tuple<Idx...>& idx) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval<deriv>(std::get<Is>(xi),
                                                                              std::get<Is>(idx))...);
    }
    
    template<deriv deriv = deriv::func,
             size_t... Is, typename... Xi, typename... Idx, typename... Coeff_Idx>
    inline auto eval_(std::index_sequence<Is...>,
                      const std::tuple<Xi...>& xi,
                      const std::tuple<Idx...>& idx,
                      const std::tuple<Coeff_Idx...>& coeff_idx) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval<deriv>(std::get<Is>(xi),
                                                                              std::get<Is>(idx),
                                                                              std::get<Is>(coeff_idx))...);
    }
    /// @}

    /// @brief Returns the value of the spline objects from
    /// precomputed basis function
    template<size_t... Is,
             typename... Basfunc, typename... Coeff_Idx,
             typename... Numeval, typename... Sizes>
    inline auto eval_from_precomputed_(std::index_sequence<Is...>,
                                       const std::tuple<Basfunc...>& basfunc,
                                       const std::tuple<Coeff_Idx...>& coeff_idx,
                                       const std::tuple<Numeval...>& numeval,
                                       const std::tuple<Sizes...>& sizes) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).eval_from_precomputed(std::get<Is>(basfunc),
                                                                               std::get<Is>(coeff_idx),
                                                                               std::get<Is>(numeval),
                                                                               std::get<Is>(sizes))...);
    }

    template<size_t... Is,
             typename... Basfunc, typename... Coeff_Idx, typename... Xi>
    inline auto eval_from_precomputed_(std::index_sequence<Is...>,
                                       const std::tuple<Basfunc...>& basfunc,
                                       const std::tuple<Coeff_Idx...>& coeff_idx,
                                       const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).eval_from_precomputed(std::get<Is>(basfunc),
                                                                               std::get<Is>(coeff_idx),
                                                                               std::get<Is>(xi)[0].numel(),
                                                                               std::get<Is>(xi)[0].sizes())...);
    }
    /// @}

    /// @brief Returns the knot indicies of knot spans containing `xi`
    template<size_t... Is, typename... Xi>
    inline auto find_knot_indices_(std::index_sequence<Is...>,
                                   const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).find_knot_indices(std::get<Is>(xi))...);
    }

    /// @brief Returns the values of the spline objects' basis functions in the points `xi`
    /// @{
    template<deriv deriv = deriv::func, size_t... Is, typename... Xi>
    inline auto eval_basfunc_(std::index_sequence<Is...>, const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).eval_basfunc(std::get<Is>(xi))...);
    }

    template<deriv deriv = deriv::func,
             size_t... Is, typename... Xi, typename... Idx>
    inline auto eval_basfunc_(std::index_sequence<Is...>,
                              const std::tuple<Xi...>& xi,
                              const std::tuple<Idx...>& idx) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).eval_basfunc(std::get<Is>(xi), std::get<Is>(idx))...);
    }
    /// @}

    /// @brief Returns the indices of the spline objects'
    /// coefficients corresponding to the knot indices `idx`
    template<size_t... Is, typename... Idx>
    inline auto eval_coeff_indices_(std::index_sequence<Is...>,
                                    const std::tuple<Idx...>& idx) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).eval_coeff_indices(std::get<Is>(idx))...);
    }

    /// @brief Returns the spline objects with uniformly refined
    /// knot and coefficient vectors
    template<size_t... Is>
    inline auto& uniform_refine_(std::index_sequence<Is...>,
                                 int numRefine = 1, int dim = -1)
    {
      (std::get<Is>(BoundaryCore::bdr_).uniform_refine(numRefine, dim), ...);
      return *this;
    }
    
  public:
    /// @brief Returns the values of the spline objects in the points `xi`
    /// @{
    template<deriv deriv = deriv::func, typename... Xi>
    inline auto eval(const std::tuple<Xi...>& xi) const
    {
      return eval_<deriv>(std::make_index_sequence<BoundaryCore::sides()>{}, xi);
    }
    
    template<deriv deriv = deriv::func,
             typename... Xi, typename... Idx>
    inline auto eval(const std::tuple<Xi...>& xi,
                     const std::tuple<Idx...>& idx) const
    {
      return eval_<deriv>(std::make_index_sequence<BoundaryCore::sides()>{}, xi, idx);
    }
    
    template<deriv deriv = deriv::func,
             typename... Xi, typename... Idx, typename... Coeff_Idx>
    inline auto eval(const std::tuple<Xi...>& xi,
                     const std::tuple<Idx...>& idx,
                     const std::tuple<Coeff_Idx...>& coeff_idx) const
    {
      return eval_<deriv>(std::make_index_sequence<BoundaryCore::sides()>{}, xi, idx, coeff_idx);
    }
    /// @}
    
    /// @brief Returns the value of the spline objects from
    /// precomputed basis function
    template<typename... Basfunc, typename... Coeff_Idx,
             typename... Numeval, typename... Sizes>
    inline auto eval_from_precomputed(const std::tuple<Basfunc...>& basfunc,
                                      const std::tuple<Coeff_Idx...>& coeff_idx,
                                      const std::tuple<Numeval...>& numeval,
                                      const std::tuple<Sizes...>& sizes) const
    {
      return eval_from_precomputed_(std::make_index_sequence<BoundaryCore::sides()>{},
                                    basfunc, coeff_idx, numeval, sizes);
    }

    template<typename... Basfunc, typename... Coeff_Idx, typename... Xi>
    inline auto eval_from_precomputed(const std::tuple<Basfunc...>& basfunc,
                                      const std::tuple<Coeff_Idx...>& coeff_idx,
                                      const std::tuple<Xi...>& xi) const
    {
      return eval_from_precomputed_(std::make_index_sequence<BoundaryCore::sides()>{},
                                    basfunc, coeff_idx, xi);
    }
    /// @}

    /// @brief Returns the knot indicies of knot spans containing `xi`
    template<typename... Xi>
    inline auto find_knot_indices(const std::tuple<Xi...>& xi) const
    {
      return find_knot_indices_(std::make_index_sequence<BoundaryCore::sides()>{}, xi);
    }

    /// @brief Returns the values of the spline objects' basis
    /// functions in the points `xi` @{
    template<deriv deriv = deriv::func, typename... Xi>
    inline auto eval_basfunc(const std::tuple<Xi...>& xi) const
    {
      return eval_basfunc_(std::make_index_sequence<BoundaryCore::sides()>{}, xi);
    }

    template<deriv deriv = deriv::func, typename... Xi, typename... Idx>
    inline auto eval_basfunc(const std::tuple<Xi...>& xi,
                             const std::tuple<Idx...>& idx) const
    {
      return eval_basfunc_(std::make_index_sequence<BoundaryCore::sides()>{}, xi, idx);
    }
    /// @}

    /// @brief Returns the indices of the spline objects'
    /// coefficients corresponding to the knot indices `idx`
    template<typename... Idx>
    inline auto eval_coeff_indices(const std::tuple<Idx...>& idx) const
    {
      return eval_coeff_indices_(std::make_index_sequence<BoundaryCore::sides()>{}, idx);
    }

    /// @brief Returns the spline objects with uniformly refined
    /// knot and coefficient vectors
    inline auto& uniform_refine(int numRefine = 1, int dim = -1)
    {
      uniform_refine_(std::make_index_sequence<BoundaryCore::sides()>{});
      return *this;
    }    
  };

  /// @brief Boundary
  template<typename spline_t>
  using Boundary = BoundaryCommon<BoundaryCore<spline_t, spline_t::parDim()>>;
  
  /// @brief Print (as string) a Boundary object
  template<typename spline_t>
  inline std::ostream& operator<<(std::ostream& os,
                                  const Boundary<spline_t>& obj)
  {
    obj.pretty_print(os);
    return os;
  }
  
} // namespace iganet
