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
    : public utils::Serializable, private utils::FullQualifiedName
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
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               boundaryspline_t(std::array<int64_t, 0>{}, init, options),
               boundaryspline_t(std::array<int64_t, 0>{}, init, options),
             }               
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 1>&,
                 enum init init = init::zeros,
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               boundaryspline_t(std::array<int64_t, 0>{}, init, options),
               boundaryspline_t(std::array<int64_t, 0>{}, init, options),
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
    inline virtual void pretty_print(std::ostream& os = std::cout) const noexcept override
    {
      os << name()
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
    : public utils::Serializable, private utils::FullQualifiedName
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
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[1]}), init, options),
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[1]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[0]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,1>({ncoeffs[0]}), init, options)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 2>& kv,
                 enum init init = init::zeros,
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[1]}), init, options),
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[1]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[0]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,1>({kv[0]}), init, options)
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
    inline virtual void pretty_print(std::ostream& os = std::cout) const noexcept override
    {
      os << name()
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
    : public utils::Serializable, private utils::FullQualifiedName
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
                              std::array<torch::Tensor, 2>>;
    
    /// @brief Default constructor
    BoundaryCore() = default;
    
    /// @brief Constructor
    BoundaryCore(const std::array<int64_t, 3>& ncoeffs,
                 enum init init = init::zeros,
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[1], ncoeffs[2]}), init, options),
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[1], ncoeffs[2]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[2]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[2]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[1]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,2>({ncoeffs[0], ncoeffs[1]}), init, options)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 3>& kv,
                 enum init init = init::zeros,
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[1], kv[2]}), init, options),
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[1], kv[2]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[2]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[2]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[1]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,2>({kv[0], kv[1]}), init, options)
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
    inline virtual void pretty_print(std::ostream& os = std::cout) const noexcept override
    {
      os << name()
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
    : public utils::Serializable, private utils::FullQualifiedName
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
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}), init, options),
               std::tuple_element_t<0,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}), init, options),
               std::tuple_element_t<3,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}), init, options),
               std::tuple_element_t<3,boundaryspline_t>(std::array<int64_t,3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}), init, options)
             }
             )
    {}

    /// @brief Constructor
    BoundaryCore(const std::array<std::vector<typename spline_t::value_type>, 4>& kv,
                 enum init init = init::zeros,
                 Options<typename spline_t::value_type> options = Options<typename spline_t::value_type>{})
      : bdr_(
             {
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[1], kv[2], kv[3]}), init, options),
               std::tuple_element_t<0,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[1], kv[2], kv[3]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[2], kv[3]}), init, options),
               std::tuple_element_t<1,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[2], kv[3]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[3]}), init, options),
               std::tuple_element_t<2,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[3]}), init, options),
               std::tuple_element_t<3,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[2]}), init, options),
               std::tuple_element_t<3,boundaryspline_t>(std::array<std::vector<typename spline_t::value_type>,3>({kv[0], kv[1], kv[2]}), init, options)
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
    inline virtual void pretty_print(std::ostream& os = std::cout) const noexcept override
    {
      os << name()
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
  class BoundaryCommon
    : public BoundaryCore
  {
  public:
    /// @brief Constructors from the base class
    using BoundaryCore::BoundaryCore;

  private:
    /// @brief Returns all coefficients of all spline objects as a
    /// single tensor
    ///
    /// @result Tensor of coefficients
    template<std::size_t... Is>
    inline torch::Tensor as_tensor_(std::index_sequence<Is...>) const
    {
      return torch::cat({std::get<Is>(BoundaryCore::bdr_).as_tensor()...});
    }

    /// @brief Returns the size of the single tensor representation of
    /// all spline objects
    ///
    /// @result Size of the tensor
    template<std::size_t... Is>
    inline int64_t as_tensor_size_(std::index_sequence<Is...>) const
    {
      return std::apply([]( auto... v ){ return ( v + ... ); },
                        std::make_tuple(std::get<Is>(BoundaryCore::bdr_).as_tensor_size()...));
    }

    /// @brief Sets all coefficients of all spline objects from a
    /// single tensor
    ///
    /// @result Updates spline object
    template<std::size_t... Is>
    inline auto& from_tensor_(std::index_sequence<Is...>,
                              const torch::Tensor& coeffs)
    {
      std::size_t counter(0);
      auto lambda = [&counter](std::size_t increment){ return counter+=increment; };

      (std::get<Is>(BoundaryCore::bdr_).from_tensor( coeffs.index({torch::indexing::Slice(counter,
                                                                                          lambda(std::get<Is>(BoundaryCore::bdr_).ncumcoeffs() *
                                                                                                 std::get<Is>(BoundaryCore::bdr_).geoDim()))}) )
        , ...);
      
      return *this;
    }

  public:    
    /// @brief Returns all coefficients of all spline objects as a
    /// single tensor
    ///
    /// @result Tensor of coefficients
    inline torch::Tensor as_tensor() const
    {
      return as_tensor_(std::make_index_sequence<BoundaryCore::sides()>{});
    }

    /// @brief Returns the size of the single tensor representation of
    /// all spline objects
    //
    /// @result Size of the tensor
    inline int64_t as_tensor_size() const
    {
      return as_tensor_size_(std::make_index_sequence<BoundaryCore::sides()>{});
    }
    
    /// @brief Sets all coefficients of all spline objects from a
    /// single tensor
    ///
    /// @result Updated spline objects
    inline auto& from_tensor(const torch::Tensor& coeffs)
    {
      return from_tensor_(std::make_index_sequence<BoundaryCore::sides()>{}, coeffs);
    }
    
  private:
    /// @brief Returns the values of the boundary spline objects in
    /// the points `xi` @{
    template<deriv deriv = deriv::func,
             bool memory_optimized = false,
             size_t... Is, typename... Xi>
    inline auto eval_(std::index_sequence<Is...>,
                      const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval<deriv, memory_optimized>(std::get<Is>(xi))...);
    }
    
    template<deriv deriv = deriv::func,
             bool memory_optimized = false,
             size_t... Is, typename... Xi, typename... Indices>
    inline auto eval_(std::index_sequence<Is...>,
                      const std::tuple<Xi...>& xi,
                      const std::tuple<Indices...>& indices) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                                std::get<Is>(indices))...);
    }
    
    template<deriv deriv = deriv::func,
             bool memory_optimized = false,
             size_t... Is, typename... Xi, typename... Indices, typename... Coeff_Indices>
    inline auto eval_(std::index_sequence<Is...>,
                      const std::tuple<Xi...>& xi,
                      const std::tuple<Indices...>& indices,
                      const std::tuple<Coeff_Indices...>& coeff_indices) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                                std::get<Is>(indices),
                                                                                                std::get<Is>(coeff_indices))...);
    }
    /// @}

    /// @brief Returns the value of the boundary spline objects from
    /// precomputed basis function @{
    template<size_t... Is,
             typename... Basfunc, typename... Coeff_Indices,
             typename... Numeval, typename... Sizes>
    inline auto eval_from_precomputed_(std::index_sequence<Is...>,
                                       const std::tuple<Basfunc...>& basfunc,
                                       const std::tuple<Coeff_Indices...>& coeff_indices,
                                       const std::tuple<Numeval...>& numeval,
                                       const std::tuple<Sizes...>& sizes) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).eval_from_precomputed(std::get<Is>(basfunc),
                                                                               std::get<Is>(coeff_indices),
                                                                               std::get<Is>(numeval),
                                                                               std::get<Is>(sizes))...);
    }

    template<size_t... Is,
             typename... Basfunc, typename... Coeff_Indices, typename... Xi>
    inline auto eval_from_precomputed_(std::index_sequence<Is...>,
                                       const std::tuple<Basfunc...>& basfunc,
                                       const std::tuple<Coeff_Indices...>& coeff_indices,
                                       const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).eval_from_precomputed(std::get<Is>(basfunc),
                                                                               std::get<Is>(coeff_indices),
                                                                               std::get<Is>(xi)[0].numel(),
                                                                               std::get<Is>(xi)[0].sizes())...);
    }
    /// @}

    /// @brief Returns the knot indicies of boundary spline object's
    /// knot spans containing `xi`
    template<size_t... Is, typename... Xi>
    inline auto find_knot_indices_(std::index_sequence<Is...>,
                                   const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).find_knot_indices(std::get<Is>(xi))...);
    }

    /// @brief Returns the values of the boundary spline spline
    /// object's basis functions in the points `xi`
    /// @{
    template<deriv deriv = deriv::func,
             bool memory_optimized = false,
             size_t... Is, typename... Xi>
    inline auto eval_basfunc_(std::index_sequence<Is...>, const std::tuple<Xi...>& xi) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval_basfunc<memory_optimized>(std::get<Is>(xi))...);
    }

    template<deriv deriv = deriv::func,
             bool memory_optimized = false, 
             size_t... Is, typename... Xi, typename... Indices>
    inline auto eval_basfunc_(std::index_sequence<Is...>,
                              const std::tuple<Xi...>& xi,
                              const std::tuple<Indices...>& indices) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template eval_basfunc<memory_optimized>(std::get<Is>(xi), std::get<Is>(indices))...);
    }
    /// @}

    /// @brief Returns the indices of the boundary spline object's
    /// coefficients corresponding to the knot indices `indices`
    template<bool memory_optimized = false, size_t... Is, typename... Indices>
    inline auto find_coeff_indices_(std::index_sequence<Is...>,
                                    const std::tuple<Indices...>& indices) const
    {
      return std::tuple(std::get<Is>(BoundaryCore::bdr_).template find_coeff_indices<memory_optimized>(std::get<Is>(indices))...);
    }

    /// @brief Returns the boundary spline object with uniformly
    /// refined knot and coefficient vectors
    template<size_t... Is>
    inline auto& uniform_refine_(std::index_sequence<Is...>,
                                 int numRefine = 1, int dim = -1)
    {
      (std::get<Is>(BoundaryCore::bdr_).uniform_refine(numRefine, dim), ...);
      return *this;
    }

    /// @brief Returns true if both boundary spline objects are the
    /// same
    template<size_t... Is>
    inline bool is_equal(std::index_sequence<Is...>,
                         const BoundaryCommon& other) const
    {
      return ((std::get<Is>(BoundaryCore::bdr_) == std::get<Is>(other.coeffs())) &&  ...);
    }
    
    /// @brief Writes the boundary spline object into a
    /// torch::serialize::OutputArchive object
    template<size_t... Is>
    inline torch::serialize::OutputArchive& write_(std::index_sequence<Is...>,
                                                   torch::serialize::OutputArchive& archive,
                                                   const std::string& key="boundary") const
    {
      (std::get<Is>(BoundaryCore::bdr_).write(archive, key+".bdr["+std::to_string(Is)+"]"), ...);
      return archive;
    }
    
    /// @brief Loads the function space object from a
    /// torch::serialize::InputArchive object
    template<size_t... Is>
    inline torch::serialize::InputArchive& read_(std::index_sequence<Is...>,
                                                 torch::serialize::InputArchive& archive,
                                                 const std::string& key="boundary")
    {
      (std::get<Is>(BoundaryCore::bdr_).read(archive, key+".bdr["+std::to_string(Is)+"]"), ...);
      return archive;
    }
    
  public:
    /// @brief Returns the values of the spline objects in the points `xi`
    /// @{
    template<deriv deriv = deriv::func,
             bool memory_optimized = false,
             typename... Xi>
    inline auto eval(const std::tuple<Xi...>& xi) const
    {
      return eval_<deriv, memory_optimized>(std::make_index_sequence<BoundaryCore::sides()>{}, xi);
    }
    
    template<deriv deriv = deriv::func,
             bool memory_optimized = false,
             typename... Xi, typename... Indices>
    inline auto eval(const std::tuple<Xi...>& xi,
                     const std::tuple<Indices...>& indices) const
    {
      return eval_<deriv, memory_optimized>(std::make_index_sequence<BoundaryCore::sides()>{}, xi, indices);
    }
    
    template<deriv deriv = deriv::func,
             bool memory_optimized = false,
             typename... Xi, typename... Indices, typename... Coeff_Indices>
    inline auto eval(const std::tuple<Xi...>& xi,
                     const std::tuple<Indices...>& indices,
                     const std::tuple<Coeff_Indices...>& coeff_indices) const
    {
      return eval_<deriv, memory_optimized>(std::make_index_sequence<BoundaryCore::sides()>{}, xi, indices, coeff_indices);
    }
    /// @}
    
    /// @brief Returns the value of the spline objects from
    /// precomputed basis function @{
    template<typename... Basfunc, typename... Coeff_Indices,
             typename... Numeval, typename... Sizes>
    inline auto eval_from_precomputed(const std::tuple<Basfunc...>& basfunc,
                                      const std::tuple<Coeff_Indices...>& coeff_indices,
                                      const std::tuple<Numeval...>& numeval,
                                      const std::tuple<Sizes...>& sizes) const
    {
      return eval_from_precomputed_(std::make_index_sequence<BoundaryCore::sides()>{},
                                    basfunc, coeff_indices, numeval, sizes);
    }

    template<typename... Basfunc, typename... Coeff_Indices, typename... Xi>
    inline auto eval_from_precomputed(const std::tuple<Basfunc...>& basfunc,
                                      const std::tuple<Coeff_Indices...>& coeff_indices,
                                      const std::tuple<Xi...>& xi) const
    {
      return eval_from_precomputed_(std::make_index_sequence<BoundaryCore::sides()>{},
                                    basfunc, coeff_indices, xi);
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
    template<deriv deriv = deriv::func,
             bool memory_optimized = false, 
             typename... Xi>
    inline auto eval_basfunc(const std::tuple<Xi...>& xi) const
    {
      return eval_basfunc_<memory_optimized>(std::make_index_sequence<BoundaryCore::sides()>{}, xi);
    }

    template<deriv deriv = deriv::func,
             bool memory_optimized = false, 
             typename... Xi, typename... Indices>
    inline auto eval_basfunc(const std::tuple<Xi...>& xi,
                             const std::tuple<Indices...>& indices) const
    {
      return eval_basfunc_<memory_optimized>(std::make_index_sequence<BoundaryCore::sides()>{}, xi, indices);
    }
    /// @}

    /// @brief Returns the indices of the spline objects'
    /// coefficients corresponding to the knot indices `indices`
    template<bool memory_optimized = false, typename... Indices>
    inline auto find_coeff_indices(const std::tuple<Indices...>& indices) const
    {
      return find_coeff_indices_<memory_optimized>(std::make_index_sequence<BoundaryCore::sides()>{}, indices);
    }

    /// @brief Returns the spline objects with uniformly refined
    /// knot and coefficient vectors
    inline auto& uniform_refine(int numRefine = 1, int dim = -1)
    {
      uniform_refine_(std::make_index_sequence<BoundaryCore::sides()>{});
      return *this;
    }

    /// @brief Returns true if both boundary objects are the same
    inline bool operator==(const BoundaryCommon& other) const
    {
      return is_equal(std::make_index_sequence<BoundaryCore::sides()>{}, other);
    }

    /// @brief Returns true if both boundaryt objects are different
    inline bool operator!=(const BoundaryCommon& other) const
    {
      return *this != other;
    }

    /// @brief Writes the boundary spline object into a
    /// torch::serialize::OutputArchive object
    inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                  const std::string& key="boundary") const
    {
      write_(std::make_index_sequence<BoundaryCore::sides()>{}, archive, key);
      return archive;
    }
    
    /// @brief Loads the boundary spline object from a
    /// torch::serialize::InputArchive object
    inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                const std::string& key="boundary")
    {
      read_(std::make_index_sequence<BoundaryCore::sides()>{}, archive, key);
      return archive;
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
