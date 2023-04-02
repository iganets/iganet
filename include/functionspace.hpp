/**
   @file include/functionspace.hpp

   @brief Function spaces

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <bspline.hpp>

namespace iganet {
  
  /// @brief Tensor-product function space
  template<typename... BSpline_t>
  class FunctionSpace : public std::tuple<FunctionSpace<BSpline_t>...>
  {
  private:
    /// @brief Base class
    using Base = std::tuple<FunctionSpace<BSpline_t>...>;
    
    /// @brief Dimension of the parametric domain
    static constexpr std::array<short_t, sizeof...(BSpline_t)> domainDim_ = {BSpline_t::parDim()...};
    
    /// @brief Dimension of the physical domain
    static constexpr std::array<short_t, sizeof...(BSpline_t)> targetDim_ = {BSpline_t::geoDim()...};

    /// @brief Type of the boundary B-spline objects
    using Boundary_t = std::tuple<Boundary<BSpline_t>...>;

    /// @brief Boundary B-spline objects
    Boundary_t boundary_;

  public:   
    /// @brief Constructor
    FunctionSpace(enum init init,
                  const std::array<int64_t, BSpline_t::parDim()>&... ncoeffs)
      : Base({ncoeffs, init}...),
        boundary_({ncoeffs, init}...)
    {}

    FunctionSpace(enum init init,
                  const std::array<std::vector<typename BSpline_t::value_type>,
                  BSpline_t::parDim()>&... kv)
      : Base({kv, init}...),
        boundary_({kv, init}...)
    {}

    template<typename... Args>
    FunctionSpace(enum init init, const Args&... args)
      : Base({args, init}...),
        boundary_({args, init}...)
    {}
  };

  /// @brief Function space 
  template<typename BSpline_t>
  class FunctionSpace<BSpline_t> : public BSpline_t
  {
  private:
    /// @brief Dimension of the parametric domain
    static constexpr short_t domainDim_ = BSpline_t::parDim();

    /// @brief Dimension of the physical domain
    static constexpr short_t targetDim_ = BSpline_t::geoDim();

    /// @brief Type of the boundary B-spline objects
    using Boundary_t = Boundary<BSpline_t>;

    /// @brief Boundary B-spline objects
    Boundary_t boundary_;
    
  public:
    /// @brief Default constructor
    FunctionSpace() = default;
    
    /// @brief Constructor
    FunctionSpace(enum init init,
                  const std::array<int64_t, domainDim_>& ncoeffs)
      : BSpline_t(ncoeffs, init),
        boundary_(ncoeffs, init)
    {
    }

    /// @brief Constructor
    FunctionSpace(enum init init,
                  std::array<std::vector<typename BSpline_t::value_type>,
                  BSpline_t::parDim()> kv)
      : BSpline_t(kv, init),
        boundary_(kv, init)
    {
    }

    /// @brief Returns the parametric dimension
    inline static constexpr short_t domainDim()
    {
      return domainDim_;
    }

    /// @brief Returns the target dimension
    inline static constexpr short_t targetDim()
    {
      return targetDim_;
    }

    /// @brief Returns a constant reference to the boundary B-spline object
    inline const auto& boundary() const
    {
      return boundary_;
    }

    /// @brief Returns a non-constant reference to the boundary B-spline object
    inline auto& boundary()
    {
      return boundary_;
    }

    /// @brief Serialization to JSON
    nlohmann::json to_json() const override
    {
      return "not yet implemented";
    }
  };

#if 0
  /// @brief Alias for Taylor-Hood like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using TH1 = std::tuple<S1<real_t, 1, bspline_t, degree+1, degree-1>,
                         S1<real_t, 1, bspline_t, degree,   degree-1>>;

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using TH2 = std::tuple<S2<real_t, 1, bspline_t, degree+1, degree-1>,
                         S2<real_t, 1, bspline_t, degree+1, degree-1>,
                         S2<real_t, 1, bspline_t, degree,   degree-1>>;

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using TH3 = std::tuple<S3<real_t, 1, bspline_t, degree+1, degree-1>,
                         S3<real_t, 1, bspline_t, degree+1, degree-1>,
                         S3<real_t, 1, bspline_t, degree+1, degree-1>,
                         S3<real_t, 1, bspline_t, degree,   degree-1>>;

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using TH4 = std::tuple<S4<real_t, 1, bspline_t, degree+1, degree-1>,
                         S4<real_t, 1, bspline_t, degree+1, degree-1>,
                         S4<real_t, 1, bspline_t, degree+1, degree-1>,
                         S4<real_t, 1, bspline_t, degree+1, degree-1>,
                         S4<real_t, 1, bspline_t, degree,   degree-1>>;

  /// @brief Alias for Nedelec like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using NE1 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1,
                                       degree-1>,
                         S1<real_t, 1, bspline_t, degree, degree-1>>;

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1}^{p,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using NE2 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1,
                                       degree,   degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1,
                                       degree-1, degree  >,
                         S2<real_t, 1, bspline_t, degree, degree-1>>;

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1}^{p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using NE3 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1, degree+1,
                                       degree,   degree-1, degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1, degree+1,
                                       degree-1, degree,   degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1, degree+1,
                                       degree-1, degree-1, degree>,
                         S3<real_t, 1, bspline_t, degree, degree-1>>;

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using NE4 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1, degree+1, degree+1,
                                       degree,   degree-1, degree-1, degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1, degree+1, degree+1,
                                       degree-1, degree,   degree-1, degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1, degree+1, degree+1,
                                       degree-1, degree-1, degree,   degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree+1, degree+1, degree+1,
                                       degree-1, degree-1, degree-1, degree>,
                         S4<real_t, 1, bspline_t, degree, degree-1>>;

  /// @brief Alias for Raviart-Thomas like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using RT1 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1,
                                       degree-1>,
                         S1<real_t, 1, bspline_t, degree, degree-1>>;
  
  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p}^{p,p-1} \otimes
  /// S_{p,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using RT2 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree,
                                       degree,   degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree,   degree+1,
                                       degree-1, degree  >,
                         S2<real_t, 1, bspline_t, degree, degree-1>>;

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p}^{p,p-1,p-1} \otimes
  /// S_{p,p+1,p}^{p-1,p,p-1} \otimes
  /// S_{p,p,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using RT3 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree,   degree,
                                       degree,   degree-1, degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree,   degree+1, degree,
                                       degree-1, degree,   degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree,   degree,   degree+1,
                                       degree-1, degree-1, degree>,
                         S3<real_t, 1, bspline_t, degree, degree-1>>;

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p,p}^{p,p-1,p-1,p-1} \otimes
  /// S_{p,p+1,p,p}^{p-1,p,p-1,p-1} \otimes
  /// S_{p,p,p+1,p}^{p-1,p-1,p,p-1} \otimes
  /// S_{p,p,p,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename real_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree>
  using RT4 = std::tuple<FunctionSpace<real_t, 1, bspline_t,
                                       degree+1, degree,   degree,   degree,
                                       degree,   degree-1, degree-1, degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree,   degree+1, degree,   degree,
                                       degree-1, degree,   degree-1, degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree,   degree,   degree+1, degree,
                                       degree-1, degree-1, degree,   degree-1>,
                         FunctionSpace<real_t, 1, bspline_t,
                                       degree,   degree,   degree,   degree+1,
                                       degree-1, degree-1, degree-1, degree>,
                         S4<real_t, 1, bspline_t, degree, degree-1>>;
#endif
  
} // namespace iganet
