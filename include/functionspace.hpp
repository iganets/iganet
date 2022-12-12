/**
   @file include/functionspace.hpp

   @brief Function spaces

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <bspline.hpp>

#pragma once

namespace iganet {

  /// @brief Function space
  ///
  /// This class implements the functionality of a function space
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t...> class FunctionSpace;

  /// @brief Function space (specialization for univariate domains)
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree0,
           short_t cont0>
  class FunctionSpace<real_t, TargetDim, bspline_t,
                      degree0, cont0>
    : public bspline_t<real_t, TargetDim, degree0>
  {
  private:
    /// @brief Dimension of the parametric domain
    static constexpr short_t domainDim_ = 1;

    /// @brief Dimension of the physical domain
    static constexpr short_t targetDim_ = TargetDim;

  public:
    /// @brief Constructor
    FunctionSpace(const std::array<int64_t, domainDim_>& ncoeffs,
                  enum init init = init::zeros)
      : bspline_t<real_t, TargetDim, degree0>(ncoeffs, init)
    {
      std::cout << "Continuity=" << cont0 << std::endl;
      // TODO: Perform knot insertion until reduced continuity is reached
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
  };

  /// @brief Function space (specialization for bivariate domains)
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree0, short_t degree1,
           short_t cont0, short_t cont1>
  class FunctionSpace<real_t, TargetDim, bspline_t,
                      degree0, degree1, cont0, cont1>
    : public bspline_t<real_t, TargetDim, degree0, degree1>
  {
  private:
    /// @brief Dimension of the parametric domain
    static constexpr short_t domainDim_ = 2;

    /// @brief Dimension of the physical domain
    static constexpr short_t targetDim_ = TargetDim;

  public:
    /// @brief Constructor
    FunctionSpace(const std::array<int64_t, domainDim_>& ncoeffs,
                  enum init init = init::zeros)
      : bspline_t<real_t, TargetDim, degree0, degree1>(ncoeffs, init)
    {
      // TODO: Perform knot insertion until reduced continuity is reached
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
  };

  /// @brief Function space (specialization for trivariate domains)
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree0, short_t degree1, short_t degree2,
           short_t cont0, short_t cont1, short_t cont2>
  class FunctionSpace<real_t, TargetDim, bspline_t,
                      degree0, degree1, degree2, cont0, cont1, cont2>
    : public bspline_t<real_t, TargetDim, degree0, degree1, degree2>
  {
  private:
    /// @brief Dimension of the parametric domain
    static constexpr short_t domainDim_ = 3;

    /// @brief Dimension of the physical domain
    static constexpr short_t targetDim_ = TargetDim;

  public:
    /// @brief Constructor
    FunctionSpace(const std::array<int64_t, domainDim_>& ncoeffs,
                  enum init init = init::zeros)
      : bspline_t<real_t, TargetDim, degree0, degree1, degree2>(ncoeffs, init)
    {
      // TODO: Perform knot insertion until reduced continuity is reached
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
  };

  /// @brief Function space (specialization for quadvariate domains)
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree0, short_t degree1, short_t degree2, short_t degree3,
           short_t cont0, short_t cont1, short_t cont2, short_t cont3>
  class FunctionSpace<real_t, TargetDim, bspline_t,
                      degree0, degree1, degree2, degree3, cont0, cont1, cont2, cont3>
    : public bspline_t<real_t, TargetDim, degree0, degree1, degree2, degree3>
  {
  private:
    /// @brief Dimension of the parametric domain
    static constexpr short_t domainDim_ = 4;

    /// @brief Dimension of the physical domain
    static constexpr short_t targetDim_ = TargetDim;

  public:
    /// @brief Constructor
    FunctionSpace(const std::array<int64_t, domainDim_>& ncoeffs,
                  enum init init = init::zeros)
      : bspline_t<real_t, TargetDim, degree0, degree1, degree2, degree3>(ncoeffs, init)
    {
      // TODO: Perform knot insertion until reduced continuity is reached
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
  };

  /// @brief Alias for \f$ S_{p}^{q}\f$
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree, short_t cont = degree-1>
  using S1 = FunctionSpace<real_t, TargetDim, bspline_t,
                           degree,
                           cont>;
  
  /// @brief Alias for \f$ S_{p,p}^{p-1,p-1} :=
  /// S_{p}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree, short_t cont = degree-1>
  using S2 = FunctionSpace<real_t, TargetDim, bspline_t,
                           degree, degree,
                           degree-1, degree-1>;
  
  /// @brief Alias for \f$ S_{p,p,p}^{p-1,p-1,p-1} :=
  /// S_{p}^{p-1} \otimes S_{p}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree, short_t cont = degree-1>
  using S3 = FunctionSpace<real_t, TargetDim, bspline_t,
                           degree, degree, degree,
                           degree-1, degree-1, degree-1>;
  
  /// @brief Alias for \f$ S_{p,p,p,p}^{p-1,p-1,p-1,p-1} :=
  /// S_{p}^{p-1} \otimes S_{p}^{p-1} \otimes S_{p}^{p-1} \otimes
  /// S_{p}^{p-1} \f$
  template<typename real_t,
           short_t TargetDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t degree, short_t cont = degree-1>
  using S4 = FunctionSpace<real_t, TargetDim, bspline_t,
                           degree, degree, degree, degree,
                           degree-1, degree-1, degree-1, degree-1>;

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

} // namespace iganet
