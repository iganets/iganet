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
  ///
  /// @note This class is not meant for direct use in
  /// applications. Instead use S1, S2, S3, S4, TH1, TH2, TH3, TH4,
  /// NE1, NE2, NE3, NE, RT1, RT2, RT3, or RT4.
  template<typename... BSpline_t>
  class FunctionSpace : public std::tuple<FunctionSpace<BSpline_t>...>
  {
  protected:
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
      : Base({init, ncoeffs}...),
        boundary_({ncoeffs, init}...)
    {      
    }

    FunctionSpace(enum init init,
                  const std::array<std::vector<typename BSpline_t::value_type>,
                  BSpline_t::parDim()>&... kv)
      : Base({init, kv}...),
        boundary_({kv, init}...)
    {}

    template<typename... Args>
    FunctionSpace(enum init init, const Args&... args)
      : Base({init, args}...),
        boundary_({args, init}...)
    {}
  };

  /// @brief Function space
  ///
  /// @note This class is not meant for direct use in
  /// applications. Instead use S1, S2, S3, S4, TH1, TH2, TH3, TH4,
  /// NE1, NE2, NE3, NE, RT1, RT2, RT3, or RT4.
  template<typename BSpline_t>
  class FunctionSpace<BSpline_t> : public BSpline_t
  {
  protected:
    /// @brief Base class
    using Base = BSpline_t;
    
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
      return Base::to_json();
    }
  };

  /// @brief Spline function space \f$ S_{p}^{p-1} \f$
  template<typename BSpline_t, short_t... Cs>
  class S1
    : public FunctionSpace<typename BSpline_t::template
                           derived_self_type_t<typename BSpline_t::value_type,
                                               BSpline_t::geoDim(),
                                               BSpline_t::degree(0)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)>>;
  public:
    /// @brief Constructor
    /// @{
    S1(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
       enum init init = init::zeros)
      : Base(init, ncoeffs)
    {
      if constexpr (sizeof...(Cs) == 1) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          Base::uniform_refine(BSpline_t::degree(0)-1-c, 0);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  
    S1(std::array<std::vector<typename BSpline_t::value_type>,
                 BSpline_t::parDim()> kv,
       enum init init = init::zeros)
      : Base(init, kv)
    {
      if constexpr (sizeof...(Cs) == 1) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          Base::uniform_refine(BSpline_t::degree(0)-1-c, 0);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  /// @}
  };

  template<typename BSpline_t, short_t... Cs>
  class S2
    : public FunctionSpace<typename BSpline_t::template
                           derived_self_type_t<typename BSpline_t::value_type,
                                               BSpline_t::geoDim(),
                                               BSpline_t::degree(0),
                                               BSpline_t::degree(1)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1)>>;
  public:
    /// @brief Constructor
    /// @{
    S2(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
       enum init init = init::zeros)
      : Base(init, ncoeffs)
    {
      if constexpr (sizeof...(Cs) == 2) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          Base::uniform_refine(BSpline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(1)-1)
          Base::uniform_refine(BSpline_t::degree(1)-1-c, 1);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  
    S2(std::array<std::vector<typename BSpline_t::value_type>,
                 BSpline_t::parDim()> kv,
       enum init init = init::zeros)
      : Base(init, kv)
    {
      if constexpr (sizeof...(Cs) == 2) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          Base::uniform_refine(BSpline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(1)-1)
          Base::uniform_refine(BSpline_t::degree(1)-1-c, 1);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  /// @}
  };

  template<typename BSpline_t, short_t... Cs>
  class S3
    : public FunctionSpace<typename BSpline_t::template
                           derived_self_type_t<typename BSpline_t::value_type,
                                               BSpline_t::geoDim(),
                                               BSpline_t::degree(0),
                                               BSpline_t::degree(1),
                                               BSpline_t::degree(2)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2)>>;
  public:
    /// @brief Constructor
    /// @{
    S3(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
       enum init init = init::zeros)
      : Base(init, ncoeffs)
    {
      if constexpr (sizeof...(Cs) == 3) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          Base::uniform_refine(BSpline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(1)-1)
          Base::uniform_refine(BSpline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(2)-1)
          Base::uniform_refine(BSpline_t::degree(2)-1-c, 2);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  
    S3(std::array<std::vector<typename BSpline_t::value_type>,
                 BSpline_t::parDim()> kv,
       enum init init = init::zeros)
      : Base(init, kv)
    {
      if constexpr (sizeof...(Cs) == 3) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          Base::uniform_refine(BSpline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(1)-1)
          Base::uniform_refine(BSpline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(2)-1)
          Base::uniform_refine(BSpline_t::degree(2)-1-c, 2);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  /// @}
  };

  template<typename BSpline_t, short_t... Cs>
  class S4
    : public FunctionSpace<typename BSpline_t::template
                           derived_self_type_t<typename BSpline_t::value_type,
                                               BSpline_t::geoDim(),
                                               BSpline_t::degree(0),
                                               BSpline_t::degree(1),
                                               BSpline_t::degree(2),
                                               BSpline_t::degree(3)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2),
                                                   BSpline_t::degree(3)>>;
  public:
    /// @brief Constructor
    /// @{
    S4(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
       enum init init = init::zeros)
      : Base(init, ncoeffs)
    {
      if constexpr (sizeof...(Cs) == 4) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          Base::uniform_refine(BSpline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(1)-1)
          Base::uniform_refine(BSpline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(2)-1)
          Base::uniform_refine(BSpline_t::degree(2)-1-c, 2);
        if constexpr (const auto c = std::get<3>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(3)-1)
          Base::uniform_refine(BSpline_t::degree(3)-1-c, 3);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  
    S4(std::array<std::vector<typename BSpline_t::value_type>,
                       BSpline_t::parDim()> kv,
       enum init init = init::zeros)
      : Base(init, kv)
    {
      if constexpr (sizeof...(Cs) == 4) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(0)-1)
          std::get<0>(*this).uniform_refine(BSpline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(1)-1)
          std::get<1>(*this).uniform_refine(BSpline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(2)-1)
          std::get<2>(*this).uniform_refine(BSpline_t::degree(2)-1-c, 2);
        if constexpr (const auto c = std::get<3>(std::make_tuple(Cs...));
                      c != BSpline_t::degree(3)-1)
          std::get<3>(*this).uniform_refine(BSpline_t::degree(3)-1-c, 3);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  /// @}
  };
  
  /// @brief Taylor-Hood like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename BSpline_t>
  class TH1 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)>>;
  public:
    /// @brief Constructor
    /// @{
    TH1(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine();
    }

    TH1(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };
  
  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename BSpline_t>
  class TH2 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1)>>;
  public:
    /// @brief Constructor
    /// @{
    TH2(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
    }

    TH2(const std::array<std::vector<typename BSpline_t::value_type>,
        BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
    }
    /// @}
  }; 

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename BSpline_t>
  class TH3 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2)>>;
  public:
    /// @brief Constructor
    /// @{
    TH3(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
    }

    TH3(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv, kv)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
    }
    /// @}
  };
  
  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename BSpline_t>
  class TH4 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2),
                                                       BSpline_t::degree(3)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2),
                                                   BSpline_t::degree(3)>>;
  public:
    /// @brief Constructor
    /// @{
    TH4(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
      std::get<3>(*this).uniform_refine();
    }

    TH4(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv, kv, kv)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
      std::get<3>(*this).uniform_refine();
    }
    /// @}
  };
  
  /// @brief Alias for Nedelec like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename BSpline_t>
  class NE1 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)>>;
  public:
    /// @brief Constructor
    /// @{
    NE1(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine();
    }

    NE1(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1}^{p,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename BSpline_t>
  class NE2 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1)>>;
  public:
    /// @brief Constructor
    /// @{
    NE2(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine(1, 1);
      std::get<1>(*this).uniform_refine(1, 0);
    }

    NE2(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv)
    {
      std::get<0>(*this).uniform_refine(1, 1);
      std::get<1>(*this).uniform_refine(1, 0);
    }
    /// @}
  };   

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1}^{p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename BSpline_t>
  class NE3 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2)>>;
  public:
    /// @brief Constructor
    /// @{
    NE3(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1);
    }

    NE3(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv, kv)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1);
    }
    /// @}
  };

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename BSpline_t>
  class NE4 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2),
                                                       BSpline_t::degree(3)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2),
                                                   BSpline_t::degree(3)>>;
  public:
    /// @brief Constructor
    /// @{
    NE4(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 3);
      std::get<3>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 2);
    }

    NE4(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv, kv, kv)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 3);
      std::get<3>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 2);
    }
    /// @}
  };
  
  /// @brief Alias for Raviart-Thomas like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename BSpline_t>
  class RT1 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)>>;
  public:
    /// @brief Constructor
    /// @{
    RT1(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs)
    {
      std::get<0>(*this).uniform_refine();
    }

    RT1(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };
  
  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p}^{p,p-1} \otimes
  /// S_{p,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename BSpline_t>
  class RT2 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1)>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1)>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1)>>;
  public:
    /// @brief Constructor
    /// @{
    RT2(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs)
    {
    }

    RT2(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv)
    {
    }
    /// @}
  };

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p}^{p,p-1,p-1} \otimes
  /// S_{p,p+1,p}^{p-1,p,p-1} \otimes
  /// S_{p,p,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename BSpline_t>
  class RT3 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2)>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2)>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2)>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2)>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2)>>;
  public:
    /// @brief Constructor
    /// @{
    RT3(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs, ncoeffs)
    {
    }

    RT3(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv, kv)
    {
    }
    /// @}
  };

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p,p}^{p,p-1,p-1,p-1} \otimes
  /// S_{p,p+1,p,p}^{p-1,p,p-1,p-1} \otimes
  /// S_{p,p,p+1,p}^{p-1,p-1,p,p-1} \otimes
  /// S_{p,p,p,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename BSpline_t>
  class RT4 : public FunctionSpace<typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0)+1,
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2),
                                                       BSpline_t::degree(3)>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1)+1,
                                                       BSpline_t::degree(2),
                                                       BSpline_t::degree(3)>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2)+1,
                                                       BSpline_t::degree(3)>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2),
                                                       BSpline_t::degree(3)+1>,
                                   typename BSpline_t::template
                                   derived_self_type_t<typename BSpline_t::value_type,
                                                       BSpline_t::geoDim(),
                                                       BSpline_t::degree(0),
                                                       BSpline_t::degree(1),
                                                       BSpline_t::degree(2),
                                                       BSpline_t::degree(3)>>
  {
  private:
    using Base = FunctionSpace<typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0)+1,
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2),
                                                   BSpline_t::degree(3)>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1)+1,
                                                   BSpline_t::degree(2),
                                                   BSpline_t::degree(3)>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2)+1,
                                                   BSpline_t::degree(3)>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2),
                                                   BSpline_t::degree(3)+1>,
                               typename BSpline_t::template
                               derived_self_type_t<typename BSpline_t::value_type,
                                                   BSpline_t::geoDim(),
                                                   BSpline_t::degree(0),
                                                   BSpline_t::degree(1),
                                                   BSpline_t::degree(2),
                                                   BSpline_t::degree(3)>>;
  public:
    /// @brief Constructor
    /// @{
    RT4(const std::array<int64_t, BSpline_t::parDim()>& ncoeffs,
        enum init init = init::zeros)
      : Base(init, ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs)
    {
    }

    RT4(const std::array<std::vector<typename BSpline_t::value_type>,
                         BSpline_t::parDim()>& kv,
        enum init init = init::zeros)
      : Base(init, kv, kv, kv, kv, kv)
    {
    }
    /// @}
  };
  
} // namespace iganet
