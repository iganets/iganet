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
#include <zip.hpp>

namespace iganet {

#define TUPLE_WRAPPER(FunctionSpace)                                    \
  namespace detail {                                                    \
    template<typename T>                                                \
    struct tuple<FunctionSpace<T>>                                      \
    {                                                                   \
      using type = typename tuple<typename FunctionSpace<T>::Base>::type; \
    };                                                                  \
  }

  namespace detail {

    // Forward declaration
    template<typename... spline_t>
    class FunctionSpace;

    /// @brief Tuple wrapper
    /// @{
    template<typename... Ts>
    struct tuple
    {
      using type = std::tuple<Ts...>;
    };

    template<typename... Ts>
    struct tuple<std::tuple<Ts...>>
    {
      using type = typename tuple<Ts...>::type;
    };

    template<typename... Ts>
    struct tuple<FunctionSpace<Ts...>>
    {
      using type = typename tuple<Ts...>::type;
    };
    /// @}

    /// @brief Function space type dispatcher
    /// @{
    template<typename... Ts>
    struct FunctionSpace_dispatch;

    template<typename... Ts>
    struct FunctionSpace_dispatch<std::tuple<Ts...>> {
      using type = FunctionSpace<Ts...>;
    };
    /// @}

    /// @brief Function space type
    template<typename... Ts>
    using FunctionSpace_t =
      typename FunctionSpace_dispatch<decltype(std::tuple_cat(std::declval<typename tuple<Ts>::type>()...))>::type;

    /// @brief Tensor-product function space
    ///
    /// @note This class is not meant for direct use in
    /// applications. Instead use S1, S2, S3, S4, TH1, TH2, TH3, TH4,
    /// NE1, NE2, NE3, NE, RT1, RT2, RT3, or RT4.
    template<typename... spline_t>
    class FunctionSpace : public std::tuple<spline_t...>
    {
    protected:
      /// @brief Dimension of the parametric domain
      static constexpr std::array<short_t, sizeof...(spline_t)> parDim_ = {spline_t::parDim()...};

      /// @brief Dimension of the physical domain
      static constexpr std::array<short_t, sizeof...(spline_t)> geoDim_ = {spline_t::geoDim()...};

      /// @brief Type of the boundary spline objects
      using Boundary_t = std::tuple<Boundary<spline_t>...>;

      /// @brief Boundary spline objects
      Boundary_t boundary_;

    public:
      /// @brief Base class
      using Base = std::tuple<spline_t...>;

      /// @brief Value type
      using value_type = typename std::common_type<typename spline_t::value_type...>::type;

      /// @brief Default constructor
      FunctionSpace() = default;

      /// @brief Copy constructor
      FunctionSpace(const FunctionSpace&) = default;

      /// @brief Move constructor
      FunctionSpace(FunctionSpace&&) = default;

      /// @brief Constructor
      /// @{
      FunctionSpace(const std::array<int64_t, spline_t::parDim()>&... ncoeffs,
                    enum init init = init::zeros,
                    core<value_type> core = iganet::core<value_type>{})
        : Base({ncoeffs, init, core}...),
          boundary_({ncoeffs, init, core}...)
      {}

      FunctionSpace(const std::array<std::vector<typename spline_t::value_type>,
                    spline_t::parDim()>&... kv,
                    enum init init = init::zeros,
                    core<value_type> core = iganet::core<value_type>{})
        : Base({kv, init, core}...),
          boundary_({kv, init, core}...)
      {}      
      /// @}

      /// @brief Returns the parametric dimension
      inline static constexpr auto parDim()
      {
        return parDim_;
      }

      /// @brief Returns the geometric dimension
      inline static constexpr auto geoDim()
      {
        return geoDim_;
      }

      /// @brief Returns a constant reference to the boundary spline object
      inline const auto& boundary() const
      {
        return boundary_;
      }

      /// @brief Returns a non-constant reference to the boundary spline object
      inline auto& boundary()
      {
        return boundary_;
      }
      
    private:
      /// @brief Returns the dimension of all bases
      template<size_t... Is>
      int64_t basisDim_(std::index_sequence<Is...>) const
      {
        return (std::get<Is>(*this).ncumcoeffs() + ...);
      }

      /// @brief Returns the dimension of all bases at the boundary
      template<size_t... Is>
      int64_t boundaryBasisDim_(std::index_sequence<Is...>) const
      {
        return (std::get<Is>(boundary_).ncumcoeffs() + ...);
      }

      /// @brief Serialization to JSON
      template<size_t... Is>
      nlohmann::json to_json_(std::index_sequence<Is...>) const
      {
        auto json_this = nlohmann::json::array();
        auto json_boundary = nlohmann::json::array();
        (json_this.push_back(std::get<Is>(*this).to_json()), ...);
        (json_boundary.push_back(std::get<Is>(boundary_).to_json()), ...);

        auto json = nlohmann::json::array();
        for (auto [t,b] : zip(json_this, json_boundary)) {
          auto json_inner = nlohmann::json::array();
          json_inner.push_back(t);
          json_inner.push_back(b);
          json.push_back(json_inner);
        }        
        
        return json;
      }
      
    public:      
      /// @brief Returns the dimension of all bases
      int64_t basisDim() const
      {
        return basisDim_(std::make_index_sequence<sizeof...(spline_t)>{});
      }

      /// @brief Returns the dimension of all bases at the boundary
      int64_t boundaryBasisDim() const
      {
        return boundaryBasisDim_(std::make_index_sequence<sizeof...(spline_t)>{});
      }

      /// @brief Serialization to JSON
      nlohmann::json to_json() const
      {
        return to_json_(std::make_index_sequence<sizeof...(spline_t)>{});
      }
    };

    /// @brief Function space
    ///
    /// @note This class is not meant for direct use in
    /// applications. Instead use S1, S2, S3, S4, TH1, TH2, TH3, TH4,
    /// NE1, NE2, NE3, NE, RT1, RT2, RT3, or RT4.
    template<typename spline_t>
    class FunctionSpace<spline_t> : public spline_t
    {
    protected:
      /// @brief Dimension of the parametric domain
      static constexpr short_t parDim_ = spline_t::parDim();

      /// @brief Dimension of the physical domain
      static constexpr short_t geoDim_ = spline_t::geoDim();

      /// @brief Type of the boundary spline objects
      using Boundary_t = Boundary<spline_t>;

      /// @brief Boundary spline objects
      Boundary_t boundary_;

    public:
      /// @brief Base class
      using Base = spline_t;

      /// @brief Value type
      using value_type = typename spline_t::value_type;

      /// @brief Default constructor
      FunctionSpace() = default;

      /// @brief Copy constructor
      FunctionSpace(const FunctionSpace&) = default;

      /// @brief Move constructor
      FunctionSpace(FunctionSpace&&) = default;

      /// @brief Constructor
      /// @{
      FunctionSpace(const std::array<int64_t, parDim_>& ncoeffs,
                    enum init init = init::zeros,
                    core<value_type> core = iganet::core<value_type>{})
        : Base(ncoeffs, init, core),
          boundary_(ncoeffs, init)
      {}

      FunctionSpace(std::array<std::vector<typename spline_t::value_type>,
                    spline_t::parDim()> kv,
                    enum init init = init::zeros,
                    core<value_type> core = iganet::core<value_type>{})
        : Base(kv, init, core),
          boundary_(kv, init, core)
      {}
      /// @}

      /// @brief Returns the parametric dimension
      inline static constexpr short_t parDim()
      {
        return parDim_;
      }

      /// @brief Returns the geometric dimension
      inline static constexpr short_t geoDim()
      {
        return geoDim_;
      }

      /// @brief Returns a constant reference to the boundary spline object
      inline const auto& boundary() const
      {
        return boundary_;
      }

      /// @brief Returns a non-constant reference to the boundary spline object
      inline auto& boundary()
      {
        return boundary_;
      }

      /// @brief Returns the dimension of the basis at the boundary
      int64_t boundaryBasisDim() const
      {
        return boundary_.ncumcoeffs();
      }

      /// @brief Returns the dimension of the basis
      int64_t basisDim() const
      {
        return spline_t::ncumcoeffs();
      }

      /// @brief Serialization to JSON
      nlohmann::json to_json() const override
      {
        auto json = nlohmann::json::array();
        json.push_back(Base::to_json());
        json.push_back(boundary_.to_json());
        return json;
      }
    };
  } // namespace detail

  template<typename... Ts>
  using FunctionSpace = detail::FunctionSpace_t<Ts...>;

  /// @brief Spline function space \f$ S_{p}^{p-1} \f$
  template<typename spline_t, short_t... Cs>
  class S1
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0)>>;

    /// @brief Constructor
    /// @{
    S1() = default;
    S1(S1&&) = default;
    S1(const S1&) = default;

    S1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, init, core)
    {
      if constexpr (sizeof...(Cs) == 1) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S1(std::array<std::vector<typename spline_t::value_type>,
                 spline_t::parDim()> kv,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, init, core)
    {
      if constexpr (sizeof...(Cs) == 1) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  /// @}
  };

  TUPLE_WRAPPER(S1);

  template<typename spline_t, short_t... Cs>
  class S2
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0),
                                               spline_t::degree(1)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0),
                                                   spline_t::degree(1)>>;

    /// @brief Constructor
    /// @{
    S2() = default;
    S2(S2&&) = default;
    S2(const S2&) = default;

    S2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, init, core)
    {
      if constexpr (sizeof...(Cs) == 2) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S2(std::array<std::vector<typename spline_t::value_type>,
                 spline_t::parDim()> kv,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, init, core)
    {
      if constexpr (sizeof...(Cs) == 2) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
    /// @brief Returns the parametric dimension
    ///
    /// @result Number of parametric dimensions
    inline static constexpr short_t parDim()
    {
      return spline_t::parDim();
    }

    /// @brief Returns the geometric dimension
    ///
    /// @result Number of geometric dimensions
    inline static constexpr short_t geoDim()
    {
      return spline_t::geoDim();
    }

  /// @}
  };

  TUPLE_WRAPPER(S2);

  template<typename spline_t, short_t... Cs>
  class S3
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0),
                                               spline_t::degree(1),
                                               spline_t::degree(2)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0),
                                                   spline_t::degree(1),
                                                   spline_t::degree(2)>>;

    /// @brief Constructor
    /// @{
    S3() = default;
    S3(S3&&) = default;
    S3(const S3&) = default;

    S3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, init, core)
    {
      if constexpr (sizeof...(Cs) == 3) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          Base::uniform_refine(spline_t::degree(2)-1-c, 2);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S3(std::array<std::vector<typename spline_t::value_type>,
                 spline_t::parDim()> kv,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, init, core)
    {
      if constexpr (sizeof...(Cs) == 3) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          Base::uniform_refine(spline_t::degree(2)-1-c, 2);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  /// @}
  };

  TUPLE_WRAPPER(S3);

  template<typename spline_t, short_t... Cs>
  class S4
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0),
                                               spline_t::degree(1),
                                               spline_t::degree(2),
                                               spline_t::degree(3)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0),
                                                   spline_t::degree(1),
                                                   spline_t::degree(2),
                                                   spline_t::degree(3)>>;

    /// @brief Constructor
    /// @{
    S4() = default;
    S4(S4&&) = default;
    S4(const S4&) = default;

    S4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, init, core)
    {
      if constexpr (sizeof...(Cs) == 4) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          Base::uniform_refine(spline_t::degree(2)-1-c, 2);
        if constexpr (const auto c = std::get<3>(std::make_tuple(Cs...));
                      c != spline_t::degree(3)-1)
          Base::uniform_refine(spline_t::degree(3)-1-c, 3);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S4(std::array<std::vector<typename spline_t::value_type>,
                       spline_t::parDim()> kv,
       enum init init = init::zeros,
       core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, init, core)
    {
      if constexpr (sizeof...(Cs) == 4) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          std::get<0>(*this).uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          std::get<1>(*this).uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          std::get<2>(*this).uniform_refine(spline_t::degree(2)-1-c, 2);
        if constexpr (const auto c = std::get<3>(std::make_tuple(Cs...));
                      c != spline_t::degree(3)-1)
          std::get<3>(*this).uniform_refine(spline_t::degree(3)-1-c, 3);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
  /// @}
  };

  TUPLE_WRAPPER(S4);

  /// @brief Taylor-Hood like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename spline_t>
  class TH1 : public FunctionSpace<S1<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                      spline_t::degree(0)+1>>,
                                   S1<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1>>,
                               S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)>>>;

    /// @brief Constructor
    /// @{
    TH1() = default;
    TH1(TH1&&) = default;
    TH1(const TH1&) = default;

    TH1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine();
    }
    
    TH1(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH1);

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename spline_t>
  class TH2 : public FunctionSpace<S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1>>,
                                   S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1>>,
                                   S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)>>>;

    /// @brief Constructor
    /// @{
    TH2() = default;
    TH2(TH2&&) = default;
    TH2(const TH2&) = default;

    TH2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
    }

    TH2(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH2);

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename spline_t>
  class TH3 : public FunctionSpace<S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>>;

    /// @brief Constructor
    /// @{
    TH3() = default;
    TH3(TH3&&) = default;
    TH3(const TH3&) = default;

    TH3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
    }

    TH3(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH3);

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename spline_t>
  class TH4 : public FunctionSpace<S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2),
                                                          spline_t::degree(3)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>>;

    /// @brief Constructor
    /// @{
    TH4() = default;
    TH4(TH4&&) = default;
    TH4(const TH4&) = default;

    TH4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
      std::get<3>(*this).uniform_refine();
    }

    TH4(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
      std::get<3>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH4);

  /// @brief Alias for Nedelec like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename spline_t>
  class NE1 : public FunctionSpace<S1<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1>>,
                                   S1<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1>>,
                               S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)>>>;

    /// @brief Constructor
    /// @{
    NE1() = default;
    NE1(NE1&&) = default;
    NE1(const NE1&) = default;

    NE1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine();
    }

    NE1(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(NE1);

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1}^{p,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename spline_t>
  class NE2 : public FunctionSpace<S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1>>,
                                   S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1>>,
                                   S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)>>>;

    /// @brief Constructor
    /// @{
    NE2() = default;
    NE2(NE2&&) = default;
    NE2(const NE2&) = default;

    NE2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine(1, 1);
      std::get<1>(*this).uniform_refine(1, 0);
    }

    NE2(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine(1, 1);
      std::get<1>(*this).uniform_refine(1, 0);
    }
    /// @}
  };

  TUPLE_WRAPPER(NE2);

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1}^{p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename spline_t>
  class NE3 : public FunctionSpace<S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>>;

    /// @brief Constructor
    /// @{
    NE3() = default;
    NE3(NE3&&) = default;
    NE3(const NE3&) = default;

    NE3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1);
    }

    NE3(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1);
    }
    /// @}
  };

  TUPLE_WRAPPER(NE3);

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename spline_t>
  class NE4 : public FunctionSpace<S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2),
                                                          spline_t::degree(3)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>>;

    /// @brief Constructor
    /// @{
    NE4() = default;
    NE4(NE4&&) = default;
    NE4(const NE4&) = default;

    NE4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 3);
      std::get<3>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 2);
    }

    NE4(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 3);
      std::get<3>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 2);
    }
    /// @}
  };

  TUPLE_WRAPPER(NE4);

  /// @brief Alias for Raviart-Thomas like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename spline_t>
  class RT1 : public FunctionSpace<S1<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1>>,
                                   S1<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1>>,
                               S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)>>>;

    /// @brief Constructor
    /// @{
    RT1() = default;
    RT1(RT1&&) = default;
    RT1(const RT1&) = default;

    RT1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, init, core)
    {
      std::get<0>(*this).uniform_refine();
    }

    RT1(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, init, core)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(RT1);

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p}^{p,p-1} \otimes
  /// S_{p,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename spline_t>
  class RT2 : public FunctionSpace<S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1)>>,
                                   S2<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1)+1>>,
                                   S2<typename spline_t::template
                                   derived_self_type_t<typename spline_t::value_type,
                                                       spline_t::geoDim(),
                                                       spline_t::degree(0),
                                                       spline_t::degree(1)>>>
  {
  public:
    using Base = FunctionSpace<S2<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0)+1,
                                  spline_t::degree(1)>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)>>>;

    /// @brief Constructor
    /// @{
    RT2() = default;
    RT2(RT2&&) = default;
    RT2(const RT2&) = default;

    RT2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, init, core)
    {
    }

    RT2(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, init, core)
    {
    }
    /// @}
  };

  TUPLE_WRAPPER(RT2);

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p}^{p,p-1,p-1} \otimes
  /// S_{p,p+1,p}^{p-1,p,p-1} \otimes
  /// S_{p,p,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename spline_t>
  class RT3 : public FunctionSpace<S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1),
                                                          spline_t::degree(2)>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2)>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2)+1>>,
                                   S3<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>>;

    /// @brief Constructor
    /// @{
    RT3() = default;
    RT3(RT3&&) = default;
    RT3(const RT3&) = default;

    RT3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, core)
    {
    }

    RT3(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, init, core)
    {
    }
    /// @}
  };

  TUPLE_WRAPPER(RT3);

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p,p}^{p,p-1,p-1,p-1} \otimes
  /// S_{p,p+1,p,p}^{p-1,p,p-1,p-1} \otimes
  /// S_{p,p,p+1,p}^{p-1,p-1,p,p-1} \otimes
  /// S_{p,p,p,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename spline_t>
  class RT4 : public FunctionSpace<S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0)+1,
                                                          spline_t::degree(1),
                                                          spline_t::degree(2),
                                                          spline_t::degree(3)>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1)+1,
                                                          spline_t::degree(2),
                                                          spline_t::degree(3)>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2)+1,
                                                          spline_t::degree(3)>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2),
                                                          spline_t::degree(3)+1>>,
                                   S4<typename spline_t::template
                                      derived_self_type_t<typename spline_t::value_type,
                                                          spline_t::geoDim(),
                                                          spline_t::degree(0),
                                                          spline_t::degree(1),
                                                          spline_t::degree(2),
                                                          spline_t::degree(3)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>>;

    /// @brief Constructor
    /// @{
    RT4() = default;
    RT4(RT4&&) = default;
    RT4(const RT4&) = default;

    RT4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, core)
    {
    }

    RT4(const std::array<std::vector<typename spline_t::value_type>,
                         spline_t::parDim()>& kv,
        enum init init = init::zeros,
        core<typename spline_t::value_type> core = iganet::core<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, kv, init, core)
    {
    }
    /// @}
  };

  TUPLE_WRAPPER(RT4);

} // namespace iganet
