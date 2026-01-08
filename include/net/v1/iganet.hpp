/**
   @file net/v1/iganet.hpp

   @brief Isogeometric analysis networks (deprecated V1)

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core/options.hpp>
#include <net/generator.hpp>
#include <net/collocation.hpp>
#include <net/optimizer.hpp>
#include <splines/boundary.hpp>
#include <splines/functionspace.hpp>
#include <utils/container.hpp>
#include <utils/fqn.hpp>
#include <utils/tuple.hpp>
#include <utils/zip.hpp>

namespace iganet::v1 {

  /// @brief IgANetOptions
  struct IgANetOptions {
  TORCH_ARG(int64_t, max_epoch) = 100;
  TORCH_ARG(int64_t, batch_size) = 1000;
  TORCH_ARG(double, min_loss) = 1e-4;
  TORCH_ARG(double, min_loss_change) = 0;
  TORCH_ARG(double, min_loss_rel_change) = 1e-3;
  };

/// @brief IgA base class (no reference data)
///
/// This class implements the base functionality of IgANets for the
/// case that no reference solution is required
template <typename GeometryMap, typename Variable>
  requires FunctionSpaceType<GeometryMap> && FunctionSpaceType<Variable>
class [[deprecated("Use novel IgANet implementation")]] IgABaseNoRefData {
public:
  /// @brief Value type
  using value_type = std::common_type_t<typename GeometryMap::value_type,
                                        typename Variable::value_type>;

  /// @brief Type of the geometry map function space(s)
  using geometryMap_type = GeometryMap;

  /// @brief Type of the variable function space(s)
  using variable_type = Variable;

  /// @brief Type of the geometry map collocation points
  using geometryMap_collPts_type =
      std::pair<typename GeometryMap::eval_type,
                typename GeometryMap::boundary_eval_type>;

  /// @brief Type of the variable collocation points
  using variable_collPts_type =
      std::pair<typename Variable::eval_type,
                typename Variable::boundary_eval_type>;

  /// @brief Indicates whether this class provides a geometry map
  bool static constexpr has_GeometryMap = true;

  /// @brief Indicates whether this class provides reference data
  bool static constexpr has_RefData = false;

  /// @brief Indicates whether this class provides a solution
  bool static constexpr has_Solution = true;

protected:
  /// @brief Spline representation of the geometry map
  GeometryMap G_;

  /// @brief Spline representation of the solution
  Variable u_;

private:
  /// @brief Constructor: number of spline coefficients (different for Geometry
  /// and Variable types)
  template <std::size_t... GeometryMapNumCoeffs, std::size_t... Is,
            std::size_t... VariableNumCoeffs, std::size_t... Js>
  IgABaseNoRefData(
      std::tuple<std::array<int64_t, GeometryMapNumCoeffs>...>
          geometryMapNumCoeffs,
      std::index_sequence<Is...>,
      std::tuple<std::array<int64_t, VariableNumCoeffs>...> variableNumCoeffs,
      std::index_sequence<Js...>,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : // Construct the different spline objects individually
        G_(std::get<Is>(geometryMapNumCoeffs)..., init::greville, options),
        u_(std::get<Js>(variableNumCoeffs)..., init::random, options) {}

public:
  /// @brief Default constructor
  explicit IgABaseNoRefData(
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : G_(), u_() {}

  /// @brief Constructor: number of spline coefficients (same for geometry map
  /// and variables)
  /// @{
  template <std::size_t NumCoeffs>
  explicit IgABaseNoRefData(
      std::array<int64_t, NumCoeffs> numCoeffs,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABaseNoRefData(std::tuple{numCoeffs}, std::tuple{numCoeffs},
                         options) {}

  template <std::size_t... NumCoeffs>
  explicit IgABaseNoRefData(
      std::tuple<std::array<int64_t, NumCoeffs>...> numCoeffs,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABaseNoRefData(numCoeffs, numCoeffs, options) {}
  /// @}

  /// @brief Constructor: number of spline coefficients (different for
  /// geometry map and variables)
  /// @{
  template <std::size_t GeometryMapNumCoeffs, std::size_t VariableNumCoeffs>
  IgABaseNoRefData(
      std::array<int64_t, GeometryMapNumCoeffs> geometryMapNumCoeffs,
      std::array<int64_t, VariableNumCoeffs> variableNumCoeffs,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABaseNoRefData(std::tuple{geometryMapNumCoeffs},
                         std::tuple{variableNumCoeffs}, options) {}

  template <std::size_t... GeometryMapNumCoeffs,
            std::size_t... VariableNumCoeffs>
  IgABaseNoRefData(
      std::tuple<std::array<int64_t, GeometryMapNumCoeffs>...>
          geometryMapNumCoeffs,
      std::tuple<std::array<int64_t, VariableNumCoeffs>...> variableNumCoeffs,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABaseNoRefData(
            geometryMapNumCoeffs,
            std::make_index_sequence<sizeof...(GeometryMapNumCoeffs)>{},
            variableNumCoeffs,
            std::make_index_sequence<sizeof...(VariableNumCoeffs)>{}, options) {
  }
  /// @}

  /// @brief Destructor
  virtual ~IgABaseNoRefData() = default;

  /// @brief Returns a constant reference to the spline
  /// representation of the geometry map
  inline const GeometryMap &G() const { return G_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the geometry map
  inline GeometryMap &G() { return G_; }

  /// @brief Returns a constant reference to the spline
  /// representation of the solution
  inline const Variable &u() const { return u_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the solution
  inline Variable &u() { return u_; }

private:
  /// @brief Returns the geometry map collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <std::size_t... Is>
  geometryMap_collPts_type
  geometryMap_collPts(enum collPts collPtsType,
                      std::index_sequence<Is...>) const {
    geometryMap_collPts_type collPts;

    switch (collPtsType) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = G_.template boundary<Is>().greville()),
       ...);
      break;

    case collPts::greville_interior:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = G_.template boundary<Is>().greville()),
       ...);
      break;

    case collPts::greville_ref1:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().clone().uniform_refine().greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
            G_.template boundary<Is>().clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_interior_ref1:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().clone().uniform_refine().greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
            G_.template boundary<Is>().clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_ref2:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().clone().uniform_refine(2, -1).greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = G_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(2, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_interior_ref2:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().clone().uniform_refine(2, -1).greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = G_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(2, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_ref3:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().clone().uniform_refine(3, -1).greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = G_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(3, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_interior_ref3:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts.first) =
            G_.template space<Is>().clone().uniform_refine(3, -1).greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = G_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(3, -1)
                                           .greville()),
       ...);
      break;      

    default:
      throw std::runtime_error("Invalid collocation point specifier");
    }

    return collPts;
  }

  /// @brief Returns the variable collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <std::size_t... Is>
  variable_collPts_type variable_collPts(enum collPts collPtsType,
                                         std::index_sequence<Is...>) const {
    variable_collPts_type collPts;

    switch (collPtsType) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = u_.template boundary<Is>().greville()),
       ...);
      break;

    case collPts::greville_interior:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = u_.template boundary<Is>().greville()),
       ...);
      break;

    case collPts::greville_ref1:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().clone().uniform_refine().greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
            u_.template boundary<Is>().clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_interior_ref1:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().clone().uniform_refine().greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) =
            u_.template boundary<Is>().clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_ref2:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().clone().uniform_refine(2, -1).greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = u_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(2, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_interior_ref2:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().clone().uniform_refine(2, -1).greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = u_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(2, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_ref3:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().clone().uniform_refine(3, -1).greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = u_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(3, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_interior_ref3:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
            u_.template space<Is>().clone().uniform_refine(3, -1).greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = u_.template boundary<Is>()
                                           .clone()
                                           .uniform_refine(3, -1)
                                           .greville()),
       ...);
      break;      

    default:
      throw std::runtime_error("Invalid collocation point specifier");
    }

    return collPts;
  }

public:
  /// @brief Returns the geometry map collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  virtual geometryMap_collPts_type
  geometryMap_collPts(enum collPts collPts) const {
    if constexpr (GeometryMap::nspaces() == 1)

      switch (collPts) {

      case collPts::greville:
        return {G_.space().greville(/* interior */ false),
                G_.boundary().greville()};

      case collPts::greville_interior:
        return {G_.space().greville(/* interior */ true),
                G_.boundary().greville()};

      case collPts::greville_ref1:
        return {
            G_.space().clone().uniform_refine().greville(/* interior */ false),
            G_.boundary().clone().uniform_refine().greville()};

      case collPts::greville_interior_ref1:
        return {
            G_.space().clone().uniform_refine().greville(/* interior */ true),
            G_.boundary().clone().uniform_refine().greville()};

      case collPts::greville_ref2:
        return {G_.space().clone().uniform_refine(2, -1).greville(
                    /* interior */ false),
                G_.boundary().clone().uniform_refine(2, -1).greville()};

      case collPts::greville_interior_ref2:
        return {G_.space().clone().uniform_refine(2, -1).greville(
                    /* interior */ true),
                G_.boundary().clone().uniform_refine(2, -1).greville()};

      case collPts::greville_ref3:
        return {G_.space().clone().uniform_refine(3, -1).greville(
                    /* interior */ false),
                G_.boundary().clone().uniform_refine(3, -1).greville()};

      case collPts::greville_interior_ref3:
        return {G_.space().clone().uniform_refine(3, -1).greville(
                    /* interior */ true),
                G_.boundary().clone().uniform_refine(3, -1).greville()};

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }

    else
      return geometryMap_collPts(
          collPts, std::make_index_sequence<GeometryMap::nspaces()>{});
  }

  /// @brief Returns the variable collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  virtual variable_collPts_type variable_collPts(enum collPts collPts) const {
    if constexpr (Variable::nspaces() == 1)

      switch (collPts) {

      case collPts::greville:
        return {u_.space().greville(/* interior */ false),
                u_.boundary().greville()};

      case collPts::greville_interior:
        return {u_.space().greville(/* interior */ true),
                u_.boundary().greville()};

      case collPts::greville_ref1:
        return {
            u_.space().clone().uniform_refine().greville(/* interior */ false),
            u_.boundary().clone().uniform_refine().greville()};

      case collPts::greville_interior_ref1:
        return {
            u_.space().clone().uniform_refine().greville(/* interior */ true),
            u_.boundary().clone().uniform_refine().greville()};

      case collPts::greville_ref2:
        return {u_.space().clone().uniform_refine(2, -1).greville(
                    /* interior */ false),
                u_.boundary().clone().uniform_refine(2, -1).greville()};

      case collPts::greville_interior_ref2:
        return {u_.space().clone().uniform_refine(2, -1).greville(
                    /* interior */ true),
                u_.boundary().clone().uniform_refine(2, -1).greville()};

      case collPts::greville_ref3:
        return {u_.space().clone().uniform_refine(3, -1).greville(
                    /* interior */ false),
                u_.boundary().clone().uniform_refine(3, -1).greville()};

      case collPts::greville_interior_ref3:
        return {u_.space().clone().uniform_refine(3, -1).greville(
                    /* interior */ true),
                u_.boundary().clone().uniform_refine(3, -1).greville()};        

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }

    else
      return variable_collPts(collPts,
                              std::make_index_sequence<Variable::nspaces()>{});
  }
};

/// @brief IgA base class
///
/// This class implements the base functionality of IgANets
template <typename GeometryMap, typename Variable>
  requires FunctionSpaceType<GeometryMap> && FunctionSpaceType<Variable>
class [[deprecated("Use novel IgANet implementation")]] IgABase : public IgABaseNoRefData<GeometryMap, Variable> {
public:
  /// @brief Base type
  using Base = IgABaseNoRefData<GeometryMap, Variable>;

  /// @brief Value type
  using value_type = Base::value_type;

  /// @brief Type of the geometry map function space(s)
  using geometryMap_type = GeometryMap;

  /// @brief Type of the variable function space(s)
  using variable_type = Variable;

  /// @brief Type of the geometry map collocation points
  using geometryMap_collPts_type = Base::geometryMap_collPts_type;

  /// @brief Type of the variable collocation points
  using variable_collPts_type = Base::variable_collPts_type;

  /// @brief Indicates whether this class provides a geometry map
  bool static constexpr has_GeometryMap = true;

  /// @brief Indicates whether this class provides a reference solution
  bool static constexpr has_RefData = true;

  /// @brief Indicates whether this class provides a solution
  bool static constexpr has_Solution = true;

protected:
  /// @brief Spline representation of the reference data
  Variable f_;

private:
  /// @brief Constructor: number of spline coefficients (different for Geometry
  /// and Variable types)
  template <std::size_t... GeometryMapNumCoeffs, std::size_t... Is,
            std::size_t... VariableNumCoeffs, std::size_t... Js>
  IgABase(
      std::tuple<std::array<int64_t, GeometryMapNumCoeffs>...>
          geometryMapNumCoeffs,
      std::index_sequence<Is...>,
      std::tuple<std::array<int64_t, VariableNumCoeffs>...> variableNumCoeffs,
      std::index_sequence<Js...>,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : // Construct the different spline objects individually
        Base(geometryMapNumCoeffs, variableNumCoeffs, options),
        f_(std::get<Js>(variableNumCoeffs)..., init::zeros, options) {}

public:
  /// @brief Default constructor
  explicit IgABase(
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : Base(), f_() {}

  /// @brief Constructor: number of spline coefficients (same for geometry map
  /// and variables)
  /// @{
  template <std::size_t NumCoeffs>
  explicit IgABase(
      std::array<int64_t, NumCoeffs> numCoeffs,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(std::tuple{numCoeffs}, std::tuple{numCoeffs}, options) {}

  template <std::size_t... NumCoeffs>
  explicit IgABase(
      std::tuple<std::array<int64_t, NumCoeffs>...> numCoeffs,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(numCoeffs, numCoeffs, options) {}
  /// @}

  /// @brief Constructor: number of spline coefficients (different for
  /// geometry map and variables)
  /// @{
  template <std::size_t GeometryMapNumCoeffs, std::size_t VariableNumCoeffs>
  IgABase(std::array<int64_t, GeometryMapNumCoeffs> geometryMapNumCoeffs,
          std::array<int64_t, VariableNumCoeffs> variableNumCoeffs,
          iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(std::tuple{geometryMapNumCoeffs}, std::tuple{variableNumCoeffs},
                options) {}

  template <std::size_t... GeometryMapNumCoeffs,
            std::size_t... VariableNumCoeffs>
  IgABase(
      std::tuple<std::array<int64_t, GeometryMapNumCoeffs>...>
          geometryMapNumCoeffs,
      std::tuple<std::array<int64_t, VariableNumCoeffs>...> variableNumCoeffs,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(geometryMapNumCoeffs,
                std::make_index_sequence<sizeof...(GeometryMapNumCoeffs)>{},
                variableNumCoeffs,
                std::make_index_sequence<sizeof...(VariableNumCoeffs)>{},
                options) {}
  /// @}

  /// @brief Returns a constant reference to the spline
  /// representation of the reference data
  inline const Variable &f() const { return f_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the reference data
  inline Variable &f() { return f_; }
};
  
  /// @brief IgANet
///
/// This class implements the core functionality of IgANets
template <typename Optimizer, typename GeometryMap, typename Variable,
          template <typename, typename> typename IgABase = IgABase>
  requires OptimizerType<Optimizer> && FunctionSpaceType<GeometryMap> &&
               FunctionSpaceType<Variable>
class [[deprecated("Use novel IgANet implementation")]] IgANet : public IgABase<GeometryMap, Variable>,
               utils::Serializable,
               private utils::FullQualifiedName {
public:
  /// @brief Base type
  using Base = IgABase<GeometryMap, Variable>;

  /// @brief Type of the optimizer
  using optimizer_type = Optimizer;

  /// @brief Type of the optimizer options
  using optimizer_options_type = optimizer_options_type<Optimizer>::type;

protected:
  /// @brief IgANet generator
  IgANetGenerator<typename Base::value_type> net_;

  /// @brief Optimizer
  std::unique_ptr<optimizer_type> opt_;

  /// @brief Options
  IgANetOptions options_;

public:
  /// @brief Default constructor
  explicit IgANet(const IgANetOptions &defaults = {},
                  iganet::Options<typename Base::value_type> options =
                      iganet::Options<typename Base::value_type>{})
      : // Construct the base class
        Base(),
        // Construct the optimizer
        opt_(std::make_unique<optimizer_type>(net_->parameters())),
        // Set options
        options_(defaults) {}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (same for geometry map and
  /// variables)
  /// @{
  template <std::size_t NumCoeffs>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::array<int64_t, NumCoeffs> numCoeffs, IgANetOptions defaults = {},
         iganet::Options<typename Base::value_type> options =
             iganet::Options<typename Base::value_type>{})
      : IgANet(layers, activations, std::tuple{numCoeffs},
               std::tuple{numCoeffs}, defaults, options) {}

  template <std::size_t... NumCoeffs>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::tuple<std::array<int64_t, NumCoeffs>...> numCoeffs,
         IgANetOptions defaults = {},
         iganet::Options<typename Base::value_type> options =
             iganet::Options<typename Base::value_type>{})
      : IgANet(layers, activations, numCoeffs, numCoeffs, defaults, options) {}
  /// @}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (different for geometry map and
  /// variables)
  /// @{
  template <std::size_t GeometryMapNumCoeffs, std::size_t VariableNumCoeffs>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::array<int64_t, GeometryMapNumCoeffs> geometryMapNumCoeffs,
         std::array<int64_t, VariableNumCoeffs> variableNumCoeffs,
         IgANetOptions defaults = {},
         iganet::Options<typename Base::value_type> options =
             iganet::Options<typename Base::value_type>{})
      : IgANet(layers, activations, std::tuple{geometryMapNumCoeffs},
               std::tuple{variableNumCoeffs}, defaults, options) {}

  template <std::size_t... GeometryMapNumCoeffs,
            std::size_t... VariableNumCoeffs>
  IgANet(
      const std::vector<int64_t> &layers,
      const std::vector<std::vector<std::any>> &activations,
      std::tuple<std::array<int64_t, GeometryMapNumCoeffs>...>
          geometryMapNumCoeffs,
      std::tuple<std::array<int64_t, VariableNumCoeffs>...> variableNumCoeffs,
      IgANetOptions defaults = {},
      iganet::Options<typename Base::value_type> options =
          iganet::Options<typename Base::value_type>{})
      : // Construct the base class
        Base(geometryMapNumCoeffs, variableNumCoeffs, options),
        // Construct the deep neural network
        net_(utils::concat(std::vector<int64_t>{inputs(/* epoch */ 0).size(0)},
                           layers,
                           std::vector<int64_t>{Base::u_.as_tensor_size()}),
             activations, options),

        // Construct the optimizer
        opt_(std::make_unique<optimizer_type>(net_->parameters())),

        // Set options
        options_(defaults) {}

  /// @brief Returns a constant reference to the IgANet generator
  inline const IgANetGenerator<typename Base::value_type> &net() const {
    return net_;
  }

  /// @brief Returns a non-constant reference to the IgANet generator
  inline IgANetGenerator<typename Base::value_type> &net() { return net_; }

  /// @brief Returns a constant reference to the optimizer
  inline const optimizer_type &optimizer() const { return *opt_; }

  /// @brief Returns a non-constant reference to the optimizer
  inline optimizer_type &optimizer() { return *opt_; }

  /// @brief Resets the optimizer
  ///
  /// @param[in] resetOptions Flag to indicate whether the optimizer options
  /// should be resetted
  inline void optimizerReset(bool resetOptions = true) {
    if (resetOptions)
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
    else {
      std::vector<optimizer_options_type> options;
      for (auto &group : opt_->param_groups())
        options.push_back(
            static_cast<optimizer_options_type &>(group.options()));
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
      for (auto [group, options] : utils::zip(opt_->param_groups(), options))
        static_cast<optimizer_options_type &>(group.options()) = options;
    }
  }

  /// @brief Resets the optimizer
  inline void optimizerReset(const optimizer_options_type &optimizerOptions) {
    opt_ =
        std::make_unique<optimizer_type>(net_->parameters(), optimizerOptions);
  }

  /// @brief Returns a non-constant reference to the optimizer options
  inline optimizer_options_type &optimizerOptions(std::size_t param_group = 0) {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type &>(
          opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Returns a constant reference to the optimizer options
  inline const optimizer_options_type &
  optimizerOptions(std::size_t param_group = 0) const {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type &>(
          opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type &options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type &>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type &&options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type &>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type &options,
                                    std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type &>(opt_->param_group().options()) =
          options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type &&options,
                                    std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type &>(opt_->param_group().options()) =
          options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Returns a constant reference to the options structure
  inline const auto &options() const { return options_; }

  /// @brief Returns a non-constant reference to the options structure
  inline auto &options() { return options_; }

  /// @brief Returns the network inputs
  ///
  /// In the default implementation the inputs are the controll
  /// points of the geometry and the reference spline objects. This
  /// behavior can be changed by overriding this virtual function in
  /// a derived class.
  virtual torch::Tensor inputs(int64_t epoch) const {
    if constexpr (Base::has_GeometryMap && Base::has_RefData)
      return torch::cat({Base::G_.as_tensor(), Base::f_.as_tensor()});
    else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
      return Base::G_.as_tensor();
    else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
      return Base::f_.as_tensor();
    else
      return torch::empty({0});
  }

  /// @brief Initializes epoch
  virtual bool epoch(int64_t) = 0;

  /// @brief Computes the loss function
  virtual torch::Tensor loss(const torch::Tensor &, int64_t) = 0;

  /// @brief Trains the IgANet
  virtual void train(
#ifdef IGANET_WITH_MPI
      c10::intrusive_ptr<c10d::ProcessGroupMPI> pg =
          c10d::ProcessGroupMPI::createProcessGroupMPI()
#endif
  ) {
    torch::Tensor inputs, outputs, loss;
    typename Base::value_type previous_loss(-1.0);

    // Loop over epochs
    for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch) {

      // Update epoch and inputs
      if (this->epoch(epoch))
        inputs = this->inputs(epoch);

      auto closure = [&]() {
        // Reset gradients
        net_->zero_grad();

        // Execute the model on the inputs
        outputs = net_->forward(inputs);

        // Compute the loss value
        loss = this->loss(outputs, epoch);

        // Compute gradients of the loss w.r.t. the model parameters
        loss.backward({}, true, false);

        return loss;
      };

#ifdef IGANET_WITH_MPI
      // Averaging the gradients of the parameters in all the processors
      // Note: This may lag behind DistributedDataParallel (DDP) in performance
      // since this synchronizes parameters after backward pass while DDP
      // overlaps synchronizing parameters and computing gradients in backward
      // pass
      std::vector<c10::intrusive_ptr<::c10d::Work>> works;
      for (auto &param : net_->named_parameters()) {
        std::vector<torch::Tensor> tmp = {param.value().grad()};
        works.emplace_back(pg->allreduce(tmp));
      }

      waitWork(pg, works);

      for (auto &param : net_->named_parameters()) {
        param.value().grad().data() =
            param.value().grad().data() / pg->getSize();
      }
#endif

      // Update the parameters based on the calculated gradients
      opt_->step(closure);

      typename Base::value_type current_loss =
          loss.item<typename Base::value_type>();
      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": "
                        << current_loss << std::endl;

      if (current_loss < options_.min_loss() ||
          std::abs(current_loss - previous_loss) < options_.min_loss_change() ||
          std::abs(current_loss - previous_loss) / current_loss <
              options_.min_loss_rel_change() ||
          loss.isnan().item<bool>()) {
        Log(log::info) << "Total epochs: " << epoch
                       << ", loss: " << current_loss << std::endl;
        return;
      }
      previous_loss = current_loss;
    }
    Log(log::info) << "Max epochs reached: " << options_.max_epoch()
                   << ", loss: " << previous_loss << std::endl;
  }

  /// @brief Trains the IgANet
  template <typename DataLoader>
  void train(DataLoader &loader
#ifdef IGANET_WITH_MPI
             ,
             c10::intrusive_ptr<c10d::ProcessGroupMPI> pg =
                 c10d::ProcessGroupMPI::createProcessGroupMPI()
#endif
  ) {
    torch::Tensor inputs, outputs, loss;
    typename Base::value_type previous_loss(-1.0);

    // Loop over epochs
    for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch) {

      typename Base::value_type current_loss(0);

      for (auto &batch : loader) {
        inputs = batch.data;

        if (inputs.dim() > 0) {
          if constexpr (Base::has_GeometryMap && Base::has_RefData) {
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).t());
            Base::f_.from_tensor(inputs
                                     .slice(1, Base::G_.as_tensor_size(),
                                            Base::G_.as_tensor_size() +
                                                Base::f_.as_tensor_size())
                                     .t());
          } else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).t());
          else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
            Base::f_.from_tensor(
                inputs.slice(1, 0, Base::f_.as_tensor_size()).t());

        } else {
          if constexpr (Base::has_GeometryMap && Base::has_RefData) {
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).flatten());
            Base::f_.from_tensor(inputs
                                     .slice(1, Base::G_.as_tensor_size(),
                                            Base::G_.as_tensor_size() +
                                                Base::f_.as_tensor_size())
                                     .flatten());
          } else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).flatten());
          else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
            Base::f_.from_tensor(
                inputs.slice(1, 0, Base::f_.as_tensor_size()).flatten());
        }

        this->epoch(epoch);

        auto closure = [&]() {
          // Reset gradients
          net_->zero_grad();

          // Execute the model on the inputs
          outputs = net_->forward(inputs);

          // Compute the loss value
          loss = this->loss(outputs, epoch);

          // Compute gradients of the loss w.r.t. the model parameters
          loss.backward({}, true, false);

          return loss;
        };

        // Update the parameters based on the calculated gradients
        opt_->step(closure);

        current_loss += loss.item<typename Base::value_type>();
      }
      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": "
                        << current_loss << std::endl;

      if (current_loss < options_.min_loss() ||
          std::abs(current_loss - previous_loss) < options_.min_loss_change() ||
          std::abs(current_loss - previous_loss) / current_loss <
              options_.min_loss_rel_change() ||
          loss.isnan().item<bool>()) {
        Log(log::info) << "Total epochs: " << epoch
                       << ", loss: " << current_loss << std::endl;
        return;
      }
      previous_loss = current_loss;
    }
    Log(log::info) << "Max epochs reached: " << options_.max_epoch()
                   << ", loss: " << previous_loss << std::endl;
  }

  /// @brief Evaluate IgANet
  void eval() {
    torch::Tensor inputs = this->inputs(0);
    torch::Tensor outputs = net_->forward(inputs);
    Base::u_.from_tensor(outputs);
  }

  /// @brief Returns the IgANet object as JSON object
  inline nlohmann::json to_json() const override {
    return "Not implemented yet";
  }

  /// @brief Returns a constant reference to the parameters of the IgANet object
  inline std::vector<torch::Tensor> parameters() const noexcept {
    return net_->parameters();
  }

  /// @brief Returns a constant reference to the named parameters of the IgANet
  /// object
  inline torch::OrderedDict<std::string, torch::Tensor>
  named_parameters() const noexcept {
    return net_->named_parameters();
  }

  /// @brief Returns the total number of parameters of the IgANet object
  inline std::size_t nparameters() const noexcept {
    std::size_t result = 0;
    for (const auto &param : this->parameters()) {
      result += param.numel();
    }
    return result;
  }

  /// @brief Registers a parameter
  torch::Tensor& register_parameter(std::string name, torch::Tensor tensor, bool requires_grad = true) {
    return net_->register_parameter(name, tensor, requires_grad);
  }

  /// @brief Returns a string representation of the IgANet object
  inline void pretty_print(std::ostream &os) const noexcept override {
    os << name() << "(\n"
       << "net = " << net_ << "\n";
    if constexpr (Base::has_GeometryMap)
      os << "G = " << Base::G_ << "\n";
    if constexpr (Base::has_RefData)
      os << "f = " << Base::f_ << "\n";
    if constexpr (Base::has_Solution)
      os << "u = " << Base::u_ << "\n)";
  }

  /// @brief Saves the IgANet to file
  inline void save(const std::string &filename,
                   const std::string &key = "iganet") const {
    torch::serialize::OutputArchive archive;
    write(archive, key).save_to(filename);
  }

  /// @brief Loads the IgANet from file
  inline void load(const std::string &filename,
                   const std::string &key = "iganet") {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    read(archive, key);
  }

  /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "iganet") const {
    if constexpr (Base::has_GeometryMap)
      Base::G_.write(archive, key + ".geo");
    if constexpr (Base::has_RefData)
      Base::f_.write(archive, key + ".ref");
    if constexpr (Base::has_Solution)
      Base::u_.write(archive, key + ".out");

    net_->write(archive, key + ".net");
    torch::serialize::OutputArchive archive_net;
    net_->save(archive_net);
    archive.write(key + ".net.data", archive_net);

    torch::serialize::OutputArchive archive_opt;
    opt_->save(archive_opt);
    archive.write(key + ".opt", archive_opt);

    return archive;
  }

  /// @brief Loads the IgANet from a torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "iganet") {
    if constexpr (Base::has_GeometryMap)
      Base::G_.read(archive, key + ".geo");
    if constexpr (Base::has_RefData)
      Base::f_.read(archive, key + ".ref");
    if constexpr (Base::has_Solution)
      Base::u_.read(archive, key + ".out");

    net_->read(archive, key + ".net");
    torch::serialize::InputArchive archive_net;
    archive.read(key + ".net.data", archive_net);
    net_->load(archive_net);

    opt_->add_parameters(net_->parameters());
    torch::serialize::InputArchive archive_opt;
    archive.read(key + ".opt", archive_opt);
    opt_->load(archive_opt);

    return archive;
  }

  /// @brief Returns true if both IgANet objects are the same
  bool operator==(const IgANet &other) const {
    bool result(true);

    if constexpr (Base::has_GeometryMap)
      result *= (Base::G_ == other.G());
    if constexpr (Base::has_RefData)
      result *= (Base::f_ == other.f());
    if constexpr (Base::has_Solution)
      result *= (Base::u_ == other.u());

    return result;
  }

  /// @brief Returns true if both IgANet objects are different
  bool operator!=(const IgANet &other) const { return *this != other; }

#ifdef IGANET_WITH_MPI
private:
  /// @brief Waits for all work processes
  static void waitWork(c10::intrusive_ptr<c10d::ProcessGroupMPI> pg,
                       std::vector<c10::intrusive_ptr<c10d::Work>> works) {
    for (auto &work : works) {
      try {
        work->wait();
      } catch (const std::exception &ex) {
        Log(log::error) << "Exception received during waitWork: " << ex.what()
                        << std::endl;
        pg->abort();
      }
    }
  }
#endif
};

/// @brief Print (as string) a IgANet object
template <typename Optimizer, typename GeometryMap, typename Variable>
  requires OptimizerType<Optimizer> && FunctionSpaceType<GeometryMap> &&
           FunctionSpaceType<Variable>
inline std::ostream &
operator<<(std::ostream &os,
           const IgANet<Optimizer, GeometryMap, Variable> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief IgANetCustomizable
///
/// This class implements a customizable variant of IgANets that
/// provides types and attributes for precomputing indices and basis
/// functions
template <typename GeometryMap, typename Variable>
  requires FunctionSpaceType<GeometryMap> && FunctionSpaceType<Variable>
class [[deprecated("Use novel IgANetCustomizable implementation")]] IgANetCustomizable {
public:
  /// @brief Type of the knot indices of the geometry map in the interior
  using geometryMap_interior_knot_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_knot_indices<functionspace::interior>(
                       std::declval<typename GeometryMap::eval_type>()));

  /// @brief Type of the knot indices of the geometry map at the boundary
  using geometryMap_boundary_knot_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_knot_indices<functionspace::boundary>(
                       std::declval<
                           typename GeometryMap::boundary_eval_type>()));

  /// @brief Type of the knot indices of the variables in the interior
  using variable_interior_knot_indices_type =
      decltype(std::declval<Variable>()
                   .template find_knot_indices<functionspace::interior>(
                       std::declval<typename Variable::eval_type>()));

  /// @brief Type of the knot indices of boundary_eval_type type at the boundary
  using variable_boundary_knot_indices_type =
      decltype(std::declval<Variable>()
                   .template find_knot_indices<functionspace::boundary>(
                       std::declval<typename Variable::boundary_eval_type>()));

  /// @brief Type of the coefficient indices of geometry type in the interior
  using geometryMap_interior_coeff_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_coeff_indices<functionspace::interior>(
                       std::declval<typename GeometryMap::eval_type>()));

  /// @brief Type of the coefficient indices of geometry type at the boundary
  using geometryMap_boundary_coeff_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_coeff_indices<functionspace::boundary>(
                       std::declval<
                           typename GeometryMap::boundary_eval_type>()));

  /// @brief Type of the coefficient indices of variable type in the interior
  using variable_interior_coeff_indices_type =
      decltype(std::declval<Variable>()
                   .template find_coeff_indices<functionspace::interior>(
                       std::declval<typename Variable::eval_type>()));

  /// @brief Type of the coefficient indices of variable type at the boundary
  using variable_boundary_coeff_indices_type =
      decltype(std::declval<Variable>()
                   .template find_coeff_indices<functionspace::boundary>(
                       std::declval<typename Variable::boundary_eval_type>()));
};
  
} // namespace iganet::v1
