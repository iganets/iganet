/**
   @file include/igabase.hpp

   @brief Isogeometric analysis base class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

namespace iganet {

  /// @brief Enumerator for the collocation point specifier
enum class collPts : short_t {
  greville = 0,         /*!< Greville points */
  greville_interior = 1 /*!< Greville points in the interior */
};

  /// @brief IgA base class
  ///
  /// This class implements the base functionality of IgA solvers and nets
  template <typename GeometryMap, typename Variable>
  class IgABase {
  public:
  /// @brief Value type
  using value_type =
      typename std::common_type<typename GeometryMap::value_type,
                                typename Variable::value_type>::type;

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

  protected:
  /// @brief Spline representation of the geometry map
  GeometryMap G_;

  /// @brief Spline representation of the reference data
  Variable f_;

  /// @brief Spline representation of the solution
  Variable u_;

  /// @brief Specifier for collocation points of the geometry map
  collPts geometryMap_collPts_;

  /// @brief Specifier for collocation points of the variables
  collPts variable_collPts_;

  private:
  /// @brief Constructor: number of spline coefficients (different for Geometry
  /// and Variable types)
  template <typename... GeometryMapSplines, size_t... Is,
            typename... VariableSplines, size_t... Js>
  IgABase(std::tuple<GeometryMapSplines...> geometryMap_splines,
         std::index_sequence<Is...>,
         std::tuple<VariableSplines...> variable_splines,
         std::index_sequence<Js...>,
         iganet::Options<value_type> options = iganet::Options<value_type>{})
      : // Construct the different spline objects individually
        G_(std::get<Is>(geometryMap_splines)..., init::greville, options),
        f_(std::get<Js>(variable_splines)..., init::zeros, options),
        u_(std::get<Js>(variable_splines)..., init::random, options),

        // Set collocation point specifiers
        geometryMap_collPts_(collPts::greville),
        variable_collPts_(collPts::greville) {}

public:
  /// @brief Default constructor
  explicit IgABase(
      iganet::Options<value_type> options = iganet::Options<value_type>{})
    : G_(), f_(), u_(),
        geometryMap_collPts_(collPts::greville),
        variable_collPts_(collPts::greville) {}

  /// @brief Constructor: number of spline coefficients (same for geometry map and
  /// variables)
  template <typename... Splines>
  IgABase(std::tuple<Splines...> splines,
         iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(splines, splines, options) {}

  /// @brief Constructor: number of spline coefficients (different for
  /// geometry map and variables)
  template <typename... GeometryMapSplines, typename... VariableSplines>
  IgABase(std::tuple<GeometryMapSplines...> geometryMap_splines,
         std::tuple<VariableSplines...> variable_splines,
         iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(geometryMap_splines,
               std::make_index_sequence<sizeof...(GeometryMapSplines)>{},
               variable_splines,
               std::make_index_sequence<sizeof...(VariableSplines)>{},
               options) {}

  /// @brief Returns a constant reference to the spline
  /// representation of the geometry map
  inline const GeometryMap &G() const { return G_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the geometry map
  inline GeometryMap &G() { return G_; }

  /// @brief Returns a constant reference to the spline
  /// representation of the reference data
  inline const Variable &f() const { return f_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the reference data
  inline Variable &f() { return f_; }

  /// @brief Returns a constant reference to the spline
  /// representation of the solution
  inline const Variable &u() const { return u_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the solution
  inline Variable &u() { return u_; }

  /// @brief Sets the collocation point specifier for the geometry map
  /// and returns the specifier
  inline enum collPts geometryMap_collPts(enum collPts collPts) {
    geometryMap_collPts_ = collPts;
    return collPts;
  }

  /// @brief Sets the collocation point specifier for the variables
  /// and returns the specifier
  inline enum collPts variable_collPts(enum collPts collPts) {
    variable_collPts_ = collPts;
    return collPts;
  }

  /// @brief Returns the collocation point specifier for the geometry map
  inline enum collPts geometryMap_collPts() const {
    return geometryMap_collPts_;
  }

  /// @brief Returns the collocation point specifier for the variables
  inline enum collPts variable_collPts() const { return variable_collPts_; }

private:
  /// @brief Returns the geometry map collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <size_t... Is>
  geometryMap_collPts_type
  geometryMap_collPts(std::index_sequence<Is...>) const {
    geometryMap_collPts_type collPts;

    switch (geometryMap_collPts_) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
        std::get<Is>(G_).greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = std::get<Is>(G_.boundary()).greville()),
       ...);
      break;

    case collPts::greville_interior:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts.first) =
        std::get<Is>(G_).greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = std::get<Is>(G_.boundary()).greville()),
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
  template <size_t... Is>
   variable_collPts_type variable_collPts(std::index_sequence<Is...>) const {
     variable_collPts_type collPts;

    switch (variable_collPts_) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
        std::get<Is>(f_).greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = std::get<Is>(f_.boundary()).greville()),
       ...);
      break;

    case collPts::greville_interior:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts.first) =
        std::get<Is>(f_).greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts.second) = std::get<Is>(f_.boundary()).greville()),
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
  virtual geometryMap_collPts_type geometryMap_collPts(int64_t epoch) const {
    if constexpr (GeometryMap::dim() == 1)

      switch (geometryMap_collPts_) {

      case collPts::greville:
        return {G_.greville(/* interior */ false), G_.boundary().greville()};

      case collPts::greville_interior:
        return {G_.greville(/* interior */ true), G_.boundary().greville()};

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }

    else
      return geometryMap_collPts(
          std::make_index_sequence<GeometryMap::dim()>{});
  }

  /// @brief Returns the variable collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  virtual variable_collPts_type variable_collPts(int64_t epoch) const {
    if constexpr (Variable::dim() == 1)

      switch (variable_collPts_) {

      case collPts::greville:
        return {f_.greville(/* interior */ false), f_.boundary().greville()};

      case collPts::greville_interior:
        return {f_.greville(/* interior */ true), f_.boundary().greville()};

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }

    else
      return variable_collPts(std::make_index_sequence<Variable::dim()>{});
  }
    
  };

} // namespace iganet
